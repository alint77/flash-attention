"""Regression tests for the Hopper self-attention sliding-window fast paths.

Unlike test_flash_attn.py's general varlen tests, these calls share the exact same
cu_seqlens object for Q/K and omit seqused, making the optimized dispatch eligible.
Run with the same FLASH_ATTENTION_DISABLE_* settings used to build the extension.
"""

import os

import pytest
import torch

from flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    get_scheduler_metadata,
)


def _disabled(feature):
    return os.getenv(f"FLASH_ATTENTION_DISABLE_{feature}", "FALSE") == "TRUE"


pytestmark = pytest.mark.skipif(
    any(_disabled(feature) for feature in ("LOCAL", "VARLEN", "HDIM64")),
    reason="requires local varlen attention with head dimension 64",
)
DTYPES = [
    pytest.param(torch.bfloat16, id="bf16"),
    pytest.param(
        torch.float16,
        id="fp16",
        marks=pytest.mark.skipif(_disabled("FP16"), reason="FP16 disabled in build"),
    ),
]
LENGTHS = [
    pytest.param([0, 1, 63, 64, 65, 127, 128], id="short"),
    pytest.param([129, 191, 255, 256], id="middle"),
    pytest.param([257, 383, 511, 917], id="long"),
    pytest.param([0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 917], id="mixed"),
]
WINDOWS = [
    (0, 0), (64, 64), (127, 128), (128, 128), (129, 128),
    (128, 129), (512, 512), (-1, 64), (64, -1),
]


@pytest.fixture(autouse=True)
def hopper_fp32_reference():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _inputs(lengths, dtype):
    torch.manual_seed(1729)
    cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    tensors = [torch.randn(sum(lengths), 4, 64, device="cuda", dtype=dtype) for _ in range(4)]
    return cu, tensors


def _reference(q, k, v, do, lengths, window):
    """Dense FP32 attention and autograd, computed separately for each sequence."""
    outputs, lses, gradients = [], [], []
    start = 0
    for length in lengths:
        if length == 0:
            continue
        end = start + length
        qq, kk, vv = [
            x[start:end].float().detach().transpose(0, 1).requires_grad_()
            for x in (q, k, v)
        ]
        row = torch.arange(length, device=q.device)[:, None]
        col = torch.arange(length, device=q.device)[None, :]
        keep = torch.ones((length, length), device=q.device, dtype=torch.bool)
        if window[0] >= 0:
            keep &= col >= row - window[0]
        if window[1] >= 0:
            keep &= col <= row + window[1]
        scores = ((qq @ kk.transpose(-1, -2)) * 0.125).masked_fill(~keep, -torch.inf)
        out = scores.softmax(-1) @ vv
        grads = torch.autograd.grad(out, (qq, kk, vv), do[start:end].float().transpose(0, 1))
        outputs.append(out.detach().transpose(0, 1))
        lses.append(scores.detach().logsumexp(-1))
        gradients.append([g.transpose(0, 1) for g in grads])
        start = end
    return (
        torch.cat(outputs), torch.cat(lses, dim=1),
        [torch.cat(parts) for parts in zip(*gradients)],
    )


def _forward(q, k, v, cu, lengths, window, **kwargs):
    return _flash_attn_forward(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=kwargs.pop("cu_k", cu),
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), softmax_scale=0.125,
        window_size_left=window[0], window_size_right=window[1], num_splits=1, **kwargs,
    )[:2]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("lengths", LENGTHS)
@pytest.mark.parametrize("window", WINDOWS)
def test_local_varlen_partition_and_window_boundaries(dtype, lengths, window):
    cu, (q, k, v, do) = _inputs(lengths, dtype)
    ref_out, ref_lse, ref_grads = _reference(q, k, v, do, lengths, window)
    fwd_tol, bwd_tol = (0.02, 0.03) if dtype == torch.bfloat16 else (0.003, 0.004)
    # A distinct pointer with identical contents selects the general fallback.
    # Validate it independently, so agreement cannot hide a shared numerical error.
    for cu_k in (cu, cu.clone()):
        out, lse = _forward(q, k, v, cu, lengths, window, cu_k=cu_k)
        torch.testing.assert_close(out.float(), ref_out, atol=fwd_tol, rtol=fwd_tol)
        torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
        if _disabled("BACKWARD"):
            continue
        grads = [torch.empty_like(q) for _ in range(3)]
        # Reuse poisoned destinations to catch missed direct stores and stale state.
        for _ in range(3):
            for grad in grads:
                grad.fill_(float("nan"))
            _flash_attn_backward(
                do, q, k, v, out, lse, cu, cu_k, None, None, max(lengths), max(lengths),
                *grads, 0.125, window_size_left=window[0], window_size_right=window[1],
            )
            for actual, expected in zip(grads, ref_grads):
                torch.testing.assert_close(actual.float(), expected, atol=bwd_tol, rtol=bwd_tol)


@pytest.mark.parametrize("window", [(64, 64), (128, 128)])
def test_local_varlen_reused_scheduler_metadata(window):
    lengths = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 917]
    cu, (q, k, v, do) = _inputs(lengths, torch.bfloat16)
    used = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    metadata = get_scheduler_metadata(
        len(lengths), max(lengths), max(lengths), 4, 4, 64, used,
        cu_seqlens_q=cu, window_size=window, num_splits=1,
    )
    # The same metadata drives the default tile on every invocation, including
    # after fast-path calls and changes to Q/K/V. Shapes and lengths stay fixed.
    for _ in range(3):
        for tensor in (q, k, v):
            tensor.normal_()
        ref_out, ref_lse, _ = _reference(q, k, v, do, lengths, window)
        for kwargs in ({}, {"scheduler_metadata": metadata}):
            out, lse = _forward(q, k, v, cu, lengths, window, **kwargs)
            torch.testing.assert_close(out.float(), ref_out, atol=0.02, rtol=0.02)
            torch.testing.assert_close(lse, ref_lse, atol=1e-4, rtol=1e-4)
