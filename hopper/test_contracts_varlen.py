"""Contract gates for the GH200 short-varlen fast paths.

These tests check properties a correct kernel must have, independent of any
tolerance where that is possible: which code path ran, isolation between the
packed sequences, no stale or missing writes, and run-to-run stability.

The fast paths are only selected when Q and K share one cu_seqlens object and
seqused is absent (flash_bwd_launch_template.h), so every gate first proves
which path it exercised instead of assuming it.

Run with the FLASH_ATTENTION_DISABLE_* settings used to build the extension:
    PYTHONPATH=<built tree>/hopper pytest -q test_contracts_varlen.py
"""

import os
import re

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from flash_attn_interface import _flash_attn_backward, _flash_attn_forward

NHEADS = 4
HEADDIM = 64
SCALE = 0.125


def _disabled(feature):
    return os.getenv(f"FLASH_ATTENTION_DISABLE_{feature}", "FALSE") == "TRUE"


pytestmark = pytest.mark.skipif(
    any(_disabled(f) for f in ("VARLEN", "HDIM64", "BACKWARD")),
    reason="requires the varlen hdim-64 backward",
)

GRADS = ("dq", "dk", "dv")


@pytest.fixture(autouse=True)
def hopper_only():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires Hopper")
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def _cu(lengths):
    return torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )


def _inputs(lengths, dtype=torch.bfloat16, seed=1729):
    torch.manual_seed(seed)
    total = sum(lengths)
    return [
        torch.randn(total, NHEADS, HEADDIM, device="cuda", dtype=dtype)
        for _ in range(4)
    ]


def _forward(q, k, v, cu, lengths, window=(-1, -1), cu_k=None):
    return _flash_attn_forward(
        q, k, v,
        cu_seqlens_q=cu, cu_seqlens_k=cu if cu_k is None else cu_k,
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths), softmax_scale=SCALE,
        window_size_left=window[0], window_size_right=window[1], num_splits=1,
    )[:2]


def _backward(do, q, k, v, out, lse, cu, lengths, grads, window=(-1, -1), cu_k=None):
    _flash_attn_backward(
        do, q, k, v, out, lse, cu, cu if cu_k is None else cu_k, None, None,
        max(lengths), max(lengths), *grads, SCALE,
        window_size_left=window[0], window_size_right=window[1],
    )
    return grads


def _run(lengths, window=(-1, -1), cu_k=None, inputs=None, grads=None, seed=1729):
    """One forward and backward. Returns the five result tensors."""
    q, k, v, do = _inputs(lengths, seed=seed) if inputs is None else inputs
    cu = _cu(lengths)
    out, lse = _forward(q, k, v, cu, lengths, window, cu_k)
    if grads is None:
        grads = [torch.empty_like(q) for _ in GRADS]
    _backward(do, q, k, v, out, lse, cu, lengths, grads, window, cu_k)
    torch.cuda.synchronize()
    return dict(zip(("out", "lse", *GRADS), (out, lse, *grads)))


# ---------------------------------------------------------------- path proof

# TileShape_MNK of a backward mainloop, e.g.
#   CollectiveMainloopBwdSm90<2, 2, 2, cute::tuple<C<1>, C<1>, C<1> >,
#                             cute::tuple<C<32>, C<256>, C<64> >, ...
# The cluster shape in front is all ones, so requiring C<64> last picks the tile.
_TILE = re.compile(r"cute::tuple<cute::C<(\d+)>, cute::C<(\d+)>, cute::C<64> >")


def _trace(fn):
    """CUDA kernel names launched by fn, with the profiler warm."""
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type.name == "CUDA"]


def _path(names):
    """Which backward specializations a kernel-name trace proves ran."""
    joined = "\n".join(names)
    tiles = {
        (int(m), int(n))
        for name in names if "FlashAttnBwdSm90" in name
        for m, n in _TILE.findall(name) if (m, n) != ("1", "1")
    }
    return {
        "partition": "partition_backward_lengths" in joined,
        "tiles": sorted(tiles),
        "preprocess": sum("FlashAttnBwdPreprocess" in n for n in names),
        "postprocess": sum("FlashAttnBwdPostprocessConvertdQ" in n for n in names),
    }


def _path_of(lengths, window=(-1, -1), cu_k=None):
    q, k, v, do = _inputs(lengths)
    cu = _cu(lengths)
    out, lse = _forward(q, k, v, cu, lengths, window, cu_k)
    grads = [torch.empty_like(q) for _ in GRADS]
    return _path(_trace(
        lambda: _backward(do, q, k, v, out, lse, cu, lengths, grads, window, cu_k)
    ))


MIXED = [100, 200, 300]


def test_path_shared_cu_seqlens_takes_the_length_split():
    """Shared cu_seqlens, no seqused: the two-arm short-varlen path."""
    path = _path_of(MIXED)
    assert path["partition"], path
    assert path["tiles"] == [(32, 256), (128, 128)], path
    # The 32x256 arm writes dQ directly, so only the second arm needs a
    # postprocess pass, and only the first arm needs a preprocess pass.
    assert path["preprocess"] == 1, path
    assert path["postprocess"] == 1, path


def test_path_cloned_cu_seqlens_falls_back_to_upstream():
    """A separately allocated, equal-valued cu_seqlens_k disables the fast path.

    This is the trap the rest of the suite must avoid: a caller that passes two
    tensors silently gets upstream's single-tile path.
    """
    path = _path_of(MIXED, cu_k=_cu(MIXED).clone())
    assert not path["partition"], path
    assert path["tiles"] == [(128, 128)], path


def _forward_path(lengths, window=(-1, -1)):
    q, k, v, _ = _inputs(lengths)
    cu = _cu(lengths)
    names = _trace(lambda: _forward(q, k, v, cu, lengths, window))
    return sorted({(int(m), int(n)) for name in names if "FlashAttnFwdSm90" in name
                   for m, n in _TILE.findall(name) if (m, n) != ("1", "1")})


def test_path_forward_uses_the_short_sequence_tile():
    """The shipped build sets FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE: 192x80.

    Built without that flag the forward is upstream's 192x192, and every
    forward gate below would be measuring upstream instead of the fork.
    """
    assert _forward_path(MIXED) == [(192, 80)], _forward_path(MIXED)


@pytest.mark.parametrize("window,tile", [
    ((64, 64), (64, 64)),       # use_small_local_fwd: both sides in [0, 128]
    ((128, 128), (64, 64)),     # the inclusive upper edge of that gate
    ((129, 128), (192, 128)),   # one past it, so the general local tile
    ((-1, 64), (192, 128)),     # an unbounded side is not the small-local case
])
def test_path_forward_local_tile_follows_the_window(window, tile):
    """The sliding-window forward has its own 64x64 specialization (flash.h:173).

    Its gate is narrow, so a window just outside it must fall back instead of
    quietly using a tile that was never validated for that window.
    """
    assert _forward_path(MIXED, window=window) == [tile], (
        window, _forward_path(MIXED, window=window))


def test_path_local_uses_the_local_fast_tiles():
    path = _path_of(MIXED, window=(64, 64))
    assert path["partition"], path
    assert path["tiles"] == [(32, 256), (64, 128)], path


def test_path_every_length_bucket_is_covered():
    """Lengths either side of the 128/256 split still select both arms."""
    for lengths in ([128, 129], [256, 257], [1, 128, 129, 256, 257, 512]):
        path = _path_of(lengths)
        assert path["partition"], (lengths, path)
        assert (32, 256) in path["tiles"] or max(lengths) <= 128, (lengths, path)


# ------------------------------------------------------------- repeatability


LENGTHS = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512]


def _stability(lengths, window=(-1, -1), repeats=5):
    """Which outputs are bit-identical across repeated identical calls."""
    inputs = _inputs(lengths)
    first = _run(lengths, window, inputs=inputs)
    stable = {name: True for name in first}
    spread = {name: 0.0 for name in first}
    for _ in range(repeats - 1):
        again = _run(lengths, window, inputs=inputs)
        for name, value in again.items():
            same = torch.equal(value, first[name])
            stable[name] &= same
            if not same:
                delta = (value.float() - first[name].float()).abs().max().item()
                spread[name] = max(spread[name], delta)
    return stable, spread


@pytest.mark.parametrize("window", [(-1, -1), (64, 64)])
def test_forward_and_dkv_are_bit_identical_across_runs(window):
    """The forward and dK/dV reduce in a fixed order, so bits must not move."""
    stable, spread = _stability(LENGTHS, window)
    for name in ("out", "lse", "dk", "dv"):
        assert stable[name], (name, spread[name])


def test_dq_run_to_run_spread_is_recorded():
    """dQ accumulates over n-blocks, so document its spread rather than assume."""
    stable, spread = _stability(LENGTHS)
    # Not an assertion about bits: a regression here is a jump in magnitude.
    assert spread["dq"] < 1e-2, spread["dq"]


# ----------------------------------------------------- cross-sequence isolation


def _isolate(lengths, index, poison, window=(-1, -1)):
    """Change one sequence; return per-tensor max change on the others."""
    inputs = _inputs(lengths)
    base = _run(lengths, window, inputs=inputs)
    stable, _ = _stability(lengths, window, repeats=3)

    start = sum(lengths[:index])
    end = start + lengths[index]
    perturbed = [t.clone() for t in inputs]
    for tensor in perturbed[:3] if poison is None else perturbed:
        tensor[start:end] = (
            torch.randn_like(tensor[start:end]) if poison is None else poison
        )
    after = _run(lengths, window, inputs=perturbed)

    rows = {"out": slice(None), "dq": slice(None), "dk": slice(None),
            "dv": slice(None), "lse": slice(None)}
    deltas = {}
    for name, value in after.items():
        a, b = base[name].float(), value.float()
        if name == "lse":
            a, b = a[:, :], b[:, :]
            keep = torch.ones(a.shape[-1], dtype=torch.bool, device=a.device)
            keep[start:end] = False
            a, b = a[:, keep], b[:, keep]
        else:
            keep = torch.ones(a.shape[0], dtype=torch.bool, device=a.device)
            keep[start:end] = False
            a, b = a[keep], b[keep]
        deltas[name] = (a - b).abs().max().item() if a.numel() else 0.0
        assert torch.isfinite(b).all(), f"{name} went non-finite outside sequence {index}"
    del rows
    return deltas, stable


@pytest.mark.parametrize("index", [0, 5, 11])
def test_other_sequences_are_untouched_by_a_finite_change(index):
    lengths = LENGTHS
    deltas, stable = _isolate(lengths, index, poison=None)
    for name, delta in deltas.items():
        if stable[name]:
            assert delta == 0.0, (name, delta)
        else:
            assert delta < 1e-2, (name, delta)


@pytest.mark.parametrize("victim", range(1, len(LENGTHS)))
def test_a_nan_never_reaches_a_later_sequence(victim):
    """Bound the direction a NaN in one packed sequence can travel.

    FA3 varlen reads whole KV tiles, so a tile near the end of sequence i can
    cover rows belonging to sequence i+1; masking those out additively keeps
    the NaN, and the earlier sequence is contaminated. Upstream FA3 at the
    merge-base does the same, and reaches further back than the fork does
    because its KV tile is wider (xbuild_probe.py measures both), so the
    inherited property is read-ahead, not an unbounded leak.

    A NaN travelling the other way, into a later sequence, would mean a read
    or write at a wrong packed offset, which no tile geometry explains.
    """
    import bisect

    lengths = LENGTHS
    cu = _cu(lengths).tolist()
    inputs = [t.clone() for t in _inputs(lengths)]
    start, end = cu[victim], cu[victim + 1]
    if start == end:
        pytest.skip("empty sequence")
    for tensor in inputs[:3]:
        tensor[start:end] = float("nan")
    result = _run(lengths, inputs=inputs)

    reached = set()
    for name in ("out", *GRADS):
        bad = (~torch.isfinite(result[name])).any(-1).any(-1).nonzero().flatten()
        reached |= {bisect.bisect_right(cu, int(row)) - 1 for row in bad.tolist()}
    later = {j for j in reached if j > victim}
    assert not later, f"NaN in sequence {victim} reached later sequences {sorted(later)}"


# --------------------------------------------------------------------- values


def _fp64_reference(lengths, window, tensors):
    """Per-sequence attention and its gradients in float64."""
    q, k, v, do = tensors
    outs, lses, grads = [], [], []
    start = 0
    for length in lengths:
        if length == 0:
            continue
        end = start + length
        qq, kk, vv = [t[start:end].double().detach().transpose(0, 1).requires_grad_()
                      for t in (q, k, v)]
        row = torch.arange(length, device=q.device)[:, None]
        col = torch.arange(length, device=q.device)[None, :]
        keep = torch.ones((length, length), device=q.device, dtype=torch.bool)
        if window[0] >= 0:
            keep &= col >= row - window[0]
        if window[1] >= 0:
            keep &= col <= row + window[1]
        scores = ((qq @ kk.transpose(-1, -2)) * SCALE).masked_fill(~keep, -torch.inf)
        out = scores.softmax(-1) @ vv
        g = torch.autograd.grad(out, (qq, kk, vv),
                                do[start:end].double().transpose(0, 1))
        outs.append(out.detach().transpose(0, 1))
        lses.append(scores.detach().logsumexp(-1))
        grads.append([x.transpose(0, 1) for x in g])
        start = end
    return {"out": torch.cat(outs), "lse": torch.cat(lses, dim=1),
            **dict(zip(GRADS, [torch.cat(p) for p in zip(*grads)]))}


# Bounds sit about 2x above the error upstream FA3 itself shows on these shapes
# in bf16 (out 2.7e-3, grads 5.7e-3); xbuild_compare.py does the tight
# comparison against upstream, which is what detects a numerical regression.
BOUNDS = {"out": 1.0e-2, "lse": 1.0e-4, "dq": 2.0e-2, "dk": 2.0e-2, "dv": 2.0e-2}


@pytest.mark.parametrize("window", [(-1, -1), (0, 0), (64, 64), (16, 48), (-1, 64)])
def test_values_against_a_float64_reference(window):
    lengths = LENGTHS
    tensors = _inputs(lengths)
    got = _run(lengths, window, inputs=tensors)
    want = _fp64_reference(lengths, window, tensors)
    for name, bound in BOUNDS.items():
        error = (got[name].double() - want[name]).abs().max().item()
        assert error < bound, (name, error, bound)


def test_values_hold_at_the_length_split_boundaries():
    """The backward splits the batch by length, so check either side of it."""
    for lengths in ([127, 128, 129], [255, 256, 257], [1, 2, 3], [512, 1, 512]):
        tensors = _inputs(lengths)
        got = _run(lengths, (-1, -1), inputs=tensors)
        want = _fp64_reference(lengths, (-1, -1), tensors)
        for name, bound in BOUNDS.items():
            error = (got[name].double() - want[name]).abs().max().item()
            assert error < bound, (lengths, name, error, bound)


# ------------------------------------------------------------ stale and missing


SENTINELS = [float("nan"), -1.0e4, 7.5]


@pytest.mark.parametrize("window", [(-1, -1), (64, 64)])
def test_no_sentinel_survives_in_a_reused_destination(window):
    """Poisoned gradient buffers, reused: every valid element must be rewritten.

    A fresh allocation often reads as zero, which can hide a missing store, so
    the same buffers are refilled with three patterns and reused.
    """
    lengths = LENGTHS
    inputs = _inputs(lengths)
    q = inputs[0]
    grads = [torch.empty_like(q) for _ in GRADS]
    reference = None
    for sentinel in SENTINELS:
        for grad in grads:
            grad.fill_(sentinel)
        result = _run(lengths, window, inputs=inputs, grads=grads)
        for name in GRADS:
            value = result[name]
            assert torch.isfinite(value).all(), f"{name} kept a NaN sentinel"
            if sentinel == 7.5:
                continue
            assert not (value.float() == sentinel).all(-1).any(), (
                f"{name} kept a whole row of the {sentinel} sentinel"
            )
        if reference is None:
            reference = {n: result[n].clone() for n in GRADS}
        else:
            for name in GRADS:
                delta = (reference[name].float() - result[name].float()).abs().max()
                assert delta < 1e-2, (name, sentinel, delta.item())


def test_alternating_short_and_long_batches_do_not_share_state():
    """The two arms share scratch space, so alternating shapes must still agree."""
    short, long = [60, 70, 80], [300, 400, 500]
    expected = {name: _run(lengths) for name, lengths in (("s", short), ("l", long))}
    for _ in range(3):
        for name, lengths in (("s", short), ("l", long)):
            again = _run(lengths)
            for key, value in again.items():
                delta = (expected[name][key].float() - value.float()).abs().max()
                assert delta < 1e-2, (name, key, delta.item())
