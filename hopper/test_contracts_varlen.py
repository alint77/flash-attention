"""Contract gates for the GH200 short-varlen fast paths.

These tests check properties a correct kernel must have, independent of any
tolerance where that is possible: which code path ran, isolation between the
packed sequences, no stale or missing writes, and run-to-run stability.

The design follows Shah and Shrestha, "A Contract-Grade Verifier for
LLM-Generated GPU Kernels, and a Native Blackwell Backward for the
Gated-Linear-Recurrence Family" (preprint, 2026; https://github.com/RishiShah99/lethe),
which operationalises the Kernel Contracts taxonomy of Veit (arXiv:2604.22032,
2026): prefer gates no tolerance can explain away, grade against a
higher-precision reference, and keep a red team of deliberately broken builds so
that a passing suite means something. Here the load-bearing comparison is
relative, against upstream FA3 at the commit this fork branched from, since the
claim under test is "faster, not different"; see docs/protein_varlen_gh200.md
Part 6 for the measured numbers and the gaps that remain.

The fast paths are only selected when Q and K share one cu_seqlens object and
seqused is absent (flash_bwd_launch_template.h), so a caller that passes two
equal-valued tensors silently gets upstream's kernel. The gates under "path
proof" below pin that dispatch down by reading the launched kernel names back.
They establish which specialization ran for a given call shape; they do not
prove which sequences or tiles it processed, and the value gates do not re-check
their own dispatch. Tile-visit accounting would need a debug build.

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


# The device-side partition sends 128 < len <= 256 to the 32x256 direct-dQ arm
# and everything else to the 128x128 arm, so these three batches put work in one
# arm, the other, or both (partition_backward_lengths in
# flash_bwd_launch_template.h). An all-rest batch also leaves the fast arm with
# no work at all, which is its own case.
ALL_FAST = [129, 200, 256]
ALL_REST = [64, 300, 512]
BOTH_ARMS = [64, 129, 300]


@pytest.mark.parametrize("lengths", [
    ALL_FAST, ALL_REST, BOTH_ARMS,
    [128, 129],     # either side of the lower split
    [256, 257],     # either side of the upper split
    [128], [257], [1],
])
def test_path_both_backward_arms_always_launch(lengths):
    """Both launches happen for every batch, even when one bucket is empty.

    The partition writes zeros into the `seqused` array it hands each arm, so an
    arm with no work still launches and walks empty tiles rather than being
    skipped on the host. That makes the launch cost fixed, and it also means
    kernel names cannot show *which* sequences an arm processed: this gate is
    dispatch only. Proving that every sequence lands in exactly one arm exactly
    once needs per-tile visit counting in a debug build, which this suite does
    not have (see docs/protein_varlen_gh200.md Part 6).
    """
    path = _path_of(lengths)
    assert path["partition"], (lengths, path)
    assert path["tiles"] == [(32, 256), (128, 128)], (lengths, path)


# ------------------------------------------------------------- repeatability


LENGTHS = [0, 1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512]
# Around the new 80-column KV tile and the 192-row Q tile, and their multiples.
TILE_EDGES = [79, 80, 81, 159, 160, 161, 191, 192, 193, 239, 240, 320, 400]
SEEDS = (1729, 20261013, 7)

# What each output is expected to do across identical runs. This is fixed by the
# design, not inferred from the run: dK and dV have one owner per element, the
# forward reduces in a fixed order, and dQ is the only output that can still be
# summed over n-blocks with atomics (mainloop_bwd_sm90_tma_gmma_ws.hpp:983).
EXACT = ("out", "lse", "dk", "dv")
APPROXIMATE = ("dq",)


def _stability(lengths, window=(-1, -1), repeats=5, seed=1729):
    """Max run-to-run movement of each output over every pair of repeats."""
    inputs = _inputs(lengths, seed=seed)
    runs = [_run(lengths, window, inputs=inputs) for _ in range(repeats)]
    for run in runs:
        for name, value in run.items():
            assert torch.isfinite(value).all(), f"{name} is not finite"
    spread = {}
    for name in runs[0]:
        worst = 0.0
        for later in runs[1:]:
            if not torch.equal(later[name], runs[0][name]):
                worst = max(worst, (later[name].double()
                                    - runs[0][name].double()).abs().max().item())
        spread[name] = worst
    return spread


@pytest.mark.parametrize("window", [(-1, -1), (64, 64)])
@pytest.mark.parametrize("lengths", [LENGTHS, ALL_FAST, ALL_REST])
def test_forward_and_dkv_are_bit_identical_across_runs(window, lengths):
    """The forward and dK/dV reduce in a fixed order, so bits must not move."""
    spread = _stability(lengths, window)
    for name in EXACT:
        assert spread[name] == 0.0, (name, spread[name])


def test_dq_spread_stays_small():
    """dQ can be summed over n-blocks, so bound its movement instead of its bits.

    Upstream FA3 shows 4.9e-4 here and the fork about 1e-6, because direct
    stores replace most of the atomic accumulation; xbuild_compare.py is what
    holds the fork to upstream's figure. This bound only catches a blow-up.
    """
    assert _stability(LENGTHS)["dq"] < 1e-2


# ----------------------------------------------------- cross-sequence isolation


def _isolate(lengths, index, which, window=(-1, -1), seed=1729):
    """Change one sequence's `which` inputs; max change on the other sequences."""
    inputs = _inputs(lengths, seed=seed)
    base = _run(lengths, window, inputs=inputs)

    start = sum(lengths[:index])
    end = start + lengths[index]
    perturbed = [t.clone() for t in inputs]
    for slot in which:
        perturbed[slot][start:end] = torch.randn_like(perturbed[slot][start:end])
    after = _run(lengths, window, inputs=perturbed)

    deltas = {}
    for name, value in after.items():
        a, b = base[name].double(), value.double()
        assert torch.isfinite(b).all(), f"{name} went non-finite"
        axis = -1 if name == "lse" else 0
        keep = torch.ones(a.shape[axis], dtype=torch.bool, device=a.device)
        keep[start:end] = False
        a, b = (a[..., keep], b[..., keep]) if name == "lse" else (a[keep], b[keep])
        deltas[name] = (a - b).abs().max().item() if a.numel() else 0.0
    return deltas


# Sequence 7 of LENGTHS is 129 long, so it is the one victim that belongs to the
# direct-dQ arm; 5 and 11 are rest-arm victims and 1 is a single token.
@pytest.mark.parametrize("index", [1, 5, 7, 11])
@pytest.mark.parametrize("which,label", [((0, 1, 2), "qkv"), ((3,), "dout")])
def test_other_sequences_are_untouched_by_a_finite_change(index, which, label):
    """Tolerance-free isolation: the expectation is fixed, not read off the run.

    dQ is allowed to move because its n-block sum can reorder; everything else
    has one owner per element and must not move a bit.
    """
    deltas = _isolate(LENGTHS, index, which)
    for name in EXACT:
        assert deltas[name] == 0.0, (label, name, deltas[name])
    for name in APPROXIMATE:
        assert deltas[name] < 1e-2, (label, name, deltas[name])


@pytest.mark.parametrize("lengths", [ALL_FAST, ALL_REST, BOTH_ARMS])
def test_isolation_holds_within_each_arm(lengths):
    """Both backward arms, and a batch that splits across the two."""
    for index in range(len(lengths)):
        deltas = _isolate(lengths, index, (0, 1, 2))
        for name in EXACT:
            assert deltas[name] == 0.0, (lengths, index, name, deltas[name])


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


def _check_values(lengths, window, seed=1729):
    """Compare against float64 built from inputs snapshotted *before* the call.

    Taking the snapshot first matters: a kernel or wrapper that modified its own
    input would otherwise be graded against the modified input and agree with
    itself. The snapshot doubles as an immutability check.
    """
    tensors = _inputs(lengths, seed=seed)
    before = [t.clone() for t in tensors]
    got = _run(lengths, window, inputs=tensors)
    for name, (now, then) in zip(("q", "k", "v", "dout"), zip(tensors, before)):
        assert torch.equal(now, then), f"the call modified its own {name}"
    want = _fp64_reference(lengths, window, before)
    errors = {name: (got[name].double() - want[name]).abs().max().item()
              for name in BOUNDS}
    for name, bound in BOUNDS.items():
        assert errors[name] < bound, (lengths, window, name, errors[name], bound)
    return errors


@pytest.mark.parametrize("window", [(-1, -1), (0, 0), (64, 64), (16, 48), (-1, 64)])
@pytest.mark.parametrize("seed", SEEDS)
def test_values_against_a_float64_reference(window, seed):
    _check_values(LENGTHS, window, seed)


@pytest.mark.parametrize("lengths", [
    [127, 128, 129], [255, 256, 257], [1, 2, 3], [512, 1, 512],
    ALL_FAST, ALL_REST, BOTH_ARMS, TILE_EDGES,
])
def test_values_hold_at_every_boundary(lengths):
    """Either side of the length split, and around the 80 and 192 tile edges."""
    _check_values(lengths, (-1, -1))
    _check_values(lengths, (64, 64))


@pytest.mark.parametrize("lengths", [LENGTHS, ALL_FAST, ALL_REST])
def test_a_zero_window_has_an_exact_answer(lengths):
    """With window (0, 0) each query attends to itself alone.

    Then out == V, dV == dout and dQ == dK == 0 exactly, with no reduction to
    reorder and no softmax to approximate, so this needs no tolerance at all.
    """
    tensors = _inputs(lengths)
    q, k, v, do = tensors
    got = _run(lengths, (0, 0), inputs=tensors)
    assert torch.equal(got["out"], v), "out should be V"
    assert torch.equal(got["dv"], do), "dV should be dout"
    for name in ("dq", "dk"):
        worst = got[name].double().abs().max().item()
        assert worst < 1e-5, (name, worst)


# ------------------------------------------------------------ stale and missing


SENTINELS = [float("nan"), -1.0e4, 7.5]


@pytest.mark.parametrize("window", [(-1, -1), (64, 64)])
@pytest.mark.parametrize("lengths", [LENGTHS, ALL_FAST, ALL_REST])
def test_no_sentinel_survives_in_a_reused_destination(window, lengths):
    """Poisoned gradient buffers, reused: every valid element must be rewritten.

    A fresh allocation often reads as zero, which can hide a missing store, so
    the same buffers are refilled with three patterns and reused. Passing this
    shows the elements were written, not that they were written correctly; the
    float64 gates above are what check the values.
    """
    inputs = _inputs(lengths)
    grads = [torch.empty_like(inputs[0]) for _ in GRADS]
    reference = None
    for sentinel in SENTINELS:
        for grad in grads:
            grad.fill_(sentinel)
        result = _run(lengths, window, inputs=inputs, grads=grads)
        for name in GRADS:
            value = result[name]
            assert torch.isfinite(value).all(), f"{name} kept a NaN sentinel"
            if sentinel != 7.5:
                assert not (value.float() == sentinel).any(), (
                    f"{name} kept the {sentinel} sentinel")
        if reference is None:
            reference = {n: result[n].clone() for n in GRADS}
        else:
            for name in GRADS:
                delta = (reference[name].double() - result[name].double()).abs().max()
                assert delta < 1e-2, (name, sentinel, delta.item())


@pytest.mark.parametrize("first,second", [
    (ALL_FAST, ALL_REST),     # one arm idle, then the other
    (ALL_REST, ALL_FAST),
    (ALL_FAST, BOTH_ARMS),
    ([60, 70, 80], [300, 400, 500]),   # both in the rest arm, different shapes
])
def test_alternating_batches_do_not_share_state(first, second):
    """The two arms reuse scratch and the scheduler semaphore between launches.

    Alternating an all-fast batch with an all-rest one leaves each arm idle in
    turn, which is where a missed semaphore reset or stale scratch shows up.
    """
    expected = {0: _run(first), 1: _run(second)}
    for _ in range(3):
        for index, lengths in enumerate((first, second)):
            again = _run(lengths)
            for key, value in again.items():
                delta = (expected[index][key].double() - value.double()).abs().max()
                assert delta < 1e-2, (lengths, key, delta.item())


def test_results_do_not_depend_on_being_profiled_or_synchronized():
    """Queue several calls with different shapes before synchronizing once.

    Every other gate synchronizes after each call, and the path gates run under
    the profiler; both can hide a race. Here the launches are left to overlap.
    """
    cases = [ALL_FAST, ALL_REST, BOTH_ARMS, LENGTHS]
    expected = [_run(lengths) for lengths in cases]
    pending = []
    for lengths in cases:
        q, k, v, do = _inputs(lengths)
        cu = _cu(lengths)
        out, lse = _forward(q, k, v, cu, lengths)
        grads = [torch.empty_like(q) for _ in GRADS]
        _backward(do, q, k, v, out, lse, cu, lengths, grads)
        pending.append(dict(zip(("out", "lse", *GRADS), (out, lse, *grads))))
    torch.cuda.synchronize()
    for lengths, want, got in zip(cases, expected, pending):
        for key in want:
            delta = (want[key].double() - got[key].double()).abs().max()
            assert delta < 1e-2, (lengths, key, delta.item())
