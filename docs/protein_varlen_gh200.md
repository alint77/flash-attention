# FlashAttention-3 varlen tuning for protein language models (SM90 / GH200)

This document describes a change to the FlashAttention-3 backward pass on Hopper: replacing
the non-persistent `SingleTileScheduler` with a **persistent n-block varlen tile scheduler**
for the non-causal varlen case. It covers the motivation, the measurements, the two bugs that
make the naive version silently wrong, and the conditions under which the change is a
regression rather than a win.

## Summary

Two changes to the FlashAttention-3 varlen path, both scoped to non-causal `headdim <= 64`:

| | upstream | this fork | speedup |
|---|---|---|---|
| forward | 142 TFLOP/s | **173 TFLOP/s** | **+21.8%** |
| backward | 130 TFLOP/s | **141 TFLOP/s** | **+8.4%** |
| **fwd + bwd** | **132 TFLOP/s** | **149 TFLOP/s** | **+12.5%** |

![protein varlen benchmark](assets/protein_varlen_gh200.png)

Protein-LM regime: 65,536 tokens/pass, ~347 sequences, median length 163, mean 187, max ~900
(lognormal, UniRef-like), H=16, D=64, bf16, non-causal. Mean of 3 seeds, 60 timed iterations
each after 20 warmup.

Hardware: a single JUPITER **GH200** (Grace-Hopper, 132 SMs, 228 KB smem/SM, 680 W enforced
cap, measured 3.64 TB/s HBM copy, ~605 TFLOP/s achieved bf16 GEMM), CUDA 13, PyTorch 2.12.
That is a machine balance of **~166 FLOP/byte**, against ~250 for an H100 SXM — the 680 W cap
is a ~24% compute derate with no bandwidth penalty, leaving the part ~35% more bandwidth-rich
per FLOP.

> **The original hypothesis, and why it was wrong.** This investigation started from the
> observation that `hopper/tile_size.h` is commented *"benchmarked on H100 SXM"*, and the guess
> that a machine with a ~35% different compute/bandwidth ratio would want a different tile.
> **It does not.** At seqlen 8192, 37 configurations were swept on GH200 and none beat the H100
> table, because FA3 sits at ~97% of the achievable GEMM ceiling there — there is no roofline
> headroom to exploit. The gains documented below are *workload-shape* effects (tile
> quantization and empty CTAs on short ragged sequences), not hardware-balance effects, and
> would likely reproduce on an H100. No H100 control was available, so **nothing in this
> document is claimed as GH200-specific.**

Ablated, so the two changes can be judged separately:

| variant | forward | backward | fwd+bwd |
|---|---|---|---|
| upstream | 142 | 130 | 132 |
| + persistent bwd | 142 (−0.1%) | 141 (**+8.5%**) | 141 (+6.5%) |
| + persistent bwd + short-seq fwd tiles | 173 (**+21.8%**) | 141 (+8.4%) | 149 (**+12.5%**) |

TFLOP/s; higher is better.

The two are independent: the forward flag does not touch the backward, and vice versa.

**Correctness.** Both changes produce gradients identical to the stock kernel
(`dQ` 0.0156 / `dK` 0.0078 / `dV` 0.0078 max abs error vs per-sequence fp32 SDPA — the same
error the stock kernel has), verified up to 1,200 tiles. 30 steps of single-GPU pretraining
give loss identical to 4 decimal places at every step and **bit-identical eval loss**.

---

## Part 1 — Short-sequence forward tiles (opt-in)

`FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE` switches the `headdim <= 64` non-causal forward from
upstream's `{192, 192, RS=false, IntraWGOverlap=true}` to `{192, 80, RS=true,
IntraWGOverlap=false}`, and raises the KV pipeline from `kStages=2` to `3`.

A per-axis sweep at the protein regime (m=192, RS=true, overlap=false, stages=2) shows a smooth
unimodal optimum in the KV tile width:

```
tile_n   32     48     64     80     96    112    128    160    192
TFLOP/s  152.5  167.9  173.7  177.2  175.1  172.4  171.9  161.3  148.8
                              ^^^^^
```

`IntraWGOverlap=false` wins everywhere (+1 to +7 TF/s), `tile_m` 192 > 128 > 64
(175 / 168 / 143), and `kStages` 3 > 4 > 2 (~+1.5 TF/s).

The mechanism is **scheduler load balance across ragged sequences**, not padding arithmetic:
`tile_n` 64, 96 and 192 pad a median-169 sequence identically, yet differ by 25 TF/s.
`tile_n=192` gives exactly one KV iteration and no pipelining (worst, 148.8); the optimum sits
around three iterations. Shared-memory bank conflicts are *not* the constraint — the fastest
config has the *most* conflicts (53.4%) and `tile_n=64` the fewest (18%) while being slower.

### Why it is opt-in: the crossover

This is a *tile-quantization* win and it inverts at long sequence length. D=64 non-causal,
fixed-length varlen, 65,536 tokens per pass:

| seqlen | upstream | short-seq tiles | speedup |
|---|---|---|---|
| 128 | 97.9 TF/s | 118.9 | **+21.4%** |
| 256 | 129.5 | 158.4 | **+22.3%** |
| 512 | 258.0 | 268.0 | +3.9% |
| 1024 | 316.3 | 325.0 | +2.8% |
| 2048 | 379.7 | 375.4 | −1.1% |
| 4096 | 430.5 | 402.9 | −6.4% |
| 8192 | 463.4 | 418.7 | −9.6% |
| 16384 | 445.2 | 399.1 | **-10.4%** |

Break-even is near seqlen 1500. Upstream's own comment two lines below the tile table says as
much — *"Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen"* —
so this is not a bug in upstream's choice, it is a different operating point.

`tile_size_fwd_sm90()` has no `seqlen` parameter and its result becomes *template* parameters,
so the tile is baked per `(headdim, causal, ...)` at compile time. Gating on sequence length
properly would need two instantiations of the kernel plus a host-side dispatch. Until then the
flag defaults to upstream behaviour so nothing regresses silently. `kStages` is tied to the same
flag because a deeper pipeline costs shared memory for *every* SM90 forward config, not just
this one.

---

## Part 2 — Persistent backward scheduler (default on)

### Motivation

The target workload is masked-LM pretraining of a ModernBERT-style encoder, where sequences
are packed varlen with a **mean length around 190 and a long tail**:

```
H = 16 heads, D = 64, 65,536 tokens/batch, 337 sequences
mean seqlen 194, max 815, lognormal(log 176, 0.5)
```

Profiling one fwd+bwd iteration of this shape (total 1.746 ms) gave:

| kernel | ms | share |
|---|---|---|
| `FlashAttnBwdSm90` (main backward) | 0.966 | 55.4% |
| forward recompute | 0.453 | 26.0% |
| `BwdPreprocess` | 0.185 | 10.6% |
| `BwdPostprocessConvertdQ` | 0.136 | 7.8% |

The main backward kernel dominates, and its SASS stall profile **inverts** the forward's:

| stall source | forward | backward |
|---|---|---|
| `MUFU.EX2` (softmax) | 18.5% | 2.0% |
| branch / sync | negligible | **38.3%** |
| `BRA` | — | **25.8%** |
| GMMA | 5.6% | 14.2% |
| global memory | — | 1.7% |

The forward is math-bound on softmax transcendentals. The backward is bound on control flow.

A gate test isolated the cause. Holding the kernel completely fixed and only raising the
median sequence length **176 → 704** moved `BRA` **26.0% → 14.4%** and GMMA **14.6% → 24.4%**.
So the control-flow cost is *per tile* and amortises over longer loop bodies — the short-sequence
regime pays it disproportionately.

---

### What is actually being wasted

The backward's `SingleTileScheduler` launches a **rectangular grid**:

```
grid = ceil_div(max_seqlen_k, kBlockN) × batch × num_heads
```

Every sequence gets as many n-blocks as the *longest* sequence in the batch. Sequences shorter
than the maximum produce CTAs where `m_block_max <= m_block_min`; those CTAs run their prologue,
call `epilogue.store_zero()`, and exit.

For the workload above, with `kBlockN = 128` and `max_seqlen_k = 815` (7 n-blocks):

```
launched CTAs        37,744     (7 × 337 × 16)
real non-empty tiles 10,816     (28.7%)
empty CTAs           26,928     (71.3%)
```

**71% of the grid does no work.** Nsight Compute confirms the launched grid exactly:
`Grid Size = 37,744`, `Waves Per SM = 285.94`.

A persistent scheduler enumerates only the real tiles across a resident grid of one CTA per SM:
**132 CTAs × ~82 tiles each, 1 wave**.

---

### The change

The port is far smaller than it first appears, because most of the machinery already exists
upstream:

* `flash_bwd_kernel_sm90.h` already wraps the whole body in a
  `get_initial_work / is_valid / get_next_work` loop — `SingleTileScheduler` simply runs it once.
* `mainloop_bwd_sm90_tma_gmma_ws.hpp` already defines
  `NumProducerThreads = cutlass::NumThreadsPerWarp * 2` (warp 0 loads K/V/Q/dO, warp 1 runs
  `store_dq`), which is exactly the participant count a persistent scheduler needs.
* `mma()` already tracks the `barrier_KV` phase as `work_idx % 2` and increments `work_idx`.
* The `BwdNamedBarriers::KVEmpty` handshake exists, commented out, with the note:
  *"We're not currently using this bc we're not using persistent scheduler."*

#### Reusing the forward's scheduler over n-blocks

`VarlenDynamicPersistentTileScheduler`'s first template parameter is named `kBlockM`, but it is
only ever used as `ceil_div(seqlen, ·)`. **Passing `kBlockN` makes it decompose over n-blocks** —
no new scheduler and no new metadata kernel are required. The backward's `scheduler_args` already
passes `cu_seqlens_k` and `seqlen_k`, and `Prepared = false` computes block counts from
`cu_seqlens` on the fly.

```cpp
// hopper/flash_bwd_launch_template.h
using SchedulerPersistentBwd = flash::VarlenDynamicPersistentTileScheduler<
    kBlockN, kBlockM, CollectiveMainloop::NumMmaThreads, CollectiveMainloop::NumProducerThreads,
    false /*Split*/, false /*PackGQA*/, true /*WarpSpecialized*/,
    false /*LPT*/, false /*Sort*/, false /*Prepared*/>;

static constexpr bool UsePersistentBwd =
    (Arch >= 90) && Varlen && !Is_causal && !Is_local && !GQA;
```

Barrier IDs do not collide: the scheduler uses cutlass *reserved* barriers
(`StreamkBarrier0/1` → hardware 4, 5), while `BwdNamedBarriers` are user barriers offset by
`ReservedNamedBarrierCount = 8` → hardware 8–15.

#### The work counter

The persistent scheduler's `prefetch_next_work` does
`atomicAdd(params.tile_count_semaphore, 1)`. The backward never allocated one — the lines were
commented out. Both API files need it:

```cpp
// hopper/flash_api_stable.cpp   (and the same in flash_api.cpp)
Tensor tile_count_semaphore = torch::stable::new_zeros(
    q, {1}, std::make_optional(torch::headeronly::ScalarType::Int));
params.tile_count_semaphore = static_cast<int*>(tile_count_semaphore.data_ptr());
```

> **Watch out.** The build uses **`flash_api_stable.cpp`**, not `flash_api.cpp`. Patching only the
> latter compiles and links fine, then faults at runtime with
> `Invalid __global__ atomic of size 4 bytes … Access to 0x0 is out of bounds` inside
> `prefetch_next_work`. Compute-sanitizer names the host frame — read it.

---

### The trap: `TensorStorage` is a union

This is the bug worth remembering, because it produces **silently wrong gradients** and passes
compute-sanitizer cleanly.

`flash_bwd_kernel_sm90.h` declares:

```cpp
struct TensorStorage : cute::aligned_struct<128> {
    union {
        typename CollectiveMainloop::TensorStorage mainloop;   // sQ, sK, sV, sdO, sdS …
        typename CollectiveEpilogue::TensorStorage epilogue;   // sdK, sdV
    };
} tensors;
```

The epilogue's `sdK`/`sdV` **alias the entire mainloop storage**. With a single tile per CTA this
never mattered. With a persistent loop there are two independent races:

1. the producer TMAs K/V/**Q**/dO for work tile `t+1` into memory the epilogue of tile `t` is
   still writing, and
2. the epilogue of tile `t` writes `sdK`/`sdV` over `sQ`/`sK`/`sV`/`sdO` that tile `t+1` needs.

The natural-looking fix — release the buffers at the end of `mma()` — is **wrong**, because the
epilogue runs *after* `mma()` returns. It compiles, runs, reports zero sanitizer errors, and
returns corrupted `dK`/`dV`.

The correct placement is exactly where upstream's commented-out code put it:

| site | action |
|---|---|
| `mma_init()` | `NamedBarrier::arrive(NumMmaThreads + 32, KVEmpty)` — initial credit |
| `load()`, **top, before the Q/dO TMA** | `NamedBarrier::sync(NumMmaThreads + 32, KVEmpty)` |
| `epilogue_bwd.hpp::store()`, after smem is drained | `arrive(NumEpilogueThreads + 32, KVEmpty)` |

Two details matter:

* The producer wait must precede the **Q** TMA, not just the K/V TMA. `sQ` is in the same union.
  Placing it between them (where it superficially belongs, next to the K/V load) still corrupts.
* Varlen takes the **non-TMA/STG** epilogue branch (`Use_TMA = !Varlen && ...`), so the arrive
  goes after the smem→register `flash::copy`, preceded by `fence_view_async_shared()`. The TMA
  branch releases after `tma_store_wait<0>()`.

Credit accounting stays balanced across empty tiles because both sides skip them: the producer
returns before its `sync` when `m_block_max <= m_block_min`, and `store_zero()` does not arrive.

---

### Testing: why the first round of correctness tests proved nothing

Gradients were checked against per-sequence PyTorch SDPA. The initial tests all passed on the
broken version, because **they never exercised persistence at all**: with fewer tiles than SMs,
every CTA receives exactly one work tile and the persistent path is identical to the single-tile
path.

Sweeping the tile count located the bug in a single run:

| tiles | result |
|---|---|
| 32 | correct |
| 132 (= `num_sm`) | correct |
| **160** | **wrong** (15/40 sequences bad) |
| 272 | wrong (37/68) |
| 1200 | wrong (292/300) |

None of the bad sequences had *zero* gradient, which ruled out a tile-mapping bug (missing tiles)
and pointed at state corruption between tiles.

> **Rule:** any test for a persistent scheduler must launch more tiles than the GPU has SMs.

After the fix, the persistent kernel matches the `SingleTileScheduler` control exactly:

```
ctrl   out=0.0078  dQ=0.0156  dK=0.0078  dV=0.0078
pers   out=0.0078  dQ=0.0156  dK=0.0078  dV=0.0078
```

(bf16 tolerances vs fp32 SDPA reference; identical to the stock kernel's own error.)

End-to-end, 30 steps of single-GPU pretraining: **loss identical to 4 dp at every logged step,
`eval_loss` bit-identical (2.9355 / 2.8995), no NaN**. `grad_norm` differs in the 5th digit
(3.7059 vs 3.7060), as expected from a different dK/dV reduction order.

---

### Measurements

#### Main backward kernel (Nsight Compute)

| metric | `SingleTileScheduler` | persistent |
|---|---|---|
| grid | 37,744 CTAs | **132** |
| waves per SM | 285.94 | **1** |
| duration | 1.18 ms | **1.00 ms (1.18× faster)** |
| **instructions executed** | 231.4 M | **191.5 M (−17.2%)** |
| cycles | 1.540 M | 1.249 M (−18.9%) |
| compute (SM) throughput | 32.8% | 40.4% |
| memory throughput | 44.5% (1.27 TB/s) | 54.0% (1.51 TB/s) |
| executed IPC | 1.19 | 1.17 |
| warp cycles / issued inst | 8.14 | 8.56 |
| registers/thread | 168 | 168 |
| achieved occupancy | 15.14% | 15.63% |

The win is **not** better per-instruction efficiency. IPC is flat and per-warp stalls are
marginally *worse*. The kernel simply executes 17% fewer instructions, because 71% of the CTAs
it used to launch existed only to write zeros.

#### Wall-clock, isolated benchmark

Per fwd+bwd iteration, all FA kernels, median of alternating repeats:

| kernel | ctrl | persistent |
|---|---|---|
| main `FlashAttnBwdSm90` | 0.981 ms | **0.859 ms (1.14× faster)** |
| forward recompute | 0.403 | 0.415 |
| `BwdPreprocess` | 0.186 | 0.186 |
| `BwdPostprocessConvertdQ` | 0.137 | 0.137 |
| **total** | **1.712 ms** | **1.603 ms (1.07× faster)** |

##### The change is a *dispersion* win, not a short-sequence win

Superseded by the variance sweep at the top of this document, which pins the mean length and is
measured in throughput. Summary: the backward's speedup rises monotonically with the spread of
the length distribution, from −1% at zero variance to +37% at CV 1.83, tracking the empty-CTA
fraction.

#### End-to-end

Single-GPU nanoPLM pretraining, warm compile cache, step time at step 20:

```
ctrl  {340.61, 339.49} ms
pers  {339.02, 338.52} ms      →  −0.38%
```

The persistent build is faster in both repeats, but the difference (1.28 ms) is comparable to
the spread within the control arm alone (1.12 ms). **Read this as "no regression, plausibly a
small win", not as a measured speedup.** Attention is a modest fraction of a step whose GEMMs
run in fp8; the kernel-level speedup is the result that is actually resolved.

---

---

## Variance sweep: separating the two mechanisms

The cleanest experiment for telling the two changes apart. Token budget fixed at 65,536 and the
**arithmetic mean sequence length pinned to 200 for every point** (lognormal `mu` is set to
`log(200) - sigma^2/2`, so only the *spread* changes, never the mean). Two seeds, forward and
backward timed separately.

![variance sweep](assets/variance_sweep.png)

| lognormal σ | 0.00 | 0.15 | 0.30 | 0.45 | 0.60 | 0.80 | 1.00 | 1.20 |
|---|---|---|---|---|---|---|---|---|
| coeff. of variation | 0.02 | 0.15 | 0.30 | 0.47 | 0.66 | 0.96 | 1.35 | 1.83 |
| max length | 200 | 314 | 482 | 724 | 1062 | 1711 | 2648 | 3937 |
| tile fill | 100% | 68% | 51% | 33% | 22% | 14% | 9% | 6% |
| forward, upstream | 83 | 108 | 119 | 136 | 149 | 177 | 212 | 258 |
| forward, fork | 122 | 146 | 160 | 161 | 177 | 207 | 241 | 283 |
| **forward speedup** | **+47%** | +36% | +34% | +19% | +19% | +17% | +14% | **+10%** |
| backward, upstream | 125 | 116 | 118 | 122 | 129 | 144 | 164 | 184 |
| backward, fork | 124 | 116 | 120 | 129 | 145 | 174 | 209 | 253 |
| **backward speedup** | **−1%** | +1% | +2% | +5% | +12% | +21% | +28% | **+37%** |

All figures TFLOP/s; higher is better.

The two curves are near mirror images, and that is the whole story of this fork:

* **Forward — a short-sequence effect.** Best at zero variance (+47%), decaying to +10% as the tail grows. The narrow 192x80 KV tile wins because a 200-token sequence
  against a 192-wide tile wastes most of the tile; as sequences lengthen, upstream's wide tile
  amortises better and the advantage erodes. This is tile quantisation.
* **Backward — a dispersion effect.** Break-even at zero variance (−1%), climbing monotonically to +37%. Nothing here depends on sequences being short: it depends on them
  being *unequal*, because that is what fills the rectangular grid with empty CTAs. This is
  scheduling.

They are independent and compose. A protein corpus sits near sigma 0.55 (CV ~0.6, ~22% fill),
where the forward still returns ~+19% and the backward ~+12%.

Note the backward's −0.7% at sigma=0 here is milder than the −7.2% in the uniform-length table
below, because this sweep pins the mean at 200 (two KV blocks per sequence) while that one also
tests seqlen 128 (a single KV block), where the per-tile handshake overhead has the least work
to hide behind.

---

## Where this fork is slower

Both changes are operating-point trades. Measured against upstream on **uniform-length** varlen
(every sequence the same length, so tile fill is 100% and there are no empty CTAs to remove),
65,536 tokens per pass, D=64 non-causal, percent change vs upstream — negative is slower:

![uniform length regression](assets/uniform_len_regression.png)

| uniform seqlen | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| forward, upstream | 100 | 131 | 256 | 288 | 382 | 422 | 450 | 447 |
| forward, fork | 122 | 161 | 272 | 306 | 359 | 396 | 415 | 407 |
| **forward speedup** | **+22.3%** | +22.2% | +6.0% | +6.4% | −5.9% | −6.2% | −7.8% | **−9.0%** |
| backward, upstream | 112 | 190 | 281 | 338 | 407 | 441 | 450 | 464 |
| backward, fork | 104 | 185 | 276 | 332 | 380 | 420 | 433 | 445 |
| **backward speedup** | **−6.7%** | −2.8% | −1.8% | −1.9% | −6.6% | −4.7% | −3.9% | −4.2% |

All figures TFLOP/s; higher is better.

Absolute forward throughput for the same sweep (TFLOP/s):

| seqlen | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| upstream | 100.0 | 131.5 | 256.2 | 287.6 | 381.8 | 421.6 | 449.9 | 447.5 |
| short-seq tiles | 122.4 | 160.7 | 271.6 | 306.1 | 359.3 | 395.6 | 415.0 | 407.2 |

### The backward's regression is about tile *fill*, not sequence length

This is the part that is easy to get wrong. The persistent backward is **not** a "short
sequence" optimisation — it is an "empty CTA" optimisation, and uniform-length batches have no
empty CTAs at any length. The variance sweep at the top of this document isolates exactly this:
holding the mean length fixed at 200 and varying only the spread moves the backward from −1% to
+37%, monotonically with tile fill.

**Practical guidance:** if your dataloader length-buckets, sorts by length, or pads to a fixed
length, tile fill will be high and you should use upstream. This fork targets packed varlen with
a genuinely ragged length distribution, which is what protein corpora look like.

### The honest fix, not implemented

Both changes should be selected at runtime from the actual fill ratio, which the host already
knows — it has `cu_seqlens` and it computes `num_blocks_n`, so `real_tiles / rectangular_grid`
costs nothing to evaluate. A threshold near 0.5 would capture both wins and avoid both
regressions. For the backward this is straightforward (the scheduler is chosen host-side). For
the forward it is harder: `tile_size_fwd_sm90()` result becomes *template* parameters, so it
needs two kernel instantiations plus a dispatch.

---

## Status and limitations

* Gated at compile time to `Arch >= 90 && Varlen && !Is_causal && !Is_local && !GQA`.
  Causal keeps `SingleTileBwdLPTScheduler`; everything else keeps `SingleTileScheduler`.
* **The gate is wrong in principle.** The benefit depends on the *runtime* fill ratio, which the
  host already knows (it has `cu_seqlens` and computes `num_blocks_n`). A runtime switch on
  something like `real_tiles / rectangular_grid < ~0.5` would capture the win without the
  high-fill regression. That is the obvious next step and is not implemented here.
* The empty-tile path under the persistent scheduler is nearly unreachable by construction
  (per-batch block counts are exact), so `store_zero()` is effectively dead code there. It is
  left in place and the credit accounting handles it, but it is **not** covered by the tests
  above.
* Deterministic mode, GQA, split-KV and paged KV are untouched and still use the stock path.
* The forward flag (`FLASH_ATTENTION_SHORT_SEQ_TILES`) is **off by default** and regresses
  seqlen >= 2k; see the crossover table in Part 1. It affects only `headdim <= 64` non-causal,
  but `kStages=3` under the same flag affects every SM90 forward config's shared-memory budget.
* Only tested for `headdim = 64`, bf16, SM90. Other head dims should work — nothing in the
  change is head-dim specific — but they have not been run.

## Reproducing

```bash
# build (SM90, bf16, hdim64, varlen, fwd+bwd)
cd hopper
export FLASH_ATTN_CUDA_ARCHS=90
export FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE   # optional: short-seq forward tiles
for v in SPLIT PAGEDKV APPENDKV LOCAL SOFTCAP PACKGQA FP16 FP8 \
         HDIM96 HDIM128 HDIM192 HDIM256 HDIMDIFF64 HDIMDIFF192 SM80; do
  export FLASH_ATTENTION_DISABLE_$v=TRUE
done
python setup.py build_ext --inplace
```

Note that `setup.py` uses distutils' `newer_group()`, which compares `.cu` sources against the
output `.so` and **ignores headers**. Editing any `.h`/`.hpp` will silently rebuild nothing.
Force it:

```bash
rm -f build/temp.*/instantiations/*.o build/temp.*/*.o
rm -f build/lib.*/flash_attn_3/_C.abi3.so flash_attn_3/_C.abi3.so
touch instantiations/*.cu *.cpp
```

---

## Disclosure

The changes, benchmarks, profiling and this document were produced with
[Claude Code](https://claude.com/claude-code) running **Claude Opus 5** at high reasoning
effort, on the JUPITER cluster, under the repository owner's direction and review.

Every number here is from a real run on a GH200; none are estimated or extrapolated. Where a
result is weak or inside measurement noise it is labelled as such — see the end-to-end section,
where the effect is *not* separable from run-to-run variance. Where a hypothesis was falsified
it is recorded as falsified rather than dropped, including the one that motivated the entire
investigation.
