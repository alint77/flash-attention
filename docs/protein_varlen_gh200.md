# FlashAttention-3 varlen tuning for protein language models (SM90 / GH200)

This document describes a change to the FlashAttention-3 backward pass on Hopper: replacing
the non-persistent `SingleTileScheduler` with a **persistent n-block varlen tile scheduler**
for the non-causal varlen case. It covers the motivation, the measurements, the two bugs that
make the naive version silently wrong, and the conditions under which the change is a
regression rather than a win.

**Contents**

1. [Summary](#summary)
2. [Part 1 — Short-sequence forward tiles (opt-in)](#part-1--short-sequence-forward-tiles-opt-in)
3. [Part 2 — Persistent backward scheduler (default on)](#part-2--persistent-backward-scheduler-default-on)
4. [Part 3 — Operating range: when this helps and when it hurts](#part-3--operating-range-when-this-helps-and-when-it-hurts)
5. [Status and limitations](#status-and-limitations)
6. [Reproducing](#reproducing)


## Summary

Three changes to the FlashAttention-3 varlen path, all scoped to non-causal `headdim <= 64`:

| | upstream | this fork | speedup |
|---|---|---|---|
| forward | 149 TFLOP/s | **184 TFLOP/s** | **+23.4%** |
| backward | 131 TFLOP/s | **164 TFLOP/s** | **+24.9%** |
| **fwd + bwd** | **137 TFLOP/s** | **172 TFLOP/s** | **+25.4%** |

![protein varlen benchmark](assets/protein_varlen_gh200.png)

Each panel shows only the variants that can affect that stage: the backward changes do not
touch the forward, and the short-seq tiles do not touch the backward.

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

Ablated, so the forward and backward halves can be judged separately:

| variant | forward | backward | fwd+bwd |
|---|---|---|---|
| upstream | 149 | 131 | 137 |
| + backward changes (default build) | 149 (−0.3%) | 164 (**+24.9%**) | 161 (+17.6%) |
| + short-seq fwd tiles (flag) | 184 (**+23.4%**) | 162 (+23.5%) | 172 (**+25.4%**) |

TFLOP/s; higher is better. The forward flag does not touch the backward, and vice versa.

**Correctness.** Gradients match the stock kernel (`dQ` 0.0156 / `dK` 0.0078 / `dV` 0.0078 max
abs error vs per-sequence fp32 SDPA — the same error the stock kernel has). Upstream's own
suite, `hopper/test_flash_attn.py`, passes on this branch: **1584 varlen and 720 non-varlen
cases**, headdim 64/96/128/192/256 and headdim_v 64/256/512, on both a default and a
`SHORT_SEQ_TILES` build.

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

End-to-end, 80 steps of single-GPU pretraining: **loss identical at 15 of 16 logged steps** (one
differs by 0.0001), no NaN. `grad_norm` differs in the 5th digit, as expected from a different
dK/dV reduction order. Timings in [Part 3](#end-to-end-training).

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

#### On its own, this change is a *dispersion* win

Measured alone, the persistent scheduler tracks the empty-CTA fraction: break-even at zero
variance, rising to +37% at CV 1.83. It needs sequences *unequal*, not short. The direct dQ
stores in Part 2b supply the short-sequence half — see the combined curve in Part 3.

#### End-to-end

See [Part 3](#end-to-end-training).

---

## Part 2b — Direct dQ stores for short sequences (default on)

### The idea

Upstream's backward always accumulates dQ through a three-pass detour, because in general many
CTAs contribute to the same Q rows: preprocess **clears** an FP32 `dQaccum` buffer, the main
kernel **atomically reduces** partial dQ into it, and a postprocess kernel **converts** it back
to bf16 and applies the softmax scale.

None of that is necessary when a sequence's whole KV range fits in one CTA's tile. Then that
CTA is the only writer, its dQ tile is already final, and it can convert and store bf16
straight from registers. The clear, the reduction and the postprocess pass all disappear for
that sequence.

Protein batches are almost entirely such sequences, which is why this is worth so much here.

### The change

`run_mha_bwd_hdim64` splits the batch on the GPU with a tiny partition kernel, then runs the
backward twice with different tiles:

| sequences | tile | dQ path |
|---|---|---|
| ≤ 128 tokens | M128/N128 (upstream's) | direct store — one KV block |
| 129 – 256 | M32/N256 | direct store — one KV block, wider |
| > 256 | M128/N128 | original FP32 accumulate + postprocess |

The 129–256 band needs the wider N256 tile so those sequences still have a single KV owner.
Both branches keep upstream's 128-row padded statistics layout, so `softmax_d` is unchanged.
Preprocess runs once for the whole batch and clears `dQaccum` only for sequences over 256;
postprocess skips everything at or under its branch's threshold. The partition metadata lives
in two batch-sized int vectors appended to the existing work-counter allocation — no
device-to-host copies.

Gated to: SM90, `headdim == 64`, varlen, non-causal, non-local, no softcap, non-deterministic,
MHA (`h == h_k`), identical non-null `cu_seqlens_q`/`cu_seqlens_k` pointers, no `seqused`
overrides. Everything else takes the original path.

### Where the time goes

CUDA trace, averaged over 50 backward calls on the protein batch:

| stage | before | after |
|---|---|---|
| preprocess | 184.7 µs | **122.5 µs** |
| main backward (both branches) | 827.7 µs | 794.3 µs |
| postprocess | 136.3 µs | **74.3 µs** |
| partition + counter reset | 0.8 µs | 3.0 µs |
| **total** | **1149.4 µs** | **994.2 µs** |

Read this honestly: the attention backward *kernel* got ~4% faster. The rest of the ~13% is
auxiliary traffic that short sequences never needed. The main loop is still the dominant cost
and this does not close the speed-of-light gap.

---

## Part 3 — Operating range: when this helps and when it hurts

### Variance sweep: separating the two mechanisms

The cleanest experiment for telling the forward and backward mechanisms apart. Token budget fixed at 65,536 and the
**arithmetic mean sequence length pinned to 200 for every point** (lognormal `mu` is set to
`log(200) - sigma^2/2`, so only the *spread* changes, never the mean). Two seeds, forward and
backward timed separately.

![variance sweep](assets/variance_sweep.png)

| lognormal σ | 0.00 | 0.15 | 0.30 | 0.45 | 0.60 | 0.80 | 1.00 | 1.20 |
|---|---|---|---|---|---|---|---|---|
| coeff. of variation | 0.02 | 0.15 | 0.30 | 0.47 | 0.66 | 0.96 | 1.35 | 1.83 |
| max length | 200 | 314 | 482 | 724 | 1062 | 1711 | 2648 | 3937 |
| tile fill | 100% | 68% | 51% | 33% | 22% | 14% | 9% | 6% |
| forward, upstream | 84 | 106 | 125 | 137 | 152 | 182 | 224 | 271 |
| forward, fork | 125 | 153 | 164 | 177 | 190 | 213 | 255 | 290 |
| **forward speedup** | **+49%** | +44% | +32% | +30% | +25% | +17% | +14% | **+7%** |
| backward, upstream | 125 | 116 | 120 | 124 | 131 | 145 | 166 | 186 |
| backward, fork | 159 | 154 | 152 | 157 | 168 | 192 | 227 | 266 |
| **backward speedup** | **+28%** | +33% | +27% | +27% | +28% | +33% | +37% | **+43%** |

All figures TFLOP/s; higher is better.

* **Forward — a short-sequence effect.** Best at zero variance (+49%), decaying to +7% as the
  tail grows. The narrow 192x80 KV tile wins because a 200-token sequence against a 192-wide
  tile wastes most of it; as sequences lengthen, upstream's wide tile amortises better. This
  is tile quantisation.
* **Backward — two effects, covering different halves of the range.** The U shape is the
  giveaway. Direct dQ stores need sequences *short*: with the mean pinned at 200, most
  sequences stay inside one CTA's KV range at every σ, which is why the curve starts at +28%
  instead of break-even. The persistent scheduler needs them *ragged*, which is why the curve
  bottoms out near σ 0.30 and then climbs to +43% as empty CTAs appear.

A protein corpus sits near σ 0.55 (CV ~0.6, ~22% fill), where the forward returns ~+25% and
the backward ~+28%.

---

### Where this fork is slower

Both changes are operating-point trades. Measured against upstream on **uniform-length** varlen
(every sequence the same length, so tile fill is 100% and there are no empty CTAs to remove),
65,536 tokens per pass, D=64 non-causal. Throughput, so higher is better:

![uniform length regression](assets/uniform_len_regression.png)

| uniform seqlen | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| forward, upstream | 102 | 134 | 259 | 273 | 385 | 412 | 442 | 444 |
| forward, fork | 126 | 165 | 280 | 307 | 355 | 395 | 411 | 402 |
| **forward speedup** | **+22.9%** | +22.9% | +8.2% | +12.6% | −7.9% | −4.1% | −7.1% | **−9.5%** |
| backward, upstream | 113 | 191 | 265 | 346 | 411 | 443 | 454 | 465 |
| backward, fork | 125 | 217 | 274 | 345 | 396 | 429 | 442 | 452 |
| **backward speedup** | **+10.6%** | +13.6% | +3.3% | −0.4% | −3.6% | −3.1% | −2.5% | **−2.8%** |

All figures TFLOP/s; higher is better.

#### The backward's crossover is near 1k

Uniform lengths are the worst case for the scheduler — 100% tile fill, no empty CTAs to
remove — so this table isolates the direct dQ store. Below ~1k a sequence fits inside one
CTA's KV range, the dQ tile is final, and skipping the FP32 clear / reduce / convert passes
is worth +11% to +14%. Above ~1k nothing qualifies, and what remains is the scheduler's extra
producer/epilogue handshake: a flat ~3%.

**Practical guidance:** if your sequences are long *and* uniform, use upstream. This fork
targets packed varlen with short, genuinely ragged sequences, which is what protein corpora
look like.

#### The honest fix, not implemented

Both changes should be selected at runtime from the actual fill ratio, which the host already
knows — it has `cu_seqlens` and it computes `num_blocks_n`, so `real_tiles / rectangular_grid`
costs nothing to evaluate. A threshold near 0.5 would capture both wins and avoid both
regressions. For the backward this is straightforward (the scheduler is chosen host-side). For
the forward it is harder: `tile_size_fwd_sm90()` result becomes *template* parameters, so it
needs two kernel instantiations plus a dispatch.

---

### End-to-end training

> **This measurement predates the direct dQ stores (Part 2b).** It covers the persistent
> scheduler and the forward flag only, when the combined attention gain was +12.5% rather than
> +25.4%. The step-time numbers below are therefore a floor, not the current figure. The
> reasoning in it — attention is a small share of the step, so the end-to-end effect is
> under 1% — is unchanged and is the part worth reading.

nanoPLM masked-LM pretraining, **single GPU, no FSDP, fp8 disabled**, 80 steps, warm compile
cache. All three arms run **sequentially on the same node** to remove node-to-node variance,
two repeats each, alternating.

| variant | median step | mean | min | speedup |
|---|---|---|---|---|
| upstream | 373.18 ms | 372.95 | 371.28 | — |
| + persistent bwd | 371.36 ms | 371.68 | 368.96 | **+0.49%** |
| + persistent bwd + short-seq fwd tiles | **369.78 ms** | 369.81 | 367.70 | **+0.91%** |

Loss is identical at 15 of the 16 logged steps (one differs by 0.0001, consistent with a
different dK/dV reduction order). The effect is small but *resolved*: the spread within an arm
is ~0.3 ms against a 3.4 ms gap between arms.

**Why under 1%, and why that is the expected answer.** Attention is only **7.8%** of a step in
this configuration (16 layers x 1.82 ms of a 373 ms step), so the 12.5% attention speedup measured at the time could not
buy more than ~0.9% end to end. Predicting the step time from the isolated kernel numbers:

| variant | predicted | measured | error |
|---|---|---|---|
| + persistent bwd | +0.47% | +0.49% | 0.02 pp |
| + both | +0.87% | **+0.91%** | 0.05 pp |

Agreement to within 0.05 percentage points says the kernel measurements are real and that
nothing else in the step regressed to absorb the gain.

**Do not read this as the end-to-end figure for a real training run.** It is one GPU with no
FSDP communication to overlap against, and fp8 is off, which slows the GEMMs and therefore
*shrinks* attention's share of the step. With fp8 enabled, or at a scale where attention is a
larger fraction, the same kernel speedup is worth proportionally more. A multi-GPU measurement
has not been made.


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
* The direct dQ store adds a second main-kernel launch and a device partition kernel, and
  mutates `params.seqused_*` across the two `run_flash_bwd` calls. On all-long workloads the
  extra launches are free within measurement noise (checked at uniform 512 / 1024 / 2048), but
  it is real added dispatch complexity.
* The forward flag (`FLASH_ATTENTION_SHORT_SEQ_TILES`) is **off by default** and regresses
  seqlen >= 2k; see the crossover table in Part 1. It affects only `headdim <= 64` non-causal,
  but `kStages=3` under the same flag affects every SM90 forward config's shared-memory budget.
* The fast paths are built and tuned for `headdim = 64`, bf16, SM90. Upstream's test suite
  passes for headdim 64/96/128/192/256, but the other head dims all take the original path —
  only D=64 is actually optimised here.

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

Every number here is from a real run on a GH200; none are estimated or extrapolated. Upstream's
own test suite is part of that: `hopper/test_flash_attn.py` passes on this branch across all
five compiled head dimensions. Where a
result is weak or inside measurement noise it is labelled as such — see the end-to-end section,
where the effect is *not* separable from run-to-run variance. Where a hypothesis was falsified
it is recorded as falsified rather than dropped, including the one that motivated the entire
investigation.
