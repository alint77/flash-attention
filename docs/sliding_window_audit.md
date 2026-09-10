# Audit — sliding-window varlen forward and backward

Auditor: separate session. Target: `sliding_window_forward.patch` (198 lines, 6 files) against
`fa_push` @ `d74a3c3`. Claim under audit: **21.3–21.9% less forward GPU time** on ragged
BF16/H16/D64 batches of 65,536 tokens with window 64/64.

## Verdict

**Both patches reproduce on the audited BF16/D64/Hopper workload.** Forward time falls 21.8%
and backward time 31.2%, measured on independent builds with each patch isolated. Within the
compiled configurations, pre-existing kernels that execute retain their machine code; the added
kernels are the intended ones. The selected upstream tests have unchanged outcomes, including
the known D96 failure. Maximum errors versus FP32 match at the reported precision in the
checked cases. These results do not establish coverage of features disabled in the builds.

**Three changes incorporated in the fork** — none affect the measured binary:

1. Added the `setmaxnreg` budget `static_assert` (section 2) to **both** `flash_fwd_kernel_sm90.h`
   and `flash_bwd_kernel_sm90.h`. This exact failure mode has now bitten three separate
   configurations across two experiments (`backward_design`'s 64/224, and both rejects here) and
   it presents as a silent hang, not a compile error.
2. Passed `false` explicitly at the `get_scheduler_metadata` call site (section 5), so
   producer/consumer agreement doesn't rest on `seqused_k` happening to be a mandatory argument.
3. Corrected the documentation: the four API call sites are inert, not load-bearing; the
   backward gate deliberately has no window bound while the forward gate is `[0, 128]`.

**Value to nanoPLM: about 1.3% less training step time on the measured `FSS` configuration**, with
10 of 16 layers sliding (section 10). An all-full `F` configuration has no eligible layers and
serves as a control. Its small measured slowdown is reported explicitly below.

---

Status: **complete — static audit and independent reproduction within the recorded build scope.**
Findings below are ordered by what a reviewer needs to decide adoption.

---

## 1. Provenance: verified

- All 7 hashes in `final_hashes.json` match, including the built `_C.abi3.so`.
- `git apply --check --whitespace=error` passes against `d74a3c3`.
- Applying the patch to a fresh `d74a3c3` export reproduces all six files **byte-identically**
  to `final/hopper/` — the tree that was benchmarked, profiled and hashed. Nothing was measured
  that isn't in the patch.

## 2. The "OPEN" hangs are closed — by arithmetic, not by experiment

`ledger_register_hang.md` leaves two deadlocks unexplained and flags that the accepted config
shares `MinBlocksPerMultiprocessor=2` with one of them. That was the single biggest risk to
adoption. It resolves cleanly.

The forward kernel's register handoff compiles to a **literal unbounded spin**
(`sass_final_main_0.txt:293`):

```
0x900: NOP
0x910: USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xa0   # math WG asks for 160 regs/thread
0x920: PLOP3.LUT P0, PT, PT, PT, UP0, 0x80, 0x0 # did it succeed?
0x930: @!P0 BRA 0x900                           # no -> retry, forever
```

If the CTA's register pool can never satisfy the request, the warpgroup spins at 100% SM
occupancy and never makes progress. That is exactly the reported signature: 78 s, no completed
case, GPU at 100%, 833 MiB resident.

The pool is `ptxas`'s reported registers/thread x block threads — **not** `65536 / minBlocks`,
because `ptxas` rounds the per-thread count *down* to warp granularity under `__launch_bounds__`.
Both rejected variants sized their request against the un-rounded number and lost one step:

| Variant | minBlocks | Threads | ptxas REG | Pool | `setmaxnreg` request | Margin |
|---|---:|---:|---:|---:|---|---:|
| `m128n64r112` (Mma 112 / Load 24) | 2 | 384 | 80 | 30,720 | 256x112 + 128x24 = 31,744 | **−1,024** |
| `m64n64r128b3` (Mma 128 / Load 40) | 3 | 256 | 80 | 20,480 | 128x128 + 128x40 = 21,504 | **−1,024** |
| **accepted** (Mma 160 / Load 56) | 2 | 256 | **128** | 32,768 | 128x160 + 128x56 = 27,648 | **+5,120** |

Both misses are exactly 1,024 registers = 8 regs/thread x 128 threads = one `setmaxnreg`
granularity step. Same mechanism as `backward_design`'s 64/224 timeout.

The accepted config is **not** near that boundary. `32768/256 = 128` divides exactly, so no
rounding is lost, and it holds 5,120 registers of slack — it would tolerate `ptxas` dropping to
112 regs/thread before it could hang. (The separate figure `128 x 256 x 2 = 65,536` is the
*occupancy* budget consuming the whole register file; that is by design and is not a fragility.)

**Implemented and verified in both directions.** The assert below is now in the push candidate.
It compiles clean across all 957 ptxas instantiations of the feature-restricted audit build
(flags listed in section 9), and a negative control
(forcing `MinBlocksPerMultiprocessor = 2` onto the stock configs, reproducing the rejected
variant's shape) fails at compile time with exactly the intended message:

```
flash_fwd_kernel_sm90.h(98): error: static assertion failed with
"setmaxnreg request exceeds the CTA register pool: the kernel would hang."
```

So it is neither vacuous nor over-tight. For calibration, these are the margins it enforces —
note how many upstream configs sit exactly on the line, which is why the rounding-down behaviour
matters so much:

| Config | Pool | Request | Margin |
|---|---:|---:|---:|
| fwd, 1 MMA WG | 63,488 | 39,936 | +23,552 |
| fwd, 2 MMA WG (TMA) | 64,512 | 64,512 | **0** |
| fwd, 2 MMA WG (non-TMA) | 64,512 | 64,512 | **0** |
| fwd, 3 MMA WG | 65,536 | 65,536 | **0** |
| fwd, SmallLocal (new) | 32,768 | 27,648 | +5,120 |
| bwd, 2 MMA WG | 64,512 | 64,512 | **0** |
| bwd, 3 MMA WG | 65,536 | 65,536 | **0** |
| *rejected* m128n64r112 | 30,720 | 31,744 | **−1,024** |
| *rejected* m64n64r128b3 | 20,480 | 21,504 | **−1,024** |

**Implemented guard.** This assert rejects both oversized register requests and passes the
accepted configuration, under the allocation assumption below:

```cpp
// ptxas rounds regs/thread down to a multiple of 8 under __launch_bounds__, and Hopper caps at 255.
static constexpr uint32_t kRegsPerThreadBudget =
    std::min(255u, 65536u / MinBlocksPerMultiprocessor / MaxThreadsPerBlock) & ~7u;
static_assert(MmaRegisterRequirement * (NumMmaWarpGroups * 128)
            + LoadRegisterRequirement * 128
            <= kRegsPerThreadBudget * MaxThreadsPerBlock,
              "setmaxnreg request exceeds the CTA register pool: the kernel will spin forever.");
```

This assumes `ptxas` allocates the full launch-bounds cap, which holds in every build in this
directory (255 / 128 / 80 each equal the cap) because a `setmaxnreg` kernel pins to it. If a
future build came in under the cap the assert would be optimistic, not wrong-way-round.

## 3. Grid x2 tile ownership: safe

`grid_dims.x *= 2` is sound for any grid size. `tile_scheduler.hpp:765` seeds each CTA with
`blockIdx.x`; `:786` hands out subsequent work as
`atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x)`. Ownership is unique regardless of
grid width. CTAs that draw no work exit immediately (`ISETP.GE` / `@P0 EXIT` right after the
register handoff), so surplus CTAs cost a launch, not a deadlock.

The semaphore is zeroed by the prepare kernel (`flash_prepare_scheduler.cu:77`), which runs on
the same stream ahead of the main kernel. It is skipped only when
`skip_scheduler_metadata_computation` is set — which the eligibility gate excludes. No garbage
read at 2x grid.

`Enable_cluster` requires `!Is_local`, so `ClusterM == 1` on this path and the doubled grid
cannot break cluster divisibility.

## 4. The device-side gate cannot misfire

`LocalSmallTile` in the kernel infers eligibility from the tile shape rather than from the
`SmallLocal` template argument, so it is worth checking it can't fire on an unintended kernel.
It requires `TileShape_MNK_PV == (64, 64, 64)`, i.e. `kBlockM == 64 && kHeadDimV == 64 &&
kBlockN == 64`. Scanning every BF16 return of `tile_size_fwd_sm90`, the only other `{64, 64}`
is `headdim <= 64 && headdim_v == 512`, whose PV tile is `(64, 512, 64)` — excluded by the
`kHeadDimV` term. FP8 is excluded outright. So `LocalSmallTile` is true iff `small_local` was
passed. Host and device agree.

## 5. Host tile queries and scheduler-metadata agreement

The original patch threaded `use_small_local_fwd(params)` through four `tile_size_fwd_sm90`
call sites in each API implementation. The helper was false at all four; the merged version
now passes `false` explicitly at the external-metadata producer:

| Call site | Why the helper is always false there |
|---|---|
| `get_pagedkv_tma` | early-returns before the call unless `page_table != nullptr`; the gate requires `!page_table` |
| `get_pack_gqa` | early-returns before the call unless `h != h_k`; the gate requires `h == h_k` |
| `get_num_splits` | invoked from `params.num_splits = ... get_num_splits(params) ...`, so `params.num_splits` is still 0; the gate requires `== 1` |
| `mha_fwd_get_scheduler_metadata` | the original helper was false because `seqused_k` is required; the merged code passes `false` explicitly |

The `get_num_splits` one has a mild consequence worth knowing: the split heuristic therefore
sizes itself with the 192x128 tile rather than 64x64. On any workload where it returns `> 1` the
gate's `num_splits == 1` term would then silently disable the fast path. It returns 1 for the
protein/nanoPLM regime — proven empirically, since the fast path demonstrably fires in the
reproduction below (outputs change at window 64/64 and the kernel is 21.8% faster).

This is harmless — those sites keep computing the upstream tile, which is what upstream did —
but the original experiment notes incorrectly described them as load-bearing. The call that
actually matters is
`flash_fwd_launch_template.h:188`, which already calls `prepare_varlen_num_blocks` with the
*templated* `kBlockM/kBlockN`, so the metadata the kernel consumes tracks `SmallLocal`
automatically.

The README's claim that external metadata "deliberately selects the existing kernel" is correct,
and is in fact enforced twice: the producer (`get_scheduler_metadata`) can never emit a 64/64
layout because `seqused_k` is mandatory, and the consumer (`mha_fwd`) can never select the
64/64 kernel because `skip_scheduler_metadata_computation` is set. A 64/64 metadata layout can
never be fed to a 192/128 kernel. The safety is real; it just doesn't come from these hunks.

**The fork now makes this agreement explicit.** Both API implementations pass `false` at the
external-metadata producer, with a comment explaining that metadata reuse selects the default
kernel. A future change to `seqused_k` eligibility therefore cannot silently change this
producer's tile. The other three inert sites retain the default tile as before.

## 6. Where the speedup comes from

Upstream picks a 192x128 tile for local D64. With window 64/64 most of each 128-wide KV tile is
masked out, so the baseline spends most of its MMA throughput on scores it immediately discards.
The ledger's own decomposition confirms the mechanism and shows both halves are necessary:

| Step | Time |
|---|---:|
| baseline M192/N128 | ~0.375–0.380 ms |
| M64/N64 tile alone (255 regs, 1 block/SM) | ~0.399 ms — *worse* |
| + 160 regs, 2 blocks/SM, 2x grid | ~0.304 ms |
| + 3 pipeline stages | ~0.298 ms |

Shrinking the tile alone regresses: at 255 regs/thread only one CTA is resident and achieved
occupancy is 7.83%. The register reduction is what converts the smaller tile into a win, by
restoring the occupancy the small tile gave up (15.65%, 0.31 → 0.67 eligible warps/scheduler).
Same shape of insight as the backward work: match the resource footprint to the actual window.


---

## 7. Scope: there are **two** new experiments, not one

`agent_space/sliding_window/` contains the separate **backward** experiment claiming
**29.5–32.6%** less backward time on the same ragged local batches. The forward work depends on
it: `sliding_forward/final/` — the tree that was built, hashed and benchmarked — is
`d74a3c3 + sliding_window_backward.patch + sliding_window_forward.patch`. I verified this by
diffing: `final/hopper` is byte-identical to a fresh `d74a3c3` with both patches applied (only
`setup.py` differs, a local edit replacing a `git submodule` call with an assert).

Their forward *baseline* (`sliding_window/baseline_full/`) is unpatched `d74a3c3`. So the
headline forward table is not a strict single-patch A/B — it is (fork) vs (fork + both patches).
Attribution still holds, because the backward patch touches only `flash_bwd_launch_template.h`,
`flash_bwd_postprocess_kernel.h` and `mainloop_bwd_sm90_tma_gmma_ws.hpp`, and `bench.py` calls
only `_flash_attn_forward`. My independent rebuild isolates the forward patch on its own.

Its provenance is equally clean: all 16 hashes in `sliding_window/final_hashes.json` verify,
`scoped_full/` (the tree behind the headline backward table) is source-identical to
`d74a3c3 + sliding_window_backward.patch`, and `baseline_full/` is source-identical to unpatched
`d74a3c3`. Only build artifacts (`.o`, `.so`, `build.ninja`) differ, as expected.

### The backward patch is the more delicate of the two

It is only 22 inserted lines, but they are load-bearing in a way the forward patch is not.

`run_mha_bwd_hdim64` splits the batch into two launches that **share one `dq_accum` buffer**:
sequences of length **129–256** on M32/N256, and lengths **<=128 or >256** on a second tile.
Both launches must pad that buffer
identically or the second one reads and writes at the wrong offsets — silent wrong gradients,
not a crash. In the non-local path both already pad to 128 (M32/N256 is special-cased to 128;
M128/N128 gets 128 from `kBlockM`). The new local remainder is **M64/N128**, which would
otherwise pad to 64. The patch therefore has to add the same `(Is_local && kBlockM==64 &&
kBlockN==128)` term in three places that must agree:

| Site | Consumer |
|---|---|
| `flash_bwd_launch_template.h:48` `QPad` | `total_q_padded_rounded`, preprocess tile, preprocess grid |
| `mainloop_bwd_sm90_tma_gmma_ws.hpp:63` `SeqlenInfo_t` | the mainloop's dQ-accum offsets |
| `flash_bwd_postprocess_kernel.h:173` (new `QPad` template arg) | postprocess's dQ-accum offsets |

All three now agree. The `Arch >= 90` term added to `QPad` is new relative to upstream but is
consistent with the mainloop (which is SM90-only by `static_assert`).

I checked the padding arithmetic holds for every length, not just the sampled ones. The local
remainder writes up to `ceil_div(s, 64) * 64` rows for a sequence of length `s`, into a region
sized `round_up(s, 128)`. Writing `s = 128a + r` with `0 < r <= 128`:
`ceil_div(s,64)*64 = 128a + 64*ceil(r/64) <= 128a + 128 = round_up(s,128)`. Never overruns, and
it is tight at `r = 65..128`. The postprocess grid is `ceil_div(seqlen_q, kBlockM=64)` with
`QPad = 128` offsets, matching the mainloop.

The local remainder also flips `dQ_swapAB` to `true` (the README's "transposed dQ MMA operands"),
and that same flag is threaded into `PostprocessKernel`, so the accumulator layout agrees on both
sides of the launch.

For direct dQ stores, `skip_dq_short = kBlockN` controls which sequences skip FP32 clear/convert.
The local remainder launch has `kBlockN = 128` and contains both <=128 and >256 lengths. Its
<=128 sequences use direct stores and skip clear/convert; >256 sequences use the accumulator
and postprocess. The 129–256 launch uses direct stores for its entire partition.

**Two questions raised by the static review**, subsequently checked in section 9:

1. `run_mha_bwd_hdim64`'s gate dropped `!params.is_local` **without adding any window-size
   bound**, unlike the forward gate's `[0, 128]`. Any local D64 varlen self-attention now takes
   the partitioned path at any window width. That is plausibly fine (the mask is applied by the
   existing local-mask code, and tile choice doesn't depend on window), but it is untested
   territory that the original sweeps at window 0/16/64/128/256 only partly covered.
2. `UsePersistentBwd` is now enabled for local. The n-block persistent scheduler must produce
   correct per-n-block m-ranges under a local mask. Floating-point dQ accumulation can vary
   with atomic-add ordering, and `Deterministic` stays excluded — this is
   the same class as the `TensorStorage` aliasing trap and deserves the gradient check, not a
   reading.

The independent `d74a3c3 + backward patch` rebuild and gradient checks are complete (section 9).

## 8. Does this help nanoPLM training?

Verified against the model code, not the config prose:

| Gate requirement | nanoPLM ModernBERT | |
|---|---|---|
| `d == dv == 64` | `hidden_size 1024 / 16 heads` = 64 | OK |
| BF16 | yes | OK |
| `h == h_k` (MHA) | `num_kv_heads` commented out | OK |
| same `cu_seqlens` pointer for q and k | `cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens` — same object (`modeling.py:614`) | OK |
| no `seqused_*` | not passed | OK |
| window in `[0, 128]` per side | `local_attention: 128` -> `sliding_window = local_attention // 2` = 64 -> `(64, 64)` (`config.py:328`) | OK |

So the fast path *would* fire — on sliding layers only. Which layers those are is set by
`attn_layer_pattern`:

- **The config I ran the 4-GPU end-to-end test with (`fa_e2e_nofp8_4gpu.yaml`) sets
  `attn_layer_pattern: 'F'` — all 16 layers full attention, zero sliding layers. This patch
  changes nothing at all in that run.**
- With the default `global_attn_every_n_layers: 3`, full attention lands on `i % 3 == 0`, so a
  16-layer model gets 6 full and **10 sliding** layers — those 10 would use both new kernels.

Section 10 reports the completed 4-GPU experiment with the `FSS` sliding pattern, measured
directly rather than extrapolated from kernel percentages.

One caveat if `use_paired_head_attention` is ever enabled (it is `false` in this config):
`_pair_flash_window_size` (`modeling.py:77`) doubles the window, so `(64, 64)` becomes
`(128, 128)` — still inside the gate, but exactly at its edge. `_pair_varlen_qkv`
(`modeling.py:504`) preserves `head_dim` and passes a single `paired_cu` object for both q and k,
so the other gate terms still hold. But `local_attention` above 256 would fall out of the fast
path silently, with no warning.

---

## 9. Independent reproduction

Fresh builds of `d74a3c3` (base), `d74a3c3 + forward patch` (cand) and
`d74a3c3 + backward patch` (cand_bwd), built by me from the patches with their build flags.
These are SM90 BF16 builds with local attention and backward enabled across the five tested
head dimensions. `SPLIT`, `PAGEDKV`, `APPENDKV`, `SOFTCAP`, `PACKGQA`, `FP16`, `FP8`, and `SM80`
are disabled (`sfaudit/build.sbatch` and `sfaudit/one.sbatch`). The compile and disassembly
claims below apply to this scope; they do not validate those omitted features. In particular,
the new local backward dispatch also accepts FP16 when built, but these runs did not test it.

Run on `jpbo-010-07`, a **900 W** GH200 (their runs were on a 680 W node), two fresh processes
per arm per seed, interleaved base/cand/base/cand on one pinned GPU. Absolute times therefore
differ slightly from their tables; the ratio is the comparable quantity.

**Forward — claim 21.3-21.9%:**

| Ragged seed | base (ms) | + forward patch (ms) | Time reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.3777 | 0.2950 | 21.91% | 1.281x |
| 1 | 0.3774 | 0.2970 | 21.30% | 1.271x |
| 2 | 0.3787 | 0.2945 | 22.24% | 1.286x |

Mean 21.82%. **Reproduces.** Note my `base` arm is unpatched `d74a3c3`, so this isolates the
forward patch on its own — their table bundled both patches. Same answer, so the attribution
in section 7 is confirmed empirically, not just by inspection.

**Backward — claim 29.5-32.6%:**

| Ragged seed | base (ms) | + backward patch (ms) | Time reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.2868 | 0.8823 | 31.43% | 1.458x |
| 1 | 1.3076 | 0.8770 | 32.93% | 1.491x |
| 2 | 1.2514 | 0.8843 | 29.33% | 1.415x |

Mean 31.23%. **Reproduces.**

### 9a. Closing the window-bound gap I flagged

Their backward validation stops at window 128/128, but the gate they relaxed has **no** window
bound. I ran 40 configurations against an independent FP32 reference on both arms: windows
`0/0, 0/1, 1/0, 64/64, 256/256, 512/512, 1024/1024, 2048/2048, 4096/4096, -1/-1` crossed with four
length sets — ragged lognormal, all-short (3..250), all-long (777..2048), and a "straddle" set
hitting every boundary of the 256-token partition and the M64/N128 remainder tiling
(1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 258, 383, 384, 385, 511, 512, 513, 1023, 1024, 1025).

Zero failures on both arms, and the patched arm's max gradient error equals the unpatched arm's
to five decimal places in **every** configuration (worst cand/base error ratio 1.00). The
checks found no regression over this sweep. Sequence lengths stop at 2048, so windows 2048
and 4096 normalize to full attention here; they do not test genuinely local 4096-wide windows
on longer sequences.

### 9b. Upstream suite

The unpatched baseline reproduces the D96 failure exactly:
`1 failed, 4607 passed, 3072 skipped, 2776 deselected`, the failing case being
`test_flash_attn_varlen_output[1-3-96-False-False-True-0.0-False-False-mha-dtype0]` — identical
counts and identical test id to their run. **The known failure is confirmed pre-existing in the
fork by direct measurement**, not merely by reading `UPSTREAM_D96.md`.

All three arms produce the identical line:

| Arm | Result |
| --- | --- |
| `d74a3c3` (base) | 1 failed, 4607 passed, 3072 skipped, 2776 deselected |
| `+ forward patch` | 1 failed, 4607 passed, 3072 skipped, 2776 deselected |
| `+ backward patch` | 1 failed, 4607 passed, 3072 skipped, 2776 deselected |

Same failing test id in all three. Neither patch introduces, masks, or changes a single test
outcome across headdim 64/96/128/192/256.

These upstream tests build separate Q/K `cu_seqlens` arrays and pass `seqused`, so they do not
enter the new self-attention fast paths. Fast-path correctness was checked by the dedicated
validators. The fork now also includes `hopper/test_flash_attn_local_varlen.py`: it uses one
shared `cu_seqlens` object without `seqused`, tests all-short/all-middle/all-long/mixed batches,
checks window boundaries on both sides of 128, and compares output, LSE, and gradients with an
independent FP32 reference. Distinct-pointer fallback and external metadata reuse are covered.
BF16 and FP16 are parametrized; FP16 cases skip when that dtype is disabled in the build.

Review follow-up: this repository test passed **38 cases, with 36 FP16 cases skipped** on one
GH200 in Booster job 1739584. It used the previously audited `sfaudit/merge` extension; all ten
changed kernel/API source files match the fork. The allocation was released after testing.

Two things to be precise about here:

- This proves the failure is pre-existing **in the fork**. That it is also pre-existing in
  Dao-AILab upstream is their claim in `sliding_window/UPSTREAM_D96.md`, which I did not
  independently rebuild.
- Earlier in this session the suite was reported fully green against the fork. That is not a
  contradiction and no patch broke anything: those runs used a narrower build scope, and this
  failing case is a **local** backward D96 case that only gets selected once `LOCAL` and
  `BACKWARD` are both enabled — which this build does, across all five head dimensions. The
  unpatched baseline arm here fails identically, which is the decisive control.

### 9c. Unchanged machine code within the compiled scope

Rather than argue from benchmarks that non-local workloads are unaffected, I disassembled both
`.so` files and compared kernel by kernel:

- **211 shared kernels, 0 differing**, across 633,550 SASS instructions.
- **1 kernel added, 0 removed.** The addition is exactly the intended specialization:
  `FlashAttnFwdSm90<CollectiveMainloopFwdSm90<3, ..., tuple<C<64>,C<64>,C<64>>, 64, bfloat16_t, ...>>`
  — 3 stages, M64/N64/64, BF16.

Every pre-existing kernel in these binaries is bit-identical machine code, including forward,
backward, and auxiliary kernels. This covers the compiled non-local, non-D64, and causal paths.
It provides no binary evidence for FP8 or paged paths, which were disabled, or for the other
omitted features listed above. Kernel disassembly also does not measure host dispatch overhead.

(Method note: `cuobjdump` embeds an `identifier = <source path>` line per kernel, which differs
purely because my two build trees have different directory names. Filtering that line is
required; without it 9 kernels appear to differ, including cub's `EmptyKernel<void>()`, a no-op
whose behaviour cannot change — which is what exposed the artefact.)

### 9d. What actually changes numerically, and the gate confirmed by measurement

Saving outputs from each arm in separate processes (loading two FA `.so` files into one process
aborts in `Dispatcher::registerLibrary`) and comparing elementwise:

**Forward is bit-deterministic** — the same binary run twice gives 0 of 78 tensors differing. So
every forward difference below is the path change, not noise:

| Window | Fast path expected | Tensors differing |
| --- | --- | ---: |
| `(-1,-1)` non-local | no | **0 of 2** |
| `(0,0)` degenerate (output is exactly `v`) | eligible, but result is tiling-invariant | 0 of 12 |
| `(1,1)`, `(16,32)`, `(64,64)`, `(127,128)`, `(128,128)` | yes | **all 12 of 12 each** |
| `(512,512)` above the bound | no | **0 of 2** |

Differences appear exactly on `0 <= window <= 128` and vanish at 512 and at non-local. **The
eligibility gate is confirmed by measurement, not just by reading it.** Worst forward divergence
0.015625 at window `(1,1)`, which is 1-2 BF16 ULP.

**Backward `dq` is nondeterministic by construction** (FP32 `atomicAdd` accumulation), so a
cross-arm diff alone proves nothing. Controls:

| Comparison | Tensors differing | Worst diff |
| --- | ---: | ---: |
| base vs base (same binary, 2 runs) | 14 of 117 | 0.000977 |
| cand_bwd vs cand_bwd (same binary, 2 runs) | 14 of 117 | 0.001953 |
| base vs cand_bwd | 27 of 117 | 0.003906 |

`dk` and `dv` are **bitwise identical in all 39 comparisons**; only `dq` ever moves. The patched
arm is exactly as reproducible as the baseline (14 vs 14). About half the cross-arm differences
are inherent nondeterminism — including the non-local `(-1,-1)` case, whose worst location
(`512_512/dq`) is the same one the base-vs-base control picks. The remainder is the genuine path
change at 1 BF16 ULP.

Note this also shows the backward gate really does have no window bound: `(512,512)` takes the
new path in backward while the forward gate correctly rejects it. That asymmetry is intentional
and is documented in section 7 and the README.

Running their own `validate.py` against **both** arms gives the apples-to-apples accuracy answer
— max output error vs an independent FP32 reference, same 14 cases:

`cross .007869 | fixed .002732 | gqa .004845 | long .003711 | strided .004644 | w-1_-1 .004593 |
w-1_0 .008537 | w-1_64 .004593 | w0_0 0 | w128_128 .004593 | w16_32 .004593 | w256_256 .004593 |
w64_-1 .004593 | w64_64 .004593`

**Identical maximum errors to six decimals in every checked case, both arms PASS, worst ratio
1.00.** This is evidence of comparable accuracy for these cases, not bitwise equality of their
outputs or a guarantee for untested inputs.

### 9e. Backward patch collateral: fully accounted for

Same disassembly treatment for `base` vs `cand_bwd`:

- **199 shared kernels, 0 differing.**
- **11 kernels renamed with byte-identical bodies.** Adding the `QPad` template parameter to
  `FlashAttnBwdPostprocessConvertdQ` changes every instantiation's mangled name; the machine code
  is unchanged.
- **1 kernel whose body genuinely changed**: the M32/N256 `PostprocessConvertdQ`, whose
  `SeqlenInfo` padding goes 32 -> 128. **It is never launched.** Its only call site
  (`flash_bwd_launch_template.h:359`) passes `all_direct_dq = true`, and line 259 returns before
  the postprocess launch. The compiler instantiates it, nothing runs it. The change in fact
  removes a latent inconsistency: that instantiation previously padded by 32 while the mainloop
  padded by 128 — harmless only because it is dead.
- **4 genuinely new kernels**: the two local backward mainloops (M32/N256 and M64/N128) and their
  two matching postprocess kernels. Exactly the intended additions.

Within these feature-restricted builds, neither patch alters the machine code of kernels that
execute on pre-existing paths.

## 10. End-to-end: nanoPLM ModernBERT, 4x GH200, sliding-window pattern

`attn_layer_pattern: 'FSS'` -> 10 of 16 layers sliding (verified on the compute node:
`sliding_attention: 10, full_attention: 6`, order `FSSFSSFSSFSSFSSF`, `sliding_window = 64` so
FA3 receives window `(64, 64)`, head_dim 64). 24 training runs, 80 steps each, median `dt` over
steps 20-75.

**A design flaw I hit and corrected.** The first job ran all six arms in the same order every
rep, so position in the job was confounded with arm identity — step time fell monotonically
373.5 -> 372.2 -> 369.9 -> 367.4 exactly along the arm order. I ran a second job with the order
reversed and fitted `dt ~ arm + position + job + cold_start`, which decorrelates them. The first
arm of each job is genuinely slow (**+1.82 ms**, se 0.68 — cache/clock ramp), which a linear
position term cannot absorb.

All entries are **candidate minus baseline step time**: negative is faster, positive slower.
Every percentage uses the observed SWA baseline median, 369.9425 ms, as its denominator.
The control is the within-model contrast `ctlF_both - ctlF_base`, with covariance retained when
computing its standard error.

| specification | + forward | + backward | + both | all-full control |
|---|---:|---:|---:|---:|
| without cold-start term | −0.26% | −1.03% | −1.45% | +0.40% |
| **+ cold-start term** | **−0.14%** | **−0.90%** | **−1.32%** | +0.28% |
| drop position 1 | −0.19% | −0.96% | −1.38% | +0.22% |

The cold-start model gives SE 0.1434 percentage points for each treatment and the control
contrast (n=24, dof=15, residual sd 0.71 ms). These uncertainties are conditional on the OLS
model, including its shared linear drift and independent-residual assumptions; the runs were
ordered and then reversed, not randomized.

**Conclusions:**

- **Both patches together: ~1.3% less training step time** (−1.32%, model SE 0.14 percentage points).
- **Backward alone carries it: ~0.9%.** Consistent across every specification and both job
  orderings.
- **Forward alone is not resolved** (−0.14%, model SE 0.14 percentage points). Forward is a small
  share of total step time; the earlier all-full-attention run measured attention at 7.8% of a
  step, but that fraction should not be assumed exact for the sliding configuration. Do not
  present the fitted forward-only coefficient as an established training speedup.
- **The control exposes residual uncertainty.** Baseline and combined builds should have no
  kernel-induced difference on the all-full `'F'` config. The measured control instead reads
  **+0.28% step time** after correction (t=1.96, 15 dof); without the cold-start term it reads
  +0.40% (t=2.54, 16 dof), which is significant at the two-sided 5% level under that model.
  Dropping first positions gives +0.22% (t=1.55, 14 dof). It is therefore incorrect to say the
  control is always insignificant or proves a universal noise floor. The ~1.3% combined gain
  persists across the three specifications, with this residual uncertainty noted.

Raw logs: `e2eswa/e2e_*.log` (forward order) and `e2eswa/rev_e2e_*.log` (reverse order).
The 24 extracted medians are retained in [data/sliding_window_e2e_runs.json](data/sliding_window_e2e_runs.json).
Run `python docs/analyze_sliding_window_e2e.py` from the repo root (requires NumPy) to regenerate
all three rows, the control contrast, and the model standard errors without the scratch logs.

---

## Status / what remains

| Check | State |
|---|---|
| Hash + patch provenance (both experiments) | done, clean — 7/7 fwd, 16/16 bwd |
| Register-pool / hang mechanism | done, closed |
| Grid x2 ownership, semaphore, cluster | done, safe |
| Gate reachability (host + device) | done; 8 inert hunks noted |
| Backward patch static read | done; both flagged items now checked empirically |
| nanoPLM eligibility | done |
| **End-to-end, 4 GPU, sliding pattern** | done — ~1.3% step time, drift-corrected |
| **Independent rebuild of all three arms** | done, clean |
| **Independent timing reproduction** | done — fwd 21.82%, bwd 31.23%, both reproduce |
| **Upstream suite, all 3 arms** | done — identical results, D96 failure pre-existing |
| **Backward window-bound gap (my addition)** | done — 40 configs, 0 failures, accuracy ratio 1.00 |
| **SASS comparison within the build scope** | done — 211/211 shared kernels bit-identical, +1 new |
| **Long-running hang stress** | done — 231,232 iters / ~4.6M launches / 29 min, no stall |

**The reproduction is complete within the recorded scope.** Every row above is done; the
pre-existing D96 test failure remains, and omitted build features are not covered by these runs.

Hang stress detail: 231,232 iterations of 20 forward launches each (~4.6M launches) on freshly
regenerated ragged batches over 29 minutes, no stall. This was expected to pass — the SASS shows
the math warpgroup's `TRY_ALLOC` spin is bounded by the producer's unconditional `DEALLOC` with no
barrier between them, and `.CTAPOOL` is per-CTA so a co-resident block cannot starve it — but a
`setmaxnreg` race that hung 1-in-10^4 launches would not have shown up in the 20-replay
benchmarks, so it was worth the wall time.
