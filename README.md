# FlashAttention-3 varlen, tuned for protein language model training

*A fork of [flash-attention](https://github.com/Dao-AILab/flash-attention) for SM90 / Hopper / GH200.*

This fork adapts the FlashAttention-3 **varlen** path (SM90 / Hopper / GH200) to the regime
protein language models actually train in: **many short sequences packed into one pass**.
A UniRef-style batch is ~65,536 tokens made of ~350 sequences with a **median length near 165**
and a long tail out past 900 — nothing like the multi-thousand-token sequences upstream's
defaults are tuned for.

Two changes, both scoped to non-causal varlen with `headdim <= 64`:

1. **Persistent n-block backward scheduler** (on by default in this fork). Upstream's backward
   launches a *rectangular* grid — every sequence gets as many KV blocks as the *longest*
   sequence in the batch — so on a protein batch **71% of the CTAs are empty tiles that only
   write zeros**. This replaces that with a persistent scheduler over the real tiles:
   37,744 CTAs → 132, and 17% fewer instructions executed.
2. **Short-sequence forward tiles** (opt-in: `FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE`). A narrow
   KV tile (192×80, `IntraWGOverlap=false`) plus a deeper pipeline (`kStages=3`), which suits
   short sequences but **regresses long ones**, hence the flag.

![protein varlen benchmark](docs/assets/protein_varlen_gh200.png)

| | upstream | this fork | speedup |
|---|---|---|---|
| forward | 142 TFLOP/s | **173 TFLOP/s** | **+21.8%** |
| backward | 130 TFLOP/s | **141 TFLOP/s** | **+8.4%** |
| **fwd + bwd** | **132 TFLOP/s** | **149 TFLOP/s** | **+12.5%** |

GH200 (680 W cap), bf16, D=64, non-causal, 65,536 tokens/pass, mean of 3 seeds. Throughput,
so higher is better. Gradients match upstream exactly and 80 steps of pretraining give an
identical loss curve.

### The two changes pull in opposite directions

Holding the token budget and the **arithmetic mean length fixed at 200** and varying only the
*spread* of the length distribution separates the two mechanisms cleanly:

![variance sweep](docs/assets/variance_sweep.png)

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

* The **forward** win is about sequences being *short*. It is largest at zero variance (+47%)
  and decays as the tail lengthens, because long sequences prefer upstream's wide KV tile.
* The **backward** win is about the distribution being *ragged*. It is break-even at zero
  variance and grows monotonically with it, because dispersion is what fills the rectangular
  grid with empty CTAs.

A protein corpus (σ ≈ 0.55, CV ≈ 0.6) sits where both are positive, which is why the fork
combines them. Neither mechanism has anything to do with the other, and either can be used
without the other.

### When this fork is slower

Both changes buy short-and-ragged throughput by giving up something else, and neither is a
free win. **If your sequences are all roughly the same length, this fork is slower than
upstream** — at every length tested.

![uniform length regression](docs/assets/uniform_len_regression.png)

Uniform-length varlen, 65,536 tokens/pass, D=64 non-causal:

| uniform seqlen | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| forward, upstream | 100 | 131 | 256 | 288 | 382 | 422 | 450 | 447 |
| forward, fork | 122 | 161 | 272 | 306 | 359 | 396 | 415 | 407 |
| **forward speedup** | **+22.3%** | +22.2% | +6.0% | +6.4% | −5.9% | −6.2% | −7.8% | **−9.0%** |
| backward, upstream | 112 | 190 | 281 | 338 | 407 | 441 | 450 | 464 |
| backward, fork | 104 | 185 | 276 | 332 | 380 | 420 | 433 | 445 |
| **backward speedup** | **−6.7%** | −2.8% | −1.8% | −1.9% | −6.6% | −4.7% | −3.9% | −4.2% |

All figures TFLOP/s; higher is better.

Two separate effects:

* **The backward regresses whenever tile fill is high.** Uniform lengths mean the rectangular
  grid has *no* empty CTAs, so there is nothing to eliminate and the change only pays for the
  extra producer/epilogue handshake it needs. It wins only when lengths are *dispersed* — +8.5%
  on a protein batch, and up to +37% at 6% fill, but it is break-even to slightly negative once
  fill approaches 100%. It is on by default here because this is a protein-LM fork; **if your batches are
  length-bucketed or padded to a common length, use upstream.**
* **The forward regresses past ~1.5k tokens**, which is tile quantization, and is why it is
  behind a flag.

Mechanism, full crossover tables, and the two bugs that make the naive backward port *silently
return wrong gradients*: **[docs/protein_varlen_gh200.md](docs/protein_varlen_gh200.md)**.

### Where this started: GH200 is not an H100

The work began on the [JUPITER](https://www.fz-juelich.de/en/ias/jsc/jupiter) booster, whose
nodes carry four GH200 Grace-Hopper superchips. Measured on one of them:

| | JUPITER GH200 | H100 SXM |
|---|---|---|
| power cap | **680 W** (enforced, flat) | 900 W |
| HBM bandwidth (achieved copy) | **3.64 TB/s** | ~3.35 TB/s |
| bf16 GEMM (achieved) | **~605 TFLOP/s** | ~750–990 TFLOP/s |
| **machine balance** | **~166 FLOP/byte** | **~250 FLOP/byte** |
| SMs / smem per SM | 132 / 228 KB | 132 / 228 KB |

The 680 W cap is a ~24% compute derate with no bandwidth penalty, which leaves the GH200
roughly **35% more bandwidth-rich per FLOP** than an H100. Upstream's forward tile table in
`hopper/tile_size.h` carries the comment *"benchmarked on H100 SXM"*. The starting hypothesis
was that a machine with a materially different compute/bandwidth ratio should prefer a
different tile shape.

**That hypothesis was falsified, and it is worth saying so plainly.** At seqlen 8192 the H100
table is already optimal on GH200 — 37 configurations were swept and nothing beat it — because
FA3 sits at ~97% of the achievable GEMM ceiling there, leaving no headroom for a roofline
argument. The wins in this fork are **not** attributable to the GH200's balance. They come from
the *shape of the workload*: short, ragged, varlen sequences, where the costs are tile
quantization and empty CTAs rather than the FLOP:byte ratio. They would very likely reproduce
on an H100. **No H100 control was available**, so nothing here is claimed as GH200-specific.

The hardware ratio was the question that started the investigation; the answer turned out to
be about sequence-length distribution instead.

### Build

```bash
cd hopper
export FLASH_ATTENTION_SHORT_SEQ_TILES=TRUE   # optional; forward tiles for short seqs
python setup.py install
```

---

### Disclosure

This fork's changes, benchmarks, profiling and documentation were produced with
[Claude Code](https://claude.com/claude-code) running **Claude Opus 5** at high reasoning
effort, working on the JUPITER cluster under my direction and review. Every performance number
in this README and in `docs/` is from an actual run on a GH200 — nothing is estimated or
extrapolated. Correctness claims are backed by gradient checks against per-sequence PyTorch
SDPA and by end-to-end training runs; the specific checks and their limits are listed in
[docs/protein_varlen_gh200.md](docs/protein_varlen_gh200.md).

---

### Upstream

This is a fork of [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention).
Installation instructions, the full feature matrix, supported architectures and the
FlashAttention-4 / CuTeDSL documentation all live upstream — see the
[upstream README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md).
Everything here other than the two changes described above is upstream's work, under
upstream's [LICENSE](LICENSE).

If you use FlashAttention, please cite the original papers:

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** —
  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
  ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135))
- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning** —
  Tri Dao ([paper](https://tridao.me/publications/flash2/flash2.pdf))
- **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** —
  Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao
  ([arXiv:2407.08608](https://arxiv.org/abs/2407.08608))
