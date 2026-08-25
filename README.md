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
| forward | 0.484 ms (142 TF/s) | **0.397 ms (173 TF/s)** | **−17.9%** |
| backward | 1.324 ms (130 TF/s) | **1.221 ms (141 TF/s)** | **−7.8%** |
| **fwd + bwd** | **1.817 ms (132 TF/s)** | **1.615 ms (149 TF/s)** | **−11.2%** |

GH200 (680 W cap), bf16, D=64, non-causal, 65,536 tokens/pass, mean 3 seeds. Gradients match
upstream exactly and 30 steps of pretraining give bit-identical eval loss.

### When this fork is slower

Both changes buy short-and-ragged throughput by giving up something else, and neither is a
free win. **If your sequences are all roughly the same length, this fork is slower than
upstream** — at every length tested.

![uniform length regression](docs/assets/uniform_len_regression.png)

Uniform-length varlen, 65,536 tokens/pass, D=64 non-causal, % vs upstream (negative = slower):

| uniform seqlen | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| backward (default) | **−7.2** | −2.9 | −1.9 | −2.0 | **−7.1** | −5.0 | −4.0 | −4.3 |
| forward (flag on) | +18.3 | +18.2 | +5.7 | +6.0 | −6.3 | −6.6 | −8.4 | **−9.9** |
| fwd+bwd (flag on) | +1.9 | +6.2 | −2.2 | −2.9 | −3.1 | −5.2 | −5.1 | **−8.2** |

Two separate effects:

* **The backward regresses whenever tile fill is high.** Uniform lengths mean the rectangular
  grid has *no* empty CTAs, so there is nothing to eliminate and the change only pays for the
  extra producer/epilogue handshake it needs. It wins only when lengths are *dispersed* — +7.9%
  on a protein batch (28.7% fill), +12.6% at 11.5% fill, but −1.1% at 64.6% fill and worse at
  100%. It is on by default here because this is a protein-LM fork; **if your batches are
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
