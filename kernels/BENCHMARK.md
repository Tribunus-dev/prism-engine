# Truthful NF4-teacher vs ternary-student benchmark

Protocol + tooling for an honest head-to-head on **performance** and **accuracy**.
The scoring lives in `compilation::bench_metrics` (perplexity, throughput stats,
comparison) and `compilation::distill_core` (teacher↔student agreement). The CPU
accuracy ground-truth + kernel-parity reference is `tools/nf4tile640_ref.rs`.

## The one rule
**Performance numbers must come from the Mac (Metal/ANE).** A `.cimage` forward
requires a `metal::Device` (`CimageDeployment::load`), so Linux CPU can validate
*correctness and plumbing* but not the throughput that matters. Never compare
tok/s across machines. Accuracy (perplexity/logits) is deterministic and can be
scored anywhere the forward runs.

## Fairness controls (identical for both models)
- Same machine, plugged in, thermally settled; cooldown between runs.
- Same prompt set, tokenizer, context length, batch size, `max_tokens`.
- **Greedy decode** (temperature 0) for determinism when scoring accuracy.
- Warmup: discard the first N iterations (JIT/caches/thermal); measure M ≥ 20.
- Report **median + p90 + p99**, never a lone mean (`throughput_stats` does this).
- Same build profile (`--release`), same feature flags except the model format.

## Performance metrics (`bench_metrics::ModelRunMetrics`)
- **Prefill tok/s** and **decode tok/s** (report both — they scale differently).
- **TTFT** (time to first token).
- **Effective bpw** from on-disk size (`effective_bpw`) — the compression axis.
- Peak resident memory (from the OS; record alongside).

## Accuracy metrics
- **Perplexity** on a fixed held-out corpus (`perplexity` / `token_nll`) — the
  headline. Report each model's absolute PPL and the ratio (`compare`).
- **Teacher↔student agreement**: `distill_core::top1_agreement` and
  `kd_divergence` (logit KL) on the same inputs — how faithfully the student
  tracks the teacher, independent of absolute PPL.
- **Weight-fidelity ground truth** (`tools/nf4tile640_ref.rs`, verified on Linux):
  NF4 teacher reconstructs Gaussian weights at **rel-L2 ≈ 0.13**; the ternary
  student was **≈ 0.5** in `quant_lab`. Expect the teacher to lead on PPL; the
  distillation goal is for the student to recover most of that gap at ~2 bpw vs
  ~4.25 bpw. Report the actual gap — don't assume the student matches.

## How to run
1. **Correctness first (any host):** `rustc -O tools/nf4tile640_ref.rs && /tmp/nf4ref`
   confirms the NF4 dequant+GEMV math and gives the fidelity ground-truth;
   `cargo test -p tribunus-compute-core --lib bench_metrics` / `distill_core`
   checks the scorers.
2. **On the Mac:** run the same prompt set through both `.cimage` files via the
   server/runner, capturing per-iteration prefill/decode timings + per-token
   logits. Feed timings to `throughput_stats`, logits to `perplexity` +
   `top1_agreement` + `kd_divergence`, then `bench_metrics::compare`.
3. Emit one table: teacher vs student × {prefill, decode, TTFT, bpw, PPL,
   top-1 agree, KL}. That table is the truthful benchmark.

## Status
- **Verified on Linux:** `bench_metrics` (perplexity/throughput math),
  `distill_core` (KD/agreement), `nf4tile640_ref` (dequant+GEMV parity +
  fidelity). `compute-core` builds under `--features backend-cpu` (cmake+clang).
- **Mac-only:** the live forward that produces the logits and timings (Metal/ANE);
  a fair run needs both `.cimage` files on the same Apple-Silicon box.
