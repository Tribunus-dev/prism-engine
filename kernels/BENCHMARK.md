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

## Speculative-decoding projection (`--spec-max-k`, `--spec-verify-factor`)

The student is a natural **draft model** for the teacher (same architecture +
tokenizer, ~2.6× cheaper per step). From the per-position logits the PPL pass
already captures, `prism-bench-ab` projects "student drafts k, teacher verifies
in one pass" — teacher-quality output, faster than teacher-alone decode:

- **Acceptance per position** (`bench_metrics::greedy_acceptance` /
  `sampling_acceptance`): greedy = argmax agreement; sampling =
  `Σ_v min(p_t, p_s)` (Leviathan-style rejection sampling at T=1).
- **Expected tokens/cycle** (`expected_tokens_per_cycle`): empirical windowed
  `mean_p [1 + Σ_i Π_j a[p+j]]` including the teacher's bonus token. This
  respects **bursty agreement** — the i.i.d. formula `(1−α^{k+1})/(1−α)`
  (`expected_tokens_iid`) is reported conceptually but the table uses the
  empirical number (the alternating-acceptance unit test shows i.i.d. can
  overestimate by ~17% at the same mean α).
- **Cycle cost** = `k · (c_student/c_teacher) + verify_factor`, in teacher
  steps, with the cost ratio taken from THIS run's measured decode medians.
- **Speedup** = tokens/cycle ÷ cycle cost. The table sweeps k = 1..`--spec-max-k`
  and marks the best greedy k.

**Projection caveats (also printed by the tool):**
1. Offline approximation — both models are teacher-forced on the fixed eval
   stream, not on a self-drafted trajectory.
2. The verify pass is **modeled** (`--spec-verify-factor`, default 1.0 = the
   memory-bound ideal of one weight read for k+1 positions). Prism has **no
   multi-token verification path today** — the megakernel is single-token
   (KERNEL_AUDIT.md); a batched verify falls out of the per-op forward work
   (PER_OP_FORWARD_PLAN.md). Until that exists this table is a feasibility
   estimate, not a measurement.
3. A projected speedup < 1 at every k means spec decode is not worth building
   for this model pair — that is a valid, publishable outcome.

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
- **Verified on Linux:** `bench_metrics` (perplexity/throughput math + the
  spec-decode projection: acceptance, windowed tokens/cycle, speedup table —
  9 unit tests), `distill_core` (KD/agreement), `nf4tile640_ref` (dequant+GEMV
  parity + fidelity). `compute-core` builds under `--features backend-cpu`
  (cmake+clang).
- **Mac-only:** the live forward that produces the logits and timings (Metal/ANE);
  a fair run needs both `.cimage` files on the same Apple-Silicon box.
