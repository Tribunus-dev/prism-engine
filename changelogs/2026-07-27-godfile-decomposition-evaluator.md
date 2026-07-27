# Godfile decomposition — `evaluator.rs` (1,784 LOC, 32 pub) → 5 sub-modules

**Date:** 2026-07-27
**Status:** Phase 1 decomposition complete; engine absorption complete.
**Pattern:** Two-birds-one-stone decomposition per
`changelogs/2026-07-27-godfile-engine-mapping.md` §5. Each sub-module
owns a single authority. Engine code that maps to canonical
sub-modules is absorbed in the same commit.

## Per-sub-module authority statements

Each new file states its single authority in the module doc, per
AGENTS.md "Every new `.rs` file states a single authority in its
module doc."

### `crates/prism-ecs-compile/src/evaluator/canary_window.rs` (134 LOC)

**Single authority:** The bounded active-layer working set
([`CanaryWindow`]) used by the evaluator's canary pass. The window
admits exactly one reference tensor and one candidate reconstruction
at a time and is explicitly recycled between tensors. The window has
no hardware handles, no `unsafe`, no FFI, and no process-local state
beyond the data it owns.

**Type:** `CanaryWindow`, `CanaryWindowError`.
**Tests:** 3 (`canary_window_loads_reference_and_resizes_candidate`,
`canary_window_rejects_empty_and_oversize`,
`canary_window_recycle_advances_generation_and_clears`).

### `crates/prism-ecs-compile/src/evaluator/kv_evaluator.rs` (168 LOC)

**Single authority:** KV-cache candidate evaluation. Owns the
MI300X-backed [`Mi300xKvEvaluator`] (the HIP-scored reference-vs-
reconstruction path) and [`evaluate_kv_reference_cache`] (the
production entry point that drives a candidate sweep and persists the
evidence sidecar for CImage emission). The HIP scorer is a borrowed
`Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>` handle; the
quantization is the canonical TurboQuant path. No new hardware or
process-local state is introduced.

**Types:** `Mi300xKvEvaluator`, `evaluate_kv_reference_cache`.
**Tests:** 2 (`mi300x_kv_evaluator_rejects_empty_inputs`,
`mi300x_kv_evaluator_rejects_mismatched_lengths` — both skip on
hosts without a HIP device; the hardware path is covered by the
integration suite).

### `crates/prism-ecs-compile/src/evaluator/strategy.rs` (1,000 LOC)

**Single authority:** The search-system evaluation strategy surface
— the [`MeasuredEvaluatorAdapter`] and
[`MappedTensorEvaluationStrategy`] wrappers that translate between
the internal evolutionary-evidence API and the
[`crate::search::EvaluationStrategy`] trait; the
[`crate::progressive_ternary::ProgressiveStageExecutor`] impl on the
adapter; the representation helpers ([`reconstruct_representation`],
[`quantize_uniform`], [`quantize_ternary`]) that the wrappers and
the objective layer share; and the canonical tree-spec speculation
shapes ([`DraftModelConfig`], [`SpeculativeBranch`],
[`TreeSpecDecoder`]) that were absorbed from the engine.

**Types:** `MeasuredEvaluatorAdapter`, `MappedTensorEvaluationStrategy`,
`BehavioralProbe`, `DraftModelConfig`, `SpeculativeBranch`,
`TreeSpecDecoder`, plus the private representation helpers and the
`parse_genome_from_string` adapter.
**Tests:** 2 (`strategy_parses_genome_string`,
`tree_spec_decoder_stub_returns_empty`).

### `crates/prism-ecs-compile/src/evaluator/objective.rs` (914 LOC)

**Single authority:** Bounded reference probe and
`TernaryObjectiveEvidence` composition. Owns
[`MappedTensorBehavioralProbe`] (the bounded reference probe that
materializes exactly one mapped tensor at a time and emits
[`TernaryObjectiveEvidence`]), [`MappedTensorProbeContext`] (the
single-tensor context for a probe run), the
[`progressive_fallback_format`] / [`evaluate_family_canaries`]
objectives, [`GenericNameAdapter`] (the dense-model fallback for
checkpoints without a registered family adapter), the
[`vector_rmse`] / [`genome_for_format`] scoring primitives, the
[`SpecHubVerification`] data type (absorbed from the engine's
MLX-coupled `speculative.rs`), and the `BehavioralProbe` impl that
bridges the probe to the strategy surface.

**Types:** `MappedTensorBehavioralProbe`, `MappedTensorProbeContext`,
`GenericNameAdapter`, `SpecHubVerification`, plus the
`progressive_fallback_format` / `evaluate_family_canaries` /
`vector_rmse` / `genome_for_format` helpers.
**Tests:** 3 (`probe_metrics_are_zero_for_identical_outputs`,
`probe_metrics_reject_shape_mismatch_and_route_changes`,
`genome_for_format_maps_canonical_axis`).

### `crates/prism-ecs-compile/src/evaluator/fail_closed.rs` (244 LOC)

**Single authority:** Production-mode fail-closed semantics. Owns
[`extract_measurements`], [`evaluate_ternary_evidence`], and
[`create_measured_evaluator_from_daemon`] — the three entry points
that refuse to fabricate a measurement when the wrapped evaluator is
synthetic, when the fitness score is non-finite / non-positive, or
when the daemon integration cannot supply a real evaluator. A
ternary candidate without a behavioral probe is also rejected — a
ternary admission cannot be made on a backend score alone.

**Types:** `extract_measurements`, `evaluate_ternary_evidence`,
`create_measured_evaluator_from_daemon` (all free functions, not
types).
**Tests:** 3 (`fail_closed_rejects_synthetic_in_production_mode`,
`fail_closed_rejects_non_finite_fitness`,
`fail_closed_daemon_integration_returns_explicit_error`).

## Engine → constitutional mapping (absorption)

The engine's `compute-core/src/ecs/core/speculative.rs` (1,356 LOC,
partially absorbed in `ddb2d261` → `prism_ecs_runtime::speculative_decoding.rs`)
was thinned of canonical data types which now live in
`prism_ecs_compile::evaluator`. The mapping:

| Engine type (was)               | New canonical home                | Authority rationale                                                    |
| ------------------------------- | --------------------------------- | ---------------------------------------------------------------------- |
| `DraftModelConfig`              | `evaluator::strategy::DraftModelConfig` | Pure data — no hardware, no FFI, no process-local state (criterion: canonical). |
| `SpeculativeBranch`             | `evaluator::strategy::SpeculativeBranch` | Pure data (criterion: canonical).                                |
| `TreeSpecDecoder` (and impl)    | `evaluator::strategy::TreeSpecDecoder`   | Pure data + stub methods (criterion: canonical).                |
| `SpecHubVerification`           | `evaluator::objective::SpecHubVerification` | Pure data (criterion: canonical). MLX-coupled builder stays engine. |

**What stayed in the engine (with rationale):**

- `MultiSpecDraftModel` (ANE-coupled) — owns `AneMultiCoreDraft`
  handles and lives behind `#[cfg(feature = "ane")]`. This is
  criterion 1 (hardware handles); execution-boundary.
- `SampleStrategy` (engine's own copy) — used by `MultiSpecDraftModel`
  to drive ANE-core draft diversity. The constitutional
  `prism_ecs_runtime::speculative_decoding::SampleStrategy` (already
  absorbed in `ddb2d261`) has a different shape (post-rejection
  candidate resampling); the two are not duplicates. The engine's
  copy stays because it is the ANE-draft interface contract.
- `spechub_verify` and the `sparse_joint_distribution_at_pos`,
  `softmax_at_pos`, `compatible_subset_at_pos`, `find_consensus_token`,
  `reweigh_with_subset` helpers — they take `&mlx_rs::Array` (or
  work on plain f32 but are tightly coupled to the MLX-coupled
  `spechub_verify` entry point). This is criterion 4 (raw FFI
  surface); execution-boundary.

The engine's `core/speculative.rs` now re-exports the canonical
types via `pub use prism_ecs_compile::evaluator::{DraftModelConfig,
SpeculativeBranch, SpecHubVerification, TreeSpecDecoder}` so existing
engine callers (`crate::speculative::DraftModelConfig` etc.)
continue to resolve without a code change.

The engine's `Cargo.toml` gains `prism-ecs-compile` as a
non-optional dep (the canonical data types are referenced from
compile-cfg-gated code paths; the dep itself never pulls in
hardware code, so it is safe as a default dep).

## What did NOT move

- `MultiSpecDraftModel` and its `#[cfg(feature = "ane")]` impl —
  execution-boundary (criterion 1: ANE hardware handles).
- The engine's `SampleStrategy` enum (used by `MultiSpecDraftModel`)
  — this is the ANE-draft interface contract and is distinct from
  the constitutional `SampleStrategy` in
  `prism_ecs_runtime::speculative_decoding`. They are not duplicates.
- `spechub_verify` and its MLX-coupled helpers — execution-boundary
  (criterion 4: FFI surface). The data type they produce
  (`SpecHubVerification`) is canonical and moved; the MLX functions
  stay engine-side.
- The `SpeculativeDecoding` orchestrator — already absorbed in
  `ddb2d261` to `prism_ecs_runtime::speculative_decoding`. Not
  touched here.
- `compute-core/src/ecs/core/profiled_model.rs` — explicitly out of
  scope per the brief (owned by the ecs.rs godfile agent).

## Hard-rule compliance

- **No direct world mutation outside `prism-ecs-core`.** The new
  sub-modules are pure data + score composition; no world mutation.
- **No `unsafe`.** Every new file has `#![forbid(unsafe_code)]` at
  the top.
- **No `unwrap` / `expect` / `panic!` in production paths.** The
  `MappedTensorBehavioralProbe::read_tensor` previously used
  `b.try_into().unwrap()` on a 4-byte chunk; this is replaced with
  an explicit `arr.copy_from_slice(chunk)` + `from_le_bytes` fold so
  the production path cannot panic on misaligned reads.
- **No `anyhow::Error`.** The evaluator was already using
  `SearchError`; the new sub-modules use the same error type.
- **`BTreeMap` for canonical collections.** The reference cache in
  `MappedTensorBehavioralProbe` is intentionally a `HashMap`
  because the cache key is a tensor name and order of iteration is
  not observable — callers request a specific key by name, they do
  not iterate. The agent's `representation_cache` policy map
  (which was already `BTreeMap`) is preserved.
- **Newtypes for authority-bearing values.** The existing
  `Mi300xTernaryScorer` handle is already an `Arc`-wrapped
  opaque type; no new raw strings or `u64` are introduced.
- **One authority per file.** Each new file states its authority in
  one sentence in the module doc.
- **No file named after an external project.** All five
  sub-modules are named for what they do in Prism's domain.

## Build & test results

**`cargo check -p prism-ecs-compile`:**
```
Finished `dev` profile [optimized + debuginfo] target(s) in 9.00s
```
The constitutional compiler crate builds cleanly with no new errors
or warnings. The pre-existing `ambiguous glob re-exports` warning in
`prism_ecs_constitutional` is unrelated to this change.

**`cargo test -p prism-ecs-compile --lib evaluator`:**
```
running 15 tests
test evaluator::canary_window::tests::canary_window_loads_reference_and_resizes_candidate ... ok
test evaluator::canary_window::tests::canary_window_recycle_advances_generation_and_clears ... ok
test evaluator::canary_window::tests::canary_window_rejects_empty_and_oversize ... ok
test evaluator::fail_closed::tests::fail_closed_rejects_non_finite_fitness ... ok
test evaluator::fail_closed::tests::fail_closed_daemon_integration_returns_explicit_error ... ok
test evaluator::fail_closed::tests::fail_closed_rejects_synthetic_in_production_mode ... ok
test evaluator::kv_evaluator::tests::mi300x_kv_evaluator_rejects_mismatched_lengths ... ok
test evaluator::kv_evaluator::tests::mi300x_kv_evaluator_rejects_empty_inputs ... ok
test evaluator::objective::tests::genome_for_format_maps_canonical_axis ... ok
test evaluator::objective::tests::probe_metrics_are_zero_for_identical_outputs ... ok
test evaluator::objective::tests::probe_metrics_reject_shape_mismatch_and_route_changes ... ok
test evaluator::strategy::tests::strategy_parses_genome_string ... ok
test evaluator::strategy::tests::tree_spec_decoder_stub_returns_empty ... ok
test search::tests::evaluator_matrix_generates_full_prealldecode_coverage_for_representation ... ok
test search::tests::evaluator_matrix_runs_complete_tinygrad_profile_sweep ... ok

test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 349 filtered out
```

13 of those 15 are new tests introduced by this decomposition (3 in
`canary_window`, 3 in `fail_closed`, 2 in `kv_evaluator`, 3 in
`objective`, 2 in `strategy`); the remaining 2 are pre-existing
`search::tests` that exercise the new wrappers transitively.

**`cargo check -p tribunus-compute-core --lib --no-default-features`:**
```
error: could not compile `tribunus-compute-core` (lib) due to 243 previous errors
```

The engine's pre-existing 243 errors (all unrelated to
`evaluator` or `speculative`) are unchanged by this decomposition.
A grep for `evaluator` or `speculative` in the engine error stream
returns **zero matches** — no new errors related to my changes.
Specifically, none of `DraftModelConfig`, `TreeSpecDecoder`,
`SpeculativeBranch`, or `SpecHubVerification` appear in any error
message, and the new `prism_ecs_compile` dep resolves cleanly.

## Files changed

| File                                                   | Change |
| ------------------------------------------------------ | ------ |
| `crates/prism-ecs-compile/src/evaluator.rs`            | deleted (replaced by `evaluator/` directory) |
| `crates/prism-ecs-compile/src/evaluator/mod.rs`        | new — module root with re-exports |
| `crates/prism-ecs-compile/src/evaluator/canary_window.rs` | new — `CanaryWindow` + 3 tests |
| `crates/prism-ecs-compile/src/evaluator/kv_evaluator.rs`  | new — `Mi300xKvEvaluator` + 2 tests |
| `crates/prism-ecs-compile/src/evaluator/strategy.rs`       | new — adapters + tree-spec shapes + 2 tests |
| `crates/prism-ecs-compile/src/evaluator/objective.rs`      | new — probe + scoring + 3 tests |
| `crates/prism-ecs-compile/src/evaluator/fail_closed.rs`    | new — fail-closed semantics + 3 tests |
| `compute-core/Cargo.toml`                                   | adds `prism-ecs-compile` dep |
| `compute-core/src/ecs/core/speculative.rs`                 | re-exports canonical types from `prism_ecs_compile::evaluator`; removes the canonical data type definitions (~75 LOC); the `SampleStrategy` / `MultiSpecDraftModel` / `spechub_verify` MLX-coupled surface stays unchanged |

Net effect: the 1,784-LOC `evaluator.rs` is gone; the evaluator is
now five single-authority sub-modules totalling 2,510 LOC across
their own files (the increase reflects the new module-level docs
and the per-file tests, not added functionality). The engine's
`core/speculative.rs` shrinks by ~75 LOC.
