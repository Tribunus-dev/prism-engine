# Sub-split — `evaluator/strategy.rs` (1,000 LOC) → 3 sub-modules by authority

**Date:** 2026-07-27
**Status:** Decomposition complete; tests pass.
**Pattern:** Per-authority split of the `evaluator/strategy.rs` godfile
into a directory of three single-authority sub-modules. The split
follows the `prism-constitutional-rust-ecs` skill's
`module-discipline.md` and the brief in the dispatch message.

## Sub-module layout

The 1,000-LOC `crates/prism-ecs-compile/src/evaluator/strategy.rs`
godfile (introduced during the `evaluator.rs` godfile decomposition
in commit `d0453c4f`) is replaced by a directory of three
single-authority sub-modules plus a `mod.rs` re-export hub. The
`evaluator/mod.rs` re-export surface is unchanged.

| Sub-module                          | LOC  | Public items | Single authority |
| ----------------------------------- | ---- | ------------ | ---------------- |
| `strategy/behavioral.rs`            | 142  | 4            | Behavioral probe trait + tree-spec speculation shapes (absorbed from engine) |
| `strategy/progressive.rs`           | 141  | 0 (`pub(crate)` only) | Bounded representation-reconstruction helpers used by progressive stage evaluation |
| `strategy/mapped.rs`                | 747  | 2            | Mapped-tensor search-system evaluation strategy family |
| `strategy/mod.rs`                   | 64   | 6 (re-exports) | Module root with re-exports for sibling modules |

All four files are under the 900-LOC hard rule and under the
35-public-items hard rule.

## Per-sub-module authority statements

Each new file states its single authority in the module doc, per
AGENTS.md "Every new `.rs` file states a single authority in its
module doc." The authorities are non-overlapping and the
dependency direction is one-way: `mapped` depends on
`behavioral` (for the [`BehavioralProbe`] trait) and on
`progressive` (for [`parse_genome_from_string`] and
[`reconstruct_representation`]); `behavioral` and `progressive`
do not depend on `mapped`.

### `strategy/behavioral.rs` (142 LOC)

**Single authority:** The abstract behavioral probe trait
([`BehavioralProbe`]) consumed by the constitutional strategy
surface, and the canonical tree-spec speculation shapes
([`DraftModelConfig`], [`SpeculativeBranch`],
[`TreeSpecDecoder`]) absorbed from
`compute-core/src/ecs/core/speculative.rs` in commit
`d0453c4f`. The trait is the contract that the objective layer
implements; the shapes are the contract that the engine's
draft/target orchestrator consumes.

**Public types:** `BehavioralProbe`, `DraftModelConfig`,
`SpeculativeBranch`, `TreeSpecDecoder`.
**Public methods on `TreeSpecDecoder`:** `propose`, `verify`
(stubs — the actual algorithms are engine-side and MLX-coupled).
**Test:** 1 (`tree_spec_decoder_stub_returns_empty`).

### `strategy/progressive.rs` (141 LOC)

**Single authority:** The bounded representation-reconstruction
helpers used by progressive stage evaluation to map a reference
tensor through a candidate representation and back to a
comparable form. The module owns
[`reconstruct_representation`] (the bounded reconstructor that
returns a real reconstruction — not a ternary label with a
different name — so admission can compare actual divergence),
[`quantize_uniform`] (the symmetric uniform quantizer),
[`quantize_ternary`] (the row-grouped ternary quantizer with
packing-aware grouping), and [`parse_genome_from_string`] (the
JSON → [`CandidateGenome`] adapter that bridges the
search-system's string-based API to the internal
[`CandidateGenome`] shape).

**Public items:** 0 (every helper is `pub(crate)` and re-exported
through `strategy/mod.rs` so `super::objective` can keep
importing `super::strategy::{parse_genome_from_string,
quantize_ternary, reconstruct_representation}` without churn).
**Test:** 1 (`strategy_parses_genome_string`).

### `strategy/mapped.rs` (747 LOC)

**Single authority:** The mapped-tensor search-system evaluation
strategy family — the constitutional [`MeasuredEvaluatorAdapter`]
and [`MappedTensorEvaluationStrategy`] adapters, plus the
workload and backend evaluation plumbing
([`evaluate_workload_profile_impl`],
[`evaluate_workload_profile_on_graph_impl`],
[`evaluate_backend_impl`]) that drives the bounded reference
probe through mixed-precision graph candidates, SpatialIR
lowering, and ANE/Metal/Accelerate backend dispatch. Fail-closed
semantics for the [`ProgressiveStageExecutor`] impl live in
`super::fail_closed` so the fail-closed authority is isolated
to one place.

**Public types:** `MeasuredEvaluatorAdapter`,
`MappedTensorEvaluationStrategy`.
**Public methods on `MeasuredEvaluatorAdapter`:** `new`,
`with_behavioral_probe`, `with_mapped_tensor_probe`,
`is_synthetic` (plus the trait impls
[`ProgressiveStageExecutor::evaluate`] and
[`SearchEvaluationStrategy::evaluate` / `name` /
`progressive_executor`]).
**Public methods on `MappedTensorEvaluationStrategy`:** `new`
(plus the trait impls [`SearchEvaluationStrategy::evaluate`,
`name`, `is_measured`, `evaluate_workload_profile`,
`evaluate_workload_profile_on_graph`, `evaluate_backend`]).
**Tests:** 0 (all evaluator-level tests for these types live in
sibling modules; the strategy is exercised transitively through
the `search::tests` matrix tests).

### `strategy/mod.rs` (64 LOC)

**Single authority:** The module root that re-exports the
three sub-modules' public surface so external callers
(`evaluator/mod.rs`, `super::objective`, `super::fail_closed`)
see a stable API. The re-exports are:

- `pub use behavioral::{BehavioralProbe, DraftModelConfig,
  SpeculativeBranch, TreeSpecDecoder};`
- `pub use mapped::{MeasuredEvaluatorAdapter,
  MappedTensorEvaluationStrategy};`
- `pub(crate) use progressive::{parse_genome_from_string,
  quantize_ternary, reconstruct_representation};`

## Engine absorption (preserved, not re-evaluated)

The engine's `compute-core/src/ecs/core/speculative.rs` keeps
the MLX-coupled helpers (criterion 4: FFI surface) and the
ANE-coupled `MultiSpecDraftModel` (criterion 1: hardware
dispatch path). The canonical data types — `DraftModelConfig`,
`SpeculativeBranch`, `TreeSpecDecoder` — were absorbed in
commit `d0453c4f` and re-export from this module. This split
does not re-evaluate that decision; the engine's
`MultiSpecDraftModel` and `spechub_verify` family stay
engine-side.

## Hard-rule compliance

- **No direct world mutation outside `prism-ecs-core`.** The
  three new sub-modules are pure data + score composition; no
  world mutation.
- **No `unsafe`.** Every new file has `#![forbid(unsafe_code)]`
  at the top.
- **No `unwrap` / `expect` / `panic!` in production paths.** The
  `MappedTensorEvaluationStrategy` impl uses
  `shape.last().copied().unwrap_or(reference.len()).max(1)`
  (line ~796 of mapped.rs) and `shape.last().copied().unwrap_or
  (reference.len())` (line ~880 of mapped.rs) — these are
  guarded defaults that substitute the reference length when
  the shape vec is empty, not a true `unwrap` of a known-safe
  value. They match the pre-decomposition behavior. The
  `is_synthetic` method on `MeasuredEvaluatorAdapter` does no
  `unwrap`; it only does `name.contains("Synthetic") ||
  name.contains("synthetic")`.
- **No `anyhow::Error`.** The strategy was already using
  `SearchError`; the new sub-modules use the same error type
  and `Result<_, String>` for the search-system surface.
- **`BTreeMap` for canonical collections.** The new sub-modules
  do not introduce any new canonical collections. The
  `mixed_precision_graphs` collection iterated by
  `evaluate_workload_profile_impl` is owned by
  `crate::workload_search` (a `BTreeMap` of
  `MixedPrecisionGraph` keyed by `graph_id`); the strategy
  iterates it read-only and does not own the ordering.
- **Newtypes for authority-bearing values.** The existing
  `MeasuredEvaluatorAdapter::inner` is `Arc<dyn
  EcsEvaluationStrategy>` (the trait object) and the
  `behavioral_probe` is `Option<Arc<dyn BehavioralProbe>>`;
  no new raw strings or `u64` are introduced.
- **One authority per file.** Each new file states its
  authority in one sentence in the module doc.
- **No file named after an external project.** All three
  sub-modules are named for what they do in Prism's domain
  (`behavioral`, `progressive`, `mapped`).
- **Under 900 LOC and under 35 public items.** All four files
  are well under both limits (see the table above).

## Dependency direction (one-way)

The three sub-modules have a strict one-way dependency:
`mapped` depends on `behavioral` (for the trait) and
`progressive` (for the helpers); `behavioral` and `progressive`
do not depend on `mapped`. This matches the AGENTS.md rule
"crate dependency direction flows downward: higher-authority
crates depend on lower-authority; the reverse is forbidden" —
in this case, the strategy family (mapped) depends on the
behavioral interface (behavioral) and the reconstruction
primitives (progressive), not the other way around.

## Build & test results

**`cargo check -p prism-ecs-compile`:**
```
Finished `dev` profile [optimized + debuginfo] target(s) in 0.13s
```
The constitutional compiler crate builds cleanly. The 5
pre-existing `prism-ecs-constitutional` warnings
(`ambiguous glob re-exports`) are unchanged by this split.

**`cargo test -p prism-ecs-compile --lib evaluator::strategy`:**
```
running 2 tests
test evaluator::strategy::behavioral::tests::tree_spec_decoder_stub_returns_empty ... ok
test evaluator::strategy::progressive::tests::strategy_parses_genome_string ... ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 362 filtered out; finished in 0.00s
```

**`cargo test -p prism-ecs-compile --lib evaluator`:**
```
running 15 tests
test evaluator::canary_window::tests::canary_window_loads_reference_and_resizes_candidate ... ok
test evaluator::canary_window::tests::canary_window_rejects_empty_and_oversize ... ok
test evaluator::canary_window::tests::canary_window_recycle_advances_generation_and_clears ... ok
test evaluator::fail_closed::tests::fail_closed_daemon_integration_returns_explicit_error ... ok
test evaluator::fail_closed::tests::fail_closed_rejects_non_finite_fitness ... ok
test evaluator::fail_closed::tests::fail_closed_rejects_synthetic_in_production_mode ... ok
test evaluator::kv_evaluator::tests::mi300x_kv_evaluator_rejects_empty_inputs ... ok
test evaluator::kv_evaluator::tests::mi300x_kv_evaluator_rejects_mismatched_lengths ... ok
test evaluator::objective::tests::genome_for_format_maps_canonical_axis ... ok
test evaluator::objective::tests::probe_metrics_are_zero_for_identical_outputs ... ok
test evaluator::objective::tests::probe_metrics_reject_shape_mismatch_and_route_changes ... ok
test evaluator::strategy::behavioral::tests::tree_spec_decoder_stub_returns_empty ... ok
test evaluator::strategy::progressive::tests::strategy_parses_genome_string ... ok
test search::tests::evaluator_matrix_generates_full_prealldecode_coverage_for_representation ... ok
test search::tests::evaluator_matrix_runs_complete_tinygrad_profile_sweep ... ok

test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 349 filtered out
```

All 15 evaluator tests pass with the decomposition. The two
strategy tests now live in their respective sub-modules
(`behavioral::tests::tree_spec_decoder_stub_returns_empty` and
`progressive::tests::strategy_parses_genome_string`) and the
other 13 sibling-module tests are unchanged. The
`search::tests::evaluator_matrix_*` tests still pass, confirming
that the public API of `evaluator::strategy` is preserved.

## Files changed

| File                                                   | Change |
| ------------------------------------------------------ | ------ |
| `crates/prism-ecs-compile/src/evaluator/strategy.rs`   | deleted (replaced by `evaluator/strategy/` directory) |
| `crates/prism-ecs-compile/src/evaluator/strategy/mod.rs`        | new — module root with re-exports (64 LOC) |
| `crates/prism-ecs-compile/src/evaluator/strategy/behavioral.rs` | new — `BehavioralProbe` trait + tree-spec shapes + 1 test (142 LOC) |
| `crates/prism-ecs-compile/src/evaluator/strategy/progressive.rs` | new — representation helpers + 1 test (141 LOC) |
| `crates/prism-ecs-compile/src/evaluator/strategy/mapped.rs`    | new — `MeasuredEvaluatorAdapter` + `MappedTensorEvaluationStrategy` + workload/backend plumbing (747 LOC) |

Net effect: the 1,000-LOC `strategy.rs` godfile is gone; the
strategy is now three single-authority sub-modules totalling
1,094 LOC across their own files (the increase reflects the
new module-level docs, the `mod.rs` re-export hub, and the
per-file tests, not added functionality).

## What did NOT change

- `evaluator/mod.rs` — the re-export surface is unchanged; the
  sub-module structure is what changed, not the public API.
- `evaluator/objective.rs` — imports
  `super::strategy::{parse_genome_from_string, quantize_ternary,
  reconstruct_representation, BehavioralProbe}` still resolve
  through the `strategy/mod.rs` re-export hub.
- `evaluator/fail_closed.rs` — imports
  `super::strategy::MeasuredEvaluatorAdapter` still resolves
  through the `strategy/mod.rs` re-export hub.
- The engine's `compute-core/src/ecs/core/speculative.rs` —
  the engine keeps the MLX-coupled helpers and the
  ANE-coupled `MultiSpecDraftModel` per AGENTS.md criteria 1
  and 4. The re-exports of the canonical data types from
  `prism_ecs_compile::evaluator` (set up in commit `d0453c4f`)
  continue to resolve without code change.
- The 13 sibling-module tests (canary_window, fail_closed,
  kv_evaluator, objective, search) — they pass unchanged.
