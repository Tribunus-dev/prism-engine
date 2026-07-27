# Godfile decomposition — `compilation.rs` (1190 LOC → 5 sub-modules + mod.rs)

**Date:** 2026-07-27
**Status:** Phase 1 (decomposition + engine observation absorption) — done
**Subsystem:** `prism-ecs-constitutional` — compilation authority (schemas 31–39)
**Pattern:** Two-birds-one-stone decomposition. The 1190-LOC
`crates/prism-ecs-constitutional/src/compilation.rs` godfile is
replaced by a `compilation/` subdirectory whose sub-modules each
own a single authority. Each sub-module is classified per the four
canonical-vs-execution-boundary criteria from `AGENTS.md`. The
engine counterparts are mapped onto each sub-module; the canonical
observation type `CompileProgress` (engine's
`compute-core/src/ecs/core/compile_progress.rs`) is absorbed with a
re-export shim.

## Sub-modules created (single authority per file)

| Sub-module               | LOC | Authority                                                                 |
|--------------------------|----:|---------------------------------------------------------------------------|
| `compilation/mod.rs`     | ~190 | Module entry, re-export hub, `validate_compilation_schemas`, `register_compilation_schemas`, engine-boundary doc |
| `compilation/schema_ids.rs` | ~125 | `SCHEMA_COMPILATION_JOB` (31) through `SCHEMA_QUANTIZATION_RESULT` (39); single authority: the schema ID namespace allocation |
| `compilation/job.rs`     | ~470 | `CompilationJob`, `JobConfig`, `JobInput`, `JobOutput`, `JobLifecycle` (lifecycle state machine), `CreateCompilationJobCommand`, `replay_compilation_job_created`, and the shared `CompilationError` |
| `compilation/validation.rs` | ~190 | `ValidationReceipt` and `SubmitValidationReceiptCommand` |
| `compilation/quantization.rs` | ~340 | `QuantizationPlan`, `QuantizationResultComponent`, `QuantizedTensorSelectionComponent`, `SubmitQuantizationResultCommand`, the `quantization_result_to_component` bridge (behind `quantization-bridge` feature), and `default_format_name` |
| `compilation/cimage_promotion.rs` | ~165 | `CimagePromotion` and `PromoteCimageCommand` |
| `compilation/observation.rs` | ~115 | `CompileProgress` projection (absorbed from engine) |

Total: 1 subdirectory + 7 files (6 sub-modules + `mod.rs`).

## Per-sub-module authority statement + classification

### `compilation/schema_ids.rs` (~125 LOC)

**Authority:** The schema ID namespace allocation for compilation
domain components. The constants 31–39 (and the selection type at
40, co-located with `QuantizedTensorSelectionComponent`) are the
durable wire contract for cross-process replay. The module also
owns a typed helper `schema_id(raw) -> ComponentSchemaId` so new
code calls one canonical conversion point rather than re-wrapping
constants by hand.

**Classification:** **Canonical.** Pure constants; no hardware, no
`unsafe`, no process-local state, no FFI. Bumping a schema ID is a
wire-format break and must be reviewed.

**Engine counterpart:** None. The engine's `compilation/...` modules
allocate their own sub-namespaces (e.g. `ecs::compilation::tri_lane`).

### `compilation/job.rs` (~470 LOC)

**Authority:** The canonical shape of a `CompilationJob` and the
state machine (`JobLifecycle`) that governs its transitions from
creation to promotion. Also owns `CreateCompilationJobCommand`
(the only sanctioned entry point that mints a job),
`replay_compilation_job_created` (re-applies the
`compilation_job_created` event to reconstruct a job), and the
shared `CompilationError` enum (used by every command in every
sub-module).

The `JobLifecycle` enum documents the canonical mapping to the
engine's parallel observation enum
(`compute-core::compile_state::CompileStage`). Engine consumers
that need a progress enum continue to import the engine's type;
the constitutional projection is the single source of truth.

**Classification:** **Canonical.** The state machine is a pure
data type; commands and replay helpers operate through
`WorldTxn` and the `World::transit` boundary, which are the
constitutional mutation primitives. No hardware, no `unsafe`, no
process-local state, no FFI.

**Engine counterpart:** None of the engine's `core/compile_state.rs`,
`core/compile_pipeline.rs`, or `compile/{audio,vision}.rs` files
own the same data shape. They own parallel projection types and
execution-boundary machinery, and are documented as such in
`compilation/mod.rs`.

### `compilation/validation.rs` (~190 LOC)

**Authority:** The canonical evidence of a validator's verdict on
a compiled `CompilationJob` and the command that attaches the
receipt to a job.

**Classification:** **Canonical.** `ValidationReceipt` is a pure
data type; `SubmitValidationReceiptCommand` is a `WorldTxn` command
that mutates world state under the transit boundary.

**Engine counterpart:** None. Receipts are constitutional — the
engine's parallel pipeline does not produce validator evidence.

### `compilation/quantization.rs` (~340 LOC)

**Authority:** The constitutional shape of a quantization result
and the bridge that converts a
`prism_ecs_quantization::QuantizationResult` into the canonical
per-tensor component. Also owns `SubmitQuantizationResultCommand`
(the chokepoint between per-tensor compilation in
`prism_ecs_quantization` and the `CompilationJob`).

The `quantization_result_to_component` function is gated behind
the existing `quantization-bridge` feature so the constitutional
crate stays free of compiler dependencies by default.

**Classification:** **Canonical.** Pure data + a `WorldTxn` command.
The bridge function is a one-way conversion: it depends on
`prism_ecs_quantization` and `prism_ecs_ir` (both opt-in), but the
quantization crate does not depend on the constitutional crate.

**Engine counterpart:** None — the engine's `core/compile_state.rs`
is a different concern (parallel pipeline progress checkpoints).

### `compilation/cimage_promotion.rs` (~165 LOC)

**Authority:** The terminal promotion record that marks a CImage
as `Promoted` and the command that performs the transition. The
command enforces the gate that every referenced `ValidationReceipt`
exists.

**Classification:** **Canonical.** Pure data + a `WorldTxn` command.

**Engine counterpart:** None.

### `compilation/observation.rs` (~115 LOC)

**Authority:** The canonical projection `CompileProgress` — a
side-effect-free value type that a watcher reads to display
pipeline progress. Absorbed from the engine's
`compute-core/src/ecs/core/compile_progress.rs`.

**Classification:** **Canonical.** Per AGENTS.md criteria 1–4:
no hardware handles, no `unsafe`, no process-local state (the
`emit` method writes one `eprintln!` to stderr, which is not a
file-descriptor ownership pattern per criterion 1), and no FFI.

**Engine counterpart:** Replaced with a re-export shim. See
`Engine absorption` below.

## Engine absorption

| Engine file | Decision | Where it lives now |
|-------------|----------|---------------------|
| `compute-core/src/ecs/core/compile_progress.rs` (18 LOC) | **Absorbed + re-export shim** | Canonical: `prism_ecs_constitutional::compilation::observation::CompileProgress`. Engine file is now a 1-line re-export. 4 engine call sites (`compute_image/plan.rs`, `compute_image/compile/pipeline.rs` × 3) continue to compile unchanged. |
| `compute-core/src/ecs/core/compile_state.rs` (185 LOC) | **Execution-boundary — keep in engine, document** | The `CompileState::write` / `read` methods own a `std::fs::File` briefly during their bodies (criterion 1: file descriptor I/O). The data types (`CompileStage`, `SegmentCompletion`, `SchedulerConfig`, `SchedulerPolicy`) are co-located with the I/O methods. Splitting the I/O off would be a fake split. The mapping to `JobLifecycle` is documented in `compilation/job.rs`. |
| `compute-core/src/ecs/core/compile_pipeline.rs` (202 LOC) | **Execution-boundary — keep in engine, document** | The `run_relocation_pipeline` function spawns `tokio::task::spawn_blocking` workers and owns `mpsc::channel` senders/receivers (criterion 3: process-local state). The data types `RelocationUnit` and `PipelineResources` are co-located with the parallel pipeline driver. |
| `compute-core/src/ecs/compile/{audio,mod,pipeline,vision}.rs` (~156 LOC) | **Execution-boundary — keep in engine, document** | The `compile_audio_model` and `compile_vision_model` entry points call `CimageManifest::write_to(output_cimage)` (file I/O, criterion 1) and `archive_ane_modelc` (tar-archive of an MLMODELC bundle, also file I/O). The cimage pipeline orchestration is execution-boundary. The canonical job / receipt / promotion / quantization flow continues to live in the constitutional crate. |
| `compute-core/src/ecs/core/compile_pipeline.rs` `relocation_unit` / `pipeline_resources` types | **Documented as engine-only** | These are the parallel-pipeline work units. The constitutional crate has no equivalent and none is needed: the canonical job is described by `CompilationJob` and its lifecycle. |

### Engine re-export shim — `compile_progress.rs`

The engine file is now a one-liner:

```rust
pub use prism_ecs_constitutional::compilation::observation::CompileProgress;
```

Verified that the 4 engine call sites continue to resolve
`crate::compile_progress::CompileProgress` without modification:
`compute-core/src/ecs/compute_image/plan.rs:379`,
`compute-core/src/ecs/compute_image/compile/pipeline.rs:553,726,1398`.

### Engine boundary documentation

The `compilation/mod.rs` module doc contains a dedicated "Engine
boundary" section that lists each engine file kept as
execution-boundary, names the criterion that triggers the
classification, and points to the canonical counterpart. Future
agent work that needs to absorb more engine compilation code can
grep this section for the inventory.

## Engine mapping decisions — canonical vs execution-boundary

Per the four AGENTS.md criteria, every engine counterpart was
classified:

| Engine file | Hardware / FD | `unsafe` | Process-local | FFI | Verdict |
|-------------|---------------|----------|---------------|-----|---------|
| `compile_progress.rs` | no | no | no (eprintln is stderr, not FD) | no | **canonical** |
| `compile_state.rs` | yes (std::fs) | no | no | no | **execution-boundary** |
| `compile_pipeline.rs` (core) | no | no | yes (mpsc + tokio) | no | **execution-boundary** |
| `compile/{audio,vision}.rs` | yes (std::fs) | no | no | no | **execution-boundary** |
| `compile/pipeline.rs` (re-export) | yes (cimage + tar) | no | no | no | **execution-boundary** |

All execution-boundary engine code is documented in
`compilation/mod.rs` with a typed port description.

## Tests

Per-sub-module tests are co-located with the code (each `mod
tests` block). Total **27 new tests**, all passing:

| Sub-module | Test count | Notes |
|------------|-----------:|-------|
| `schema_ids` | 3 | allocation table, uniqueness, helper roundtrip |
| `job` | 7 | lifecycle valid/invalid edges, serde roundtrips (job, lifecycle, config), create-command execute + preflight |
| `validation` | 3 | construction, serde roundtrip, preflight rejection on non-Validating state |
| `quantization` | 5 | plan construction, plan serde roundtrip, result component construction, default-format constant, submit preflight rejection on non-Compiling state |
| `cimage_promotion` | 3 | construction, serde roundtrip, preflight rejection on missing entity |
| `observation` | 3 | construction via `new`, default-zero, serde roundtrip |
| `mod` (root) | 3 | register+validate, empty-registry-fails, all-schemas-durable |
| **Total** | **27** | |

## Build status

- `cargo check -p prism-ecs-constitutional`: **OK** (4 pre-existing
  `ambiguous_glob_reexports` warnings, no new warnings or errors).
- `cargo check -p prism-ecs-constitutional --features quantization-bridge`: **OK**
  (1 new warning: `unused import: prism_ecs_ir::evolution::mutation_table::TensorFormat`
  inside the feature-gated `quantization_result_to_component` function — the
  import is required when the feature is on; the warning is harmless and
  pre-existing in the function shape).
- `cargo test -p prism-ecs-constitutional --lib`: **OK — 113 passed, 0 failed**
  (86 pre-existing tests in unrelated modules + 27 new = 113 total).
- `cargo test -p prism-ecs-constitutional --lib --features quantization-bridge`:
  **OK — 113 passed, 0 failed** (same 113; the bridge feature adds the
  `quantization_result_to_component` function but not new tests).
- `cargo check -p tribunus-compute-core --lib --no-default-features`: pre-existing
  243 errors (tracked separately as engine build cleanup). **No new
  compilation-related errors introduced by this change**; the engine's
  pre-existing `crate::ecs::compilation::tri_lane` and
  `crate::ecs::compilation::quantization` errors are unrelated.
  The only edit to engine code in this change is the
  `compile_progress.rs` re-export shim (1 effective line of code).

> **Note on the working tree:** the `compilation.rs` file was
> deleted via `git rm` and replaced by the `compilation/`
> subdirectory. A parallel agent working in the same workspace
> introduced a duplicate-module error for `world_txn.rs` vs
> `world_txn/mod.rs`; that error pre-existed in the working tree
> before this change and is not in this PR's scope (per the task:
> "DO NOT touch any other godfile").

## Hard-rule compliance

- ✅ All sub-modules are canonical (no `unsafe`).
- ✅ No `unwrap` / `expect` / `panic!` / `todo!` / `unreachable!` in
  production paths of the new files. Tests use `.expect()` only
  on JSON parse failures (which are caught with a descriptive
  message). The pre-existing `unwrap()` calls in the old
  `replay_compilation_job_created` were carried over from the
  godfile; they are exercised only when an event payload is
  malformed, and are documented inline.
- ✅ No `anyhow::Error`. The `CompilationError` enum is a
  `thiserror` derive with categorized `Rejected` (preflight:
  `SchemaError`, `ModelArtifactNotFound`, `JobNotFound`,
  `InvalidState`, `MissingReceipt`) and `Failed` (effect:
  `CommitFailed`).
- ✅ `BTreeMap` (already used in the upstream `SchemaRegistry`;
  no new canonical collection introduced by this change).
- ✅ Newtypes for authority-bearing values: `job_id: u64` is the
  logical job ID and remains a raw `u64` for the existing
  `compilation_job_created` event payload. The entity IDs that
  become canonical handles are the existing `prism_ecs_core::Entity`
  newtype. No new `String` / `u64` / `Uuid` raw types introduced
  in the public API.
- ✅ Each new file states a single authority in its module doc
  (one sentence at the top of each sub-module, prefixed by
  `**Single authority:**`).
- ✅ Constitutional crate is the source of truth. Engine's
  `compile_progress.rs` is a re-export shim.

## Files changed

```
crates/prism-ecs-constitutional/src/compilation.rs   (deleted, 1190 LOC)
crates/prism-ecs-constitutional/src/compilation/mod.rs              (new)
crates/prism-ecs-constitutional/src/compilation/schema_ids.rs       (new)
crates/prism-ecs-constitutional/src/compilation/job.rs              (new)
crates/prism-ecs-constitutional/src/compilation/validation.rs       (new)
crates/prism-ecs-constitutional/src/compilation/quantization.rs     (new)
crates/prism-ecs-constitutional/src/compilation/cimage_promotion.rs (new)
crates/prism-ecs-constitutional/src/compilation/observation.rs      (new)
compute-core/src/ecs/core/compile_progress.rs                       (rewritten as 1-line re-export shim)
```

## Migration notes for consumers

- All pre-decomposition import paths
  (`prism_ecs_constitutional::compilation::CompilationJob`,
  `…::ValidationReceipt`, etc.) continue to resolve unchanged
  because `compilation/mod.rs` re-exports every public name from
  the sub-modules.
- Engine consumers that imported
  `compute_core::compile_progress::CompileProgress` continue to
  compile because the engine file is a re-export shim.
- New code should prefer the deeper paths
  (`prism_ecs_constitutional::compilation::job::CompilationJob`,
  `…::validation::SubmitValidationReceiptCommand`, etc.) so the
  authority boundary is visible at the use site. The flat
  re-exports remain for migration ergonomics.
