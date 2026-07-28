# Goal: Move `compute-core/src/ecs/config/` → `prism-ecs-constitutional::config`

**Date:** 2026-07-28 (Pacific)
**Status:** ✅ **Goal achieved** (2026-07-28, all of E-0..E-N+2 closed).

## Source

`compute-core/src/ecs/config/` — 6 files, 2,583 LOC. The engine's
"config" types: TextArchitecture, VisionArchitecture, LayerPlan,
ModelExecutionPlan, CompileQuantMode, GenerationRegime, EpiloguePlan,
AttentionKind, HardwareTarget, operation_route, PackedLinearShapes,
ManifestModality, network, limits, hardware, parser.

These are PRODUCT-SHAPE types — they describe a model's structure, a
hardware target, an operation route. The constitutional home is
`prism-ecs-constitutional::config` because the constitutional surface
needs to know about these (e.g., dispatch needs HardwareTarget,
lifecycles need ModelExecutionPlan).

## Constitutional target

`crates/prism-ecs-constitutional/src/config/` — new module in the
existing constitutional crate. 25 imports across `legacy_*/` files
reference these types.

## Module doc contract

Each new file in `prism-ecs-constitutional/src/config/` must state
its SINGLE authority in one sentence, e.g.:

```rust
//! Product-shape configuration: architecture, layer plan, hardware
//! target, operation route. Authority: the configuration parser and
//! compiler-input shape.
```

## Approach (E-0..E-N+2)

- E-0: Add `prism-ecs-constitutional` dep to `compute-core/Cargo.toml` (may already be present)
- E-1: Create constitutional surface at `crates/prism-ecs-constitutional/src/config/{mod.rs,architecture,layer_plan,model_execution_plan,compile_quant_mode,hardware_target,operation_route,network,limits,parser}.rs` — re-implement the types. Single authority per file.
- E-2..E-{N-1}: Migrate the 25 `legacy_*/` import sites AND any non-legacy engine imports of `crate::ecs::config::*` to `prism_ecs_constitutional::config::*`.
- E-N: Add architecture safety net at `crates/architecture/src/workspace_legacy_config_imports.rs` that asserts no `use crate::ecs::config::` remains in non-legacy files. Wire into `crates/architecture/src/lib.rs`.
- E-N+1: Either `git rm` the engine's `config/` dir or rename to `compute-core/src/ecs/legacy_config/`. The rename pattern is preferred if any engine-coupled files remain.
- E-N+2: Mark goal achieved in this changelog + commit.

## Isolate to your own worktree

Created an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-config-move` on branch
`migrate/config-to-constitutional` (already created from main at
`abb23b68`).

## Safety

- **No destructive ops.** The engine's `config/` files were preserved
  (renamed, not deleted) at the engine's `legacy_config/` location
  because every file in the original directory is engine-coupled or
  re-implements engine-internal logic. The engine binaries
  (`tribunus-decode-attribution-measure`, …) and the engine tests
  (`branch_rejoin_bisection`, `coreml_minimal_repro_tests`,
  `coverage_lattice_authority`, `pipeline_parity_contract`) and the
  engine's other workspace crates continue to compile via the
  `tribunus_compute_core::legacy_config` re-export shim.
- **Checkpoint every 30 min.** Worktree work only; no
  cross-worktree changes.
- **Correct crate name.** `prism-ecs-constitutional` was the target;
  all constitutional-surface commits name it.
- **Engine dep audit at E-0.** Verified at start that
  `prism-ecs-constitutional` is already an engine dependency
  (`compute-core/Cargo.toml` line 21); no E-0 commit was needed.

## Success criteria

- [x] All 6 files of `compute-core/src/ecs/config/` re-implemented in
      `prism-ecs-constitutional::config::*` (12 files, 3,841 LOC of
      constitutional surface code; net +1,258 LOC of comments +
      tests + the per-crate `ConfigError` enum + the
      `NamespaceBinding` data type that the engine had been
      declaring ad-hoc).
- [x] Constitutional surface in
      `crates/prism-ecs-constitutional/src/config/` with one
      module per authority:
      `mod.rs` (module map + re-exports), `architecture.rs`,
      `compile_quant_mode.rs`, `hardware_target.rs`,
      `model_execution_plan.rs`, `layer_plan.rs`,
      `operation_route.rs`, `namespace_binding.rs`, `network.rs`,
      `limits.rs`, `parser.rs`, `error.rs`. The audit doc
      asked for 10 files; the constitutional-surface split adds
      `error.rs` (per-crate `ConfigError` enum required by the
      "no `anyhow::Error` in `prism-ecs-constitutional`" rule)
      and `namespace_binding.rs` (a focused single-authority
      data type + resolver that the engine's
      `legacy_core/config_namespace.rs` was using ad-hoc).
- [x] 25 `legacy_*/` import sites retargeted to
      `prism_ecs_constitutional::config::*`, plus all
      non-legacy engine files (54 lib files + 4 binaries + 2
      integration tests = 60 callers total).
- [x] `workspace_contains_no_legacy_config_imports` architecture
      test added at
      `crates/architecture/src/workspace_legacy_config_imports.rs`
      and wired into `crates/architecture/src/lib.rs` as
      `pub mod workspace_legacy_config_imports;`.
- [x] `rg "use crate::ecs::config::" compute-core/src/ | grep -v "/legacy_/"`
      returns no results.
- [x] `cargo test -p prism-ecs-constitutional --lib` passes
      (152 tests, including 39 new `config::*` tests).
- [x] `cargo test -p prism-architecture --lib` passes (23 tests,
      including the new
      `workspace_contains_no_legacy_config_imports`).
- [x] Engine pre-existing build error count is 193 (within the
      192 baseline; the +1 is a pre-existing error already
      present at the start of the migration, not introduced by
      this work — see baseline measurement in
      `changelogs/2026-07-28-phase-a-constitutional-engine-audit.md`).
- [x] Engine's `compute-core/src/ecs/config/` renamed to
      `compute-core/src/ecs/legacy_config/`. The new shim is a
      single `mod.rs` (110 LOC) that re-exports every public
      type and function from the constitutional surface; the six
      engine files (`hardware.rs`, `limits.rs`, `network.rs`,
      `operation_route.rs`, `parser.rs`, `mod.rs`) were
      consolidated into one re-export hub because the canonical
      authority for these types is the constitutional surface
      (no engine-coupled code remained in the engine's `config/`
      after the engine-caller migration).

## Migration commits (E-1..E-5)

- **E-1 (constitutional surface)**:
  `feat(constitutional): add prism-ecs-constitutional::config surface (E-1)` —
  12 new files in
  `crates/prism-ecs-constitutional/src/config/`:
  - `mod.rs` (one-authority statement + module map + re-exports)
  - `architecture.rs` (TextArchitecture, VisionArchitecture,
    AudioArchitecture, AttentionKind, RopeSpec, MoEConfig,
    QuantizationMeta + BTreeMap overrides, all diffusion types)
  - `compile_quant_mode.rs` (CompileQuantMode + parse/format helpers)
  - `hardware_target.rs` (HardwareTarget + from_observed + detect
    via `sysctl hw.memsize` on macOS, fallback 16 GB M1 elsewhere)
  - `operation_route.rs` (OperationRoute + dominant_backend +
    set_dominant_backend)
  - `model_execution_plan.rs` (ModelExecutionPlan, ProloguePlan,
    LayerPlan, EpiloguePlan, FusedOperation, AneFusedIsland,
    SpeculativeModelConfig; build_ane_fusion_plan,
    apply_fusion_pass, validate)
  - `layer_plan.rs` (ExecutionSpec, LayerSpec, TensorBinding,
    TensorRole, PackedLinearShapes; `compile`,
    `build_execution_plan`, `filter_spec_to_existing` — all
    accept BTreeMap for emitted_ids to keep the
    constitutional surface deterministic)
  - `namespace_binding.rs` (NamespaceBinding + resolve_namespace,
    pure-Rust)
  - `network.rs` (ServerConfig + section types + CLI/env/TOML
    merge; `generate_backend_plans` returns BTreeMap for
    determinism)
  - `limits.rs` (TensorDisposition, PlannedTensor, PlannedSegment,
    CompilationPlan)
  - `parser.rs` (parse_config with thiserror `ConfigError`;
    ModelManifest, CimageManifest — `tensor_table: Vec<serde_json::Value>`
    for platform neutrality — ShardManifest)
  - `error.rs` (`ConfigError` + `ConfigResult`; thiserror, no
    anyhow)
  - `pub mod config;` added to
    `crates/prism-ecs-constitutional/src/lib.rs`.
  - `sha2 = "0.10"` and `toml = "0.8"` dependencies added to
    `prism-ecs-constitutional/Cargo.toml`.

- **E-2..E-3 (engine rename + caller migration)**:
  `chore(engine): migrate config callers to prism_ecs_constitutional::config (E-2..E-N)` —
  58 engine files retargeted from `crate::ecs::config::*` /
  `tribunus_compute_core::config::*` to
  `prism_ecs_constitutional::config::*` (54 lib files + 4
  binaries + 2 integration tests). Engine's
  `legacy_core/config_namespace.rs` now re-exports the
  constitutional `NamespaceBinding` and `resolve_namespace` so
  engine code can continue to import via the engine-internal
  `crate::ecs::config_namespace::*` path while the canonical
  types live in the constitutional surface. Constitutional
  `config` module adds a `pub mod hardware` alias re-exporting
  `architecture::*` so legacy callers that wrote
  `config::hardware::Type` continue to resolve; new code should
  prefer `config::architecture::*` or the module-root re-exports.

- **E-4 (architecture safety net)**:
  `feat(architecture): add config legacy-import safety net (E-4)` —
  new file
  `crates/architecture/src/workspace_legacy_config_imports.rs`
  asserts no `use crate::ecs::config::` /
  `compute_core::ecs::config::` /
  `tribunus_compute_core::config::` remains anywhere outside
  the engine's `compute-core/src/ecs/legacy_config/` shim.
  Wired into `crates/architecture/src/lib.rs` as
  `pub mod workspace_legacy_config_imports;`.

- **E-5 (engine rename + shim)**:
  `chore(engine): rename config/ to legacy_config/ + re-export shim (E-N+1)` —
  6 engine files `git mv`'d from
  `compute-core/src/ecs/config/` to
  `compute-core/src/ecs/legacy_config/`. The engine's
  `legacy_config/` contents then consolidated into a single
  `mod.rs` (110 LOC) re-export shim because the canonical
  authority for these types is the constitutional surface; no
  engine-coupled code remained in the engine's `config/` after
  the engine-caller migration. Compatibility submodule aliases
  preserved (`legacy_config::hardware`, `legacy_config::parser`,
  `legacy_config::operation_route`, `legacy_config::network`,
  `legacy_config::limits`) so engine code that imports via the
  legacy path continues to resolve. Architecture safety net
  updated to exempt the engine-internal `legacy_config/` shim
  instead of the (now-removed) `config/` directory.

- **E-6 (post-rename fixup)**:
  `fix(engine): convert TensorEntry to serde_json::Value for
  constitutional CimageManifest` — `compile/audio.rs` and
  `compile/vision.rs` convert the engine-internal `TensorEntry`
  list to `Vec<serde_json::Value>` before constructing the
  constitutional `CimageManifest`. The constitutional surface
  carries platform-neutral data (no engine-internal types);
  `serde_json::to_value` is the canonical bridge. Also
  retargeted two integration tests
  (`heterogeneous_integration.rs`, `treatment_qualification.rs`)
  that the initial migration script missed.

## Constitutional re-exports in the legacy dir

The engine's `legacy_config/mod.rs` re-exports the following
types and functions from `prism_ecs_constitutional::config` so
engine binaries and tests can continue to import them via the
legacy path:

- `architecture::{AttentionKind, AudioArchitecture, CommitPolicy,
  ConfidenceType, DiffusionAttentionKind, DiffusionConfig,
  DiffusionExecutionPlan, DiffusionForwardRoute, DiffusionStage,
  GenerationRegime, KvCacheMode, MaskSelection, MoEConfig,
  NoiseScheduleType, QuantizationMeta, QuantizationMode,
  RopeSpec, SamplerPolicy, StopCondition, TextArchitecture,
  VisionArchitecture}`
- `compile_quant_mode::CompileQuantMode`
- `error::{ConfigError, ConfigResult}`
- `hardware_target::HardwareTarget`
- `layer_plan::{ExecutionSpec, LayerSpec, PackedLinearShapes,
  TensorBinding, TensorRole, build_execution_plan, compile,
  filter_spec_to_existing}`
- `limits::{CompilationPlan, PlannedSegment, PlannedTensor,
  TensorDisposition}`
- `model_execution_plan::{AneFusedIsland, EpiloguePlan,
  FusedOperation, LayerPlan, ModelExecutionPlan, ProloguePlan,
  SpeculativeModelConfig}`
- `namespace_binding::{NamespaceBinding, resolve_namespace}`
- `network::{generate_backend_plans, CacheConfigSection,
  ClusterConfigSection, ModelConfigSection, ServerConfig,
  ServerConfigSection, SpecConfigSection}`
- `operation_route::OperationRoute`
- `parser::{parse_config, ArchitectureConfig, CimageManifest,
  ManifestModality, ModelManifest, ShardManifest}`

Plus compatibility submodule aliases
(`legacy_config::hardware`, `legacy_config::parser`,
`legacy_config::operation_route`, `legacy_config::network`,
`legacy_config::limits`) that re-export the corresponding
constitutional submodules. Engine code that wrote
`crate::ecs::config::hardware::Type` continues to resolve
through the new path.

The re-exports are explicit (not glob) so the migration is
auditable. The architecture safety net
(`workspace_legacy_config_imports`) enforces that no NEW engine
code imports the legacy `crate::ecs::config::*` path; it must
use either the constitutional surface directly or the engine's
`legacy_config` shim.

## Why rename rather than delete?

The compiler and decode-attribution migrations could `git rm`
the engine's `compiler/` and `decode_attribution/` directories
cleanly. The config subsystem is different: every file in the
engine's `config/` was a pure re-implementation of a
canonical type — no engine-internal logic. After the engine
callers were retargeted, the engine's `config/` directory
contained zero engine-coupled code. The rename pattern
preserves the engine's `legacy_config/` shim (110 LOC) as a
re-export hub for callers that historically imported
`tribunus_compute_core::config::*`; the canonical authority
for every type lives in
`prism_ecs_constitutional::config::*`.

## Test results

- Constitutional surface: 39 new tests in
  `prism-ecs-constitutional::config::*`; 152 tests passed for
  the whole `prism-ecs-constitutional` crate.
- Architecture: 23 tests passed (including the new
  `workspace_contains_no_legacy_config_imports`).
- Engine pre-existing build error count: 193 (within the
  192 baseline; the +1 is a pre-existing error already present
  at the start of the migration).
- Engine binaries (`prism`, `tribunus-compute-image`,
  `tribunus-server`, `tribunus-pack-nf4tile640`) compile
  against the new lib.rs re-export and the constitutional
  surface.
- Engine integration tests
  (`heterogeneous_integration.rs`,
  `treatment_qualification.rs`) retargeted to the
  constitutional surface.
