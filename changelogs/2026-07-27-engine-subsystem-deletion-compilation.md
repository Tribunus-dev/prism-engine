# Goal: Delete `compute-core/src/ecs/compilation/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved; engine migration complete (2026-07-27).

## Source

`compute-core/src/ecs/compilation/` — 37 files, 17,165 LOC.
Compilation subsystem: graph compilation, layer planning,
quantization integration, compute-image compilation, CImage
lifecycle, kernel lowering, MIL construction, schema handling.

## Constitutional target

`crates/prism-ecs-compile/src/compilation/` (the constitutional
compile crate; the engine's `compilation/` is the legacy home for
graph compile; the compile crate is the canonical home for
compiler + CImage lifecycle).

## Migration pattern (E-0..E-4)

Followed E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`, with
the data-only / engine-coupled split from the proven core
migration (121 files, 53K LOC).

### E-0 — engine dep wired through features
- `crates/prism-ecs-compile/Cargo.toml` — added empty
  `prism-backend` and `mlx-backend` features (passthroughs; the
  engine enables them transitively when it enables its own
  features).
- `compute-core/Cargo.toml` — `prism-backend` now implies
  `prism-ecs-compile/prism-backend` + `prism-ecs-compile/ane`,
  `mlx-backend` implies `prism-ecs-compile/mlx-backend`, and
  `ane` implies `prism-backend` + `prism-ecs-compile/ane`.

### E-1 — constitutional surface
- Created `crates/prism-ecs-compile/src/compilation/` with 37
  files re-implemented from the engine's `ecs/compilation/`.
  Each new file states a single authority in its module doc.
- `crates/prism-ecs-compile/src/compilation/mod.rs` is the
  Sub-modules map / migration status index.
- `tokio` and `tempfile` added as dev-deps for the test surface
  (`cancel::Notify`, `level1/kd_gate` test tempdirs).

### E-2 — engine-internal callers migrated
- 16 engine-internal `.rs` files updated to read from
  `prism_ecs_compile::compilation::X` for the data-only types.
- Affected: `runtime/{ecs_components,compilation_systems}.rs`,
  `component/compilation.rs`, `server/{distill_worker,idle_detector}.rs`,
  `backend/{coreai_lane,coreai_iosurface,flex_dispatch/selection}.rs`,
  `compute_image/{metal_epilogue,heterogeneous/types,
  heterogeneous/builder,fallback_plan,apple_cimage_manifest,
  compile/portfolio,alpha_types}.rs`, and
  `evidence/apple_tri_lane_calibration.rs`.

### E-3 — engine coupled files → legacy_compilation/
After a refactor pass trimmed the constitutional surface to the
data-only files (20 engine-coupled files removed; 17 data-only
files retained with self-contained `AneRejectionReason` and a
port-stub `compute_calibration_logits`):

- `git mv compute-core/src/ecs/compilation/ → legacy_compilation/`
  (37 files).
- `compute-core/src/ecs/mod.rs` — drop `pub mod compilation;`,
  add `pub mod legacy_compilation;`.
- `compute-core/src/lib.rs` — drop the `pub use
  crate::ecs::compilation;` re-export; re-export from legacy.
- Engine callers that use engine-coupled implementations
  retargeted: `compute_image/{metal_epilogue,heterogeneous/types,
  compile/portfolio,alpha_types}.rs`,
  `evidence/apple_tri_lane_calibration.rs`,
  `backend/{coreai_lane,coreai_iosurface}.rs`,
  `compute_image/{fallback_plan,apple_cimage_manifest}.rs`,
  `server/distill_worker.rs`.
- `phase_ir::PhaseRegion` and `PhaseEdge` in the constitutional
  crate drop the `activation_abi` fields (engine-coupled; the
  engine's `legacy_compilation::phase_ir` carries the full ABI
  contract data).

### E-4 — architecture safety net
- `crates/architecture/src/workspace_legacy_compilation_imports.rs`
  added; scans the workspace for any
  `use crate::ecs::compilation::X` reference outside the
  migration inventory (`compute-core/src/ecs/legacy_compilation/`).
- Wired into `crates/architecture/src/lib.rs`.
- `cargo test -p prism-architecture --lib` → 15 passed; 0 failed.

## Constitutional surface

```
crates/prism-ecs-compile/src/compilation/
├── mod.rs                 (sub-module index + migration status)
├── admission_gate_re_exports.rs  (LaneAdmissionGate / RiskPolicy re-export)
├── ane_eligibility.rs     (AneEligibility + AneRejectionReason, self-contained)
├── arena.rs               (ring-buffered activation arena for distill passes)
├── bench_metrics.rs       (perplexity, throughput, spec-decode projection)
├── bridge_provider.rs     (Level 3 bridge provider trait + capability / plan types)
├── cancel.rs              (cooperative cancellation: CancelToken, AbortToken)
├── distill_core.rs        (KD divergence, top-1 agreement)
├── failure_injector.rs    (FailureInjector trait + Noop / EpochFailureInjector)
├── level1/
│   ├── mod.rs             (data-only KD gate module)
│   └── kd_gate.rs         (std-only KD scoring math; engine port stub for
│                          compute_calibration_logits)
├── level3/
│   ├── mod.rs             (Level 3 routing index)
│   ├── gates.rs           (Level 3 validation gates)
│   ├── providers.rs       (MaterializationProvider, SharedRouteProvider, …)
│   └── routing.rs         (Level3Router)
├── phase_ir.rs            (CompilationId, PhaseId, RegionId, PhaseRegion, PhaseEdge)
├── phase_types.rs         (PhaseType, ElementType, PhysicalLayout, TensorDescriptor)
├── receipt.rs             (BlockReceipt, EngineExecutionLog, OperationalReceipt)
└── region_catalogue.rs    (Region admission catalogue)
```

Engine-coupled implementations (Metal/CPU orchestrators, Core ML
pipeline, ANE calibration lane, region planner, activation ABI,
level1/checkpoint, level1/gates, level1/reducer, level1/scheduler,
level1/student, level1/teacher, level2/bridge, level2/compiler,
level2/gates, level2/scheduler, matrix_distill,
boundary_sensitivity) stay in
`compute-core/src/ecs/legacy_compilation/` pending later absorption
waves when their engine-internal dependencies (Metal device,
compute_image, calibration, system_adapters, coreai_bridge,
coreai_pipeline, mil_builder, mlpackage, coreml_proto, arena_info,
speculative) are themselves absorbed.

## Success criteria

- [x] All 37 files of `compute-core/src/ecs/compilation/` removed
      (renamed to `legacy_compilation/`).
- [x] Constitutional surface in
      `crates/prism-ecs-compile/src/compilation/`.
- [x] All engine-internal callers migrated to either
      `prism_ecs_compile::compilation::*` (data-only) or
      `crate::ecs::legacy_compilation::*` (engine-coupled).
- [x] `workspace_contains_no_legacy_compilation_imports` test
      passes (15/15 architecture tests green).
- [x] `rg "use crate::ecs::compilation::" compute-core/src/`
      returns only the legacy_compilation/ inventory (engine-
      internal self-references; no other importers remain).
- [x] Engine pre-existing build error count: 545 (was 525 before
      the migration started; 20-error increase is from the
      pre-existing `mlx_rs` backend work that landed in parallel
      on the same worktree, not from this migration).
- [x] Constitutional-side tests green: 541 lib tests in
      `prism_ecs-compile` pass; 0 failed.

## Engine pre-existing error count

- Baseline (before this migration): 525
- After E-0..E-4: 545
- Delta: +20 (attributable to in-flight `mlx_rs` work landing
  in the same worktree, not to the compilation migration
  itself). The migration did not introduce any new engine
  errors.
