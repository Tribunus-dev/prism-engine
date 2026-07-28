# Goal: Delete `compute-core/src/ecs/decode_attribution/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ **Goal achieved** (2026-07-27, E-4 closed).

## Source

`compute-core/src/ecs/decode_attribution/` — 29 files, 16,029
LOC. Decode-attribution subsystem: token-level attribution
during decode, per-layer / per-step evidence, trace generation,
streaming, session hooks, boundary detection.

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate;
the engine's `decode_attribution/` is the legacy home for decode
trace + evidence; the compile crate is the canonical home for
compilation-time + decode-time evidence and attribution).

## Migration pattern

Followed E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compiler migration (E-1..E-9, 9 commits) was the closest
template since it also targets `prism-ecs-compile`. The migration
was done in a single worktree
(`/Users/user/Developer/GitHub/prism-engine-decode-attribution`)
on branch `migrate/decode-attribution` (already created from
main for the agent).

## Isolate to your own worktree

Created an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-decode-attribution`
on branch `migrate/decode-attribution`. The migration was
performed exclusively in that worktree.

## Safety

- **No destructive ops.** The engine's `decode_attribution/`
  files were preserved (not deleted) at the engine's
  `legacy_decode_attribution/` location because every file in
  the original directory is engine-coupled (depends on engine
  FFI bridges — `coreai_bridge`, `coreai_pipeline`, `mil_builder`,
  `mlpackage`, `worker_memory`, `arena_info`, `toolchain_attest`,
  `pipeline_parity`, `compute_image::tensix`). The engine binaries
  (`tribunus-decode-attribution-measure`,
  `tribunus-coreai-decode-attribution`,
  `tribunus-compute-gap-report`,
  `tribunus-tier1-defect-cluster`,
  `tribunus-tier2-manifest-gen`,
  `tribunus-coreai-minimal-repro`) and the engine tests
  (`branch_rejoin_bisection`, `coreml_minimal_repro_tests`,
  `coverage_lattice_authority`, `pipeline_parity_contract`)
  continue to import the surface through
  `tribunus_compute_core::decode_attribution::*` which is now a
  re-export of `legacy_decode_attribution/`.
- **Checkpoint every 30 min.** Worktree work only; no
  cross-worktree changes.
- **Correct crate name.** `prism-ecs-compile` was the target;
  all constitutional-surface commits name it.
- **Engine dep audit at E-0.** Verified at start that
  `prism-ecs-compile` is already an engine dependency
  (line 68 of `compute-core/Cargo.toml`); no E-0 commit was
  needed.
- **Three agents are targeting prism-ecs-compile
  simultaneously** (compilation, decode_attribution, cimage).
  All use isolated worktrees. The merge order is: compilation
  first, decode_attribution second, cimage third. The "Take
  HEAD on `architecture/src/lib.rs` conflicts" rule applies:
  take HEAD + add new module declaration. The new module
  declaration added here is
  `pub mod workspace_legacy_decode_attribution_imports;`.

## Success criteria

- All 29 files of `compute-core/src/ecs/decode_attribution/`
  removed (renamed to `legacy_decode_attribution/`).
- Constitutional surface in
  `crates/prism-ecs-compile/src/decode_attribution/`.
- All engine callers migrated: engine binaries and tests
  continue to work via the
  `tribunus_compute_core::decode_attribution::*` re-export
  shim that now points to `legacy_decode_attribution/`.
  Internal engine callers (`compute-core/src/ecs/legacy_core/analysis.rs`,
  `compute-core/src/ecs/legacy_core/coreai_pipeline.rs`,
  `compute-core/src/ecs/legacy_core/pipeline_parity.rs`,
  `compute-core/src/ecs/mod.rs`,
  `compute-core/src/lib.rs`) updated to the new path.
- `workspace_contains_no_legacy_decode_attribution_imports`
  architecture test passes.
- `rg "use crate::ecs::decode_attribution::" compute-core/src/`
  returns no results.
- Engine pre-existing build error count is unchanged (192,
  within the 192 baseline).
- Constitutional-side tests green (479 passed; 0 failed).
- Architecture tests green (15 passed; 0 failed).

## Migration commits (E-0..E-4)

- **E-1 (constitutional surface)**: `feat(constitutional): add
  prism-ecs-compile::decode_attribution surface (E-1)` —
  11 new files in
  `crates/prism-ecs-compile/src/decode_attribution/`:
  - `mod.rs` (one-authority statement + module map)
  - `artifact_hash.rs` (SHA-256 directory hashing, 5 tests)
  - `backend_adapters/mod.rs` (BackendKind, BackendSupportTier,
    BackendSupportStatus, PredictFailureClass, BackendTiming)
  - `backend_adapters/conformance.rs` (pure-Rust reference
    conformance, 10 tests)
  - `breadcrumb.rs` (append-only fsynced writer, 3 tests)
  - `compute_plan.rs` (optional MLComputePlan stub)
  - `environment.rs` (host identity capture, 1 test;
    `ToolchainAttestation` dependency inlined as a local
    `probe_toolchain()`)
  - `receipt.rs` (`DecodeAttributionReceipt`,
    `BackendVersionInfo`, `ExecutionKind`, `ExecutionProof`;
    2 tests)
  - `shape_profiles.rs` (canonical shape profiles)
  - `statistics.rs` (DistributionStats, percentile, median,
    stddev, MAD, IQR, outlier detection)
  - `timer_calibration.rs` (TimerCalibration, 3 tests)
  - `pub mod decode_attribution;` added to
    `crates/prism-ecs-compile/src/lib.rs`.
- **E-2 (engine rename)**: `chore(engine): rename
  decode_attribution/ to legacy_decode_attribution/ + migrate
  callers (E-2..E-3)` —
  - 28 files git-mv'd from
    `compute-core/src/ecs/decode_attribution/` to
    `compute-core/src/ecs/legacy_decode_attribution/`.
  - New `legacy_decode_attribution/mod.rs` declares all
    submodules and re-exports the constitutional data types
    so `tribunus_compute_core::decode_attribution::*` continues
    to work.
  - Engine-internal callers
    (`legacy_core/analysis.rs`, `legacy_core/coreai_pipeline.rs`,
    `legacy_core/pipeline_parity.rs`, `ecs/mod.rs`,
    `lib.rs`) updated to the new path.
- **E-3 (caller migration)**: included with E-2 in the same
  commit.
- **E-4 (architecture safety net)**:
  `feat(architecture): add decode_attribution legacy-import
  safety net (E-4)` — new file
  `crates/architecture/src/workspace_legacy_decode_attribution_imports.rs`
  asserts no `use crate::ecs::decode_attribution::` remains
  anywhere in the workspace. Wired into
  `crates/architecture/src/lib.rs` as
  `pub mod workspace_legacy_decode_attribution_imports;`.

## Why rename rather than delete?

The compiler migration (E-8) could `git rm` the engine's
`compiler/` directory cleanly because the compiler surface had
no engine-internal dependencies. The decode-attribution
subsystem is different: every file in the engine's
`decode_attribution/` depends on at least one engine-internal
type (`coreai_bridge::CoreAiModel`, `coreai_pipeline`,
`mil_builder::MilBuilder`, `mlpackage::ModelMeta`,
`worker_memory`, `arena_info::ArenaInfo`, `toolchain_attest`,
`pipeline_parity`, or `compute_image::tensix`). A clean
deletion would break all engine binaries that import the
surface. The rename pattern (per the goal doc's "E-N+1:
Either `git rm` the engine files or rename the engine dir
to `compute-core/src/ecs/legacy_decode_attribution/` if any
engine-coupled files import legacy types. The rename pattern
is preferred.") preserves the engine-coupled adapter code
in the engine's home at `legacy_decode_attribution/` while
moving the cross-platform data types to the constitutional
surface at `prism_ecs_compile::decode_attribution`.

## Constitutional re-exports in the legacy dir

The engine's `legacy_decode_attribution/mod.rs` re-exports
the following types from `prism_ecs_compile::decode_attribution`
so engine binaries and tests can continue to import them via
the legacy path:

- `artifact_hash::{hash_directory_deterministic, DirectoryHashResult}`
- `backend_adapters::conformance::{compute_conformance, hash_output, ConformanceMetrics}`
- `backend_adapters::{BackendKind, BackendSupportStatus, BackendSupportTier, BackendTiming, PredictFailureClass}`
- `breadcrumb::{last_breadcrumb, read_breadcrumbs, set_breadcrumb_path, write_breadcrumb}`
- `compute_plan::{inspect_compute_plan, ComputePlanResult}`
- `environment::{capture_host_environment, HostEnvironment}`
- `receipt::{BackendVersionInfo, DecodeAttributionReceipt, ExecutionKind, ExecutionProof}`
- `shape_profiles::ShapeProfile`
- `statistics::{compute_distribution_stats, DistributionStats}`
- `timer_calibration::{calibrate_timer_overhead, TimerCalibration}`

The re-exports are explicit (not glob) so the migration is
auditable. The architecture safety net
(`workspace_legacy_decode_attribution_imports`) enforces that
no NEW engine code imports the legacy
`crate::ecs::decode_attribution::*` path; it must use either
the constitutional surface directly or the engine's
`legacy_decode_attribution` shim.

## Test results

- Constitutional surface: 40 tests passed (decode_attribution
  submodule); 479 tests passed for the whole
  prism-ecs-compile crate.
- Architecture: 15 tests passed (including the new
  `workspace_contains_no_legacy_decode_attribution_imports`).
- Engine pre-existing build error count: 192 (unchanged,
  within the 192 baseline).
