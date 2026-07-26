# Compute-core Absorption — Phase 5: Documentation Update (2026-07-25)

**Scope:** Phase 5 of the `compute-core.legacy/` absorption plan.
Update the project documentation to reflect the new state after the
absorption work from Phases 0–4C follow-up.

**Status:** Complete. No code changes — this is a docs-only commit.

## Files updated

| File | Net change | What changed |
|---|---|---|
| `changelogs/2026-07-25-compute-core-legacy-integration-plan.md` | **new** (246 lines) | Master integration plan: per-phase status, commit pointers, original changelog references, pre-existing build issues, end state. |
| `AGENTS.md` | +16 / -6 | Project layout section: each constitutional crate now names the new modules it owns (`admission_gates`, `buffer_lifetime_plan`, `engine_systems`, `engine_receipts`, `attention_sink`, `kernel_generation`, `compile_pipeline`, `compile_planning`, `hardware_tuning`, `fusion_analysis`, `fusion_scheduling`, `cimage_pipeline/`, `cimage_packer/`, `cimage_validation/`, `text_architecture_extract`, `mil_builder`, `mil_layer_programs`, `manifest`); engine now described as `compute-core/` (renamed from `compute-core.legacy/`) and a sibling workspace member; `compute-core/compute-core.legacy/` noted as the pre-absorption archaeology snapshot. Testing instructions note the engine's pre-existing build errors and direct callers to focused crate commands. Pre-existing build issues (`prism-metal-runtime` not in workspace; `prism-metal-runtime` → `tribunus-compute-core` dep is broken) called out. |
| `CAMPAIGN.md` | +89 / -7 | Subsystem Registry: cleaned up duplicate rows for #3 (Model Deployment & Residency was repeated 3×), #12 (Persistence & Projections was repeated 3×); added 10 new subsystem rows (14–23) for the absorbed surfaces (Engine Receipts, Attention Sinks, GGUF Manifest Extraction, ANE MIL Builder, Compile-Phase Admission Gates, Buffer Lifetime Planning, Hardware Tuning & Kernel Generation, Engine Singleton Systems, Text Architecture Extraction, Engine Runtime WorldTxn). Each new row links to the source commit and the original changelog. Workspace Baseline snapshot adds a "compute-core.legacy absorption" paragraph. Project Absorption Backlog adds a new "Resolved during the 2026-07-25 `compute-core.legacy` absorption" sub-section with the per-file original → re-implementation mappings (commit `472d9754`, `14e8edb1`, `b7d92c40`, `7cd96e16`, `ebcaf2bc`). |
| `/Users/user/.minimax/skills/prism-constitutional-rust-ecs/references/project-absorption.md` | +47 / -5 | Concrete Violations table: removed the `compute-core.legacy/` row, added new `compute-core/` row (engine is in active absorption; do not add new external-named files) and a new `compute-core/compute-core.legacy/` sub-tree row (vendored archaeology). `crates/prism-gguf/` row updated to note the new `manifest.rs` sub-module. New "Resolved during the 2026-07-25 `compute-core.legacy` absorption" sub-section lists the 14 absorbed files (10 `system/` + 3 `compute_image/` + 3 `core/` + 1 `core/mil_builder`) with their original → re-implementation mapping and the new file's Prism-domain authority statement. |

## Subsystem rows added to `CAMPAIGN.md`

The following 10 subsystems are added to the Subsystem Registry. The
schema, entity kind, and status columns reflect the per-file changelogs.

| # | Subsystem | Status | Owner | Source commit |
|---|---|---|---|---|
| 14 | **Engine Receipts** | `Shadow` | runtime | `b7d92c40` |
| 15 | **Attention Sinks** | `Shadow` | runtime | `b7d92c40` |
| 16 | **GGUF Manifest Extraction** | `Shadow` | kernel | `b7d92c40` |
| 17 | **ANE MIL Builder** | `Canonical` | ane | `7cd96e16` |
| 18 | **Compile-Phase Admission Gates** | `Shadow` | constitutional | `472d9754` |
| 19 | **Buffer Lifetime Planning** | `Shadow` | runtime | `472d9754` |
| 20 | **Hardware Tuning & Kernel Generation** | `Shadow` | kernel | `472d9754` |
| 21 | **Engine Singleton Systems** | `Shadow` | runtime | `472d9754` |
| 22 | **Text Architecture Extraction** | `Shadow` | artifact | `472d9754` |
| 23 | **Engine Runtime WorldTxn** | `Canonical` (engine-local) | runtime | `ebcaf2bc` |

**Status convention.** A re-implementation in a constitutional crate
enters at `Shadow` and advances to `Canonical` only when (a) the
original engine file is deleted and (b) the constitutional path has a
propagation test. Subsystems 18–22 reached `Shadow` with the original
engine files deleted in the same commit (single authority).
Subsystems 14–16 left the originals in place for shadow comparison and
remain `Shadow` until coordinated deletion. Subsystem 17 is
`Canonical` because the engine file is now a re-export shim.
Subsystem 23 is engine-local and is `Canonical` within the engine.

## Cross-references

This docs update does not introduce new facts. Every claim is
cross-referenced to one of the following primary sources:

- `changelogs/2026-07-25-compute-core-absorption-phase-2-system.md`
  (commit `472d9754`): `system/` absorption — 10 files, 162 direct
  world mutations eliminated, 140 new tests.
- `changelogs/2026-07-25-compute-core-absorption-phase-3-runtime.md`
  (commit `ebcaf2bc`): 10 remaining direct world mutations → engine-
  local `WorldTxn` at `compute-core/src/ecs/runtime/world_txn.rs`.
- `changelogs/2026-07-25-compute-core-absorption-phase-3-4a-runtime-audit.md`
  (commit `c5ad9070`): read-only audit of 25+ partially-absorbed
  subsystems.
- `changelogs/2026-07-25-compute-core-absorption-phase-4b-compute-image.md`
  (commit `14e8edb1`): 3 `compute_image/` files → `cimage_pipeline/`,
  `cimage_packer/`, `cimage_validation/`.
- `changelogs/2026-07-25-compute-core-absorption-phase-4c-core.md`
  (commit `b7d92c40`): 3 `core/` files → `engine_receipts`,
  `attention_sink`, `manifest`.
- Commit `7cd96e16`: `mil_builder` absorbed (engine → prism-ane).
- Commit `ef826363`: Phase 0+1 mechanical cleanup (engine rename,
  4 shim dirs removed from `compute-core/src/ecs/mod.rs`).

## Pre-existing build issues (out of scope)

The doc updates do not address the two pre-existing build issues
called out in the integration plan:

- `crates/prism-metal-runtime/` is not in the workspace
  (`Cargo.toml` `[workspace] members` array). Tracked separately.
- `prism-metal-runtime` → `tribunus-compute-core` dependency is
  broken (the engine builds with ~219 pre-existing errors). Tracked
  separately.

## Out-of-scope work (in progress via parallel dispatches)

The following work is in progress in parallel dispatches and will be
folded into the docs at a later phase:

- Phase 2.5: `system/` mutations in the remaining 42 files (writer &
  effect boundaries).
- Phase 4B continuation: more `compute_image/` files per the Phase
  4B roadmap.
- Phase 4C continuation: more `core/` files per the Phase 4C roadmap.

## Build status

- `cargo check --workspace` — succeeds with the constitutional
  libraries clean (only pre-existing warnings, no new errors
  introduced by this commit).
- The engine (`compute-core/`) has the same ~219 pre-existing build
  errors as before this commit (per the Phase 3 changelog baseline).
  This commit does not touch the engine code.
