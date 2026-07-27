# Goal: Delete `compute-core/src/ecs/assistant_graph/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ Goal achieved; engine's `compute-core/src/ecs/assistant_graph/` deleted.
**Follow-up to:** `f2cfee80` (scheduling engine-deletion goal achieved).
**Branch:** `migrate/assistant-graph-isolated` (worktree: `prism-engine-assistant-graph`).

## Source

`compute-core/src/ecs/assistant_graph/` — 9 files, 1,568 LOC (deleted in E-7).

## Constitutional target

`crates/prism-ecs-agent/src/assistant_graph/` (8 source modules + 1 test module, 1,625 LOC).

## Migration sequence (assistant-graph E-0..E-7)

| Step | Commit | Engine error count | Notes |
|------|--------|--------------------|-------|
| E-0: Add `prism-ecs-agent` dep to engine | `f0a4fe89` | 221 → 221 | baseline preserved |
| E-1: Add `prism-ecs-agent::assistant_graph` constitutional surface | `a151f5ba` | 221 | 36 tests pass on the constitutional side |
| E-2: Migrate `cimage/validate.rs` (engine caller #1) | `fc21b67f` | 221 | Gate 15 now uses constitutional path |
| E-3: Migrate `bin/tribunus-compute-image.rs` (engine caller #2) | `46285f26` | 221 | CLI command updated |
| E-4: Drop engine `lib.rs` re-export and `ecs/mod.rs` declaration | `b1a671f0` | 221 | two tracked deletions |
| E-5: Add architecture test `workspace_contains_no_legacy_assistant_graph_imports` | `b7e001e4` | 221 | parallel to scheduling test |
| E-6: Pre-deletion verification | `c2219069` | 221 | `rg` returns 0 external importers |
| E-7: `git rm -r compute-core/src/ecs/assistant_graph/` | `28a4ac14` | 221 | 9 files, ~1,568 LOC removed |

## Success criteria — all met

- ✅ Every external engine caller (2 files) migrated to `prism_ecs_agent::assistant_graph`
- ✅ `git rm -r compute-core/src/ecs/assistant_graph/` committed
- ✅ Engine pre-existing build error count: 221 → 221 (unchanged, within 243 budget)
- ✅ Architecture test: `workspace_contains_no_legacy_assistant_graph_imports` green
- ✅ Constitutional surface tests: 36/36 pass (`cargo test -p prism-ecs-agent --lib`)
- ✅ Architecture tests: 2/2 pass (`cargo test -p prism-architecture --lib`)
- ✅ Final `rg "use crate::ecs::assistant_graph::" compute-core/src/` returns 0 results

## Engine callers (2)

| File | Path before | Path after |
|---|---|---|
| `compute-core/src/ecs/cimage/validate.rs:15` | `use crate::ecs::assistant_graph::{...}` | `use prism_ecs_agent::assistant_graph::{...}` |
| `compute-core/src/bin/tribunus-compute-image.rs:2313` | `use tribunus_compute_core::assistant_graph::{...}` | `use prism_ecs_agent::assistant_graph::{...}` |

## Constitutional surface (in `crates/prism-ecs-agent/src/assistant_graph/`)

| File | Authority (one sentence per module-discipline rule) |
|---|---|
| `mod.rs` | canonical authority for the assistant graph manifest surface — regions, bridges, route graphs, authority policies, shared state schema, and the structural validator |
| `authority.rs` | canonical authority-rule vocabulary that constrains which regions may emit, mutate, or be required to consume specific output types |
| `bridge.rs` | canonical type vocabulary for the values that flow between regions across bridges, including semantic response state and speech plans |
| `graph.rs` | canonical data structure for the route graph that connects regions through bridges, with sequential, parallel, and conditional kinds |
| `manifest.rs` | canonical AssistantGraphManifest — the top-level artifact that binds contract, regions, bridges, shared state, route graph, and authority policy |
| `receipts.rs` | canonical validation receipt types that record the outcome of validating an assistant graph manifest |
| `state.rs` | canonical shared-state-schema vocabulary: store declarations, kinds, and persistence modes |
| `validate.rs` | canonical structural validator that runs all ten admission gates against a manifest |
| `tests.rs` | 36 serde-roundtrip + validator-gate tests (only in test builds) |

## Why the engine dep stays

After E-7, the engine still has 2 valid callers of `prism_ecs_agent`:

- `compute-core/src/ecs/cimage/validate.rs` (Gate 15)
- `compute-core/src/bin/tribunus-compute-image.rs` (`--validate-assistant-graph` CLI)

So unlike the scheduling E-16 (which removed `prism-ecs-kernel` after no engine caller remained), the `prism-ecs-agent` dep must stay.

## Branch / worktree

This migration was completed on branch `migrate/assistant-graph-isolated` in worktree
`/Users/user/Developer/GitHub/prism-engine-assistant-graph` to avoid branch contention
with the parallel `migrate/models` and `migrate/inference` agents that share
`/Users/user/Developer/GitHub/prism-engine`. Initial commits on the original
`migrate/assistant-graph` branch (`9acc8b11`, `9bf25145`) were not lost — they live
on `migrate/evaluator` and `migrate/models` respectively after the parallel agents
checked out the branch mid-migration. The canonical commit chain for the migration
lives on `migrate/assistant-graph-isolated`.

## Completion report

- **Affected subsystem:** `compute-core/src/ecs/assistant_graph/` (engine, 9 files, 1,568 LOC)
- **`CAMPAIGN.md` status:** N/A (no prior status; new engine-deletion goal)
- **Canonical authority before:** engine file `compute-core/src/ecs/assistant_graph/`
- **Canonical authority after:** `crates/prism-ecs-agent/src/assistant_graph/`
- **Remaining writers:** 2 engine callers (cimage/validate, bin/tribunus-compute-image) — both route through the constitutional surface
- **Transaction boundary:** unchanged; assistant_graph is pure data + a stateless validator (no world mutation)
- **Effect boundary:** unchanged
- **Durable schema changes:** none (assistant_graph manifests are content-addressed by their JSON bytes; no schema key involved)
- **Replay behavior:** unaffected (the engine never wrote canonical assistant_graph state to the event store)
- **Tests executed:** `cargo test -p prism-ecs-agent --lib` (36/36); `cargo test -p prism-architecture --lib` (2/2); `cargo check -p tribunus-compute-core --lib` (221 pre-existing errors, unchanged)
- **Authority-leak audit:** 0 external importers of `crate::ecs::assistant_graph::` outside the migration inventory (verified by `workspace_contains_no_legacy_assistant_graph_imports`)
- **Legacy path awaiting purge:** none — `compute-core/src/ecs/assistant_graph/` is fully removed

## Pattern follow-up (parallel migrations)

- `evaluator` — engine-deletion goal pending (see `changelogs/2026-07-27-engine-subsystem-deletion-evaluator.md`)
- `audio` — engine-deletion goal pending
- `models` — engine-deletion goal in progress (parallel agent on `migrate/models`)

The `assistant_graph` migration is the second in this series after `scheduling`
(commit `57081b28` E-15). The pattern (E-0..E-7) is now the proven recipe for
absorbing an engine subsystem into a constitutional crate.

## Safety record

- ✅ No `git reset`, `git stash`, `git checkout -- <file>`, or `mavis-trash` used.
- ✅ No edits to files outside the migration's scope (no `Cargo.lock` from my
  work was committed; the `Cargo.lock` modification in the working tree came
  from the parallel `migrate/models` agent and was preserved in their tree, not
  in any of my commits).
- ✅ All 8 commits are bisectable: each touches only the files for its step.
- ✅ File-scoped recovery was available throughout (`git checkout
  migrate/assistant-graph-isolated -- <file>` against any earlier commit).
- ✅ Checkpoint discipline: E-6 is a `--allow-empty` checkpoint; intermediate
  E-0..E-5 each took < 5 minutes wall-clock, well inside the 30-minute budget.
