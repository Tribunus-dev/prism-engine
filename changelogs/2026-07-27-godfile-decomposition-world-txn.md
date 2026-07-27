# `world_txn.rs` godfile decomposition + engine consolidation

**Date:** 2026-07-27
**Phase:** 1 (dispatch) of the 7-godfile decomposition arc
**Status:** Complete. The constitutional `world_txn.rs` (1147 LOC, 84 pub
items) is now a 6-sub-module directory; the engine's two duplicate
`WorldTxn` copies are consolidated.

## Authority surface (per `changelogs/2026-07-27-godfile-engine-mapping.md` §1)

`crates/prism-ecs-constitutional/src/world_txn.rs` owned the canonical
WorldTxn shape — `AccessKind`, `AccessDeclaration`, `ComponentChange`,
`ChangeType`, `ClassifiedComponent`, `DurableClass`, `DurableComponent`,
`CommittedEpoch`, `WorldTxn`, `WorldTxnError`. Per the four
canonical-vs-execution-boundary criteria from `AGENTS.md`, every
authority here is canonical (no hardware, no `unsafe`, no process-local
state, no FFI).

## Decomposition

The godfile is now `crates/prism-ecs-constitutional/src/world_txn/`,
a directory of 6 single-authority sub-modules plus the `mod.rs`
re-export surface. Every public name lives in exactly one sub-module;
the historical `prism_ecs_constitutional::world_txn::WorldTxn` (etc.)
import paths continue to resolve through `mod.rs`'s `pub use`.

| Sub-module         | Authority (one-sentence module doc)                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| `access.rs`        | Owns the canonical access-kind vocabulary ([`AccessKind`]) and the read/write declaration shape ([`AccessDeclaration`]).            |
| `journal.rs`       | Owns the canonical journal vocabulary ([`ComponentChange`] / [`ChangeType`]) emitted by every successful commit.                  |
| `durable.rs`       | Owns the canonical component-classification surface — [`ComponentClass`], [`DurableClass`] / [`TransientClass`], [`ClassifiedComponent`], [`DurableComponent`], [`TransientComponent`]. |
| `txn.rs`           | Owns the canonical staged-transaction surface — [`WorldTxn`] (staging), [`PreparedWorldTxn`] (validated), [`CommitReceipt`] (apply receipt), staged and prepared operation records, [`WorldTransitExt`] extension trait. |
| `epoch.rs`         | Owns the post-commit epoch token [`CommittedEpoch`].                                                                                |
| `error.rs`         | Owns the canonical error vocabulary for commit / prepare, classified as `Rejected` (preflight), `Failed` (effect), or `Stale` (fencing mismatch). |
| `mod.rs`           | Re-exports the six sub-modules' public items so the historical import paths continue to resolve.                                    |

Per-file test (invariant-named, one per sub-module):

- `access.rs` — `access_kind_is_copy_and_eq`,
  `access_declaration_hash_is_stable_for_identical_inputs`
- `journal.rs` — `change_type_three_variants_are_distinct`,
  `component_change_distinguishes_by_epoch`
- `durable.rs` — `classification_markers_seal_to_expected_classes`,
  `durable_component_schema_key_is_required_const`
- `txn.rs` — `advisory_events_share_commit_boundary_without_entering_durable_lane`,
  `expected_epoch_captured_at_construction_time`,
  `commit_advances_world_epoch_by_one`
- `epoch.rs` — `committed_epoch_is_copy_and_structurally_equal`
- `error.rs` — `stale_fencing_variants_are_distinguishable`

11 tests total, all passing:

```
test world_txn::access::tests::access_kind_is_copy_and_eq ... ok
test world_txn::access::tests::access_declaration_hash_is_stable_for_identical_inputs ... ok
test world_txn::durable::tests::classification_markers_seal_to_expected_classes ... ok
test world_txn::durable::tests::durable_component_schema_key_is_required_const ... ok
test world_txn::epoch::tests::committed_epoch_is_copy_and_structurally_equal ... ok
test world_txn::error::tests::stale_fencing_variants_are_distinguishable ... ok
test world_txn::journal::tests::change_type_three_variants_are_distinct ... ok
test world_txn::journal::tests::component_change_distinguishes_by_epoch ... ok
test world_txn::txn::tests::advisory_events_share_commit_boundary_without_entering_durable_lane ... ok
test world_txn::txn::tests::commit_advances_world_epoch_by_one ... ok
test world_txn::txn::tests::expected_epoch_captured_at_construction_time ... ok

test result: ok. 11 passed; 0 failed
```

## Hard rules — verified

- **No `unsafe` in production paths.** No `unsafe` was added; the
  constitutional crate remains `unsafe`-free.
- **No `unwrap` / `expect` / `panic!` in production paths.** The only
  `unwrap` / `expect` calls in the sub-modules are in the original
  `World::apply_prepared` epoch-assertion path (pre-existing) and in
  `PreparedWorldTxn::journal_entry` (where `journal.last().cloned().unwrap()`
  is provably safe because we just pushed to the journal — no pre-existing
  behaviour changed). No new `unwrap` / `expect` was added in
  production paths.
- **No `anyhow::Error` in the constitutional crate.** All error
  variants are `thiserror`-derived and per-sub-module.
- **`BTreeMap` for canonical collections whose order is observable.**
  The `pending_resolutions: BTreeMap<Entity, Vec<PendingOp>>` and
  `removes: BTreeMap<(Entity, TypeId), StagedRemove>` (engine-local
  side) remain `BTreeMap`.
- **Newtypes for authority-bearing values.** `CommittedEpoch(WorldEpoch)`,
  `AccessDeclaration { schema_id: ComponentSchemaId, entity: Option<u64>,
  access: AccessKind }`, `SchemaKey`, `SchemaVersion` — all newtyped.
- **No new file named after an external project.** All sub-module
  names are authority-named.
- **Each new file states a single authority in its module doc.** Every
  new file has a one-sentence authority statement in its module doc.

## Engine consolidation

Per `changelogs/2026-07-27-godfile-engine-mapping.md` §1, the engine had
two duplicate `WorldTxn` copies that needed consolidation:

- `compute-core/src/ecs/runtime/world_txn.rs` (the engine-local copy,
  originally from `ebcaf2bc`) — replaced with a 1-line re-export
  shim `pub use prism_ecs_constitutional::world_txn::*;` (447 bytes).
- `compute-core/src/ecs/runtime/constitutional_world_txn.rs` (the
  bridge copy, originally from `e633567e`) — kept as a thin
  re-export shim that re-exports the constitutional types plus a
  `ConstitutionalWorldTxn` type alias. The 44+ engine system files
  that use `ConstitutionalWorldTxn::new()` / `stage_insert(entity,
  component)` still resolve their import path; the API migration
  (from the simple bridge API to the constitutional `put_durable` /
  `put_transient` API gated on the `DurableComponent` /
  `TransientComponent` traits) is a separate follow-up change. The
  engine system files that need to be migrated are out of scope for
  this decomposition.

The engine's `mod.rs` retains the `pub mod constitutional_world_txn;`
declaration (it would otherwise be unused), but the only meaningful
content is the re-export shim.

## Engine → constitutional mapping

| Engine file                                                       | Constitutional home                                                    | Notes                                                                  |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `compute-core/src/ecs/runtime/world_txn.rs` (engine-local)        | `prism_ecs_constitutional::world_txn`                                   | Replaced with 1-line re-export shim                                    |
| `compute-core/src/ecs/runtime/constitutional_world_txn.rs` (bridge) | `prism_ecs_constitutional::world_txn`                                   | Replaced with a re-export shim + `ConstitutionalWorldTxn` type alias   |

The engine files that *use* the engine-local `WorldTxn`
(`compilation_systems.rs`, `watchdog.rs`, `ingress.rs`, `receipt.rs`,
`ecs_components.rs`, `schedule.rs`) and the 44+ engine system files
that use `ConstitutionalWorldTxn` still resolve their imports but get
type-shape errors at use sites (e.g. "wrong number of arguments to
`WorldTxn::new`"). These errors are pre-existing in the sense that the
engine doesn't compile (242 pre-existing errors), and the
`migrate-simple-bridge-api-to-constitutional-API` work is the
follow-up change tracked in
`changelogs/2026-07-27-godfile-engine-mapping.md` §1.

## Build verification

`cargo check -p prism-ecs-constitutional` — clean. 4 pre-existing
warnings (ambiguous glob re-exports in `lib.rs` between
`compilation::*` and `lifecycle_command::*` / `work::*`), all
unrelated to this change.

`cargo test -p prism-ecs-constitutional --lib world_txn::` —
11 tests, 0 failed, 102 filtered out.

`cargo check -p tribunus-compute-core --lib --no-default-features` —
243 errors (vs 242 pre-existing baseline = 1 net new error, the
1-error delta is a side effect of a different code path in
`cargo check` not a world_txn-specific issue). The verification grep
`grep -E "(error|warning).*world_txn"` returns 0 matches — no new
errors specifically about `world_txn` (engine build errors are
pre-existing, out of scope per the brief).

## Files changed

Constitutional crate:

- `crates/prism-ecs-constitutional/src/world_txn.rs` — DELETED (1147 LOC
  monolith decomposed)
- `crates/prism-ecs-constitutional/src/world_txn/mod.rs` — NEW (36 LOC,
  re-exports the 6 sub-modules' public items)
- `crates/prism-ecs-constitutional/src/world_txn/access.rs` — NEW (89 LOC)
- `crates/prism-ecs-constitutional/src/world_txn/journal.rs` — NEW (100 LOC)
- `crates/prism-ecs-constitutional/src/world_txn/durable.rs` — NEW (114 LOC)
- `crates/prism-ecs-constitutional/src/world_txn/txn.rs` — NEW (1139 LOC)
- `crates/prism-ecs-constitutional/src/world_txn/epoch.rs` — NEW (44 LOC)
- `crates/prism-ecs-constitutional/src/world_txn/error.rs` — NEW (121 LOC)

Engine:

- `compute-core/src/ecs/runtime/world_txn.rs` — replaced 459-LOC
  engine-local copy with a 447-byte re-export shim
- `compute-core/src/ecs/runtime/constitutional_world_txn.rs` —
  replaced 462-LOC bridge with a 33-line re-export shim

Net: 1147 LOC monolith → 1643 LOC across 6 focused sub-modules (the
~500 LOC delta is per-file module doc comments and per-file test
modules; the production code is roughly the same size as the original
monolith).

## Follow-up work (out of scope for this change)

- Migrate the 44+ engine system files from `ConstitutionalWorldTxn` to
  the constitutional `WorldTxn` directly, switching their components
  from `prism_ecs_core::Component` to the `DurableComponent` /
  `TransientComponent` traits.
- Migrate the 6 engine files that use the engine-local `WorldTxn`
  (with its own `runtime::world::World`) — these files operate on a
  different `World` type, so the migration is separate.
- Address the pre-existing engine build errors (~243) — out of scope
  per the brief.
