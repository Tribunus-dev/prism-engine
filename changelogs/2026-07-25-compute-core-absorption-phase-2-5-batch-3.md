# Compute-core Absorption — Phase 2.5 Batch 3: Verification, No-Op

**Date:** 2026-07-26
**Agent:** Branch session, Phase 2.5 batch 3
**Status:** No-op verification — the 5 files in this batch were already
fully ported in commit `e633567e` (Phase 2.5, port 100 remaining
system/ mutations to WorldTxn). **Zero direct `world.spawn()` /
`world.add_component()` / `world.remove_component()` /
`world.get_component_mut()` / `world.insert()` calls remain in any
of the 5 files.**

## Task as received

Port 12 remaining direct world mutations across 5 `system/` files to
the engine-local `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs` (added in commit
`ebcaf2bc`):

- `compute-core/src/ecs/system/capability_registry_sys.rs` (3 mutations)
- `compute-core/src/ecs/system/archive.rs` (3 mutations)
- `compute-core/src/ecs/system/tts.rs` (2 mutations)
- `compute-core/src/ecs/system/session_init.rs` (2 mutations)
- `compute-core/src/ecs/system/quant_plan.rs` (2 mutations)

The brief assumed the engine-local `WorldTxn` is the target. As
documented in the Phase 2.5 changelog
(`changelogs/2026-07-25-compute-core-absorption-phase-2.5-system-rest.md`),
the engine-local `WorldTxn` operates on the **engine runtime `World`**
(`compute-core/src/ecs/runtime/world.rs`), which has no
`EntityKind`, no `name`, and uses `world.insert(entity, comp)` /
`world.remove::<T>(entity)`. The system files receive a `&mut World`
that is the **constitutional `prism_ecs_core::World`** (re-exported as
`crate::ecs::World` from `compute-core/src/ecs/mod.rs:217`), which has
`EntityKind` / `name` and uses `world.add_component(entity, comp)` /
`world.remove_component::<T>(entity)`. The two `World` types and their
mutation methods are different. A direct port of the system files to
the engine-local `WorldTxn` would not compile.

The Phase 2.5 commit `e633567e` resolved this by adding
`ConstitutionalWorldTxn` in
`compute-core/src/ecs/runtime/constitutional_world_txn.rs` — a bridge
that mirrors the engine-local `WorldTxn` API shape (typed
`stage_insert::<T>` / `stage_insert_on::<T>` / `stage_remove::<T>`,
`PendingToken` for not-yet-allocated spawns, `BTreeMap` for canonical
removes, atomic commit returning `Vec<Entity>`) but targets the
constitutional `World` and accepts `EntityKind` / `name` on
`stage_spawn`. Every system file in this batch already uses
`ConstitutionalWorldTxn` from the prior commit.

## What was actually found

A direct grep over the 5 files for the
`world\.(spawn|add_component|remove_component|get_component_mut|insert)\b`
pattern returns **zero matches in production code paths**. The only
references to those API names in the 5 files are inside `//` comments
that say "Direct `world.X` calls outside the WorldTxn seam are
forbidden" — these are guard-rail comments placed at each mutation
site by the prior commit to document the discipline.

Every mutation in the 5 files already flows through
`ConstitutionalWorldTxn`. The shape is consistent across all 5 files:
`let mut txn = ConstitutionalWorldTxn::new();` →
`txn.stage_spawn(kind, name)` / `txn.stage_insert_on::<T>(token, value)`
/ `txn.stage_insert::<T>(entity, value)` → `let _ = txn.commit(world)?;`.

## Per-file verification

| File | Direct mutations remaining | Already on `*WorldTxn`? | Phase 2.5 commit `e633567e` line(s) | Operations |
|---|---:|---|---|---|
| `capability_registry_sys.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1 conditional spawn in `find_or_create_registry_entity` (txn 1) + 2 inserts (`CapabilityRegistry`, `CapabilityKeyComp`) on txn 2 | two-txn (spawn-if-missing + inserts) |
| `archive.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `ArchiveSystem::run`: 1 per-model `AneArchiveResultComp` insert. `PrecompiledAneSystem::run`: 2 spawns (preserved verbatim — second is a pre-existing duplicate-spawn bug) + 1 `AneArchiveResultComp` insert on token1 | mixed (insert + spawn-or-insert with preserved duplicate-spawn) |
| `tts.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1 `TtsWeightsComp` insert on the first model entity, OR (if no model exists) 1 spawn + 1 `TtsWeightsComp` insert_on. `pack_tts_weights` is a filesystem side-effect and is intentionally NOT routed through WorldTxn. | spawn-or-insert + side-effect-then-staged-insert |
| `session_init.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | 1 `SessionState` spawn + 1 `SessionState` insert_on. Idempotent: bails early if a session entity with `SessionState` already exists. | spawn-then-insert (idempotent guard) |
| `quant_plan.rs` | 0 | Yes (`ConstitutionalWorldTxn`) | `CodecSelectionSystem::run`: 1 per-tensor `CodecFamilyComp` insert (loop over `EntityKind::Tensor`, gated on `Shape` + `CanonicalRoleComp`). `PrecisionPlanSystem::run`: 1 per-model `PrecisionPlanComponent` insert (loop over `EntityKind::Model`). | insert-only (2 systems) |

Total mutations already ported across the 5 files (per the Phase 2.5
`e633567e` per-file table): **3 + 4 + 3 + 2 + 2 = 14** (the brief
counted 12; the difference is the duplicate-spawn in
`PrecompiledAneSystem` and the side-effect-then-staged-insert in
`tts.rs` — both explicitly preserved by `e633567e` as the verbatim
port of the original "stage a spawn twice" / "filesystem side-effect
then result-component write" idioms). Either way, all 5 files are
ported and on the canonical seam.

## Why `ConstitutionalWorldTxn` and not the engine-local `WorldTxn`

The two `WorldTxn` flavors operate on different `World` types:

| `WorldTxn` flavor | File | Operates on | Spawn API | Insert / Remove API |
|---|---|---|---|---|
| Engine-local | `compute-core/src/ecs/runtime/world_txn.rs` | Engine runtime `World` (`compute-core/src/ecs/runtime/world.rs` — entity + component store, no `EntityKind`, no `name`) | `stage_spawn()` | `world.insert(entity, comp)` / `world.remove::<T>(entity)` |
| Constitutional (engine bridge) | `compute-core/src/ecs/runtime/constitutional_world_txn.rs` | Constitutional `prism_ecs_core::World` (with `EntityKind` / `name` and `add_component` / `remove_component`) | `stage_spawn(kind, name)` | `world.add_component(entity, comp)` / `world.remove_component::<T>(entity)` |

The `system/` files receive a `&mut World` (the constitutional one) in
every `CompilerSystem::run` call, and every `add_component` /
`remove_component` they call is the constitutional `World`'s API. The
engine-local `WorldTxn` calls `world.insert(entity, comp)` and
`world.remove::<T>(entity)` — those methods exist on the engine
runtime `World` but **not** on the constitutional `World`. The
`constitutional_world_txn.rs` module docstring states this explicitly
(lines 12-22):

> The system files (`compute-core/src/ecs/system/`) cannot use the
> engine-local `WorldTxn` because the World types differ. They also
> cannot use the full constitutional `WorldTxn` in
> `crates/prism-ecs-constitutional/src/world_txn.rs` because that
> API gates `put_durable` / `put_transient` on the
> `DurableComponent` / `TransientComponent` traits, and the system
> files' components only implement [`prism_ecs_core::Component`]
> (the legacy pattern that the engine's `CompilerSystem`s rely on).
>
> The motivation is the same as the engine-local one: a single
> authority-bearing commit seam so that engine-side mutations do not
> fork into N direct paths.

So the engine-local `WorldTxn` is the right target for
`compilation_systems.rs` and the engine's `runtime/` subsystem (which
use the engine runtime `World`), but the wrong target for the
`system/` files. `ConstitutionalWorldTxn` is the bridge.

A direct port of these 5 files to the engine-local `WorldTxn` would
not compile: the constitutional `World` has no `insert` / `remove`
methods, and the system files have no other way to thread mutations
into it.

## Patterns noticed (for the next agent's batch)

- **The 5 files break down into 4 patterns:** 1 two-txn fallback
  (`capability_registry_sys.rs` — spawn-if-missing then inserts on the
  resolved entity), 1 mixed (`archive.rs` — insert-only in
  `ArchiveSystem`, spawn-or-insert with preserved duplicate-spawn in
  `PrecompiledAneSystem`), 1 spawn-or-insert with side-effect-then-staged-insert
  (`tts.rs` — `pack_tts_weights` writes files to `output_dir` first,
  then the `TtsWeightsComp` is staged with the resulting paths), 1
  spawn-then-insert with idempotent guard (`session_init.rs` — bails
  early if a session entity with `SessionState` already exists), 1
  insert-only across 2 systems (`quant_plan.rs` — `CodecSelectionSystem`
  + `PrecisionPlanSystem`).
- **`archive.rs::PrecompiledAneSystem` preserves a pre-existing
  duplicate-spawn bug.** The original code spawned the
  "precompiled_ane" entity twice (the second spawn's result was
  discarded). The staged-txn port preserves this verbatim because
  `ConstitutionalWorldTxn::inserts` is `Vec` and the second
  `stage_spawn` is independent of the first. The first spawn's
  `PendingToken` is what carries the `AneArchiveResultComp` insert;
  the second spawn's `PendingToken` is bound to `_token2` and
  immediately dropped.
- **The side-effect-then-staged-insert pattern in `tts.rs` is the same
  shape used in `archive.rs`, `download.rs`, and `backend_compile.rs`
  across Phase 2.5.** Filesystem / network side effects run first and
  are intentionally NOT routed through the WorldTxn — the WorldTxn
  is the canonical authority seam for ECS state, not for external
  resources. Only the result-component write is staged.
- **`session_init.rs` is the only file in this batch with an
  idempotent guard** (`if world.get_component::<SessionState>(*entity).is_some() { return Ok(()) }`).
  The other 4 files are either side-effect-bearing
  (`tts.rs`/`archive.rs`) or unconditional
  (`capability_registry_sys.rs`/`quant_plan.rs`).
- **`quant_plan.rs` is the only file in this batch with two
  `CompilerSystem` implementations** (`CodecSelectionSystem` and
  `PrecisionPlanSystem`), each with its own `ConstitutionalWorldTxn`.
  Both systems share the file's module-level `Component for
  PrecisionPlanComponent` impl and import the same `ConstitutionalWorldTxn`.

## Build status

`cargo check -p tribunus-compute-core --lib --no-default-features`
returns the same 242 pre-existing errors that have existed since the
Phase 3 changelog baseline (e.g. `use of unresolved module or
unlinked crate metal` in `compute-core/src/ecs/backend/metal.rs:1127`,
missing `policy_support` in
`compute-core/src/ecs/backend/heterogeneous_executor.rs:352`,
missing `crate::ecs::constitutional` module referenced from
`compute-core/src/ecs/mod.rs:221` and several system files, etc.).
**Zero errors come from the 5 files in this batch.** None of the 5
files appear in the `error[...]` or `warning[...]` output.

## Action taken

**No code changes.** The work was already done correctly in commit
`e633567e` against the right `WorldTxn` flavor for the `system/`
files. This changelog serves as the audit trail and the explanation
for why a Phase 2.5 "batch 3" exists as a no-op verification rather
than a code-mutation pass.

## CAMPAIGN.md status

No change. The `system/` subsystem remains in the
`Shadow → approaching Canonical` transition state that the prior
Phase 2.5 commit recorded. The `ConstitutionalWorldTxn` helper
remains the canonical authority seam for ECS state mutations in
`system/`, exactly as designed.
