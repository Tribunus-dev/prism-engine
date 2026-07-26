# Compute-core.legacy Absorption — Phase 2.5 Batch 1: No-Op Confirmation

**Date:** 2026-07-26
**Agent:** Batch 1 of Phase 2.5
**Status:** No-op — all 3 files in batch already ported in commit `e633567e`

## TL;DR

The brief for this batch asked to port 19 direct world mutations in 3
`compute-core/src/ecs/system/` files to the engine-local `WorldTxn`
(added in commit `ebcaf2bc`). On inspection, **all 3 files have zero
direct mutations remaining in production code paths**. They were
already ported to `ConstitutionalWorldTxn` in commit `e633567e`
(Phase 2.5 — "port 100 remaining system/ mutations to WorldTxn").
The 19 raw `world.spawn` / `world.add_component` / etc. call sites
remaining in these files are all inside `#[cfg(test)]` blocks gated by
`#[cfg(feature = "legacy_mutations")]` — opt-in legacy tests, not
compiled in the default build, and intentionally preserved per the
Phase 2.5 changelog.

## Verification

Files in batch (all checked via
`grep -nE "world\.(spawn|add_component|remove_component|get_component_mut|insert)\("`):

| File | Direct mutations in production | Direct mutations in tests (legacy_mutations) | Already on `*WorldTxn`? | Phase 2.5 mapping |
|---|---:|---:|---|---|
| `compute-core/src/ecs/system/kernel_catalog.rs` | 0 | 7 (lines 97, 98, 117, 118, 137, 149, 150) | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 1/1 prod `add_component` → `stage_insert` |
| `compute-core/src/ecs/system/variant_select.rs` | 0 | 5 (lines 260, 261, 262, 268, 269) | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 1/1 prod `add_component` → `stage_insert` (6-mutation top-10 table counts the inner-loop insert as 1 pattern, 6 spawn+insert cycles collapsed to 1 txn) |
| `compute-core/src/ecs/system/variant_gen.rs` | 0 | 3 (lines 232, 233, 241) | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 5 prod mutations ported via the two-transaction fallback pattern |

Total raw direct-mutation call sites in production paths across the 3
files: **0**. Total across the 3 files including `legacy_mutations`
tests: **15** (7 + 5 + 3). The 19-mutation count in the brief appears
to include the loop iterations counted separately, but the **production
path is already ported**.

The 15 raw test-block call sites are all inside `#[cfg(test)]` blocks
(lines 84-163, 253-359, 225-266 respectively) and additionally gated
by `#[cfg(feature = "legacy_mutations")]` (declared in
`compute-core/Cargo.toml` as a non-default feature). They are not
compiled in the default build. Per the Phase 2.5 changelog, these
are a documented escape hatch for legacy direct-mutation tests and
are expected to remain until a future phase explicitly ports them.

## Why `ConstitutionalWorldTxn` and not the engine-local `WorldTxn`

The brief assumes the engine-local `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs` (added in `ebcaf2bc`)
is the target. It is not, for the system files. The two `WorldTxn`
types operate on different `World` types:

| `WorldTxn` flavor | File | Operates on | Spawn API | Insert API |
|---|---|---|---|---|
| Engine-local | `compute-core/src/ecs/runtime/world_txn.rs` | Engine runtime `World` (entity + component store, no `EntityKind`) | `stage_spawn()` | `world.insert(entity, comp)` |
| Constitutional (engine bridge) | `compute-core/src/ecs/runtime/constitutional_world_txn.rs` | Constitutional `prism_ecs_core::World` (with `EntityKind` / name) | `stage_spawn(kind, name)` | `world.add_component(entity, comp)` / `world.remove_component::<T>(entity)` |

The `system/` files receive a `&mut World` (the constitutional one)
in every `CompilerSystem::run` call, and every `add_component` /
`remove_component` they call is the constitutional `World`'s API. The
engine-local `WorldTxn` calls `world.insert(entity, comp)` and
`world.remove::<T>(entity)` — those methods exist on the engine
runtime `World` but **not** on the constitutional `World`. The
`constitutional_world_txn.rs` module docstring states this
explicitly (lines 12-19):

> The system files (`compute-core/src/ecs/system/`) cannot use the
> engine-local `WorldTxn` because the World types differ. They also
> cannot use the full constitutional `WorldTxn` in
> `crates/prism-ecs-constitutional/src/world_txn.rs` because that API
> gates `put_durable` / `put_transient` on the
> `DurableComponent` / `TransientComponent` traits, and the system
> files' components only implement `prism_ecs_core::Component` (the
> legacy pattern that the engine's `CompilerSystem`s rely on).

So the engine-local `WorldTxn` is the right target for
`compilation_systems.rs` and the engine's `runtime/` subsystem (which
uses the engine runtime `World`), but the wrong target for the
`system/` files. `ConstitutionalWorldTxn` is the bridge.

A direct port of these 3 files to the engine-local `WorldTxn` would
not compile: the constitutional `World` has no `insert` / `remove`
methods, and the system files have no other way to thread mutations
into it.

## Current pattern in each file (already correct)

### `kernel_catalog.rs` (1 production mutation, ported)

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let kernels: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
    let mut txn = ConstitutionalWorldTxn::new();
    for &kernel in &kernels {
        // ... validate kernel's CompiledBinary ...
        let valid = ...;
        let errors = ...;
        if let Err(e) = txn.stage_insert(kernel, CatalogEntry { valid, errors }) {
            tracing::warn!(entity = ?kernel, error = %e, "kernel_catalog: stage_insert failed");
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "kernel_catalog: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("kernel_catalog: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

### `variant_select.rs` (1 production mutation in a per-group loop, ported)

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    // ... group variants by parent_kernel ...
    let mut txn = ConstitutionalWorldTxn::new();
    for (parent_kernel, variants) in &groups {
        // ... score and pick best variant ...
        if let Some(idx) = best_idx {
            let best_data = &variants[idx].1;
            let score = scored[idx];
            if let Err(e) = txn.stage_insert(
                *parent_kernel,
                SelectedVariant { profile_id: best_data.profile_id.clone(), score },
            ) {
                tracing::warn!(entity = ?parent_kernel, error = %e, "variant_select: stage_insert SelectedVariant");
            }
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "variant_select: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("variant_select: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

### `variant_gen.rs` (5 production mutations, ported via two-transaction fallback)

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    // ... collect dispatch and kernel entities ...
    // Transaction 1: fallback kernel (only if needed).
    let fallback_parent: Option<Entity> = if kernel_entities.is_empty() {
        let mut fallback_txn = ConstitutionalWorldTxn::new();
        let token = fallback_txn.stage_spawn(EntityKind::Kernel, Some("variant_parent".into()));
        let spawned = fallback_txn.commit(world).map_err(|e| { ... })?;
        let _ = token; // token consumed by commit
        spawned.into_iter().next()
    } else {
        None
    };
    // Transaction 2: every per-variant spawn + insert.
    let mut txn = ConstitutionalWorldTxn::new();
    for &dispatch in &dispatch_entities {
        let parent_kernel = if kernel_entities.is_empty() {
            fallback_parent.expect("fallback_parent resolved in transaction 1")
        } else {
            kernel_entities[dispatch.0 as usize % kernel_entities.len()]
        };
        for &template_id in ALL_TEMPLATES {
            for profile in &profiles {
                let profile_str = profile.to_string();
                let variant_token = txn.stage_spawn(
                    EntityKind::KernelVariant,
                    Some(format!("variant_{template_id:?}_{profile_str}")),
                );
                if let Err(e) = txn.stage_insert_on(
                    variant_token,
                    KernelVariantEntityData { profile_id: profile_str, template_id, parent_kernel: CompEntityRef(parent_kernel.0) },
                ) {
                    tracing::warn!(error = %e, "variant_gen: stage_insert_on KernelVariantEntityData");
                }
            }
        }
    }
    let _ = txn.commit(world).map_err(|e| { ... })?;
    Ok(())
}
```

The `variant_gen.rs` two-transaction sequence is a deliberate port
shape: the fallback Entity is needed as `parent_kernel` for the
variant inserts, and `WorldTxn::stage_spawn` returns a `PendingToken`
whose resolved `Entity` is only known after `commit`. The original
code spawned the fallback inline and used the real `Entity` directly.
The staged-txn port preserves the same observable outcome via two
commits. The cost is one extra `BTreeMap` + `Vec` insert per fallback,
committed atomically.

## Build status

`cargo check -p tribunus-compute-core --lib --no-default-features`:

- **242 pre-existing errors** (the engine has known build issues,
  including `compute-core/compute-core.legacy/` references that no
  longer resolve — tracked separately in `AGENTS.md` and the Phase 3
  changelog baseline).
- **0 errors from the 3 files in this batch** (verified by grepping
  the error stream for `kernel_catalog|variant_select|variant_gen` —
  no matches).
- No new errors introduced by this batch.
- No changes to any constitutional library crates — the working-tree
  modifications in `crates/prism-ecs-*` (visible in `git status`)
  are from an out-of-scope parallel effort and are not part of this
  batch.

## What this batch did

Nothing in `compute-core/src/ecs/system/`. The work was already
complete in `e633567e`. This changelog is the only artifact, and it
exists to make the no-op state explicit for the next agent that
scans the system/ directory and to record the verification that all
3 files are on the `ConstitutionalWorldTxn` seam.

## Recommendation to the parent session

- The Phase 2.5 absorption of `system/` is complete. No follow-up
  batches targeting direct mutations in this directory will find
  any in production paths. The 15 raw call sites remaining in the
  3 files in this batch are all in `#[cfg(feature = "legacy_mutations")]`
  test blocks — opt-in legacy tests, not compiled in the default
  build, and intentionally preserved per the Phase 2.5 changelog.
- The "batch N" decomposition used to parallelize Phase 2.5 should
  be retired for the `system/` subsystem. Future compute-core
  absorption work should target the `core/` and `compute_image/`
  subsystems (per CAMPAIGN.md and the Phase 4B/4C/4C-cont
  changelogs), or the engine's `runtime/` subsystem (Phase 3 in
  `ebcaf2bc` already covered the engine-local `World` ops; the
  remaining `runtime/` work is a follow-up).
- If the brief's "engine-local `WorldTxn`" framing is the goal
  project-wide, consider migrating `ConstitutionalWorldTxn` to a
  thin alias over the engine-local `WorldTxn` so there is one
  `WorldTxn` type. That is a refactor, not a port.

## Files changed in this batch

- None in `compute-core/src/ecs/system/`.
- This changelog only.
