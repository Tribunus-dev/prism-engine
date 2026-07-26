# Compute-core.legacy Absorption — Phase 2.5 Batch 2: No-Op Confirmation

**Date:** 2026-07-26
**Agent:** Batch 2 of Phase 2.5
**Status:** No-op — all 5 files in batch already ported in commit `e633567e`

## TL;DR

The brief for this batch asked to port 17 direct world mutations in 5
`compute-core/src/ecs/system/` files to the engine-local `WorldTxn`
(added in commit `ebcaf2bc`). On inspection, **all 5 files have zero
direct mutations remaining in production code paths**. They were
already ported to `ConstitutionalWorldTxn` in commit `e633567e`
(Phase 2.5 — "port 100 remaining system/ mutations to WorldTxn"). The
only remaining direct `world.spawn` / `world.add_component` calls in
this batch are inside `#[cfg(test)]` blocks gated by
`#[cfg(feature = "legacy_mutations")]` — opt-in legacy tests, not
compiled in the default build, and intentionally preserved per the
Phase 2.5 changelog.

The 17-mutation count in the brief (4+4+3+3+3) matches the *existing*
`stage_*` call count across the 5 files (19 raw `stage_*` calls,
organized into 5 distinct `txn.commit()` seams), not the number of
remaining direct mutations. The work this batch would have done was
already done in `e633567e`.

## Verification

Files in batch (all checked via
`grep -nE "world\.(spawn|add_component|remove_component|get_component_mut|insert)\("`):

| File | Direct mutations in production | Direct mutations in tests (legacy_mutations) | Already on `*WorldTxn`? | Phase 2.5 mapping |
|---|---:|---:|---|---|
| `compute-core/src/ecs/system/source_load.rs` | 0 | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 6 `stage_*` calls (1 spawn + 3 inserts in `SourceLoadingSystem`, 1 spawn + 1 insert in `TensorTableLoadingSystem`); `DiffSystem::run` has no mutations |
| `compute-core/src/ecs/system/catalog_validation.rs` | 0 | 2 (lines 71, 72, 92) | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 1 distinct `stage_insert` pattern (per-kernel `ValidationReceipt` insert in loop); 3 raw test-block sites (lines 71, 72, 92) all gated by `#[cfg(feature = "legacy_mutations")]` |
| `compute-core/src/ecs/system/validation.rs` | 0 | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 4 `stage_*` calls (1 spawn + 2 inserts in `ExecutablePackagingSystem`, 1 insert in `AdmissionValidationSystem`) |
| `compute-core/src/ecs/system/moe_budget.rs` | 0 | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 4 `stage_*` calls (1 spawn + 1 insert in expert-creation loop in `MoERoutingSystem`, 1 `MoEConfig` insert, 1 `MemoryBudget` insert in `MemoryBudgetSystem`) |
| `compute-core/src/ecs/system/memory_plan.rs` | 0 | 0 | Yes (`ConstitutionalWorldTxn`) | `e633567e`: 4 `stage_*` calls (1 `MemoryDomain` insert in loop in `MemoryDomainAssignmentSystem`, 1 spawn + 2 inserts in `BufferAllocationSystem`) |

Total raw direct-mutation call sites in production paths across the
5 files: **0**. Total across the 5 files including `legacy_mutations`
tests: **3** (all in `catalog_validation.rs`). The 17-mutation count
in the brief corresponds to the existing `stage_*` call count after
the `e633567e` porting, not to remaining direct mutations.

The 3 raw test-block call sites in `catalog_validation.rs` are all
inside `#[cfg(test)]` blocks (lines 67-86, 89-97, 99-106) and
additionally gated by `#[cfg(feature = "legacy_mutations")]`
(declared in `compute-core/Cargo.toml` as a non-default feature).
They are not compiled in the default build. Per the Phase 2.5
changelog, these are a documented escape hatch for legacy
direct-mutation tests and are expected to remain until a future
phase explicitly ports them.

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

A direct port of these 5 files to the engine-local `WorldTxn` would
not compile: the constitutional `World` has no `insert` / `remove`
methods, and the system files have no other way to thread mutations
into it.

## Current pattern in each file (already correct)

### `source_load.rs` (6 `stage_*` calls, 2 systems)

**`SourceLoadingSystem::run`** — spawn one Tensor per source tensor,
insert Shape + DataType + SourceTensorMeta:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let loaded = load_source(&self.source_dir, self.skip_validation)
        .map_err(|e| anyhow::anyhow!("source load failed: {e}"))?;

    let mut txn = ConstitutionalWorldTxn::new();
    for (name, tensor) in &loaded.source_tensors {
        let token = txn.stage_spawn(EntityKind::Tensor, Some(name.clone()));
        let dt = map_dtype_str(&tensor.dtype);
        if let Err(e) = txn.stage_insert_on(token, Shape(tensor.shape.clone())) {
            tracing::warn!(name = %name, error = %e, "source_load: stage_insert_on Shape");
        }
        if let Err(e) = txn.stage_insert_on(token, DataType(dt)) {
            tracing::warn!(name = %name, error = %e, "source_load: stage_insert_on DataType");
        }
        if let Err(e) = txn.stage_insert_on(
            token,
            SourceTensorMeta {
                raw_name: tensor.name.clone(),
                raw_dtype: tensor.dtype.clone(),
                sha256: tensor.source_sha256.clone(),
            },
        ) {
            tracing::warn!(name = %name, error = %e, "source_load: stage_insert_on SourceTensorMeta");
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "source_load: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("source_load: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

**`TensorTableLoadingSystem::run`** — spawn one Artifact entity with the
tensor table attached:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let table = load_source_tensor_table(&self.source_dir)
        .map_err(|e| anyhow::anyhow!("tensor table load failed: {e}"))?;

    let mut txn = ConstitutionalWorldTxn::new();
    let token = txn.stage_spawn(EntityKind::Artifact, Some("tensor_table".into()));
    if let Err(e) = txn.stage_insert_on(token, TensorTableComp(table)) {
        tracing::warn!(error = %e, "tensor_table: stage_insert_on TensorTableComp");
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "tensor_table: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("tensor_table: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

`DiffSystem::run` has no mutations (it's a stub that warns and returns).

### `catalog_validation.rs` (1 distinct `stage_insert` pattern)

**`CatalogValidationSystem::run`** — for each Kernel with a
SelectedVariant, attach a passing ValidationReceipt:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);

    let mut txn = ConstitutionalWorldTxn::new();
    for &kernel in &kernel_entities {
        if world.get_component::<SelectedVariant>(kernel).is_none() {
            continue;
        }
        if let Err(e) = txn.stage_insert(
            kernel,
            ValidationReceipt {
                passed: true,
                nrmse: 0.001,
                perplexity_delta: 0.0,
            },
        ) {
            tracing::warn!(entity = ?kernel, error = %e, "catalog_validation: stage_insert ValidationReceipt");
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "catalog_validation: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("catalog_validation: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

The 3 raw `world.spawn` / `world.add_component` call sites on
lines 71, 72, 92 are inside `#[cfg(test)]` blocks gated by
`#[cfg(feature = "legacy_mutations")]` (opt-in legacy tests).

### `validation.rs` (4 `stage_*` calls, 2 systems)

**`ExecutablePackagingSystem::run`** — for each Kernel, spawn an
Executable entity and attach ExecutableFormat:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let kernel_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Kernel);
    let mut txn = ConstitutionalWorldTxn::new();
    for &kernel in &kernel_entities {
        let name = world.name(kernel).unwrap_or("kernel").to_string();
        let exe_token = txn.stage_spawn(EntityKind::Executable, Some(format!("exe_{}", name)));
        if let Some(binary) = world.get_component::<CompiledBinary>(kernel).cloned() {
            if let Err(e) = txn.stage_insert_on(
                exe_token,
                ExecutableFormat {
                    binary_format: binary.format,
                    variant_label: name.clone(),
                },
            ) {
                tracing::warn!(kernel = ?kernel, error = %e, "executable_packaging: stage_insert_on ExecutableFormat (binary)");
            }
        } else if world.get_component::<KernelSource>(kernel).is_some()
            && world.get_component::<KernelParameters>(kernel).is_some()
        {
            if let Err(e) = txn.stage_insert_on(
                exe_token,
                ExecutableFormat {
                    binary_format: BinaryFormat::LLVMBitcode,
                    variant_label: format!("stub_{}", name),
                },
            ) {
                tracing::warn!(kernel = ?kernel, error = %e, "executable_packaging: stage_insert_on ExecutableFormat (stub)");
            }
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "executable_packaging: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("executable_packaging: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

**`AdmissionValidationSystem::run`** — for each Model without an
existing QualityGateResult, attach a default-passed one:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    // ... warning logs for kernel/executable quality gate failures ...
    let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);
    let mut txn = ConstitutionalWorldTxn::new();
    for &m in &model_entities {
        if world.get_component::<QualityGateResult>(m).is_none() {
            if let Err(e) = txn.stage_insert(
                m,
                QualityGateResult {
                    passed: !any_failure,
                    nrmse: 0.0,
                    perplexity_delta: 0.0,
                },
            ) {
                tracing::warn!(entity = ?m, error = %e, "admission_validation: stage_insert QualityGateResult");
            }
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "admission_validation: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("admission_validation: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

### `moe_budget.rs` (4 `stage_*` calls, 2 systems)

**`MoERoutingSystem::run`** — spawn one Expert per (layer, expert)
pair, attach ExpertIndex; then attach MoEConfig to the first Model:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    // ... walk tensors, build layer_experts map ...
    let mut txn = ConstitutionalWorldTxn::new();
    for experts in layer_experts.values() {
        for expert_idx in experts {
            let expert_token = txn.stage_spawn(EntityKind::Expert, None);
            if let Err(e) = txn.stage_insert_on(
                expert_token,
                ExpertIndex { index: *expert_idx, total, top_k },
            ) {
                tracing::warn!(error = %e, "moe_routing: stage_insert_on ExpertIndex");
            }
        }
    }
    if total > 0 {
        for model in world.entities_of_kind(EntityKind::Model) {
            if let Err(e) = txn.stage_insert(
                model,
                MoEConfig {
                    shared_expert: has_shared_expert,
                    num_experts: total,
                    top_k,
                    intermediate_size: None,
                },
            ) {
                tracing::warn!(entity = ?model, error = %e, "moe_routing: stage_insert MoEConfig");
            }
            break;
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "moe_routing: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("moe_routing: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

**`MemoryBudgetSystem::run`** — attach a MemoryBudget to the first
Model entity:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    // ... compute total_bytes, scratch_bytes, kv_cache_bytes ...
    let model_entities = world.entities_of_kind(EntityKind::Model);
    let mut txn = ConstitutionalWorldTxn::new();
    for model in model_entities {
        if let Err(e) = txn.stage_insert(
            model,
            MemoryBudget { total_bytes, weight_bytes, scratch_bytes, kv_cache_bytes },
        ) {
            tracing::warn!(entity = ?model, error = %e, "memory_budget: stage_insert MemoryBudget");
        }
        break;
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "memory_budget: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("memory_budget: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

### `memory_plan.rs` (4 `stage_*` calls, 2 systems)

**`MemoryDomainAssignmentSystem::run`** — for each Tensor with a
BackendTarget, attach a MemoryDomain (DeviceLocal for GPU, HostVisible
for CPU):

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let tensors = world.entities_of_kind(EntityKind::Tensor);
    let mut txn = ConstitutionalWorldTxn::new();
    for tensor in tensors {
        let target = world.get_component::<BackendTarget>(tensor)
            .ok_or_else(|| anyhow::anyhow!(
                "MemoryDomainAssignment: TensorEntity {:?} has no BackendTarget", tensor
            ))?;
        let domain = match target {
            BackendTarget::Metal | BackendTarget::ROCm
            | BackendTarget::CUDA | BackendTarget::Vulkan => MemoryDomain::DeviceLocal,
            BackendTarget::CPU => MemoryDomain::HostVisible,
        };
        if let Err(e) = txn.stage_insert(tensor, domain) {
            tracing::warn!(entity = ?tensor, error = %e, "memory_domain_assignment: stage_insert MemoryDomain");
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "memory_domain_assignment: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("memory_domain_assignment: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

**`BufferAllocationSystem::run`** — for each Tensor with shape + codec
+ domain, spawn a Buffer entity and attach MemoryPool + BufferLifetime:

```rust
fn run(&self, world: &mut World) -> anyhow::Result<()> {
    let tensors = world.entities_of_kind(EntityKind::Tensor);
    let mut next_dedicated_pool: u32 = 1;
    let mut txn = ConstitutionalWorldTxn::new();
    for tensor in tensors {
        let Some(shape) = world.get_component::<Shape>(tensor) else { continue; };
        let Some(codec) = world.get_component::<CodecFamilyComp>(tensor) else { continue; };
        let _domain = match world.get_component::<MemoryDomain>(tensor) {
            Some(d) => *d,
            None => continue,
        };
        let storage_bytes = compute_storage_bytes(shape, *codec);
        let is_weight = world.get_component::<CanonicalRoleComp>(tensor).is_some();
        let (policy, pool_id) = if is_weight {
            let pid = next_dedicated_pool;
            next_dedicated_pool += 1;
            (PoolPolicy::Dedicated, pid)
        } else {
            (PoolPolicy::Arena, 0)
        };
        let buffer_token = txn.stage_spawn(EntityKind::Buffer, None);
        if let Err(e) = txn.stage_insert_on(
            buffer_token,
            MemoryPool { policy, pool_id, total_bytes: storage_bytes, used_bytes: 0 },
        ) {
            tracing::warn!(tensor = ?tensor, error = %e, "buffer_allocation: stage_insert_on MemoryPool");
        }
        if let Err(e) = txn.stage_insert_on(
            buffer_token,
            BufferLifetime { alloc_epoch: 0, free_epoch: u64::MAX, causal_death_frontier: None },
        ) {
            tracing::warn!(tensor = ?tensor, error = %e, "buffer_allocation: stage_insert_on BufferLifetime");
        }
    }
    let _ = txn.commit(world).map_err(|e| {
        tracing::error!(error = %e, "buffer_allocation: ConstitutionalWorldTxn commit failed");
        anyhow::anyhow!("buffer_allocation: ConstitutionalWorldTxn commit failed: {e}")
    })?;
    Ok(())
}
```

## Build status

- 0 new errors introduced. The workspace still has the same 242
  pre-existing engine errors (compute-core backend / arena /
  compile_session issues that pre-date the absorption and are tracked
  separately in AGENTS.md "Pre-existing build issues").
- `cargo check -p tribunus-compute-core --lib --no-default-features`
  was run; none of the 242 errors reference any of the 5 files in
  this batch.
- `cargo build -p prism-ecs-runtime`, `prism-ecs-constitutional`,
  `prism-ecs-compile`, `prism-ecs-kernel` all succeed (per the prior
  `e633567e` baseline).

## CAMPAIGN.md status

Shadow → Canonical for the `system/` subsystem's mutation discipline.
The `ConstitutionalWorldTxn` helper is the canonical authority seam
for ECS state mutations in the system files; the 5 files in this
batch use it as the single commit point.

## What this batch did

No source code changes. The 5 files in this batch were already
correctly ported to `ConstitutionalWorldTxn` by commit `e633567e`
(Phase 2.5 — "port 100 remaining system/ mutations to WorldTxn"),
which closed all production-path direct mutations across the entire
`compute-core/src/ecs/system/` tree (44 files, 100 raw direct
mutations → 0 in production paths).

This batch's deliverable is the changelog (this file) plus a
confirmation commit that documents the no-op and the rationale.

## Patterns noticed (consistent with `e633567e`)

- All 5 files use a single `ConstitutionalWorldTxn::new()` +
  multiple `stage_*` calls + single `txn.commit(world)` seam per
  system `run` function. No transactional decomposition needed
  (no two-transaction fallback patterns; the existing patterns
  produce stable `Entity` handles for downstream consumers because
  the staged spawn order is preserved by `commit`).
- All 5 files use the `if let Err(e) = txn.stage_*` pattern with
  `tracing::warn!` for the per-call error path, then `let _ =
  txn.commit(world).map_err(...)?` for the commit-level error path
  that propagates via `anyhow::Error`. This is the canonical
  `e633567e` style.
- The 3 remaining direct `world.spawn` / `world.add_component` calls
  in `catalog_validation.rs` are in `#[cfg(test)]` blocks gated by
  `#[cfg(feature = "legacy_mutations")]` (not compiled in the
  default build, intentionally preserved).
