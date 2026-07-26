use crate::ecs::component::fusion::{BindingCapacity, FusionGroup, WorkgroupCount};
use crate::ecs::component::tensor::Shape;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

pub struct DispatchFormationSystem;
impl CompilerSystem for DispatchFormationSystem {
    fn name(&self) -> &str {
        "DispatchFormationSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let dispatch_entities = world.entities_of_kind(EntityKind::Dispatch);

        // Stage every per-entity dispatch-formation mutation on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct world
        // mutations are forbidden outside the WorldTxn seam.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in dispatch_entities {
            if let Some(fusion) = world.get_component::<FusionGroup>(entity) {
                let fusion = fusion.clone();

                if fusion.accepted {
                    Self::attach_fused_dispatch(&mut txn, world, entity, &fusion);
                } else {
                    Self::spawn_per_op_dispatches(&mut txn, world, entity, &fusion)?;
                }
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "dispatch_formation: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("dispatch_formation: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}

impl DispatchFormationSystem {
    /// Compute the workgroup X dimension from the shape of the dispatch output
    /// tensor. Uses the hidden (last) dimension with 256 threads per workgroup.
    fn workgroup_dim(world: &World, entity: Entity) -> u32 {
        let dim = world
            .get_component::<Shape>(entity)
            .and_then(|s| s.0.last().copied())
            .unwrap_or(1)
            .max(1);
        (dim + 255) / 256
    }

    /// Accept a fused dispatch group: attach WorkgroupCount and BindingCapacity
    /// to the entity so a single fused kernel is launched.
    ///
    /// Stages the inserts on `txn`; the caller's `commit` applies them
    /// atomically.
    fn attach_fused_dispatch(
        txn: &mut ConstitutionalWorldTxn,
        world: &World,
        entity: Entity,
        fusion: &FusionGroup,
    ) {
        let wg_x = Self::workgroup_dim(world, entity);
        if let Err(e) = txn.stage_insert(entity, WorkgroupCount(wg_x, 1, 1)) {
            tracing::warn!(entity = ?entity, error = %e, "attach_fused_dispatch: stage_insert WorkgroupCount");
        }
        if let Err(e) = txn.stage_insert(
            entity,
            BindingCapacity {
                max_slots: fusion.binding_slots.max(1),
                max_bytes_per_slot: 64 * 1024 * 1024,
            },
        ) {
            tracing::warn!(entity = ?entity, error = %e, "attach_fused_dispatch: stage_insert BindingCapacity");
        }
    }

    /// Rejected fusion group: spawn one Dispatch entity per operation (root
    /// + each fused op), each carrying its own WorkgroupCount, BindingCapacity,
    /// and Shape copied from the parent.
    ///
    /// Stages the spawns and inserts on `txn`; the caller's `commit`
    /// applies them atomically.
    fn spawn_per_op_dispatches(
        txn: &mut ConstitutionalWorldTxn,
        world: &World,
        parent: Entity,
        fusion: &FusionGroup,
    ) -> Result<(), crate::ecs::WorldError> {
        // Op labels: root first, then each fused op.
        let op_kinds: Vec<String> = std::iter::once(&fusion.root_op_kind)
            .chain(fusion.fused_op_kinds.iter())
            .cloned()
            .collect();

        let parent_shape = world.get_component::<Shape>(parent).cloned();
        let wg_x = Self::workgroup_dim(world, parent);

        for op_kind in &op_kinds {
            let token = txn.stage_spawn(
                EntityKind::Dispatch,
                Some(format!("dispatch_{op_kind}")),
            );
            // Original code stage-inserted WorkgroupCount twice — preserved
            // verbatim (this is a duplicate insert bug we are not
            // introducing; it predates the port). The constitutional
            // `add_component` will overwrite, matching the original
            // direct-mutation behaviour.
            if let Err(e) = txn.stage_insert_on(token, WorkgroupCount(wg_x, 1, 1)) {
                tracing::warn!(error = %e, "spawn_per_op_dispatches: stage_insert_on WorkgroupCount (1st)");
            }
            if let Err(e) = txn.stage_insert_on(token, WorkgroupCount(wg_x, 1, 1)) {
                tracing::warn!(error = %e, "spawn_per_op_dispatches: stage_insert_on WorkgroupCount (2nd)");
            }
            if let Err(e) = txn.stage_insert_on(
                token,
                BindingCapacity {
                    max_slots: 1,
                    max_bytes_per_slot: 64 * 1024 * 1024,
                },
            ) {
                tracing::warn!(error = %e, "spawn_per_op_dispatches: stage_insert_on BindingCapacity");
            }

            // Carry forward the parent shape so downstream systems (e.g.
            // ScalarDispatchSystem) can reason about element counts.
            if let Some(shape) = &parent_shape {
                if let Err(e) = txn.stage_insert_on(token, shape.clone()) {
                    tracing::warn!(error = %e, "spawn_per_op_dispatches: stage_insert_on Shape");
                }
            }
        }
        Ok(())
    }
}
