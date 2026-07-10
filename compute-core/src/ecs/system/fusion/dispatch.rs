use crate::ecs::component::fusion::{BindingCapacity, FusionGroup, WorkgroupCount};
use crate::ecs::component::tensor::Shape;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

pub struct DispatchFormationSystem;
impl CompilerSystem for DispatchFormationSystem {
    fn name(&self) -> &str {
        "DispatchFormationSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let dispatch_entities = world.entities_of_kind(EntityKind::Dispatch);

        for entity in dispatch_entities {
            if let Some(fusion) = world.get_component::<FusionGroup>(entity) {
                let fusion = fusion.clone();

                if fusion.accepted {
                    Self::attach_fused_dispatch(world, entity, &fusion);
                } else {
                    Self::spawn_per_op_dispatches(world, entity, &fusion);
                }
            }
        }

        Ok(())
    }
}

impl DispatchFormationSystem {
    /// Compute the workgroup X dimension from the shape of the dispatch output
    /// tensor. Uses the hidden (last) dimension with 256 threads per workgroup.
    fn workgroup_dim(world: &CompWorld, entity: CompEntity) -> u32 {
        let dim = world
            .get_component::<Shape>(entity)
            .and_then(|s| s.0.last().copied())
            .unwrap_or(1)
            .max(1);
        (dim + 255) / 256
    }

    /// Accept a fused dispatch group: attach WorkgroupCount and BindingCapacity
    /// to the entity so a single fused kernel is launched.
    fn attach_fused_dispatch(world: &mut CompWorld, entity: CompEntity, fusion: &FusionGroup) {
        let wg_x = Self::workgroup_dim(world, entity);
        world.add_component(entity, WorkgroupCount(wg_x, 1, 1));
        world.add_component(
            entity,
            BindingCapacity {
                max_slots: fusion.binding_slots.max(1),
                max_bytes_per_slot: 64 * 1024 * 1024,
            },
        );
    }

    /// Rejected fusion group: spawn one Dispatch entity per operation (root
    /// + each fused op), each carrying its own WorkgroupCount, BindingCapacity,
    /// and Shape copied from the parent.
    fn spawn_per_op_dispatches(world: &mut CompWorld, parent: CompEntity, fusion: &FusionGroup) {
        // Op labels: root first, then each fused op.
        let op_kinds: Vec<String> = std::iter::once(&fusion.root_op_kind)
            .chain(fusion.fused_op_kinds.iter())
            .cloned()
            .collect();

        let parent_shape = world.get_component::<Shape>(parent).cloned();

        for op_kind in &op_kinds {
            let op_entity = world.spawn(EntityKind::Dispatch, Some(format!("{op_kind}_per_op")));

            let wg_x = Self::workgroup_dim(world, parent);
            world.add_component(op_entity, WorkgroupCount(wg_x, 1, 1));
            world.add_component(
                op_entity,
                BindingCapacity {
                    max_slots: 1,
                    max_bytes_per_slot: 64 * 1024 * 1024,
                },
            );

            // Carry forward the parent shape so downstream systems (e.g.
            // ScalarDispatchSystem) can reason about element counts.
            if let Some(shape) = &parent_shape {
                world.add_component(op_entity, shape.clone());
            }
        }
    }
}
