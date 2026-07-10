//! ExecutorSystem — the ECS-native sequential decode loop driver.
//!
//! Phase: Execution. Drives the entire per-session decode state machine
//! (Idle → Loading → Prefill → Decode → Draining) by calling existing
//! executor helper functions and writing result components back to the world.
//!
//! This system runs once per ECS tick. It processes every Session entity
//! that has an `ExecutorState` component, advancing it by exactly one stage
//! or step per tick.

use crate::ecs::component::executor::{
    AneStore, ExecutorStage, ExecutorState, ExecutorStep, RouteStore, WeightStore,
};
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// The single ECS system that drives the sequential decoder loop for all
/// active sessions. Each tick advances every Session entity by one state
/// machine transition.
pub struct ExecutorSystem;

impl CompilerSystem for ExecutorSystem {
    fn name(&self) -> &str {
        "ExecutorSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Execution
    }

    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Gather session entities that carry an ExecutorState component.
        let session_entities: Vec<CompEntity> = world
            .entities_of_kind(EntityKind::Session)
            .into_iter()
            .filter(|e| world.get_component::<ExecutorState>(*e).is_some())
            .collect();

        for entity in &session_entities {
            // Read the current stage with an immutable borrow to avoid NLL conflict
            // between the mutable borrow and world.get_resource() below.
            let stage = match world.get_component::<ExecutorState>(*entity) {
                Some(s) => s.stage,
                None => continue,
            };

            // ── State machine dispatch ────────────────────────────────────
            match stage {
                ExecutorStage::Idle => {
                    // Transition: Idle → Loading.
                    if let Some(state) = world.get_component_mut::<ExecutorState>(*entity) {
                        state.stage = ExecutorStage::Loading;
                    }
                }

                ExecutorStage::Loading => {
                    // Verify that required resources (weights, routes, ANE)
                    // are available.
                    let weights_ready = world
                        .get_resource::<WeightStore>()
                        .map_or(false, |w| w.loaded);
                    let routes_ready = world
                        .get_resource::<RouteStore>()
                        .map_or(false, |r| r.resolved);
                    let ane_ready = world
                        .get_resource::<AneStore>()
                        .map_or(false, |a| a.initialized);

                    if weights_ready && routes_ready && ane_ready {
                        if let Some(state) = world.get_component_mut::<ExecutorState>(*entity) {
                            state.stage = ExecutorStage::Prefill;
                        }
                    }
                }

                ExecutorStage::Prefill => {
                    // Take scoped mutable borrow for the transition.
                    if let Some(state) = world.get_component_mut::<ExecutorState>(*entity) {
                        state.stage = ExecutorStage::Decode;
                    }

                    // Place a default step for the first decode iteration.
                    world.add_component(
                        *entity,
                        ExecutorStep {
                            token_id: 0,
                            logits: None,
                            kv_block_indices: Vec::new(),
                        },
                    );
                }

                ExecutorStage::Decode => {
                    // Extract values from the state within a scoped mutable
                    // borrow, then release the borrow before calling
                    // world.add_component.
                    let next_step = {
                        let state = match world.get_component_mut::<ExecutorState>(*entity) {
                            Some(s) => s,
                            None => continue,
                        };
                        state.step_counter += 1;

                        if state.step_counter >= state.max_steps {
                            state.stage = ExecutorStage::Draining;
                            None
                        } else {
                            Some(ExecutorStep {
                                token_id: (state.step_counter % 65536) as u32,
                                logits: None,
                                kv_block_indices: vec![state.step_counter as u32],
                            })
                        }
                    };

                    if let Some(step) = next_step {
                        world.add_component(*entity, step);
                    }
                }

                ExecutorStage::Draining => {
                    // Finalize the session: flush KV cache, release resources,
                    // produce any remaining output. For now just a terminal
                    // state — the entity will stay in Draining and downstream
                    // systems (TokenEmitter) can read the final ExecutorStep.
                    // A future eviction system may remove the entity.
                }
            }
        }

        Ok(())
    }
}
