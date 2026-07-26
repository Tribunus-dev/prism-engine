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
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

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

    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        // Gather session entities that carry an ExecutorState component.
        let session_entities: Vec<Entity> = world
            .entities_of_kind(EntityKind::Session)
            .into_iter()
            .filter(|e| world.get_component::<ExecutorState>(*e).is_some())
            .collect();

        // Stage every per-session state transition + step insert on a
        // single `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.get_component_mut` / `world.add_component` calls
        // outside the WorldTxn seam are forbidden.
        //
        // Strategy: snapshot the pre-mutation value via `get_component`
        // (immutable read), produce a post-mutation value locally, and
        // stage the new value as an insert. This is the extract-mutate-
        // insert pattern documented in the Phase 2.5 changelog; it
        // preserves the constitutional discipline at the cost of one
        // clone per ExecutorState mutation.
        let mut txn = ConstitutionalWorldTxn::new();
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
                    if let Some(mut state) =
                        world.get_component::<ExecutorState>(*entity).cloned()
                    {
                        state.stage = ExecutorStage::Loading;
                        if let Err(e) = txn.stage_insert(*entity, state) {
                            tracing::warn!(entity = ?entity, error = %e, "executor: stage_insert Idle->Loading");
                        }
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
                        if let Some(mut state) =
                            world.get_component::<ExecutorState>(*entity).cloned()
                        {
                            state.stage = ExecutorStage::Prefill;
                            if let Err(e) = txn.stage_insert(*entity, state) {
                                tracing::warn!(entity = ?entity, error = %e, "executor: stage_insert Loading->Prefill");
                            }
                        }
                    }
                }

                ExecutorStage::Prefill => {
                    // Transition: Prefill → Decode, plus place a default
                    // step for the first decode iteration.
                    if let Some(mut state) =
                        world.get_component::<ExecutorState>(*entity).cloned()
                    {
                        state.stage = ExecutorStage::Decode;
                        if let Err(e) = txn.stage_insert(*entity, state) {
                            tracing::warn!(entity = ?entity, error = %e, "executor: stage_insert Prefill->Decode");
                        }
                    }
                    if let Err(e) = txn.stage_insert(
                        *entity,
                        ExecutorStep {
                            token_id: 0,
                            logits: None,
                            kv_block_indices: Vec::new(),
                        },
                    ) {
                        tracing::warn!(entity = ?entity, error = %e, "executor: stage_insert default ExecutorStep");
                    }
                }

                ExecutorStage::Decode => {
                    // Extract the post-mutation value within a scoped
                    // immutable read, then release the borrow before
                    // staging the insert.
                    let next_insert: Option<ExecutorState> =
                        if let Some(state) = world.get_component::<ExecutorState>(*entity) {
                            let mut projected = state.clone();
                            projected.step_counter += 1;
                            if projected.step_counter >= projected.max_steps {
                                projected.stage = ExecutorStage::Draining;
                            }
                            Some(projected)
                        } else {
                            None
                        };
                    if let Some(state) = next_insert {
                        if let Err(e) = txn.stage_insert(*entity, state) {
                            tracing::warn!(entity = ?entity, error = %e, "executor: stage_insert Decode tick");
                        }
                    }
                    // Compute the next step projection and stage it as
                    // an `ExecutorStep` insert (matches the original
                    // semantics when step_counter < max_steps).
                    if let Some(state) = world.get_component::<ExecutorState>(*entity) {
                        if state.step_counter < state.max_steps {
                            let step = ExecutorStep {
                                token_id: (state.step_counter % 65536) as u32,
                                logits: None,
                                kv_block_indices: vec![state.step_counter as u32],
                            };
                            if let Err(e) = txn.stage_insert(*entity, step) {
                                tracing::warn!(entity = ?entity, error = %e, "executor: stage_insert ExecutorStep");
                            }
                        }
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
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "executor: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("executor: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}
