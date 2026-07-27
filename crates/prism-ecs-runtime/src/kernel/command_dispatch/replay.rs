//! Canonical replay path for committed commands.
//!
//! Authority: this module owns the canonical authority for the replay
//! path — the [`apply_recovered_command`] function that re-applies a
//! committed command (loaded from the journal during restart
//! recovery) to the world, with entity-id and result-variant
//! verification. It does **not** own the live submit path (which lives
//! in [`super::submit`]), the data shapes (which live in
//! [`super::envelope`]), or the typed command implementations (which
//! live in [`super::submit`] and [`super::envelope`]).
//!
//! ## Classification
//!
//! The replay path is **canonical** for journal recovery. It does not
//! acquire the lease coordinator (no live work) and does not call
//! `command_store.complete` (the result is already durable). It only
//! re-applies the world mutation and verifies the result matches what
//! was stored. It therefore crosses criterion 3 (it takes the world
//! write lock) but does not perform any external effect.

use prism_ecs_constitutional::lifecycle_command::LifecycleCommandResult;
use prism_ecs_core::{Entity, World};

use crate::modality::{ModalityExecution, ModalityFailure};
use crate::ports::{CompletedCommand, RuntimeError};

use super::envelope::{Command, CommandDispatchContext, CommandEnvelope, CommandResult};
use super::submit::{
    execute_advance_inference, execute_bind_inference_kv, execute_cancel_txn,
    execute_create_modality_work, execute_lifecycle, execute_register_model, execute_spawn,
};

/// Replay a command that was already committed — no journal, no
/// idempotency checking, no receipt. Just apply the world mutation
/// and verify that the result matches what was stored.
pub fn apply_recovered_command(
    completed: &CompletedCommand,
    ctx: &CommandDispatchContext<'_>,
) -> Result<(), RuntimeError> {
    // Parse the envelope to extract the command
    let envelope: CommandEnvelope = serde_json::from_str(&completed.envelope_json)
        .map_err(|e| RuntimeError::Journal(format!("replay: bad envelope: {e}")))?;

    // Deserialize the stored result for verification
    let stored_result: CommandResult = serde_json::from_str(&completed.result_json)
        .map_err(|e| RuntimeError::Journal(format!("replay: bad result: {e}")))?;

    let mut world = ctx
        .world
        .write()
        .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;

    match envelope.command {
        Command::SpawnAgent {
            parent_id,
            task,
            max_steps,
        } => {
            let entity_id = execute_spawn(&mut world, parent_id, &task, max_steps)?;
            if let CommandResult::Spawned {
                entity_id: expected,
            } = &stored_result
            {
                if entity_id != *expected {
                    return Err(RuntimeError::Journal(format!(
                        "replay entity ID mismatch: generated {} but stored result has {}",
                        entity_id, expected
                    )));
                }
            }
        }
        Command::CancelAgent { agent_id } => {
            execute_cancel_txn(&mut world, agent_id)?;
        }
        Command::RegisterModel {
            name,
            source_path,
            format,
        } => {
            let entity_id = execute_register_model(&mut world, &name, &source_path, &format)?;
            if let CommandResult::Registered {
                entity_id: expected,
            } = &stored_result
            {
                if entity_id != *expected {
                    return Err(RuntimeError::Journal(format!(
                        "replay register entity ID mismatch: generated {} but stored result has {}",
                        entity_id, expected
                    )));
                }
            }
        }
        Command::AdvanceInference {
            entity,
            phase,
            prefilled_tokens,
            generated_tokens,
            kv_epoch,
            kv_tokens,
        } => {
            let _ = execute_advance_inference(
                &mut world,
                entity,
                phase,
                prefilled_tokens,
                generated_tokens,
                kv_epoch,
                kv_tokens,
            )?;
        }
        Command::BindInferenceKv {
            entity,
            epoch,
            page_ids,
            logical_context_tokens,
            capacity_tokens,
        } => {
            execute_bind_inference_kv(
                &mut world,
                entity,
                epoch,
                page_ids,
                logical_context_tokens,
                capacity_tokens,
            )?;
        }
        Command::CreateModalityWork {
            kind,
            model_path,
            prompt,
            output_path,
        } => {
            let entity_id = execute_create_modality_work(
                &mut world,
                kind,
                model_path,
                prompt,
                output_path,
            )?;
            if let CommandResult::ModalitySubmitted {
                entity_id: expected,
            } = stored_result
            {
                if entity_id != expected {
                    return Err(RuntimeError::Journal(format!(
                        "replay modality entity mismatch: generated {entity_id}, stored {expected}"
                    )));
                }
            }
        }
        Command::CompleteModalityWork {
            entity,
            output_digest,
            output_bytes,
        } => {
            world
                .add_component(
                    Entity::new(entity, 0),
                    ModalityExecution {
                        output_digest: output_digest.clone(),
                        output_bytes,
                    },
                )
                .map_err(|error| RuntimeError::Journal(error.to_string()))?;
            match stored_result {
                CommandResult::ModalityCompleted {
                    entity_id: expected,
                    output_digest: expected_digest,
                } if entity == expected && output_digest == expected_digest => {}
                _ => {
                    return Err(RuntimeError::Journal(
                        "replay modality completion mismatch".into(),
                    ))
                }
            }
        }
        Command::FailModalityWork { entity, error } => {
            world
                .add_component(
                    Entity::new(entity, 0),
                    ModalityFailure {
                        error: error.clone(),
                    },
                )
                .map_err(|error| RuntimeError::Journal(error.to_string()))?;
            match stored_result {
                CommandResult::ModalityFailed {
                    entity_id,
                    error: expected,
                } if entity == entity_id && error == expected => {}
                _ => {
                    return Err(RuntimeError::Journal(
                        "replay modality failure mismatch".into(),
                    ))
                }
            }
        }
        // Re-execute lifecycle commands so entity ID allocation stays
        // consistent across the command sequence. The execute_lifecycle
        // function performs actual state changes and returns a result
        // that we verify against the stored result.
        Command::Lifecycle(lc) => {
            let new_result = execute_lifecycle(&mut world, lc)?;
            // Verify result variant matches (entity IDs must be consistent
            // since replay runs all commands in order).
            match (&new_result, &stored_result) {
                (
                    CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                        work_entity,
                        ..
                    }),
                    CommandResult::Lifecycle(LifecycleCommandResult::WorkCreated {
                        work_entity: expected,
                        ..
                    }),
                ) if work_entity != expected => {
                    return Err(RuntimeError::Journal(format!(
                        "replay work entity ID mismatch: generated {} but expected {}",
                        work_entity, expected
                    )))
                }
                _ if std::mem::discriminant(&new_result)
                    != std::mem::discriminant(&stored_result) =>
                {
                    return Err(RuntimeError::Journal(format!(
                        "replay lifecycle result variant mismatch: got {:?} expected {:?}",
                        new_result, stored_result
                    )))
                }
                _ => {}
            }
        }
    }

    drop(world);
    Ok(())
}
