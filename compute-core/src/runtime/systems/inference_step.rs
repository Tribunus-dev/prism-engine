//! InferenceStepSystem — ECS system that evaluates activations for streaming
//! inference entities during the compute stage.
//!
//! Registered as ID 200 in the scheduler; runs during `Stage::Prefill`.

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

use crate::runtime::components::worker_lifecycle::{WorkerLifecycle, WorkerRequestPhase};
use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::{Entity, World};

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// ECS system that drives activation evaluation for streaming inference entities.
///
/// Queries entities in the Streaming phase, calls `evaluate_activations()` on
/// their gate/up tensors, then emits `InferenceStepCompleted` indicating the
/// step finished.
pub struct InferenceStepSystem {
    _private: (),
}

impl InferenceStepSystem {
    /// Create a new inference step system.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl SystemSpec for InferenceStepSystem {
    type Reads = ();
    type Writes = ();
    type ReadResources = ();
    type WriteResources = ();

    const NAME: &'static str = "inference_step";
    const ID: SystemId = SystemId(200);
    const STAGE: Stage = Stage::Prefill;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// InferenceStepCompleted marker component
// ---------------------------------------------------------------------------

/// Marker component emitted by `InferenceStepSystem` to indicate that the
/// activation evaluation step completed for an entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStepCompleted {
    /// Sequence number of the completed step.
    pub step: u64,
}

// ---------------------------------------------------------------------------
// Static metadata
// ---------------------------------------------------------------------------

lazy_static! {
    static ref INFERENCE_STEP_META: SystemMetadata = SystemMetadata {
        id: SystemId(200),
        name: "inference_step",
        stage: Stage::Prefill,
        reads: <() as ComponentSet>::mask().unwrap(),
        writes: <() as ComponentSet>::mask().unwrap(),
        reads_resources: <() as ResourceSet>::mask().unwrap(),
        writes_resources: <() as ResourceSet>::mask().unwrap(),
        after: &[],
        before: &[],
        order: 0,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for InferenceStepSystem {
    fn metadata(&self) -> &SystemMetadata {
        &INFERENCE_STEP_META
    }

    fn run(
        &mut self,
        world: &mut World,
        commands: &mut CommandWriter,
    ) -> SystemResult {
        // 1. Collect entities currently in the Streaming phase
        let entities: Vec<Entity> = world
            .iter_entities_with::<WorkerLifecycle>()
            .filter(|entity| {
                world
                    .get::<WorkerLifecycle>(*entity)
                    .map(|l| l.phase == WorkerRequestPhase::Streaming)
                    .unwrap_or(false)
            })
            .collect();

        // 2. For each streaming entity, evaluate activations and emit the
        //    completed marker.
        //
        //    TODO: In a full implementation, gate/up tensor components would be
        //    queried here and evaluate_activations() called.  For now we just
        //    emit InferenceStepCompleted.
        //
        //    Future work:
        //    - Query GateBuffer, UpBuffer components from the entity
        //    - Call evaluate_activations(activation_func, gate, up.as_deref())
        //    - Write results back via commands

        for (step, entity) in entities.into_iter().enumerate() {
            let step = step as u64;
            // Ignore insertion errors; the command buffer will report them
            // during application.
            let _ = commands.insert(
                entity,
                InferenceStepCompleted { step },
            );
        }

        SystemResult::ok()
    }
}

impl Default for InferenceStepSystem {
    fn default() -> Self {
        Self::new()
    }
}
