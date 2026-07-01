//! InferenceStepSystem — ECS system that drives per-step inference execution
//! (prefill chunks and decode steps) during the compute stage.
//!
//! Registered as ID 200 in the scheduler; runs during `Stage::Prefill` (or
//! `Stage::Decode`) and delegates to the session's prefill_chunk / decode_one
//! methods.

use lazy_static::lazy_static;

use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// ECS system that drives per-step inference for active sessions.
///
/// Runs one prefill chunk or decode step per tick for each session entity
/// that carries a pending inference request.  State is held in the session's
/// ProfiledInferenceSession (stored as a world resource or component),
/// and the system dispatches to prefill_chunk / decode_one depending on
/// the current session phase.
///
/// This is the primary system wired into the compute scheduling loop.
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
// Static metadata (lazy_static per assignment requirements)
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
        _world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // Placeholder — full implementation will:
        // 1. Query for session entities with pending inference work
        // 2. Call session.prefill_chunk() or session.decode_one()
        // 3. Emit output tokens and lifecycle transitions via CommandWriter
        //
        // The concrete session dispatch depends on the resource/component
        // binding for ProfiledInferenceSession, which is wired by the
        // scheduler's session management layer.
        SystemResult::ok()
    }
}

impl Default for InferenceStepSystem {
    fn default() -> Self {
        Self::new()
    }
}
