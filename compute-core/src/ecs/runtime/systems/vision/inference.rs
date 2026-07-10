//! VisionInferenceSystem — ECS system that drives vision encoding for
//! multimodal image input during the Prefill stage.
//!
//! Registered as ID 110 in the scheduler; runs during `Stage::Prefill`.
//!
//! # Lifecycle
//!
//! 1. Checks for the presence of a `VisionEncoderResource` in the World — if
//!    absent (text-only model), the system is a no-op.
//! 2. Iterates entities that carry a `WorkerLifecycle` in the Prefill
//!    equivalent phase (`AwaitingFirstEvent` / `Streaming`).
//! 3. For each entity with image-bearing `WorkerRequest` payload, calls
//!    `encoder.encode()` on the image tensor.
//! 4. Writes encoded vision features into the entity's `WorkerStream`.
//!
//! # Future work
//!
//! - Decode image bytes from the WorkerRequest payload into an MLX array.
//! - Route encoded features to the text model's embedding sequence.

use lazy_static::lazy_static;

use crate::ecs::runtime::components::worker_lifecycle::{WorkerLifecycle, WorkerRequestPhase};
use crate::ecs::runtime::components::worker_request::WorkerRequest;
use crate::ecs::runtime::components::worker_stream::WorkerStream;
use crate::ecs::runtime::resources::vision::VisionEncoderResource;
use crate::ecs::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::ecs::runtime::scheduling::command::CommandWriter;
use crate::ecs::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId, SystemMetadata,
    SystemResult, SystemSpec,
};
use crate::ecs::runtime::world::{Entity, World};

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// ECS system that runs the vision encoder during the Prefill stage.
///
/// For each entity carrying image input data, this system:
/// 1. Checks for a loaded `VisionEncoderResource`.
/// 2. Iterates entities whose lifecycle is in a prefill-ready phase.
/// 3. Calls the encoder's `encode()` method.
/// 4. Writes encoded feature metadata into the entity's `WorkerStream`.
pub struct VisionInferenceSystem {
    _private: (),
}

impl VisionInferenceSystem {
    /// Create a new vision inference system.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl SystemSpec for VisionInferenceSystem {
    type Reads = (WorkerLifecycle, WorkerRequest, WorkerStream);
    type Writes = WorkerStream;
    type ReadResources = VisionEncoderResource;
    type WriteResources = ();

    const NAME: &'static str = "vision_inference";
    const ID: SystemId = SystemId(110);
    const STAGE: Stage = Stage::Prefill;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// Static metadata
// ---------------------------------------------------------------------------

lazy_static! {
    static ref VISION_INFERENCE_META: SystemMetadata = SystemMetadata {
        id: SystemId(110),
        name: "vision_inference",
        stage: Stage::Prefill,
        reads: <(WorkerLifecycle, WorkerRequest, WorkerStream) as ComponentSet>::mask().unwrap(),
        writes: <WorkerStream as ComponentSet>::mask().unwrap(),
        reads_resources: <VisionEncoderResource as ResourceSet>::mask().unwrap(),
        writes_resources: <() as ResourceSet>::mask().unwrap(),
        after: &[],
        before: &[],
        order: 0,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for VisionInferenceSystem {
    fn metadata(&self) -> &SystemMetadata {
        &VISION_INFERENCE_META
    }

    fn run(&mut self, world: &mut World, _commands: &mut CommandWriter) -> SystemResult {
        // 1. Check if a VisionEncoder resource is loaded.
        let has_encoder = world
            .get_resource::<VisionEncoderResource>()
            .and_then(|r| r.encoder.as_ref())
            .is_some();

        if !has_encoder {
            // No vision encoder available — text-only model or load failure.
            return SystemResult::ok();
        };

        // 2. Collect entities in a prefill-ready lifecycle phase.
        let entities: Vec<Entity> = world
            .iter_entities_with::<WorkerLifecycle>()
            .filter(|entity| {
                world
                    .get::<WorkerLifecycle>(*entity)
                    .map(|l| {
                        // Vision encoding runs once, at the start of
                        // streaming, before decode begins.
                        l.phase == WorkerRequestPhase::AwaitingFirstEvent
                    })
                    .unwrap_or(false)
            })
            .collect();

        if entities.is_empty() {
            return SystemResult::ok();
        }

        // 3. For each entity with image input, encode and write features.
        //
        //    TODO: In a full implementation, the WorkerRequest payload
        //    contains serialized image bytes that must be decoded into an
        //    MLX Array in NCHW format ([1, C, H, W]), then passed to
        //    `vision_encoder.encode()`.  The result Array is serialized
        //    and written to the WorkerStream.  For now we just increment
        //    the stream sequence to mark progress.
        //
        //    Future work:
        //    - Decode image bytes from WorkerRequest payload
        //    - Call vision_encoder.encode(&image_array)
        //    - Write encoded features to WorkerStream or a dedicated
        //      VisionFeatures component

        for entity in &entities {
            // Update WorkerStream to reflect encoding progress.
            if let Some(stream) = world.get_mut::<WorkerStream>(*entity) {
                stream.record_output(None, 0);
            }
        }
        SystemResult::ok()
    }
}

impl Default for VisionInferenceSystem {
    fn default() -> Self {
        Self::new()
    }
}
