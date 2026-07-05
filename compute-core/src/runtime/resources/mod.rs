//! ECS resource types for Prism Engine worker supervision (Slice 2).
//!
//! Resources are per-World singletons shared across systems.  Each resource
//! in this module participates in schedule dependency resolution via its
//! stable `SchedulableResource::RESOURCE_ID`.

#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod audio;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod kv_cache_coordinator;
pub mod legacy_worker_bridge;
pub mod monotonic_clock;
pub mod npu_completion_port;
pub mod text_to_speech;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod vision;
pub mod worker_diagnostics;
pub mod worker_event_source;
pub mod worker_ingress_queue;
pub mod worker_pool;
pub mod worker_process_manager;
pub mod worker_response_registry;
pub mod worker_supervision_config;

#[cfg(feature = "mlx-backend")]
pub use audio::AudioEncoderResource;
#[cfg(feature = "mlx-backend")]
pub use kv_cache_coordinator::{KVCacheCoordinator, LiveKvCache};
pub use legacy_worker_bridge::LegacyWorkerBridge;
pub use monotonic_clock::MonotonicClockResource;
pub use npu_completion_port::NpuCompletionPort;
pub use text_to_speech::TextToSpeechResource;
#[cfg(feature = "mlx-backend")]
pub use vision::VisionEncoderResource;
pub use worker_diagnostics::WorkerDiagnosticsResource;
pub use worker_event_source::{EventKind, WorkerEventEnvelope, WorkerEventSource};
pub use worker_ingress_queue::{IngressEntry, WorkerIngressQueue};
pub use worker_pool::WorkerPoolResource;
pub use worker_process_manager::{WorkerId, WorkerProcessHandles, WorkerProcessManager};
pub use worker_response_registry::WorkerResponseRegistry;
pub use worker_supervision_config::{EcsWorkerSupervisionConfig, EcsWorkerSupervisionMode};

// ── Stable resource IDs ───────────────────────────────────────────────

pub const WORKER_POOL_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 10;
pub const WORKER_INGRESS_QUEUE_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 11;
pub const WORKER_EVENT_SOURCE_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 12;
pub const WORKER_RESPONSE_REGISTRY_RESOURCE: crate::runtime::scheduling::component_id::ResourceId =
    13;
pub const WORKER_DIAGNOSTICS_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 14;
pub const MONOTONIC_CLOCK_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 15;
pub const LEGACY_WORKER_BRIDGE_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 16;
pub const WORKER_SUPERVISION_CONFIG_RESOURCE: crate::runtime::scheduling::component_id::ResourceId =
    17;

pub const WORKER_PROCESS_MANAGER_RESOURCE: crate::runtime::scheduling::component_id::ResourceId =
    18;

pub const KV_CACHE_COORDINATOR_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 19;

pub const NPU_COMPLETION_PORT_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 20;

pub const AUDIO_ENCODER_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 21;

pub const TEXT_TO_SPEECH_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 22;
pub const VISION_ENCODER_RESOURCE: crate::runtime::scheduling::component_id::ResourceId = 23;

// ── SchedulableResource impls ─────────────────────────────────────────

use crate::runtime::scheduling::component_id::{ResourceId, SchedulableResource};

impl SchedulableResource for WorkerPoolResource {
    const RESOURCE_ID: ResourceId = WORKER_POOL_RESOURCE;
    const NAME: &'static str = "WorkerPoolResource";
}

impl SchedulableResource for WorkerIngressQueue {
    const RESOURCE_ID: ResourceId = WORKER_INGRESS_QUEUE_RESOURCE;
    const NAME: &'static str = "WorkerIngressQueue";
}

impl SchedulableResource for WorkerEventSource {
    const RESOURCE_ID: ResourceId = WORKER_EVENT_SOURCE_RESOURCE;
    const NAME: &'static str = "WorkerEventSource";
}

impl SchedulableResource for WorkerResponseRegistry {
    const RESOURCE_ID: ResourceId = WORKER_RESPONSE_REGISTRY_RESOURCE;
    const NAME: &'static str = "WorkerResponseRegistry";
}

impl SchedulableResource for WorkerDiagnosticsResource {
    const RESOURCE_ID: ResourceId = WORKER_DIAGNOSTICS_RESOURCE;
    const NAME: &'static str = "WorkerDiagnosticsResource";
}

impl SchedulableResource for MonotonicClockResource {
    const RESOURCE_ID: ResourceId = MONOTONIC_CLOCK_RESOURCE;
    const NAME: &'static str = "MonotonicClockResource";
}

impl SchedulableResource for LegacyWorkerBridge {
    const RESOURCE_ID: ResourceId = LEGACY_WORKER_BRIDGE_RESOURCE;
    const NAME: &'static str = "LegacyWorkerBridge";
}

impl SchedulableResource for EcsWorkerSupervisionConfig {
    const RESOURCE_ID: ResourceId = WORKER_SUPERVISION_CONFIG_RESOURCE;
    const NAME: &'static str = "EcsWorkerSupervisionConfig";
}

impl SchedulableResource for WorkerProcessManager {
    const RESOURCE_ID: ResourceId = WORKER_PROCESS_MANAGER_RESOURCE;
    const NAME: &'static str = "WorkerProcessManager";
}

impl SchedulableResource for NpuCompletionPort {
    const RESOURCE_ID: ResourceId = NPU_COMPLETION_PORT_RESOURCE;
    const NAME: &'static str = "NpuCompletionPort";
}
