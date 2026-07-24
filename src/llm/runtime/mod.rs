// ── Prism LLM Inference — Runtime Module ──────────────────────────────────
//
// Root of the inference runtime subsystem. Aggregates all 9 subsystems into a
// single PrismInferenceServer that orchestrates session lifecycle, weight
// residency, KV-cache management, lane dispatch, scheduling, cancellation,
// memory pressure monitoring, receipt storage, and HTTP serving.

use std::sync::Arc;
use parking_lot::RwLock;

use super::manifest::ContextProfile;
use super::server::InferenceExecutionPolicy;
use super::server::{
    CancellationHandle, CreateSessionRequest, GenerateRequest, InferenceCancelledReceipt,
    KvEpochId, KvPageId,
};
use crate::llm::manifest::SessionId;
use prism_ecs_runtime::{Command, CommandEnvelope, KernelHandle};

// ── Subsystem declarations ───────────────────────────────────────────

pub mod cancel;
pub mod kv;
pub mod lanes;
pub mod memory;
pub mod memory_hierarchy;
pub mod modality;
pub mod receipt;
pub mod residency;
pub mod scheduler;
pub mod server;
pub mod session;

// ── Re-exports from subsystems ───────────────────────────────────────

pub use cancel::CancellationManager;
pub use kv::KvManager;
pub use lanes::LaneRouter;
pub use memory::MemoryPressureMonitor;
pub use modality::{ModalityCapabilities, ModalityProvider};
pub use receipt::ReceiptStore;
pub use residency::WeightResidencyManager;
pub use scheduler::InferenceScheduler;
pub use server::HttpServer;
pub use session::SessionManager;

// ── Configuration ─────────────────────────────────────────────────────

/// Top-level configuration for the Prism LLM inference server.
///
/// This struct aggregates all configuration parameters required to
/// initialise the nine runtime subsystems.
pub struct ServerConfig {
    /// Filesystem path to the CImage artifact directory.
    pub cimage_path: String,
    /// Supported context profiles for inference.
    pub context_profiles: Vec<ContextProfile>,
    /// Default execution policy for lane selection.
    pub execution_policy: InferenceExecutionPolicy,
    /// Maximum number of concurrent sessions the server will admit.
    pub max_concurrent_sessions: u32,
    /// Optional HTTP listen address (e.g. "0.0.0.0:8080").
    /// When None, the HTTP server is not started.
    pub http_listen: Option<String>,
    /// Filesystem path for persistent receipt storage.
    pub receipt_store_path: String,
    /// Memory threshold (bytes) above which pressure is "elevated".
    pub memory_elevated_threshold_bytes: u64,
    /// Memory threshold (bytes) above which pressure is "critical".
    pub memory_critical_threshold_bytes: u64,
}

// ── Streaming event ──────────────────────────────────────────────────

/// An event emitted on the generation stream.
///
/// The server yields these tokens/events via a `tokio::sync::mpsc::Receiver`
/// returned from [`PrismInferenceServer::generate`].
pub enum GenerationStreamEvent {
    /// A generated token (decoded text fragment).
    Token(String),
    /// End-of-stream signal carrying the total token count.
    Done(u32),
    /// An error that terminated generation.
    Error(String),
    /// A status event (useful for observability).
    Status(String),
    /// Backpressure signal — the consumer is falling behind and the
    /// server has taken the configured action.
    Backpressure,
}

// ── Runtime Inference Server ─────────────────────────────────────────

/// The operational Prism LLM inference server.
///
/// Holds `Arc`-wrapped references to all nine runtime subsystems. Callers
/// obtain a server instance via [`PrismInferenceServer::new`] and then
/// drive inference through [`create_session`], [`generate`], [`cancel`],
/// and [`close_session`].
pub struct PrismInferenceServer {
    /// Manages session lifecycle (creation, state, teardown).
    pub session_manager: Arc<session::SessionManager>,
    /// Manages weight residency on device.
    pub residency_manager: Arc<residency::WeightResidencyManager>,
    /// Manages KV-cache epochs and pages.
    pub kv_manager: Arc<kv::KvManager>,
    /// Schedules prefill, decode, and auxiliary work.
    pub scheduler: Arc<scheduler::InferenceScheduler>,
    /// Routes dispatches to execution lanes.
    pub lane_router: Arc<lanes::LaneRouter>,
    /// ECS-owned Metal execution lane used by compiled model work items.
    pub ecs_metal_lane: Arc<lanes::EcsMetalLane>,
    /// Append-only event-sourced receipt store.
    pub receipt_store: Arc<receipt::ReceiptStore>,
    /// Cooperative session cancellation.
    pub cancellation_manager: Arc<cancel::CancellationManager>,
    /// Unified memory pressure monitoring.
    pub memory_monitor: Arc<memory::MemoryPressureMonitor>,
    /// Optional Axum-based HTTP server for the inference API.
    pub http_server: Arc<server::HttpServer>,
    ecs_kernel: Arc<RwLock<Option<KernelHandle>>>,
}

impl PrismInferenceServer {
    /// Constructs all nine subsystems from the given configuration.
    ///
    /// The HTTP server is initialised only when `config.http_listen` is
    /// `Some(...)`; otherwise the subsystem is created with a default
    /// placeholder address and will not be started.
    pub fn new(config: ServerConfig) -> Self {
        let session_manager = Arc::new(session::SessionManager::new());
        let residency_manager = Arc::new(residency::WeightResidencyManager::new());
        let kv_manager = Arc::new(kv::KvManager::new(4096, 32768));
        let scheduler = Arc::new(scheduler::InferenceScheduler::new());
        let lane_router = Arc::new(lanes::LaneRouter::new());
        let ecs_metal_lane = Arc::new(lanes::EcsMetalLane::new(Arc::new(
            prism_ecs_runtime::BackendExecutionRegistry::new(),
        )));
        let receipt_store = Arc::new(receipt::ReceiptStore::new(config.receipt_store_path));
        let cancellation_manager = Arc::new(cancel::CancellationManager::new());
        let memory_monitor = Arc::new(memory::MemoryPressureMonitor::new(
            config.memory_elevated_threshold_bytes,
            config.memory_critical_threshold_bytes,
        ));
        let http_listen = config
            .http_listen
            .unwrap_or_else(|| "127.0.0.1:0".to_string());
        let http_server = Arc::new(server::HttpServer::new(http_listen));

        PrismInferenceServer {
            session_manager,
            residency_manager,
            kv_manager,
            scheduler,
            lane_router,
            ecs_metal_lane,
            receipt_store,
            cancellation_manager,
            memory_monitor,
            http_server,
            ecs_kernel: Arc::new(RwLock::new(None)),
        }
    }

    /// Return the ECS-owned Metal lane for compiled kernel dispatch.
    pub fn metal_lane(&self) -> Arc<lanes::EcsMetalLane> {
        Arc::clone(&self.ecs_metal_lane)
    }

    /// Attach the canonical ECS kernel used for journaled inference/KV state.
    pub fn attach_ecs_kernel(&self, kernel: KernelHandle) {
        *self.ecs_kernel.write() = Some(kernel);
    }

    /// Propagate a KV epoch/page reservation to the canonical ECS work entity.
    pub fn bind_kv_to_ecs(
        &self,
        entity: u64,
        epoch: KvEpochId,
        page_ids: &[KvPageId],
        logical_context_tokens: u32,
        capacity_tokens: u32,
    ) -> Result<(), String> {
        let kernel = self
            .ecs_kernel
            .read()
            .clone()
            .ok_or_else(|| "ECS kernel is not attached".to_string())?;
        kernel
            .submit(CommandEnvelope::new(Command::BindInferenceKv {
                entity,
                epoch: epoch.0,
                page_ids: page_ids.iter().map(|page| page.0).collect(),
                logical_context_tokens,
                capacity_tokens,
            }))
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    /// Submit image/audio/video/Metal work to the canonical ECS world. The
    /// returned entity is the hand-off point for provider execution and
    /// completion receipts.
    pub fn submit_modality_work(
        &self,
        kind: prism_ecs_runtime::ModalityKind,
        model_path: impl Into<String>,
        prompt: impl Into<String>,
        output_path: impl Into<String>,
    ) -> Result<u64, String> {
        let kernel = self
            .ecs_kernel
            .read()
            .clone()
            .ok_or_else(|| "ECS kernel is not attached".to_string())?;
        let outcome = kernel
            .submit(CommandEnvelope::new(Command::CreateModalityWork {
                kind,
                model_path: model_path.into(),
                prompt: prompt.into(),
                output_path: output_path.into(),
            }))
            .map_err(|error| error.to_string())?;
        match outcome.result {
            prism_ecs_runtime::CommandResult::ModalitySubmitted { entity_id } => Ok(entity_id),
            other => Err(format!("unexpected modality command result: {other:?}")),
        }
    }

    /// Commit provider output provenance to the same ECS entity that was
    /// admitted for modality execution.
    pub fn complete_modality_work(
        &self,
        entity: u64,
        output_digest: impl Into<String>,
        output_bytes: u64,
    ) -> Result<(), String> {
        let kernel = self
            .ecs_kernel
            .read()
            .clone()
            .ok_or_else(|| "ECS kernel is not attached".to_string())?;
        kernel
            .submit(CommandEnvelope::new(Command::CompleteModalityWork {
                entity,
                output_digest: output_digest.into(),
                output_bytes,
            }))
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    pub fn fail_modality_work(&self, entity: u64, error: impl Into<String>) -> Result<(), String> {
        let kernel = self
            .ecs_kernel
            .read()
            .clone()
            .ok_or_else(|| "ECS kernel is not attached".to_string())?;
        kernel
            .submit(CommandEnvelope::new(Command::FailModalityWork {
                entity,
                error: error.into(),
            }))
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    /// Creates a new inference session and returns its [`SessionId`].
    ///
    /// Delegates to the session manager for admission and initial state
    /// setup. On success, a cancellation handle is registered with the
    /// cancellation manager so the session can be cancelled later.
    pub fn create_session(&self, request: CreateSessionRequest) -> Result<SessionId, String> {
        let session_id = self.session_manager.create_session(request)?;
        self.cancellation_manager.register_handle(session_id);
        Ok(session_id)
    }

    /// Starts streaming generation for an existing session.
    ///
    /// Returns a `tokio::sync::mpsc::Receiver` on which the caller can
    /// receive [`GenerationStreamEvent`] values as tokens are produced.
    ///
    /// An optional [`CancellationHandle`] is accepted so the caller can
    /// cancel generation asynchronously — the handle is registered before
    /// any work begins.
    pub fn generate(
        &self,
        _request: GenerateRequest,
        cancel: Option<CancellationHandle>,
    ) -> Result<tokio::sync::mpsc::Receiver<GenerationStreamEvent>, String> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        // If a cancellation handle was provided, register it.
        if let Some(ref handle) = cancel {
            self.cancellation_manager.register_handle(handle.session_id);
        }

        // This compatibility server has no loaded model runtime of its own.
        // Never fabricate tokens here: callers must use the ECS-backed wire
        // runtime, which owns token execution, KV binding, and receipts.
        tokio::spawn(async move {
            let _ = tx
                .send(GenerationStreamEvent::Error(
                    "no ECS-backed model runtime is attached".into(),
                ))
                .await;
        });

        Ok(rx)
    }

    /// Cancels an in-flight inference request.
    ///
    /// Delegates to the cancellation manager, which marks the session
    /// as cancelled. Downstream consumers check
    /// `cancellation_manager.is_cancelled` before proceeding with work.
    pub fn cancel(&self, handle: CancellationHandle) -> Result<InferenceCancelledReceipt, String> {
        self.cancellation_manager.cancel(&handle)
    }

    /// Closes an active session and releases its resources.
    ///
    /// Delegates to the session manager, which transitions the session
    /// to the Closed state and returns the final admission receipt.
    pub fn close_session(&self, id: SessionId) -> Result<(), String> {
        self.session_manager.close_session(&id)?;
        Ok(())
    }
}
