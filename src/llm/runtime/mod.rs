// ── Prism LLM Inference — Runtime Module ──────────────────────────────────
//
// Root of the inference runtime subsystem. Aggregates all 9 subsystems into a
// single PrismInferenceServer that orchestrates session lifecycle, weight
// residency, KV-cache management, lane dispatch, scheduling, cancellation,
// memory pressure monitoring, receipt storage, and HTTP serving.

use std::sync::Arc;

use super::manifest::ContextProfile;
use super::server::InferenceExecutionPolicy;
use super::server::{
    CancellationHandle, CreateSessionRequest, GenerateRequest, InferenceCancelledReceipt,
};
use crate::llm::manifest::SessionId;

// ── Subsystem declarations ───────────────────────────────────────────

pub mod cancel;
pub mod kv;
pub mod lanes;
pub mod memory;
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
    /// Append-only event-sourced receipt store.
    pub receipt_store: Arc<receipt::ReceiptStore>,
    /// Cooperative session cancellation.
    pub cancellation_manager: Arc<cancel::CancellationManager>,
    /// Unified memory pressure monitoring.
    pub memory_monitor: Arc<memory::MemoryPressureMonitor>,
    /// Optional Axum-based HTTP server for the inference API.
    pub http_server: Arc<server::HttpServer>,
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
            receipt_store,
            cancellation_manager,
            memory_monitor,
            http_server,
        }
    }

    /// Creates a new inference session and returns its [`SessionId`].
    ///
    /// Delegates to the session manager for admission and initial state
    /// setup. On success, a cancellation handle is registered with the
    /// cancellation manager so the session can be cancelled later.
    pub fn create_session(
        &self,
        request: CreateSessionRequest,
    ) -> Result<SessionId, String> {
        let session_id = self.session_manager.create_session(request)?;
        self.cancellation_manager
            .register_handle(session_id);
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
            self.cancellation_manager
                .register_handle(handle.session_id);
        }

        // Spawn generation work on the tokio runtime.
        // In a full implementation this would drive the scheduler, lanes,
        // and KV manager; for now we emit a placeholder token.
        let cancel_mgr = Arc::clone(&self.cancellation_manager);
        let session_id = _request.session_id;
        let max_tokens = _request.max_new_tokens;
        tokio::spawn(async move {
            for i in 0..max_tokens {
                if cancel_mgr.is_cancelled(&session_id) {
                    let _ = tx
                        .send(GenerationStreamEvent::Error("cancelled".into()))
                        .await;
                    return;
                }
                // Simulate token production.
                if tx
                    .send(GenerationStreamEvent::Token(format!("token_{}", i)))
                    .await
                    .is_err()
                {
                    return;
                }
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
            let _ = tx.send(GenerationStreamEvent::Done(max_tokens)).await;
        });

        Ok(rx)
    }

    /// Cancels an in-flight inference request.
    ///
    /// Delegates to the cancellation manager, which marks the session
    /// as cancelled. Downstream consumers check
    /// `cancellation_manager.is_cancelled` before proceeding with work.
    pub fn cancel(
        &self,
        handle: CancellationHandle,
    ) -> Result<InferenceCancelledReceipt, String> {
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
