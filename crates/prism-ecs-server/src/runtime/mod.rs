// ── Prism LLM Inference — Runtime Module ──────────────────────────────────
//
// Root of the inference runtime subsystem. Aggregates all 9 subsystems into a
// single PrismInferenceServer that orchestrates session lifecycle, weight
// residency, KV-cache management, lane dispatch, scheduling, cancellation,
// memory pressure monitoring, receipt storage, and HTTP serving.

use prism_ecs_compile::{HardwareCapabilities, MultiModelManifest};
use prism_ecs_ir::model_graph::ModelGraph;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::RwLock;

use crate::runtime::manifest::ContextProfile;
use crate::runtime::manifest::SessionId;
use crate::runtime::server::PrefillDecodeRuntime;
use crate::runtime::server_types::{
    CancellationHandle, CreateSessionRequest, GenerateRequest, InferenceCancelledReceipt,
    InferenceExecutionPolicy,
};

// ── Local dependency stubs ──────────────────────────────────────────
// Manifest types (TODO: import from prism-ecs-ir::manifest when ported)
pub mod manifest;
// Server types (TODO: extract to crate root or dedicated crate)
pub mod server_types;

// ── Subsystem declarations ───────────────────────────────────────────

pub mod cancel;
pub mod kv;
pub mod lanes;
pub mod memory;
pub mod modality;
pub mod mori;
pub mod receipt;
pub mod residency;
pub mod scheduler;
pub mod server;
pub mod session;

pub mod backend;
pub mod wire_runtime;

// ── Re-exports from subsystems ───────────────────────────────────────

pub use backend::{
    BackendCapabilities, BackendKind, CancellationToken, ExecutionRecipe, ExternalBackendSpec,
    GenerationEvent, GenerationSession, InferenceBackend, InferenceTelemetry,
    InferenceTelemetrySnapshot,
};
pub use cancel::CancellationManager;
pub use kv::KvManager;
pub use lanes::LaneRouter;
pub use memory::MemoryPressureMonitor;
pub use modality::{ModalityCapabilities, ModalityProvider};
pub use mori::{
    MoriCapabilityKey, MoriCopyPin, MoriEcs, MoriRecoveryMetadata, MoriResidency,
    MoriResidencyStage, MoriRouteDescriptor, MoriRouteStage, MoriTransferId, MoriTransferReceipt,
    MoriTransferSession, MoriTransferState,
};
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ModelRegistryEntry {
    pub model_id: String,
    pub cimage_path: PathBuf,
    /// Digest of the embedded multimodal manifest captured for recovery.
    #[serde(default)]
    pub manifest_digest: Option<String>,
    /// Native ternary promotion receipt captured with the artifact for
    /// restart-time integrity checking.
    #[serde(default)]
    pub native_ternary_promotion:
        Option<prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence>,
    /// Joint ANE/Metal search provenance captured for recovery diagnostics.
    #[serde(default)]
    pub joint_tiling_evidence: Option<prism_ecs_compile::search::JointTilingEvidence>,
    /// Declarative execution recipe selected during admission.
    #[serde(default)]
    pub execution_recipe: backend::ExecutionRecipe,
    /// Capabilities advertised by the selected backend.
    #[serde(default)]
    pub backend_capabilities: Option<backend::BackendCapabilities>,
    /// Optional external engine configuration.
    #[serde(default)]
    pub external_backend: Option<backend::ExternalBackendSpec>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Default)]
pub struct ModelRegistrySnapshot {
    pub entries: Vec<ModelRegistryEntry>,
}

fn validate_manifest_hardware(manifest: &MultiModelManifest, model_id: &str) -> Result<(), String> {
    let lanes = lanes::LaneCapabilities::host();
    let available = HardwareCapabilities {
        ane: lanes.coreml_ane,
        metal: lanes.metal,
        accelerate: lanes.accelerate,
        video_toolbox: cfg!(target_os = "macos"),
        av_foundation: cfg!(target_os = "macos"),
    };
    for model in manifest.models.values() {
        model
            .requirements
            .validate_against(available.clone())
            .map_err(|error| format!("model {model_id}/{}: {error}", model.id))?;
    }
    Ok(())
}

fn multimodal_manifest_digest(manifest: Option<&MultiModelManifest>) -> Option<String> {
    let manifest = manifest?;
    let bytes = serde_json::to_vec(manifest).ok()?;
    use sha2::{Digest, Sha256};
    Some(format!("{:x}", Sha256::digest(bytes)))
}

// ── Runtime Inference Server ─────────────────────────────────────────

/// The operational Prism LLM inference server.
///
/// Holds `Arc`-wrapped references to all nine runtime subsystems. Callers
/// obtain a server instance via [`PrismInferenceServer::new`] and then
/// drive inference through [`create_session`], [`generate`], [`cancel`],
/// and [`close_session`].
pub struct PrismInferenceServer {
    /// Namespaced CImage registry used by multimodal routing and admission.
    pub model_registry: Arc<RwLock<HashMap<String, PathBuf>>>,
    pub model_manifests: Arc<RwLock<HashMap<String, prism_ecs_compile::MultiModelManifest>>>,
    pub model_graphs: Arc<RwLock<HashMap<String, ModelGraph>>>,
    pub vision_weights: Arc<RwLock<HashMap<String, Arc<HashMap<String, Vec<f32>>>>>>,
    pub live_runtimes:
        Arc<RwLock<HashMap<String, Arc<crate::runtime::wire_runtime::WirePrefillDecodeRuntime>>>>,
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
    /// Structured counters and latency samples for inference operations.
    pub telemetry: Arc<backend::InferenceTelemetry>,
}

impl PrismInferenceServer {
    /// Constructs all nine subsystems from the given configuration.
    ///
    /// The HTTP server is initialised only when `config.http_listen` is
    /// `Some(...)`; otherwise the subsystem is created with a default
    /// placeholder address and will not be started.
    pub fn new(config: ServerConfig) -> Self {
        let model_registry = Arc::new(RwLock::new(HashMap::new()));
        let model_manifests = Arc::new(RwLock::new(HashMap::new()));
        let model_graphs = Arc::new(RwLock::new(HashMap::new()));
        let vision_weights = Arc::new(RwLock::new(HashMap::new()));
        let live_runtimes = Arc::new(RwLock::new(HashMap::new()));
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
        let telemetry = Arc::new(backend::InferenceTelemetry::default());

        PrismInferenceServer {
            model_registry,
            model_manifests,
            model_graphs,
            vision_weights,
            live_runtimes,
            session_manager,
            residency_manager,
            kv_manager,
            scheduler,
            lane_router,
            receipt_store,
            cancellation_manager,
            memory_monitor,
            http_server,
            telemetry,
        }
    }

    /// Validate and register a namespaced CImage for model dispatch.
    pub fn register_model(
        &self,
        model_id: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<(), String> {
        let model_id = model_id.into();
        if model_id.is_empty() {
            return Err("model_id must not be empty".into());
        }
        let path = path.as_ref();
        if !path.is_file() {
            return Err(format!("model path does not exist: {}", path.display()));
        }
        let inspection = prism_ecs_compile::runtime::RuntimeModel::inspect(path)
            .map_err(|error| format!("invalid CImage for {model_id}: {error}"))?;
        if let Some(manifest) = inspection.model_manifest.as_ref() {
            manifest
                .validate()
                .map_err(|error| format!("invalid multimodal manifest for {model_id}: {error}"))?;
            validate_manifest_hardware(manifest, &model_id)?;
        }
        let mut registry = self
            .model_registry
            .write()
            .map_err(|_| "model registry lock poisoned".to_string())?;
        if registry.contains_key(&model_id) {
            return Err(format!("model_id already registered: {model_id}"));
        }
        registry.insert(model_id.clone(), path.to_path_buf());
        if let Some(manifest) = inspection.model_manifest {
            self.model_manifests
                .write()
                .map_err(|_| "model manifest registry lock poisoned".to_string())?
                .insert(model_id, manifest);
        }
        Ok(())
    }

    pub fn model_path(&self, model_id: &str) -> Result<PathBuf, String> {
        self.model_registry
            .read()
            .map_err(|_| "model registry lock poisoned".to_string())?
            .get(model_id)
            .cloned()
            .ok_or_else(|| format!("unknown model_id: {model_id}"))
    }

    /// Return the validated header inspection for a registered model.  This
    /// keeps operational status, admission, and recovery on the same CImage
    /// validation path and exposes native promotion/tiling provenance without
    /// loading model payloads.
    pub fn model_inspection(
        &self,
        model_id: &str,
    ) -> Result<prism_ecs_compile::runtime::CImageInspection, String> {
        let path = self.model_path(model_id)?;
        prism_ecs_compile::runtime::RuntimeModel::inspect(&path)
            .map_err(|error| format!("cannot inspect registered model {model_id}: {error}"))
    }

    /// Capture the durable namespace-to-CImage mapping for restart recovery.
    pub fn snapshot_model_registry(&self) -> Result<ModelRegistrySnapshot, String> {
        let registry = self
            .model_registry
            .read()
            .map_err(|_| "model registry lock poisoned".to_string())?;
        let mut entries = Vec::with_capacity(registry.len());
        for (model_id, cimage_path) in registry.iter() {
            let inspection = prism_ecs_compile::runtime::RuntimeModel::inspect(cimage_path)
                .map_err(|error| format!("cannot snapshot CImage {model_id}: {error}"))?;
            entries.push(ModelRegistryEntry {
                model_id: model_id.clone(),
                cimage_path: cimage_path.clone(),
                manifest_digest: multimodal_manifest_digest(inspection.model_manifest.as_ref()),
                native_ternary_promotion: inspection.native_ternary_promotion,
                joint_tiling_evidence: inspection.joint_tiling_evidence,
                execution_recipe: backend::ExecutionRecipe::default(),
                backend_capabilities: None,
                external_backend: None,
            });
        }
        entries.sort_by(|left, right| left.model_id.cmp(&right.model_id));
        Ok(ModelRegistrySnapshot { entries })
    }

    /// Restore a previously captured namespace registry. Every CImage is
    /// reopened and revalidated before registration, including its embedded
    /// multi-model manifest and execution-plan consistency.
    pub fn restore_model_registry(&self, snapshot: &ModelRegistrySnapshot) -> Result<(), String> {
        let mut validated = Vec::with_capacity(snapshot.entries.len());
        let mut ids = std::collections::HashSet::new();
        for entry in &snapshot.entries {
            if entry.model_id.is_empty() || !ids.insert(entry.model_id.clone()) {
                return Err(format!(
                    "invalid or duplicate model_id in recovery snapshot: {}",
                    entry.model_id
                ));
            }
            if !entry.cimage_path.is_file() {
                return Err(format!(
                    "model path does not exist: {}",
                    entry.cimage_path.display()
                ));
            }
            let inspection = prism_ecs_compile::runtime::RuntimeModel::inspect(&entry.cimage_path)
                .map_err(|error| {
                    format!(
                        "invalid CImage for {} during recovery: {error}",
                        entry.model_id
                    )
                })?;
            if entry.manifest_digest
                != multimodal_manifest_digest(inspection.model_manifest.as_ref())
            {
                return Err(format!(
                    "multimodal manifest changed for {} since recovery snapshot",
                    entry.model_id
                ));
            }
            if entry.native_ternary_promotion != inspection.native_ternary_promotion {
                return Err(format!(
                    "native ternary promotion evidence changed for {} since recovery snapshot",
                    entry.model_id
                ));
            }
            if entry.joint_tiling_evidence != inspection.joint_tiling_evidence {
                return Err(format!(
                    "joint tiling evidence changed for {} since recovery snapshot",
                    entry.model_id
                ));
            }
            if let Some(manifest) = inspection.model_manifest.as_ref() {
                manifest.validate().map_err(|error| {
                    format!(
                        "invalid multimodal manifest for {} during recovery: {error}",
                        entry.model_id
                    )
                })?;
                validate_manifest_hardware(manifest, &entry.model_id)?;
            }
            validated.push((
                entry.model_id.clone(),
                entry.cimage_path.clone(),
                inspection.model_manifest,
            ));
        }
        let mut registry = self
            .model_registry
            .write()
            .map_err(|_| "model registry lock poisoned".to_string())?;
        let mut manifests = self
            .model_manifests
            .write()
            .map_err(|_| "model manifest registry lock poisoned".to_string())?;
        if validated
            .iter()
            .any(|(model_id, _, _)| registry.contains_key(model_id))
        {
            return Err("recovery snapshot overlaps an already registered model_id".into());
        }
        for (model_id, path, manifest) in validated {
            registry.insert(model_id.clone(), path);
            if let Some(manifest) = manifest {
                manifests.insert(model_id, manifest);
            }
        }
        Ok(())
    }

    /// Persist the registry snapshot with a replace-by-rename commit so a
    /// process interruption cannot leave a partially written recovery file.
    pub fn save_model_registry_snapshot(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let snapshot = self.snapshot_model_registry()?;
        let bytes = serde_json::to_vec_pretty(&snapshot)
            .map_err(|error| format!("serialize model registry snapshot: {error}"))?;
        let temp = path.with_extension("tmp");
        std::fs::write(&temp, bytes)
            .map_err(|error| format!("write model registry snapshot: {error}"))?;
        std::fs::rename(&temp, path)
            .map_err(|error| format!("commit model registry snapshot: {error}"))
    }

    /// Load and validate a persisted registry snapshot, then restore it.
    pub fn restore_model_registry_from_file(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<ModelRegistrySnapshot, String> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|error| format!("read model registry snapshot: {error}"))?;
        let snapshot: ModelRegistrySnapshot = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse model registry snapshot: {error}"))?;
        self.restore_model_registry(&snapshot)?;
        Ok(snapshot)
    }

    /// Return the embedded specialized-model registry for a namespace, when
    /// the CImage declares one.
    pub fn model_manifest(
        &self,
        model_id: &str,
    ) -> Result<Option<prism_ecs_compile::MultiModelManifest>, String> {
        Ok(self
            .model_manifests
            .read()
            .map_err(|_| "model manifest registry lock poisoned".to_string())?
            .get(model_id)
            .cloned())
    }

    /// Resolve a namespaced model entry from either its directly registered
    /// CImage bundle or another bundle that embeds the namespace.
    pub fn manifest_for_namespace(
        &self,
        namespace: &str,
    ) -> Result<Option<prism_ecs_compile::ModelManifest>, String> {
        let manifests = self
            .model_manifests
            .read()
            .map_err(|_| "model manifest registry lock poisoned".to_string())?;
        Ok(manifests
            .get(namespace)
            .and_then(|manifest| manifest.get(namespace).cloned())
            .or_else(|| {
                manifests
                    .values()
                    .find_map(|manifest| manifest.get(namespace).cloned())
            }))
    }

    pub fn manifest_containing_namespace(
        &self,
        namespace: &str,
    ) -> Result<Option<prism_ecs_compile::MultiModelManifest>, String> {
        let manifests = self
            .model_manifests
            .read()
            .map_err(|_| "model manifest registry lock poisoned".to_string())?;
        Ok(manifests.get(namespace).cloned().or_else(|| {
            manifests
                .values()
                .find(|m| m.get(namespace).is_some())
                .cloned()
        }))
    }

    pub fn has_live_runtime(&self, model_id: &str) -> Result<bool, String> {
        Ok(self
            .live_runtimes
            .read()
            .map_err(|_| "live runtime registry lock poisoned".to_string())?
            .contains_key(model_id))
    }

    /// Number of registered runtimes that can execute CPU/reference or
    /// heterogeneous generation requests.
    pub fn live_runtime_count(&self) -> Result<usize, String> {
        Ok(self
            .live_runtimes
            .read()
            .map_err(|_| "live runtime registry lock poisoned".to_string())?
            .len())
    }

    pub fn live_runtime(
        &self,
        model_id: &str,
    ) -> Result<Arc<crate::runtime::wire_runtime::WirePrefillDecodeRuntime>, String> {
        self.live_runtimes
            .read()
            .map_err(|_| "live runtime registry lock poisoned".to_string())?
            .get(model_id)
            .cloned()
            .ok_or_else(|| format!("model_id has no live runtime: {model_id}"))
    }

    /// Register the normalized graph required to construct a live engine for
    /// an already-validated CImage namespace.
    pub fn register_model_graph(&self, model_id: &str, graph: ModelGraph) -> Result<(), String> {
        self.model_path(model_id)?;
        let mut graphs = self
            .model_graphs
            .write()
            .map_err(|_| "model graph registry lock poisoned".to_string())?;
        if graphs.contains_key(model_id) {
            return Err(format!("model graph already registered: {model_id}"));
        }
        graphs.insert(model_id.to_string(), graph);
        Ok(())
    }

    pub fn register_vision_weights(
        &self,
        model_id: &str,
        weights: HashMap<String, Vec<f32>>,
    ) -> Result<(), String> {
        self.model_path(model_id)?;
        let mut registry = self
            .vision_weights
            .write()
            .map_err(|_| "vision weight registry lock poisoned".to_string())?;
        if registry.contains_key(model_id) {
            return Err(format!("vision weights already registered: {model_id}"));
        }
        registry.insert(model_id.to_string(), Arc::new(weights));
        Ok(())
    }

    pub fn vision_weights(&self, model_id: &str) -> Result<Arc<HashMap<String, Vec<f32>>>, String> {
        self.vision_weights
            .read()
            .map_err(|_| "vision weight registry lock poisoned".to_string())?
            .get(model_id)
            .cloned()
            .ok_or_else(|| format!("no vision weights registered: {model_id}"))
    }

    pub fn register_live_runtime(&self, model_id: &str, eos_id: u32) -> Result<(), String> {
        let path = self.model_path(model_id)?;
        let graph = self
            .model_graphs
            .read()
            .map_err(|_| "model graph registry lock poisoned".to_string())?
            .get(model_id)
            .cloned()
            .ok_or_else(|| format!("no graph registered for model_id: {model_id}"))?;
        let runtime = crate::runtime::wire_runtime::WirePrefillDecodeRuntime::from_cimage(
            &path, graph, eos_id,
        )?;
        let mut runtimes = self
            .live_runtimes
            .write()
            .map_err(|_| "live runtime registry lock poisoned".to_string())?;
        if runtimes.contains_key(model_id) {
            return Err(format!("live runtime already registered: {model_id}"));
        }
        runtimes.insert(model_id.to_string(), Arc::new(runtime));
        Ok(())
    }

    /// Register a live runtime for a specialist namespace embedded in an
    /// owning CImage. The specialist gets its own graph/runtime identity,
    /// while weights and execution artifacts remain backed by the owner's
    /// single CImage file.
    pub fn register_namespaced_live_runtime(
        &self,
        namespace: &str,
        owner_model_id: &str,
        graph: ModelGraph,
        eos_id: u32,
    ) -> Result<(), String> {
        if namespace.is_empty() || owner_model_id.is_empty() {
            return Err("namespace and owner_model_id must not be empty".into());
        }
        if self.has_live_runtime(namespace)? {
            return Err(format!("live runtime already registered: {namespace}"));
        }
        if self.manifest_for_namespace(namespace)?.is_none() {
            return Err(format!("no manifest entry for namespace: {namespace}"));
        }
        let path = self.model_path(owner_model_id)?;
        let runtime = crate::runtime::wire_runtime::WirePrefillDecodeRuntime::from_cimage(
            &path, graph, eos_id,
        )?;
        {
            let mut registry = self
                .model_registry
                .write()
                .map_err(|_| "model registry lock poisoned".to_string())?;
            if registry.contains_key(namespace) {
                return Err(format!("model_id already registered: {namespace}"));
            }
            registry.insert(namespace.to_string(), path);
        }
        let mut runtimes = self
            .live_runtimes
            .write()
            .map_err(|_| "live runtime registry lock poisoned".to_string())?;
        if runtimes.contains_key(namespace) {
            return Err(format!("live runtime already registered: {namespace}"));
        }
        runtimes.insert(namespace.to_string(), Arc::new(runtime));
        Ok(())
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
        request: GenerateRequest,
        cancel: Option<CancellationHandle>,
    ) -> Result<tokio::sync::mpsc::Receiver<GenerationStreamEvent>, String> {
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        // If a cancellation handle was provided, register it.
        if let Some(handle) = &cancel {
            self.cancellation_manager.register_handle(handle.session_id);
        }

        let model_id = self
            .session_manager
            .get_receipt(&request.session_id)
            .ok_or_else(|| format!("session {:?} not found", request.session_id))?
            .cimage_id
            .0;
        let runtime = self.live_runtime(&model_id)?;

        // Spawn generation work on the tokio runtime.
        let cancel_mgr = Arc::clone(&self.cancellation_manager);
        let telemetry = Arc::clone(&self.telemetry);
        telemetry.started();
        let session_id = request.session_id;
        let max_tokens = request.max_new_tokens;
        let prompt = request.prompt;
        let sampling = request.sampling;
        tokio::spawn(async move {
            struct CancellationCleanup {
                manager: Arc<cancel::CancellationManager>,
                session_id: SessionId,
            }
            impl Drop for CancellationCleanup {
                fn drop(&mut self) {
                    self.manager.clear(&self.session_id);
                }
            }
            let _cleanup = CancellationCleanup {
                manager: Arc::clone(&cancel_mgr),
                session_id,
            };
            {
                let prompt_tokens = match runtime.tokenize(&prompt) {
                    Ok(tokens) => tokens,
                    Err(error) => {
                        telemetry.failed();
                        let _ = tx.send(GenerationStreamEvent::Error(error)).await;
                        return;
                    }
                };
                let prefill_started = std::time::Instant::now();
                let mut logits = match runtime.run_prefill(&prompt_tokens) {
                    Ok(logits) => logits,
                    Err(error) => {
                        telemetry.failed();
                        telemetry.fallback(format!("prefill:{error}"));
                        let _ = tx.send(GenerationStreamEvent::Error(error)).await;
                        return;
                    }
                };
                telemetry.prefill_latency(prefill_started.elapsed().as_secs_f64() * 1000.0);
                let eos = runtime.eos_token_id();
                let mut generated = 0;
                while generated < max_tokens {
                    if cancel_mgr.is_cancelled(&session_id) {
                        telemetry.cancelled();
                        let _ = tx
                            .send(GenerationStreamEvent::Error("cancelled".into()))
                            .await;
                        return;
                    }
                    let token = match runtime.sample(&logits, &sampling) {
                        Ok(token) => token,
                        Err(error) => {
                            telemetry.failed();
                            let _ = tx.send(GenerationStreamEvent::Error(error)).await;
                            return;
                        }
                    };
                    if token == eos {
                        break;
                    }
                    match runtime.detokenize(token) {
                        Ok(text) => {
                            if tx.send(GenerationStreamEvent::Token(text)).await.is_err() {
                                return;
                            }
                        }
                        Err(error) => {
                            telemetry.failed();
                            let _ = tx.send(GenerationStreamEvent::Error(error)).await;
                            return;
                        }
                    }
                    generated += 1;
                    let decode_started = std::time::Instant::now();
                    logits = match runtime.run_decode(token) {
                        Ok(logits) => logits,
                        Err(error) => {
                            telemetry.failed();
                            telemetry.fallback(format!("decode:{error}"));
                            let _ = tx.send(GenerationStreamEvent::Error(error)).await;
                            return;
                        }
                    };
                    telemetry.decode_latency(decode_started.elapsed().as_secs_f64() * 1000.0);
                }
                telemetry.completed(generated as u64);
                let _ = tx.send(GenerationStreamEvent::Done(generated)).await;
            }
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

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use prism_ecs_compile::model_manifest::{
        ModelIoBinding, ModelIoKind, ModelManifest, ModelModality,
    };

    fn multimodal_fixture() -> MultiModelManifest {
        let mut manifest = MultiModelManifest::default();
        manifest
            .insert(ModelManifest {
                id: "vision".into(),
                modality: ModelModality::Vision,
                inputs: vec![ModelIoBinding {
                    name: "image".into(),
                    kind: ModelIoKind::ImageRgba,
                    dtype: "u8".into(),
                    shape: vec![1, 224, 224, 4],
                    optional: false,
                }],
                outputs: vec![ModelIoBinding {
                    name: "embedding".into(),
                    kind: ModelIoKind::Embedding,
                    dtype: "f32".into(),
                    shape: vec![1, 1024],
                    optional: false,
                }],
                requirements: Default::default(),
                program_names: vec!["vision_encoder".into()],
                projectors: Vec::new(),
                fusion_inputs: Vec::new(),
            })
            .unwrap();
        manifest
    }

    #[test]
    fn recovery_manifest_digest_is_stable_and_detects_configuration_changes() {
        let original = multimodal_fixture();
        let original_digest = multimodal_manifest_digest(Some(&original));
        assert_eq!(original_digest, multimodal_manifest_digest(Some(&original)));

        let mut changed = original.clone();
        changed.models.get_mut("vision").unwrap().program_names[0] = "vision_encoder_v2".into();
        assert_ne!(original_digest, multimodal_manifest_digest(Some(&changed)));
    }

    #[test]
    fn recovery_snapshot_remains_compatible_without_legacy_digest() {
        let snapshot: ModelRegistrySnapshot = serde_json::from_value(serde_json::json!({
            "entries": [{"model_id": "vision", "cimage_path": "/tmp/vision.cimage"}]
        }))
        .unwrap();
        assert_eq!(snapshot.entries[0].manifest_digest, None);
    }

    #[test]
    fn recovery_snapshot_preserves_native_provenance() {
        let evidence =
            prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence {
                cpu_canary: prism_ecs_quantization::ternarization::promotion::BackendPass::passed(),
                accelerate_reconstruction:
                    prism_ecs_quantization::ternarization::promotion::BackendPass::passed(),
                metal_packed: prism_ecs_quantization::ternarization::promotion::BackendPass::passed(
                ),
                ane_static: prism_ecs_quantization::ternarization::promotion::BackendPass::passed(),
                cimage_replay:
                    prism_ecs_quantization::ternarization::promotion::BackendPass::passed(),
                behavioral_reference:
                    prism_ecs_quantization::ternarization::promotion::BackendPass::passed(),
                activation_error: Some(0.0),
                logit_divergence: Some(0.0),
                task_loss: Some(0.0),
                router_agreement: Some(1.0),
                router_margin_error: Some(0.0),
                logit_cross_entropy: Some(0.0),
                generation_loss: Some(0.0),
                expert_balance_error: Some(0.0),
                ane_selected: true,
                packed_abi_digest: "abi".into(),
                reference_digest: "reference".into(),
            };
        let tiling = prism_ecs_compile::search::JointTilingEvidence {
            selected_configuration: None,
            selected_score: Some(1.0),
            both_backends_feasible: true,
            both_backends_measured: true,
            profiles_evaluated: Vec::new(),
        };
        let snapshot = ModelRegistrySnapshot {
            entries: vec![ModelRegistryEntry {
                model_id: "vision".into(),
                cimage_path: "/tmp/vision.cimage".into(),
                manifest_digest: None,
                native_ternary_promotion: Some(evidence.clone()),
                joint_tiling_evidence: Some(tiling.clone()),
                execution_recipe: backend::ExecutionRecipe::default(),
                backend_capabilities: None,
                external_backend: None,
            }],
        };
        let restored: ModelRegistrySnapshot =
            serde_json::from_value(serde_json::to_value(snapshot).unwrap()).unwrap();
        assert_eq!(restored.entries[0].native_ternary_promotion, Some(evidence));
        assert_eq!(restored.entries[0].joint_tiling_evidence, Some(tiling));
    }
}
