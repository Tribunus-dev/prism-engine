//! Runtime port trait definitions for future integration.
//!
//! These traits define the boundaries between the runtime kernel and
//! provider-specific infrastructure (durable storage, coordination, hardware,
//! etc.). In-memory test implementations live in [`crate::test_adapters`].

use prism_ecs_constitutional::{ClassifiedComponent, DurableClass, DurableComponent, SchemaKey};

/// Versioned snapshot of the world state for restart recovery.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SnapshotPayload {
    pub schema_version: u32,
    pub world_epoch: u64,
    pub next_entity_id: u64,
    pub last_command_sequence: u64,
    pub allocator_data: Vec<u8>,
    pub schedule_hash: [u8; 32],
    pub created_at_ms: u64,
}

/// Wrapper with integrity checksum over the payload.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WorldSnapshot {
    pub payload: SnapshotPayload,
    pub checksum: [u8; 32],
}

impl WorldSnapshot {
    /// Compute the checksum over the payload only (not self).
    pub fn compute_checksum(payload: &SnapshotPayload) -> [u8; 32] {
        let bytes = bincode::serialize(payload).unwrap_or_default();
        *blake3::hash(&bytes).as_bytes()
    }

    /// Verify that the stored checksum matches the payload.
    pub fn verify(&self) -> bool {
        self.checksum == Self::compute_checksum(&self.payload)
    }
}

/// Tick receipt store — persists schedule tick receipts for observability.
pub trait TickReceiptStore: Send + Sync {
    fn save(
        &self,
        receipt: &crate::schedule::TickReceipt,
        daemon_instance_id: &str,
    ) -> Result<(), RuntimeError>;
}

/// Snapshot store — persists and loads world snapshots.
pub trait SnapshotStore: Send + Sync {
    fn save(&self, snapshot: &WorldSnapshot) -> Result<(), RuntimeError>;
    fn load_latest(&self) -> Result<Option<WorldSnapshot>, RuntimeError>;
}

/// Authority journal port — durable append-only event log.
pub trait AuthorityJournal: Send + Sync {
    fn append(&self, batch: &[u8]) -> Result<u64, RuntimeError>;
    fn replay(&self, from_seq: u64) -> Result<Vec<Vec<u8>>, RuntimeError>;
}

/// Lease coordinator port — distributed fencing.
pub trait LeaseCoordinator: Send + Sync {
    fn acquire(&self, key: &str, ttl_ms: u64) -> Result<bool, RuntimeError>;
    fn renew(&self, key: &str, ttl_ms: u64) -> Result<bool, RuntimeError>;
    fn release(&self, key: &str) -> Result<(), RuntimeError>;
}

/// Evidence sink port — chained receipt recording.
pub trait EvidenceSink: Send + Sync {
    fn record(&self, receipt: &[u8]) -> Result<(), RuntimeError>;
}

/// Hardware dispatcher port — device-specific execution.
pub trait HardwareDispatcher: Send + Sync {
    fn dispatch(&self, payload: &[u8]) -> Result<Vec<u8>, RuntimeError>;
}

// ── WorkDispatcher — provider-neutral dispatch contract ──────────────────────

/// A fenced dispatch request — the durable record of intent to execute work
/// on a specific backend. Must carry all information needed to start, poll,
/// and cancel execution without access to the originating ECS world.
#[derive(Debug, Clone)]
pub struct DispatchRequest {
    /// Work identity from the ECS entity.
    pub work_entity: u64,
    /// Attempt number (fencing against stale dispatch handles).
    pub attempt: u32,
    /// Plan generation — monotonically increasing counter for the plan this
    /// dispatch executes. Combined with work_entity and attempt to form a
    /// deterministic dispatch identity that survives retries and recovery.
    pub plan_generation: u32,
    /// Fencing token from the lease.
    pub lease_token: String,
    /// Absolute deadline in ms since epoch.
    pub deadline_ms: u64,
    /// Backend identifier (e.g. "metal", "ane", "subprocess").
    pub backend: String,
    /// Compiler backend configuration (JSON-encoded).
    pub config: String,
    /// Input artifact path or reference.
    pub input_path: String,
    /// Output staging directory path.
    pub output_path: String,
}

/// Opaque handle representing an in-flight dispatch on a backend.
#[derive(Debug, Clone)]
pub struct DispatchHandle {
    /// Provider-assigned identifier for this work.
    pub id: String,
    /// Work entity this handle belongs to.
    pub work_entity: u64,
    /// The attempt this handle was created for.
    pub attempt: u32,
}

/// Status of an in-flight dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchStatus {
    /// Execution is in progress.
    Running,
    /// Execution completed successfully. Payload contains the result data.
    Completed(Vec<u8>),
    /// Execution failed with an error.
    Failed(String),
}

/// Errors that can occur during dispatch operations.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("dispatch start failed: {0}")]
    StartFailed(String),
    #[error("poll failed: {0}")]
    PollFailed(String),
    #[error("cancel failed: {0}")]
    CancelFailed(String),
    #[error("backend not available: {0}")]
    BackendUnavailable(String),
    #[error("stale attempt: handle attempt {handle_attempt} != current attempt {current_attempt}")]
    StaleAttempt {
        handle_attempt: u32,
        current_attempt: u32,
    },
}

/// Provider-neutral dispatch contract.
///
/// Implementations wrap compiler, hardware, subprocess, and remote execution.
/// The lifecycle is: start → poll → (collect | cancel).
pub trait WorkDispatcher: Send + Sync {
    /// Persist a fenced dispatch intent and begin execution.
    fn start(&self, request: &DispatchRequest) -> Result<DispatchHandle, DispatchError>;

    /// Poll an active dispatch for status. Must be cheap to call repeatedly.
    fn poll(&self, handle: &DispatchHandle) -> Result<DispatchStatus, DispatchError>;

    /// Cancel an active dispatch. Best-effort: the backend may have already
    /// completed the work.
    fn cancel(&self, handle: &DispatchHandle) -> Result<(), DispatchError>;
}

// ── Provider selection ─────────────────────────────────────────────────────

/// A provider exposed by the runtime composition root. `backend` is kept as
/// a neutral string because `WorkDispatcher` and remote adapters may use
/// backend names that are not represented by one Rust enum.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderDescriptor {
    pub id: String,
    pub backend: String,
    pub priority: u32,
    pub available: bool,
}

impl ProviderDescriptor {
    pub fn new(id: impl Into<String>, backend: impl Into<String>, priority: u32) -> Self {
        Self {
            id: id.into(),
            backend: backend.into(),
            priority,
            available: true,
        }
    }

    pub fn unavailable(mut self) -> Self {
        self.available = false;
        self
    }
}

/// Why a requested provider was not used.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    RequestedProviderUnavailable { provider: String },
    CandidateProviderUnavailable { provider: String },
    RequestedProviderNotSpecified,
    NoAvailableProvider,
}

/// A structured provider-selection request owned by the runtime kernel.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderSelectionRequest {
    pub operation: String,
    pub requested_provider: Option<String>,
    pub fallback_providers: Vec<String>,
}

/// Evidence for a provider decision. The receipt is returned to callers and
/// included in tick receipts; persistence remains the responsibility of the
/// existing receipt/evidence adapters.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ProviderSelectionReceipt {
    pub request_id: uuid::Uuid,
    pub operation: String,
    pub requested_provider: Option<String>,
    pub selected_provider: Option<String>,
    pub selected_backend: Option<String>,
    pub attempted_providers: Vec<String>,
    pub fallback_reason: Option<FallbackReason>,
}

impl ProviderSelectionReceipt {
    pub fn selected_backend(&self) -> Option<&str> {
        self.selected_backend.as_deref()
    }

    pub fn fell_back(&self) -> bool {
        self.fallback_reason.is_some()
    }
}

/// Provider selection authority. Implementations may inspect hardware or
/// remote health, but the decision is always requested through `KernelHandle`.
pub trait ProviderSelector: Send + Sync {
    fn select(&self, request: &ProviderSelectionRequest) -> ProviderSelectionReceipt;
}

/// Deterministic selector used by the default kernel and by focused tests.
/// Production composition roots can supply a selector backed by their
/// existing backend-health adapters.
#[derive(Debug, Clone)]
pub struct StaticProviderSelector {
    providers: Vec<ProviderDescriptor>,
}

impl StaticProviderSelector {
    pub fn new(mut providers: Vec<ProviderDescriptor>) -> Self {
        providers.sort_by_key(|provider| (provider.priority, provider.id.clone()));
        Self { providers }
    }

    pub fn providers(&self) -> &[ProviderDescriptor] {
        &self.providers
    }
}

impl Default for StaticProviderSelector {
    fn default() -> Self {
        Self::new(vec![ProviderDescriptor::new("cpu", "cpu", u32::MAX)])
    }
}

impl ProviderSelector for StaticProviderSelector {
    fn select(&self, request: &ProviderSelectionRequest) -> ProviderSelectionReceipt {
        let requested = request
            .requested_provider
            .as_deref()
            .filter(|provider| !provider.is_empty() && *provider != "auto");
        let mut attempted = Vec::new();
        let mut fallback_reason = None;
        let mut unavailable_candidate = None;

        let mut candidates = Vec::new();
        if let Some(provider) = requested {
            candidates.push(provider.to_string());
        }
        candidates.extend(request.fallback_providers.iter().cloned());
        if candidates.is_empty() {
            candidates.extend(self.providers.iter().map(|provider| provider.id.clone()));
        }

        let mut selected = None;
        for candidate in candidates {
            if attempted
                .iter()
                .any(|attempt: &String| attempt == &candidate)
            {
                continue;
            }
            attempted.push(candidate.clone());
            if let Some(provider) = self
                .providers
                .iter()
                .find(|provider| provider.id == candidate)
            {
                if provider.available {
                    selected = Some((provider.id.clone(), provider.backend.clone()));
                    if requested.is_some_and(|requested| requested != candidate) {
                        fallback_reason = Some(FallbackReason::RequestedProviderUnavailable {
                            provider: requested.unwrap_or_default().to_string(),
                        });
                    }
                    break;
                }
                unavailable_candidate = Some(candidate.clone());
            } else {
                unavailable_candidate = Some(candidate.clone());
            }
            if requested == Some(candidate.as_str()) {
                fallback_reason = Some(FallbackReason::RequestedProviderUnavailable {
                    provider: candidate,
                });
            }
        }

        if selected.is_none() {
            if fallback_reason.is_none() {
                fallback_reason = Some(FallbackReason::NoAvailableProvider);
            }
        } else if fallback_reason.is_none() {
            fallback_reason = unavailable_candidate
                .map(|provider| FallbackReason::CandidateProviderUnavailable { provider })
                .or_else(|| {
                    request
                        .requested_provider
                        .is_none()
                        .then_some(FallbackReason::RequestedProviderNotSpecified)
                });
        }

        ProviderSelectionReceipt {
            request_id: uuid::Uuid::new_v4(),
            operation: request.operation.clone(),
            requested_provider: request.requested_provider.clone(),
            selected_provider: selected.as_ref().map(|(provider, _)| provider.clone()),
            selected_backend: selected.map(|(_, backend)| backend),
            attempted_providers: attempted,
            fallback_reason,
        }
    }
}

/// A no-op test dispatcher that immediately completes.
#[allow(dead_code)]
pub struct NoopDispatcher;

impl WorkDispatcher for NoopDispatcher {
    fn start(
        &self,
        request: &DispatchRequest,
    ) -> std::result::Result<DispatchHandle, DispatchError> {
        Ok(DispatchHandle {
            id: "noop".to_string(),
            work_entity: request.work_entity,
            attempt: request.attempt,
        })
    }

    fn poll(&self, _handle: &DispatchHandle) -> std::result::Result<DispatchStatus, DispatchError> {
        Ok(DispatchStatus::Completed(vec![]))
    }

    fn cancel(&self, _handle: &DispatchHandle) -> std::result::Result<(), DispatchError> {
        Ok(())
    }
}

/// Kernel clock port — deterministic or wall-clock time.
pub trait KernelClock: Send + Sync {
    fn now_ms(&self) -> u64;
}

// ── Admission + CommandStore ──────────────────────────────────────────────

/// Admission outcome — the result of attempting to admit a command.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Admission {
    /// Command was admitted — this caller owns execution.
    Admitted { sequence: u64 },
    /// Command was already completed — here is the existing result.
    Completed {
        result: String,
        sequence: u64,
        world_epoch: u64,
    },
    /// Command is in flight by another caller.
    InFlight,
}

/// A completed command ready for replay.
#[derive(Debug, Clone)]
pub struct CompletedCommand {
    pub sequence: u64,
    pub idempotency_key: uuid::Uuid,
    pub envelope_json: String,
    pub result_json: String,
    pub world_epoch: u64,
}

/// An admitted but incomplete command.
#[derive(Debug, Clone)]
pub struct AdmittedCommand {
    pub sequence: u64,
    pub idempotency_key: uuid::Uuid,
    pub envelope_json: String,
}

/// High-water marks for recovery.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CommandWatermarks {
    pub last_committed_sequence: u64,
    pub last_admitted_sequence: u64,
    pub unresolved_count: u64,
}

/// Command store — atomic admission, completion, and replay for typed commands.
pub trait CommandStore: Send + Sync {
    /// Try to admit a command. Returns Admitted with a unique sequence on
    /// first use, Completed with the prior result on duplicate, or InFlight
    /// if another actor is currently executing this idempotency key.
    fn admit(
        &self,
        idempotency_key: &uuid::Uuid,
        envelope_json: &str,
    ) -> Result<Admission, RuntimeError>;

    /// Atomically mark an admitted command as completed with its result.
    fn complete(
        &self,
        sequence: u64,
        result_json: &str,
        world_epoch: u64,
    ) -> Result<(), RuntimeError>;

    /// Lookup a completed result by idempotency key.
    fn lookup(&self, idempotency_key: &uuid::Uuid) -> Result<Option<String>, RuntimeError>;

    /// Return all completed commands with sequence greater than the given value,
    /// ordered by sequence ascending.
    fn completed_after(&self, sequence: u64) -> Result<Vec<CompletedCommand>, RuntimeError>;

    /// Return all admitted-but-not-completed commands.
    fn unresolved(&self) -> Result<Vec<AdmittedCommand>, RuntimeError>;

    /// Return high-water marks: last committed sequence, last admitted
    /// sequence, and count of unresolved commands.
    fn high_water_marks(&self) -> Result<CommandWatermarks, RuntimeError>;

    /// Transition an admitted-but-not-completed command to a new state.
    ///
    /// Used during recovery to free idempotency keys for commands that were
    /// in-flight at the time of a crash, so they can be re-submitted.
    fn transition_state(&self, sequence: u64, target_state: &str) -> Result<(), RuntimeError>;
}

/// Durable payload for result publication.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResultPayload {
    pub result_type: String,
    pub result: String,
}

impl prism_ecs_core::Component for ResultPayload {}
impl ClassifiedComponent for ResultPayload {
    type Class = DurableClass;
}
impl DurableComponent for ResultPayload {
    const SCHEMA_KEY: SchemaKey = SchemaKey {
        namespace: "prism.runtime",
        id: 1,
        version: 1,
    };
}

// ── RuntimeError ───────────────────────────────────────────────────────────
// Defined here so ports can refer to it without cyclic dependency on kernel.
// Re-exported from crate root alongside the kernel module.

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("entity error: {0}")]
    Entity(String),
    #[error("journal error: {0}")]
    Journal(String),
    #[error("receipt error: {0}")]
    Receipt(String),
    #[error("unknown command: {0}")]
    UnknownCommand(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("epoch mismatch: expected {expected}, got {actual}")]
    EpochMismatch { expected: u64, actual: u64 },
    #[error("idempotency conflict: {0}")]
    IdempotencyConflict(String),
    #[error("lease error: {0}")]
    Lease(String),
    #[error("dispatch error: {0}")]
    Dispatch(String),
}
#[derive(Debug, Clone)]
pub struct RecoveredCommand {
    pub sequence: u64,
    pub idempotency_key: uuid::Uuid,
    pub command: crate::kernel::Command,
    pub result: crate::kernel::CommandResult,
    pub world_epoch: u64,
}

/// Result of a recovery attempt from a snapshot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RecoveryReport {
    pub recovery_state: String,
    pub snapshot_epoch: u64,
    pub snapshot_sequence: u64,
    pub replayed_commands: u64,
    pub unresolved_commands: u64,
    pub world_epoch_before: u64,
}
