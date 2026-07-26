//! Protocol types — version, generation regime, and the host↔worker
//! message-kind enums.
//!
//! This sub-module owns the *envelope* of the worker protocol: the
//! discriminated [`MessageKind`] (a [`HostCommand`] sent host→worker or
//! a [`WorkerEvent`] sent worker→host), the protocol version, and the
//! generation regime selector that lives on
//! [`crate::worker_protocol::payloads::StartGenerationPayload`].
//!
//! The 16 type-specific payload schemas live in
//! [`crate::worker_protocol::payloads`]; the [`Frame`] envelope and
//! validators live in [`crate::worker_protocol::frame`].

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

// ── Generation regime ──────────────────────────────────────────────────────

/// Distinguishes autoregressive from diffusion generation in
/// [`crate::worker_protocol::payloads::StartGenerationPayload`]. Differs
/// from a raw `String` so the dispatch surface can pattern-match on the
/// variant without parsing the wire string.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GenerationRegime {
    /// Standard autoregressive (causal) decoding.
    #[default]
    Autoregressive,
    /// Diffusion-style generation with iterative canvas commitment.
    Diffusion,
}

// ── Version ────────────────────────────────────────────────────────────────

/// Protocol version identifier. Pair of `(major, minor)` integers; the
/// runtime accepts [`V1_0`] only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u16,
    pub minor: u16,
}

/// Current protocol version: 1.0.
pub const V1_0: ProtocolVersion = ProtocolVersion { major: 1, minor: 0 };

// ── Message Kinds ──────────────────────────────────────────────────────────

/// Commands sent from the host to the worker.
///
/// Every [`Frame`](crate::worker_protocol::Frame) whose direction is
/// host→worker carries one of these as its
/// [`MessageKind::HostCommand`] variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HostCommand {
    /// Initial handshake — worker must respond with [`WorkerEvent::HelloAck`].
    Hello,
    /// Load a model by identifier.
    LoadModel,
    /// Begin token generation for a prompt.
    StartGeneration,
    /// Request cancellation of an in-flight generation.
    CancelGeneration,
    /// Unload the currently loaded model.
    UnloadModel,
    /// Liveness probe — worker should respond with a [`WorkerEvent::Heartbeat`].
    Ping,
    /// Graceful shutdown — worker should terminate after flushing.
    Shutdown,
    /// Sent by the watchdog when the worker's RSS crosses the soft ceiling.
    MemoryPressure,
}

/// Events emitted from the worker to the host.
///
/// Every [`Frame`](crate::worker_protocol::Frame) whose direction is
/// worker→host carries one of these as its [`MessageKind::WorkerEvent`]
/// variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkerEvent {
    /// Acknowledgment of [`HostCommand::Hello`].
    HelloAck,
    /// Model load operation has begun.
    ModelLoadStarted,
    /// Model load completed successfully.
    ModelLoaded,
    /// Manifest + segments resolved, arena allocated.
    ComputeImageBound,
    /// Core ML ANE programs compiled and loaded.
    AnePrepared,
    /// MLX GPU shader/library warmup done.
    GpuPrepared,
    /// Accelerate CPU lane initialized.
    CpuPrepared,
    /// KV cache pool constructed.
    KvArenaReady,
    /// At least one dry-run dispatch per lane succeeded.
    RoutesValidated,
    /// Terminal: generation may now be accepted.
    ModelReady,
    /// Generation has been accepted and is starting.
    GenerationStarted,
    /// Prefill (prompt processing) phase started.
    PrefillStarted,
    /// Prefill completed; about to enter decode loop.
    PrefillCompleted,
    /// A single output token produced during decoding.
    Token,
    /// Per-step performance metrics (latency, throughput).
    StepMetrics,
    /// Generation completed normally. Payload: [`crate::worker_protocol::payloads::GenerationCompletedPayload`].
    GenerationCompleted,
    /// Generation was cancelled by the host.
    GenerationCancelled,
    /// Generation failed with an error. Payload: [`crate::worker_protocol::payloads::GenerationFailedPayload`].
    GenerationFailed,
    /// Periodic worker health report. Payload: [`crate::worker_protocol::payloads::HeartbeatPayload`].
    Heartbeat,
    /// Model has been fully unloaded.
    ModelUnloaded,
    /// Fatal worker error — worker is about to terminate.
    WorkerFatal,
    /// Batch of research trace events from the worker.
    ResearchTraceBatch,
    /// A diffusion denoising step has started.
    DiffusionStepStarted,
    /// A diffusion denoising step has completed.
    DiffusionStepCompleted,
    /// The diffusion canvas has been updated.
    CanvasUpdated,
    /// One or more canvas positions have been committed.
    PositionsCommitted,
    /// The diffusion generation has converged.
    Converged,
    /// Diffusion generation completed.
    DiffusionGenerationCompleted,
}

/// Combined message kind.
///
/// Serializes as a flat kebab-case string (e.g. `"hello"`, `"hello-ack"`)
/// thanks to `#[serde(untagged)]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageKind {
    /// Host-to-worker command.
    HostCommand(HostCommand),
    /// Worker-to-host event.
    WorkerEvent(WorkerEvent),
}
