//! Worker protocol — the canonical authority for the framed host↔worker
//! message contract used by Prism's compute workers.
//!
//! This module owns the on-the-wire schema for the length-prefixed JSON
//! protocol that links a runtime host (the schedule) to a compute worker
//! (the backend process). It was formerly
//! `compute-core/src/ecs/core/worker_protocol.rs` (1,170 LOC); the
//! re-implementation keeps the same surface but lifts it into the
//! constitutional runtime crate and promotes the untyped identifier
//! fields to newtypes where they were authority-bearing.
//!
//! The protocol is **deserialized and emitted** by the runtime kernel
//! when it talks to a remote compute worker; the protocol does not
//! *itself* decide canonical lifecycle transitions. A protocol frame is
//! evidence that a worker said or received something; the runtime
//! converts that evidence into a constitutional command (admission,
//! step completion, terminal, etc.). Frames are durably storable through
//! the [`crate::ports::EvidenceSink`] port.
//!
//! # Module layout
//!
//! - [`types`] — [`GenerationRegime`], [`ProtocolVersion`] (and the
//!   [`V1_0`] constant), [`HostCommand`], [`WorkerEvent`], and the
//!   [`MessageKind`] enum that ties them together.
//! - [`payloads`] — the 16 type-specific payload schemas that ride inside
//!   a [`Frame::payload`] field.
//! - [`frame`] — the [`Frame`] envelope, the stateless [`validate_frame`]
//!   checker, the [`FrameValidationError`] enum, and the stateful
//!   [`ProtocolValidator`] that tracks in-flight and terminal request ids.
//!
//! # Frame contract
//!
//! Every frame carries:
//! - A [`ProtocolVersion`] — currently `V1_0`. Mismatches are rejected.
//! - A `worker_instance_id` — opaque newtype below; the validator rejects
//!   frames from a worker it does not expect.
//! - A `sequence_number` — strictly monotonic per-sender. Regressions and
//!   gaps are rejected.
//! - A `request_id` — optional; set on frames that belong to a specific
//!   generation request.
//! - A [`MessageKind`] — `HostCommand` or `WorkerEvent`, the two
//!   directions.
//! - A `payload` — type-specific schema (see the [`payloads`] sub-module).
//!
//! # Validation
//!
//! Two layers:
//!
//! 1. [`frame::validate_frame`] — stateless; checks size, version, sequence,
//!    worker id, and message kind.
//! 2. [`frame::ProtocolValidator`] — stateful; tracks in-flight and terminal
//!    request ids so that direction-aware rules (no duplicate
//!    `StartGeneration` for the same `request_id`, no events after a
//!    terminal, etc.) are enforced without external bookkeeping.
//!
//! # Tensor payloads do not cross the wire
//!
//! All tensor data stays in the worker's address space; only metadata,
//! token IDs, and control messages flow over this channel. The protocol
//! is intentionally small.

#![forbid(unsafe_code)]

pub mod frame;
pub mod payloads;
pub mod types;

#[cfg(test)]
mod tests;

pub use frame::{validate_frame, Frame, FrameValidationError, ProtocolValidator, MAX_FRAME_SIZE_BYTES};
pub use payloads::{
    ConvergedPayload, DiffusionGenerationCompletedPayload, DiffusionStepCompletedPayload,
    DiffusionStepStartedPayload, CanvasUpdatedPayload, GenerationCompletedPayload,
    GenerationFailedPayload, HeartbeatPayload, PolicySnapshotPayload, PositionsCommittedPayload,
    ResearchTraceBatchPayload, ResearchTraceEventJson, StartGenerationPayload, TokenPayload,
    WorkerFatalPayload,
};
pub use types::{GenerationRegime, HostCommand, MessageKind, ProtocolVersion, WorkerEvent, V1_0};
