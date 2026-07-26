//! Frame envelope, validation, and stateful validator for the
//! host↔worker protocol.
//!
//! This sub-module owns the on-the-wire [`Frame`] struct, the
//! stateless [`validate_frame`] checker, and the stateful
//! [`ProtocolValidator`] that tracks in-flight and terminal request
//! ids. The frame types live here rather than alongside the message
//! kinds in the parent `worker_protocol` module so the file stays
//! under the 900-LOC threshold and the single-authority discipline
//! remains crisp.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{HostCommand, MessageKind, WorkerEvent};

// ── Frame ──────────────────────────────────────────────────────────────────

/// Maximum serialized frame size in bytes (1 MB).
pub const MAX_FRAME_SIZE_BYTES: usize = 1_048_576;

/// A single framed message in the host–worker length-prefixed JSON protocol.
///
/// Every frame carries a protocol version, a worker-instance identifier, a
/// monotonically increasing sequence number, an optional request correlation
/// id, a discriminated message kind, and an arbitrary JSON payload.
///
/// The sender frames each message by serializing this struct to JSON, then
/// writing a 4-byte little-endian length prefix followed by the JSON bytes.
/// The receiver reads the length prefix, reads that many bytes, and
/// deserializes the [`Frame`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Protocol version — MUST be [`V1_0`](super::V1_0).
    pub version: super::ProtocolVersion,
    /// UUID identifying the worker instance.
    pub worker_instance_id: String,
    /// Monotonically increasing per-sender sequence number.
    pub sequence_number: u64,
    /// Opaque request correlation id (present on events that belong to a
    /// specific generation request; absent on commands and global events).
    pub request_id: Option<String>,
    /// Discriminated message type.
    pub message_kind: MessageKind,
    /// Arbitrary JSON payload whose schema is determined by `message_kind`.
    pub payload: serde_json::Value,
}

impl Frame {
    /// Create a new host-command frame.
    ///
    /// `request_id` is set to `None` — host commands do not correlate to
    /// an in-flight generation request.
    pub fn new_host_command(
        worker_id: String,
        seq: u64,
        cmd: HostCommand,
        payload: serde_json::Value,
    ) -> Self {
        Frame {
            version: super::V1_0,
            worker_instance_id: worker_id,
            sequence_number: seq,
            request_id: None,
            message_kind: MessageKind::HostCommand(cmd),
            payload,
        }
    }

    /// Create a new host-command frame with a specific request correlation id.
    ///
    /// Used for request-scoped commands such as [`HostCommand::StartGeneration`]
    /// and [`HostCommand::CancelGeneration`]. Request-less commands (Hello, Ping,
    /// Shutdown, etc.) should use [`Frame::new_host_command`] instead.
    pub fn new_host_command_with_request(
        worker_id: &str,
        seq: u64,
        request_id: &str,
        cmd: HostCommand,
        payload: serde_json::Value,
    ) -> Self {
        Frame {
            version: super::V1_0,
            worker_instance_id: worker_id.to_string(),
            sequence_number: seq,
            request_id: Some(request_id.to_string()),
            message_kind: MessageKind::HostCommand(cmd),
            payload,
        }
    }

    /// Create a new worker-event frame.
    ///
    /// `request_id` identifies the generation request this event belongs to
    /// (e.g. `"generation-abc"`). Events such as [`WorkerEvent::HelloAck`]
    /// and [`WorkerEvent::Heartbeat`] that are not tied to a request may
    /// pass an empty string or a placeholder.
    pub fn new_worker_event(
        worker_id: String,
        seq: u64,
        request_id: String,
        event: WorkerEvent,
        payload: serde_json::Value,
    ) -> Self {
        Frame {
            version: super::V1_0,
            worker_instance_id: worker_id,
            sequence_number: seq,
            request_id: Some(request_id),
            message_kind: MessageKind::WorkerEvent(event),
            payload,
        }
    }
}

// ── Validation ─────────────────────────────────────────────────────────────

/// Errors that can arise when validating a [`Frame`].
///
/// Categorised per the constitutional pattern: `Rejected` for preflight
/// failures, `Stale` for sequencing / fencing mismatches. We use a flat
/// `FrameValidationError` enum rather than a `thiserror` enum-of-enums
/// because the variants are mutually exclusive at any single call site.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum FrameValidationError {
    /// Serialized frame exceeds [`MAX_FRAME_SIZE_BYTES`].
    /// (Effect failure — wire-level reject.)
    #[error("frame exceeds max size of {max} bytes", max = MAX_FRAME_SIZE_BYTES)]
    FrameTooLarge,
    /// Frame version is not [`V1_0`](super::V1_0).
    /// (Preflight — caller is speaking a different protocol.)
    #[error("unknown protocol version")]
    UnknownVersion,
    /// Sequence number regressed or skipped — expected a specific `n`.
    /// (Stale — fencing mismatch.)
    #[error("sequence regression: expected {expected}, got {actual}")]
    SequenceRegression { expected: u64, actual: u64 },
    /// A duplicate [`StartGeneration`](HostCommand::StartGeneration) request
    /// was detected (same `request_id` already in flight).
    /// (Rejected — preflight failure.)
    #[error("duplicate start for request {0}")]
    DuplicateRequestStart(String),
    /// Worker ID did not match the expected identifier.
    /// (Rejected — preflight failure.)
    #[error("unknown worker instance {0}")]
    UnknownWorker(String),
    /// A frame arrived after a terminal message ([`Shutdown`](HostCommand::Shutdown)
    /// or [`WorkerFatal`](WorkerEvent::WorkerFatal)).
    /// (Rejected — preflight failure.)
    #[error("frame arrived after terminal message for request {0}")]
    TerminalAfterClose(String),
    /// Message kind is not recognized or is not valid for the sender direction.
    /// (Rejected — preflight failure.)
    #[error("invalid message kind for sender direction")]
    InvalidMessageKind,
    /// JSON serialization round-trip failed. (Effect — could not encode
    /// the frame to check its size.)
    #[error("frame serialization failed: {0}")]
    SerializationFailed(String),
}

/// Validate a [`Frame`] against protocol invariants.
///
/// Checks:
///
/// 1. Serialized size ≤ [`MAX_FRAME_SIZE_BYTES`].
/// 2. `version` == [`V1_0`](super::V1_0).
/// 3. `sequence_number` == `expected_next_seq` (no regressions, no gaps).
/// 4. `worker_instance_id` matches `expected_worker_id` when provided.
/// 5. `message_kind` round-trips through serde (i.e. is a recognised kind).
///
/// # Direction-aware validation
///
/// This function does **not** track frame history (which terminal messages
/// have been seen, which `request_id`s are in flight). Callers SHOULD
/// maintain their own state machine and reject frames that arrive after a
/// terminal message (returning [`TerminalAfterClose`](FrameValidationError::TerminalAfterClose))
/// or that start a duplicate generation request (returning
/// [`DuplicateRequestStart`](FrameValidationError::DuplicateRequestStart)).
pub fn validate_frame(
    frame: &Frame,
    expected_next_seq: u64,
    expected_worker_id: Option<&str>,
) -> Result<(), FrameValidationError> {
    // 1. Size check — serialize to JSON and measure.
    let serialized = serde_json::to_vec(frame)
        .map_err(|e| FrameValidationError::SerializationFailed(e.to_string()))?;
    if serialized.len() > MAX_FRAME_SIZE_BYTES {
        return Err(FrameValidationError::FrameTooLarge);
    }

    // 2. Version must be V1_0.
    if frame.version != super::V1_0 {
        return Err(FrameValidationError::UnknownVersion);
    }

    // 3. Sequence must match expected (no regression, no gaps).
    if frame.sequence_number != expected_next_seq {
        return Err(FrameValidationError::SequenceRegression {
            expected: expected_next_seq,
            actual: frame.sequence_number,
        });
    }

    // 4. Worker ID must match when expected.
    if let Some(expected_id) = expected_worker_id {
        if frame.worker_instance_id != expected_id {
            return Err(FrameValidationError::UnknownWorker(
                frame.worker_instance_id.clone(),
            ));
        }
    }

    // 5. Message kind must be a recognized variant.
    //    With #[serde(untagged)], an unknown string would already cause
    //    deserialization to fail, but we verify round-trip explicitly.
    let kind_value = serde_json::to_value(&frame.message_kind)
        .map_err(|e| FrameValidationError::SerializationFailed(e.to_string()))?;
    if !kind_value.is_string() {
        return Err(FrameValidationError::InvalidMessageKind);
    }

    Ok(())
}

// ── Stateful Protocol Validator ────────────────────────────────────────────

/// Stateful validator that tracks protocol state across frames.
///
/// Maintains the expected worker ID, next expected sequence number, and
/// active/terminal request sets so that direction-aware validation (e.g.
/// rejecting duplicate start requests or events for unknown requests) can
/// be performed without external bookkeeping.
#[derive(Debug, Clone)]
pub struct ProtocolValidator {
    /// Worker instance ID the validator expects on every frame.
    pub expected_worker_id: String,
    /// Next sequence number the validator expects.
    pub next_expected_seq: u64,
    /// Request IDs that are currently in flight (started but not yet terminal).
    pub known_requests: Vec<String>,
    /// Request IDs that have received a terminal event (completed, cancelled,
    /// or failed).
    pub terminal_requests: Vec<String>,
}

impl ProtocolValidator {
    /// Create a new validator for the given `worker_id`.
    pub fn new(worker_id: String) -> Self {
        ProtocolValidator {
            expected_worker_id: worker_id,
            next_expected_seq: 0,
            known_requests: Vec::new(),
            terminal_requests: Vec::new(),
        }
    }

    /// Run stateless checks (version, seq, worker ID, message kind) shared by
    /// both host and worker frames.
    fn validate_baseline(&self, frame: &Frame) -> Result<(), FrameValidationError> {
        // 1. Version must be V1_0.
        if frame.version != super::V1_0 {
            return Err(FrameValidationError::UnknownVersion);
        }

        // 2. Sequence must match expected (no regression, no gaps).
        if frame.sequence_number != self.next_expected_seq {
            return Err(FrameValidationError::SequenceRegression {
                expected: self.next_expected_seq,
                actual: frame.sequence_number,
            });
        }

        // 3. Worker ID must match.
        if frame.worker_instance_id != self.expected_worker_id {
            return Err(FrameValidationError::UnknownWorker(
                frame.worker_instance_id.clone(),
            ));
        }

        // 4. Message kind must be a recognized variant.
        let kind_value = serde_json::to_value(&frame.message_kind)
            .map_err(|e| FrameValidationError::SerializationFailed(e.to_string()))?;
        if !kind_value.is_string() {
            return Err(FrameValidationError::InvalidMessageKind);
        }

        Ok(())
    }

    /// Validate a host-command frame and advance internal state.
    ///
    /// Checks:
    /// - Baseline fields (version, seq, worker ID, message kind).
    /// - `message_kind` is a [`HostCommand`].
    /// - For [`HostCommand::StartGeneration`]: rejects if `request_id` is
    ///   already in `known_requests` (duplicate).
    /// - For [`HostCommand::CancelGeneration`]: rejects if `request_id` is
    ///   not in `known_requests`.
    pub fn validate_host_command(&mut self, frame: &Frame) -> Result<(), FrameValidationError> {
        self.validate_baseline(frame)?;

        // Verify this is actually a HostCommand.
        let cmd = match &frame.message_kind {
            MessageKind::HostCommand(cmd) => cmd,
            _ => return Err(FrameValidationError::InvalidMessageKind),
        };

        // Request-scoped checks.
        if let Some(req_id) = &frame.request_id {
            match cmd {
                HostCommand::StartGeneration => {
                    if self.known_requests.contains(req_id) {
                        return Err(FrameValidationError::DuplicateRequestStart(
                            req_id.clone(),
                        ));
                    }
                }
                HostCommand::CancelGeneration => {
                    if !self.known_requests.contains(req_id) {
                        return Err(FrameValidationError::UnknownWorker(req_id.clone()));
                    }
                }
                _ => {}
            }
        }

        self.next_expected_seq += 1;
        Ok(())
    }

    /// Validate a worker-event frame and advance internal state.
    ///
    /// Checks:
    /// - Baseline fields (version, seq, worker ID, message kind).
    /// - `message_kind` is a [`WorkerEvent`].
    /// - On [`WorkerEvent::GenerationStarted`]: records the `request_id` into
    ///   `known_requests`.
    /// - On terminal events ([`GenerationCompleted`](WorkerEvent::GenerationCompleted),
    ///   [`GenerationCancelled`](WorkerEvent::GenerationCancelled),
    ///   [`GenerationFailed`](WorkerEvent::GenerationFailed)):
    ///   rejects if `request_id` is unknown or already terminal;
    ///   otherwise moves from `known_requests` to `terminal_requests`.
    pub fn validate_worker_event(&mut self, frame: &Frame) -> Result<(), FrameValidationError> {
        self.validate_baseline(frame)?;

        // Verify this is actually a WorkerEvent.
        let event = match &frame.message_kind {
            MessageKind::WorkerEvent(ev) => ev,
            _ => return Err(FrameValidationError::InvalidMessageKind),
        };

        if let Some(req_id) = &frame.request_id {
            match event {
                WorkerEvent::GenerationStarted => {
                    // Duplicate start of a known request is an error.
                    if self.known_requests.contains(req_id) {
                        return Err(FrameValidationError::DuplicateRequestStart(
                            req_id.clone(),
                        ));
                    }
                    self.known_requests.push(req_id.clone());
                }
                WorkerEvent::GenerationCompleted
                | WorkerEvent::GenerationCancelled
                | WorkerEvent::GenerationFailed => {
                    // Reject unknown or already-terminated requests.
                    if !self.known_requests.contains(req_id) {
                        return Err(FrameValidationError::UnknownWorker(req_id.clone()));
                    }
                    if self.terminal_requests.contains(req_id) {
                        return Err(FrameValidationError::TerminalAfterClose(req_id.clone()));
                    }
                    // Move from known to terminal.
                    self.known_requests.retain(|id| id != req_id);
                    self.terminal_requests.push(req_id.clone());
                }
                _ => {}
            }
        }

        self.next_expected_seq += 1;
        Ok(())
    }
}
