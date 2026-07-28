//! `pipeline::event_emitter` — live evidence of compiler pipeline execution.
//!
//! This file owns the canonical authority for the [`CompilerEvent`] enum
//! (one variant per stage boundary of the compile pipeline) and the
//! [`CompilerEventStream`] that stores them in order with a verifiable
//! chain digest. The stream digest is included in runtime receipts to
//! prove which compiler pipeline produced each artifact.
//!
//! # Chain invariants
//!
//! 1. Each event has a non-decreasing timestamp (monotonic wall clock).
//! 2. The chain always starts with `ParseStarted`.
//! 3. Variants must alternate `*Started / *Complete` in the correct
//!    stage order. No stage may be skipped.
//! 4. The stream digest (SHA-256 of the serialized event list) provides
//!    a content-addressed identity that runtime receipts reference.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single compiler pipeline event — produced at every stage boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilerEvent {
    // ── Parse ──────────────────────────────────────────────────────────────
    /// Source parsing has started.
    ParseStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    /// Source parsing has completed.
    ParseComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Content digest of the parsed source.
        source_digest: String,
    },
    // ── Canonicalize ───────────────────────────────────────────────────────
    /// Canonicalization has started.
    CanonicalizeStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    /// Canonicalization has completed.
    CanonicalizeComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    // ── Schedule ───────────────────────────────────────────────────────────
    /// Region scheduling has started.
    ScheduleStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Schedule identifier or label.
        schedule: String,
    },
    /// Region scheduling has completed.
    ScheduleComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    // ── Lower ──────────────────────────────────────────────────────────────
    /// Backend lowering has started.
    LowerStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Lowering target (e.g. "metal", "ane", "mlx").
        target: String,
    },
    /// Backend lowering has completed.
    LowerComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// MLIR digest of the lowered program.
        mlir_digest: String,
    },
    // ── Compile ────────────────────────────────────────────────────────────
    /// Per-region compilation has started.
    CompileStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Identifier of the implementation being compiled.
        implementation_id: String,
    },
    /// Per-region compilation has completed.
    CompileComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Content digest of the produced artifact.
        artifact_digest: String,
    },
    // ── Validate ───────────────────────────────────────────────────────────
    /// Validation has started.
    ValidateStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    /// Validation has completed.
    ValidateComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Whether validation passed.
        passed: bool,
    },
    // ── Package ────────────────────────────────────────────────────────────
    /// Artifact packaging has started.
    PackageStarted {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    /// Artifact packaging has completed.
    PackageComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Identifier of the generation that was packaged.
        generation_id: String,
    },
    // ── Lifecycle: Bind / Evaluation / Admission / Promotion / Cancel ──────
    /// All tensor payloads have been resolved and bound.
    BindComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Number of payloads that were resolved.
        num_payloads_resolved: u64,
    },
    /// Backend evaluation of the artifact has completed.
    EvaluationComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Whether backend evaluation passed.
        passed: bool,
    },
    /// Admission gate passed.
    AdmissionPassed {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
    /// Admission gate rejected the artifact.
    AdmissionRejected {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Reason for rejection.
        reason: String,
    },
    /// Artifact was promoted to the next lifecycle stage.
    PromotionComplete {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Identifier of the generation that was promoted.
        generation_id: String,
    },
    /// Promotion failed.
    PromotionFailed {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
        /// Reason for failure.
        reason: String,
    },
    // ── Lifecycle: Cancellation ────────────────────────────────────────────
    /// The compile was cancelled by the user or upper layer.
    Cancelled {
        /// Wall-clock timestamp in microseconds since UNIX epoch.
        timestamp: u64,
    },
}

// Manual PartialEq — skip `f64`-like concerns, just structural eq on strings
// and bools so tests can compare expected event lists.
impl PartialEq for CompilerEvent {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::ParseStarted { timestamp: a }, Self::ParseStarted { timestamp: b }) => a == b,
            (
                Self::ParseComplete {
                    timestamp: a,
                    source_digest: sa,
                },
                Self::ParseComplete {
                    timestamp: b,
                    source_digest: sb,
                },
            ) => a == b && sa == sb,
            (
                Self::CanonicalizeStarted { timestamp: a },
                Self::CanonicalizeStarted { timestamp: b },
            ) => a == b,
            (
                Self::CanonicalizeComplete { timestamp: a },
                Self::CanonicalizeComplete { timestamp: b },
            ) => a == b,
            (
                Self::ScheduleStarted {
                    timestamp: a,
                    schedule: sa,
                },
                Self::ScheduleStarted {
                    timestamp: b,
                    schedule: sb,
                },
            ) => a == b && sa == sb,
            (Self::ScheduleComplete { timestamp: a }, Self::ScheduleComplete { timestamp: b }) => {
                a == b
            }
            (
                Self::LowerStarted {
                    timestamp: a,
                    target: ta,
                },
                Self::LowerStarted {
                    timestamp: b,
                    target: tb,
                },
            ) => a == b && ta == tb,
            (
                Self::LowerComplete {
                    timestamp: a,
                    mlir_digest: da,
                },
                Self::LowerComplete {
                    timestamp: b,
                    mlir_digest: db,
                },
            ) => a == b && da == db,
            (
                Self::CompileStarted {
                    timestamp: a,
                    implementation_id: ia,
                },
                Self::CompileStarted {
                    timestamp: b,
                    implementation_id: ib,
                },
            ) => a == b && ia == ib,
            (
                Self::CompileComplete {
                    timestamp: a,
                    artifact_digest: da,
                },
                Self::CompileComplete {
                    timestamp: b,
                    artifact_digest: db,
                },
            ) => a == b && da == db,
            (Self::ValidateStarted { timestamp: a }, Self::ValidateStarted { timestamp: b }) => {
                a == b
            }
            (
                Self::ValidateComplete {
                    timestamp: a,
                    passed: pa,
                },
                Self::ValidateComplete {
                    timestamp: b,
                    passed: pb,
                },
            ) => a == b && pa == pb,
            (Self::PackageStarted { timestamp: a }, Self::PackageStarted { timestamp: b }) => {
                a == b
            }
            (
                Self::PackageComplete {
                    timestamp: a,
                    generation_id: ga,
                },
                Self::PackageComplete {
                    timestamp: b,
                    generation_id: gb,
                },
            ) => a == b && ga == gb,
            (
                Self::BindComplete {
                    timestamp: a,
                    num_payloads_resolved: na,
                },
                Self::BindComplete {
                    timestamp: b,
                    num_payloads_resolved: nb,
                },
            ) => a == b && na == nb,
            (
                Self::EvaluationComplete {
                    timestamp: a,
                    passed: pa,
                },
                Self::EvaluationComplete {
                    timestamp: b,
                    passed: pb,
                },
            ) => a == b && pa == pb,
            (Self::AdmissionPassed { timestamp: a }, Self::AdmissionPassed { timestamp: b }) => {
                a == b
            }
            (
                Self::AdmissionRejected {
                    timestamp: a,
                    reason: ra,
                },
                Self::AdmissionRejected {
                    timestamp: b,
                    reason: rb,
                },
            ) => a == b && ra == rb,
            (
                Self::PromotionComplete {
                    timestamp: a,
                    generation_id: ga,
                },
                Self::PromotionComplete {
                    timestamp: b,
                    generation_id: gb,
                },
            ) => a == b && ga == gb,
            (
                Self::PromotionFailed {
                    timestamp: a,
                    reason: ra,
                },
                Self::PromotionFailed {
                    timestamp: b,
                    reason: rb,
                },
            ) => a == b && ra == rb,
            (Self::Cancelled { timestamp: a }, Self::Cancelled { timestamp: b }) => a == b,
            _ => false,
        }
    }
}

/// Ordered stream of compiler events with a session identity.
///
/// Events are appended in the order they occur. The
/// [`digest`](Self::digest) method produces a SHA-256 hash of the full
/// event list, suitable for inclusion in runtime receipts to prove
/// which compiler pipeline produced the artifact.
#[derive(Debug, Clone, Serialize)]
pub struct CompilerEventStream {
    /// Events in insertion order.
    events: Vec<CompilerEvent>,
    /// Session identifier for this compile.
    session_id: String,
}

impl CompilerEventStream {
    /// Create a new empty stream for the given compiler session.
    pub fn new(session_id: &str) -> Self {
        Self {
            events: Vec::new(),
            session_id: session_id.into(),
        }
    }

    /// Append an event to the stream.
    ///
    /// Events are stored in insertion order for chain verification.
    pub fn emit(&mut self, event: CompilerEvent) {
        self.events.push(event);
    }

    /// Borrow the event list.
    pub fn events(&self) -> &[CompilerEvent] {
        &self.events
    }

    /// Drain all events from the stream.
    pub fn drain(&mut self) -> Vec<CompilerEvent> {
        std::mem::take(&mut self.events)
    }

    /// Session identifier for this compile session.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Compute a SHA-256 digest over the whole event list.
    ///
    /// Uses `bincode` to produce a canonical encoding of the event
    /// sequence so the digest is reproducible across identical
    /// compiler pipelines. Falls back to debug-format concatenation
    /// when bincode fails (should never happen for structurally valid
    /// events).
    pub fn digest(&self) -> String {
        let mut hasher = Sha256::new();
        // Include session_id to bind the digest to a specific compile.
        hasher.update(b"session:");
        hasher.update(self.session_id.as_bytes());
        hasher.update(b"\n");
        for event in &self.events {
            // bincode for canonical serialization
            let encoded = bincode::serialize(event).unwrap_or_else(|_| {
                // Fallback: debug format (deterministic for these variants)
                format!("{:?}\n", event).into_bytes()
            });
            hasher.update(&encoded);
        }
        format!("{:x}", hasher.finalize())
    }
}

impl Default for CompilerEventStream {
    fn default() -> Self {
        Self::new("default-session")
    }
}

// ── Chain verification ─────────────────────────────────────────────────────

/// Outcome of chain verification.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainVerificationResult {
    /// The event chain is valid — all invariants hold.
    Valid,
    /// The chain is empty.
    Empty,
    /// The first event is not `ParseStarted`.
    DoesNotStartWithParse,
    /// A `*Started` event has no matching `*Complete` (or vice versa).
    Unpaired {
        /// Index of the offending event.
        index: usize,
        /// Event variant name.
        kind: &'static str,
        /// Human-readable detail of the violation.
        detail: String,
    },
    /// Two events have non-contiguous timestamps (later event is
    /// before an earlier one).
    NonContiguousTimestamp {
        /// Index of the earlier event.
        earlier_index: usize,
        /// Index of the later event.
        later_index: usize,
        /// Earlier event timestamp.
        earlier_ts: u64,
        /// Later event timestamp.
        later_ts: u64,
    },
    /// A stage appears out of the required order (e.g. `CompileStarted`
    /// before `ScheduleComplete`).
    OutOfOrder {
        /// Index of the offending event.
        index: usize,
        /// Expected variant name at this position.
        expected: &'static str,
        /// Actual variant name.
        actual: &'static str,
    },
    /// Duplicate event variant at this position.
    Duplicate {
        /// Index of the duplicate event.
        index: usize,
        /// Variant name.
        kind: &'static str,
    },
}

/// The canonical stage ordering required by the compiler pipeline.
const STAGE_ORDER: &[&str] = &[
    "ParseStarted",
    "ParseComplete",
    "CanonicalizeStarted",
    "CanonicalizeComplete",
    "ScheduleStarted",
    "ScheduleComplete",
    "LowerStarted",
    "LowerComplete",
    "CompileStarted",
    "CompileComplete",
    "ValidateStarted",
    "ValidateComplete",
    "PackageStarted",
    "PackageComplete",
    "BindComplete",
    "EvaluationComplete",
    "AdmissionPassed",
    "AdmissionRejected",
    "PromotionComplete",
    "PromotionFailed",
    "Cancelled",
];

fn event_kind(event: &CompilerEvent) -> &'static str {
    match event {
        CompilerEvent::ParseStarted { .. } => "ParseStarted",
        CompilerEvent::ParseComplete { .. } => "ParseComplete",
        CompilerEvent::CanonicalizeStarted { .. } => "CanonicalizeStarted",
        CompilerEvent::CanonicalizeComplete { .. } => "CanonicalizeComplete",
        CompilerEvent::ScheduleStarted { .. } => "ScheduleStarted",
        CompilerEvent::ScheduleComplete { .. } => "ScheduleComplete",
        CompilerEvent::LowerStarted { .. } => "LowerStarted",
        CompilerEvent::LowerComplete { .. } => "LowerComplete",
        CompilerEvent::CompileStarted { .. } => "CompileStarted",
        CompilerEvent::CompileComplete { .. } => "CompileComplete",
        CompilerEvent::ValidateStarted { .. } => "ValidateStarted",
        CompilerEvent::ValidateComplete { .. } => "ValidateComplete",
        CompilerEvent::PackageStarted { .. } => "PackageStarted",
        CompilerEvent::PackageComplete { .. } => "PackageComplete",
        CompilerEvent::BindComplete { .. } => "BindComplete",
        CompilerEvent::EvaluationComplete { .. } => "EvaluationComplete",
        CompilerEvent::AdmissionPassed { .. } => "AdmissionPassed",
        CompilerEvent::AdmissionRejected { .. } => "AdmissionRejected",
        CompilerEvent::PromotionComplete { .. } => "PromotionComplete",
        CompilerEvent::PromotionFailed { .. } => "PromotionFailed",
        CompilerEvent::Cancelled { .. } => "Cancelled",
    }
}

fn event_timestamp(event: &CompilerEvent) -> u64 {
    match event {
        CompilerEvent::ParseStarted { timestamp }
        | CompilerEvent::ParseComplete { timestamp, .. }
        | CompilerEvent::CanonicalizeStarted { timestamp }
        | CompilerEvent::CanonicalizeComplete { timestamp }
        | CompilerEvent::ScheduleStarted { timestamp, .. }
        | CompilerEvent::ScheduleComplete { timestamp }
        | CompilerEvent::LowerStarted { timestamp, .. }
        | CompilerEvent::LowerComplete { timestamp, .. }
        | CompilerEvent::CompileStarted { timestamp, .. }
        | CompilerEvent::CompileComplete { timestamp, .. }
        | CompilerEvent::ValidateStarted { timestamp }
        | CompilerEvent::ValidateComplete { timestamp, .. }
        | CompilerEvent::PackageStarted { timestamp }
        | CompilerEvent::PackageComplete { timestamp, .. } => *timestamp,
        CompilerEvent::BindComplete { timestamp, .. }
        | CompilerEvent::EvaluationComplete { timestamp, .. }
        | CompilerEvent::AdmissionPassed { timestamp }
        | CompilerEvent::AdmissionRejected { timestamp, .. }
        | CompilerEvent::PromotionComplete { timestamp, .. }
        | CompilerEvent::PromotionFailed { timestamp, .. }
        | CompilerEvent::Cancelled { timestamp } => *timestamp,
    }
}

/// Verify that a sequence of compiler events forms a valid chain.
///
/// Checks:
/// 1. Chain is non-empty.
/// 2. First event is `ParseStarted`.
/// 3. Timestamps are non-decreasing.
/// 4. Events appear in the required stage order (no missing/reordered
///    stages).
/// 5. No duplicate events within the sequence.
pub fn verify_event_chain(events: &[CompilerEvent]) -> ChainVerificationResult {
    if events.is_empty() {
        return ChainVerificationResult::Empty;
    }

    // 1. Must start with ParseStarted
    if !matches!(events[0], CompilerEvent::ParseStarted { .. }) {
        return ChainVerificationResult::DoesNotStartWithParse;
    }

    let mut last_kind_idx: Option<usize> = None;

    for (i, event) in events.iter().enumerate() {
        let kind = event_kind(event);
        let ts = event_timestamp(event);

        // 2. Timestamps must be non-decreasing
        if i > 0 {
            let prev_ts = event_timestamp(&events[i - 1]);
            if ts < prev_ts {
                return ChainVerificationResult::NonContiguousTimestamp {
                    earlier_index: i - 1,
                    later_index: i,
                    earlier_ts: prev_ts,
                    later_ts: ts,
                };
            }
        }

        // 3. Stage order: find this kind in STAGE_ORDER
        let pos = STAGE_ORDER
            .iter()
            .position(|&s| s == kind)
            .expect("event kind not in STAGE_ORDER — this is a bug in event_kind");

        // 4. Check that we haven't skipped a stage
        if let Some(last_pos) = last_kind_idx {
            if pos < last_pos {
                return ChainVerificationResult::OutOfOrder {
                    index: i,
                    expected: STAGE_ORDER[last_pos],
                    actual: kind,
                };
            }
            if pos == last_pos {
                // Duplicate: same variant twice
                return ChainVerificationResult::Duplicate { index: i, kind };
            }
            // Skip check for skipped stages — if pos > last_pos + 1 that
            // means a stage was skipped ("ParseComplete" followed by
            // "LowerStarted" without "Canonicalize*" and "Schedule*").
            // This is a missing stage.
            if pos > last_pos + 1 {
                // At least one stage between last_pos and pos is missing
                let expected_kind = STAGE_ORDER[last_pos + 1];
                return ChainVerificationResult::OutOfOrder {
                    index: i,
                    expected: expected_kind,
                    actual: kind,
                };
            }
        }

        last_kind_idx = Some(pos);
    }

    // 5. Verify we end with PackageComplete
    let last = &events[events.len() - 1];
    if !matches!(
        last,
        CompilerEvent::PackageComplete { .. }
            | CompilerEvent::PromotionComplete { .. }
            | CompilerEvent::PromotionFailed { .. }
            | CompilerEvent::AdmissionRejected { .. }
            | CompilerEvent::Cancelled { .. }
    ) {
        return ChainVerificationResult::OutOfOrder {
            index: events.len() - 1,
            expected: "PackageComplete",
            actual: event_kind(last),
        };
    }

    ChainVerificationResult::Valid
}

// ── Helpers for producing events from the compile session ──────────────────

/// Get a monotonic timestamp in microseconds since UNIX epoch.
pub fn now_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a stream with a complete valid sequence of events.
    fn valid_stream() -> CompilerEventStream {
        let mut stream = CompilerEventStream::new("test-session");
        let mut ts = 100u64;
        let mut t = || {
            let v = ts;
            ts += 100;
            v
        };
        stream.emit(CompilerEvent::ParseStarted { timestamp: t() });
        stream.emit(CompilerEvent::ParseComplete {
            timestamp: t(),
            source_digest: "abc".into(),
        });
        stream.emit(CompilerEvent::CanonicalizeStarted { timestamp: t() });
        stream.emit(CompilerEvent::CanonicalizeComplete { timestamp: t() });
        stream.emit(CompilerEvent::ScheduleStarted {
            timestamp: t(),
            schedule: "default".into(),
        });
        stream.emit(CompilerEvent::ScheduleComplete { timestamp: t() });
        stream.emit(CompilerEvent::LowerStarted {
            timestamp: t(),
            target: "metal".into(),
        });
        stream.emit(CompilerEvent::LowerComplete {
            timestamp: t(),
            mlir_digest: "def".into(),
        });
        stream.emit(CompilerEvent::CompileStarted {
            timestamp: t(),
            implementation_id: "impl-1".into(),
        });
        stream.emit(CompilerEvent::CompileComplete {
            timestamp: t(),
            artifact_digest: "artifact-xyz".into(),
        });
        stream.emit(CompilerEvent::ValidateStarted { timestamp: t() });
        stream.emit(CompilerEvent::ValidateComplete {
            timestamp: t(),
            passed: true,
        });
        stream.emit(CompilerEvent::PackageStarted { timestamp: t() });
        stream.emit(CompilerEvent::PackageComplete {
            timestamp: t(),
            generation_id: "gen-42".into(),
        });
        stream
    }

    #[test]
    fn valid_chain_verifies() {
        let stream = valid_stream();
        assert_eq!(
            verify_event_chain(stream.events()),
            ChainVerificationResult::Valid
        );
    }

    #[test]
    fn empty_chain() {
        let stream = CompilerEventStream::new("empty");
        assert_eq!(
            verify_event_chain(stream.events()),
            ChainVerificationResult::Empty
        );
    }

    #[test]
    fn does_not_start_with_parse() {
        let mut stream = CompilerEventStream::new("no-parse");
        stream.emit(CompilerEvent::CanonicalizeStarted {
            timestamp: now_micros(),
        });
        assert_eq!(
            verify_event_chain(stream.events()),
            ChainVerificationResult::DoesNotStartWithParse
        );
    }

    #[test]
    fn non_contiguous_timestamp() {
        let mut stream = CompilerEventStream::new("bad-ts");
        stream.emit(CompilerEvent::ParseStarted { timestamp: 200 });
        stream.emit(CompilerEvent::ParseComplete {
            timestamp: 100, // earlier timestamp!
            source_digest: "x".into(),
        });
        let result = verify_event_chain(stream.events());
        assert!(
            matches!(
                result,
                ChainVerificationResult::NonContiguousTimestamp { .. }
            ),
            "expected NonContiguousTimestamp, got {result:?}"
        );
    }

    #[test]
    fn out_of_order_skipped_stage() {
        let mut stream = CompilerEventStream::new("ooo");
        stream.emit(CompilerEvent::ParseStarted {
            timestamp: now_micros(),
        });
        stream.emit(CompilerEvent::ParseComplete {
            timestamp: now_micros(),
            source_digest: "x".into(),
        });
        // Skip Canonicalize*, jump to ScheduleStarted
        stream.emit(CompilerEvent::ScheduleStarted {
            timestamp: now_micros(),
            schedule: "x".into(),
        });
        let result = verify_event_chain(stream.events());
        assert!(
            matches!(result, ChainVerificationResult::OutOfOrder { .. }),
            "expected OutOfOrder for skipped stage, got {result:?}"
        );
    }

    #[test]
    fn duplicate_event() {
        let mut stream = CompilerEventStream::new("dup");
        stream.emit(CompilerEvent::ParseStarted {
            timestamp: now_micros(),
        });
        stream.emit(CompilerEvent::ParseStarted {
            // duplicate
            timestamp: now_micros(),
        });
        let result = verify_event_chain(stream.events());
        assert!(
            matches!(result, ChainVerificationResult::Duplicate { .. }),
            "expected Duplicate, got {result:?}"
        );
    }

    #[test]
    fn digest_is_deterministic() {
        let s1 = valid_stream();
        let s2 = valid_stream();
        assert_eq!(s1.digest(), s2.digest());
    }

    #[test]
    fn digest_differs_with_different_events() {
        let mut s1 = CompilerEventStream::new("s1");
        let mut s2 = CompilerEventStream::new("s2");
        let t = || now_micros();
        s1.emit(CompilerEvent::ParseStarted { timestamp: t() });
        s2.emit(CompilerEvent::ParseStarted { timestamp: t() });
        // s2 has an extra event
        s2.emit(CompilerEvent::ParseComplete {
            timestamp: t(),
            source_digest: "x".into(),
        });
        assert_ne!(s1.digest(), s2.digest());
    }

    #[test]
    fn partial_eq_structural() {
        let a = CompilerEvent::ParseStarted { timestamp: 42 };
        let b = CompilerEvent::ParseStarted { timestamp: 42 };
        let c = CompilerEvent::ParseStarted { timestamp: 99 };
        assert_eq!(a, b);
        assert_ne!(a, c);

        let a = CompilerEvent::PackageComplete {
            timestamp: 1,
            generation_id: "g1".into(),
        };
        let b = CompilerEvent::PackageComplete {
            timestamp: 1,
            generation_id: "g2".into(),
        };
        assert_ne!(a, b);
    }

    #[test]
    fn missing_final_package_complete_detected() {
        let mut stream = CompilerEventStream::new("unfinished");
        let t = || now_micros();
        stream.emit(CompilerEvent::ParseStarted { timestamp: t() });
        stream.emit(CompilerEvent::ParseComplete {
            timestamp: t(),
            source_digest: "x".into(),
        });
        stream.emit(CompilerEvent::CanonicalizeStarted { timestamp: t() });
        stream.emit(CompilerEvent::CanonicalizeComplete { timestamp: t() });
        stream.emit(CompilerEvent::ScheduleStarted {
            timestamp: t(),
            schedule: "x".into(),
        });
        stream.emit(CompilerEvent::ScheduleComplete { timestamp: t() });
        // Ends at ScheduleComplete — should be Unpaired
        let result = verify_event_chain(stream.events());
        assert!(
            matches!(result, ChainVerificationResult::OutOfOrder { .. }),
            "expected OutOfOrder for incomplete chain, got {result:?}"
        );
    }

    #[test]
    fn events_are_serializable() {
        let e = CompilerEvent::CompileComplete {
            timestamp: 12345,
            artifact_digest: "deadbeef".into(),
        };
        let encoded = bincode::serialize(&e).expect("bincode serialization");
        let decoded: CompilerEvent =
            bincode::deserialize(&encoded).expect("bincode deserialization");
        assert_eq!(e, decoded);
    }
}
