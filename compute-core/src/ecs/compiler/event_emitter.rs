//! Compiler event stream — live evidence of compiler pipeline execution.
//!
//! [`CompilerEvent`] captures each stage boundary of the compiler pipeline
//! (parse, canonicalize, schedule, lower, compile, validate, package).
//! Events are stored in order in a [`CompilerEventStream`] and can be
//! verified as a contiguous chain via [`verify_event_chain`].
//!
//! # Chain invariants
//!
//! 1. Each event has a non-decreasing timestamp (monotonic wall clock).
//! 2. The chain always starts with `ParseStarted`.
//! 3. Variants must alternate *Started / *Complete in the correct stage
//!    order. No stage may be skipped.
//! 4. The stream digest (SHA-256 of the serialized event list) provides
//!    a content-addressed identity that runtime receipts reference.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single compiler pipeline event — produced at every stage boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompilerEvent {
    // ── Parse ──────────────────────────────────────────────────────────────
    ParseStarted {
        timestamp: u64,
    },
    ParseComplete {
        timestamp: u64,
        source_digest: String,
    },
    // ── Canonicalize ───────────────────────────────────────────────────────
    CanonicalizeStarted {
        timestamp: u64,
    },
    CanonicalizeComplete {
        timestamp: u64,
    },
    // ── Schedule ───────────────────────────────────────────────────────────
    ScheduleStarted {
        timestamp: u64,
        schedule: String,
    },
    ScheduleComplete {
        timestamp: u64,
    },
    // ── Lower ──────────────────────────────────────────────────────────────
    LowerStarted {
        timestamp: u64,
        target: String,
    },
    LowerComplete {
        timestamp: u64,
        mlir_digest: String,
    },
    // ── Compile ────────────────────────────────────────────────────────────
    CompileStarted {
        timestamp: u64,
        implementation_id: String,
    },
    CompileComplete {
        timestamp: u64,
        artifact_digest: String,
    },
    // ── Validate ───────────────────────────────────────────────────────────
    ValidateStarted {
        timestamp: u64,
    },
    ValidateComplete {
        timestamp: u64,
        passed: bool,
    },
    // ── Package ────────────────────────────────────────────────────────────
    PackageStarted {
        timestamp: u64,
    },
    PackageComplete {
        timestamp: u64,
        generation_id: String,
    },
    // ── Lifecycle: Bind ────────────────────────────────────────────────────
    BindComplete {
        timestamp: u64,
        num_payloads_resolved: usize,
    },
    // ── Lifecycle: Evaluate ───────────────────────────────────────────────
    EvaluationComplete {
        timestamp: u64,
        passed: bool,
    },
    // ── Lifecycle: Admission ──────────────────────────────────────────────
    AdmissionPassed {
        timestamp: u64,
    },
    AdmissionRejected {
        timestamp: u64,
        reason: String,
    },
    // ── Lifecycle: Promotion ─────────────────────────────────────────────
    PromotionComplete {
        timestamp: u64,
        generation_id: String,
    },
    PromotionFailed {
        timestamp: u64,
        reason: String,
    },
    // ── Lifecycle: Cancellation ──────────────────────────────────────────
    Cancelled {
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
/// Events are appended in the order they occur.  The [`digest`](Self::digest)
/// method produces a SHA-256 hash of the full event list, suitable for
/// inclusion in runtime receipts to prove which compiler pipeline produced
/// the artifact.
#[derive(Debug, Clone, Serialize)]
pub struct CompilerEventStream {
    events: Vec<CompilerEvent>,
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
    /// Uses `bincode` to produce a canonical encoding of the event sequence
    /// so the digest is reproducible across identical compiler pipelines.
    /// Falls back to debug-format concatenation when bincode fails (should
    /// never happen for structurally valid events).
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
        index: usize,
        kind: &'static str,
        detail: String,
    },
    /// Two events have non-contiguous timestamps (later event is before
    /// an earlier one).
    NonContiguousTimestamp {
        earlier_index: usize,
        later_index: usize,
        earlier_ts: u64,
        later_ts: u64,
    },
    /// A stage appears out of the required order (e.g. `CompileStarted`
    /// before `ScheduleComplete`).
    OutOfOrder {
        index: usize,
        expected: &'static str,
        actual: &'static str,
    },
    /// Duplicate event variant at this position.
    Duplicate { index: usize, kind: &'static str },
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
/// 4. Events appear in the required stage order (no missing/reordered stages).
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
        //    Every event must be either the first, the next in order,
        //    or (if it's a duplicate we reject below) the same position.
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
    fn test_valid_chain_verifies() {
        let stream = valid_stream();
        assert_eq!(
            verify_event_chain(stream.events()),
            ChainVerificationResult::Valid
        );
    }

    #[test]
    fn test_empty_chain() {
        let stream = CompilerEventStream::new("empty");
        assert_eq!(
            verify_event_chain(stream.events()),
            ChainVerificationResult::Empty
        );
    }

    #[test]
    fn test_does_not_start_with_parse() {
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
    fn test_non_contiguous_timestamp() {
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
    fn test_out_of_order() {
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
    fn test_duplicate_event() {
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
    fn test_digest_is_deterministic() {
        let s1 = valid_stream();
        let s2 = valid_stream();
        assert_eq!(s1.digest(), s2.digest());
    }

    #[test]
    fn test_digest_differs_with_different_events() {
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
    fn test_partial_eq_structural() {
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
    fn test_missing_final_package_complete_detected() {
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
    fn test_events_are_serializable() {
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
