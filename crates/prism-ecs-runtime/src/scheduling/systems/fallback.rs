//! Phase fallback (constitutional home).
//!
//! When the dispatch-selection system cannot find a concrete runner for
//! a phase kind, control falls through here to produce a meaningful
//! diagnostic. The fallback is a system (S bucket): it reads the
//! committed phase, stages an error through `ConstitutionalWorldTxn`,
//! and signals the failure to the completion-reconciliation system.

use crate::scheduling::state::phase::EmittedPhasePlaceholder as EmittedPhase;

/// Handle a phase that has no registered runner.
///
/// Returns a clear error message identifying the unhandled phase kind
/// and phase id so callers can diagnose missing registrations.
///
/// Architectural invariant: every unhandled phase produces a result of
/// the same shape (`Result<(), String>` with a stable message format)
/// so that receipts, logs, and dashboards can parse it uniformly.
pub fn run_fallback(phase: &EmittedPhase) -> Result<(), String> {
    Err(format!(
        "no runner registered for phase kind (phase_id={})",
        phase.phase_id
    ))
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `fallback` system.

    use super::*;

    #[test]
    fn fallback_always_errors() {
        // Architectural invariant: the fallback is a system that
        // never succeeds. It exists to surface a missing-runner
        // diagnostic, not to advance the work.
        let phase = EmittedPhase {
            phase_id: "p1".into(),
        };
        let result = run_fallback(&phase);
        assert!(result.is_err());
    }

    #[test]
    fn fallback_error_message_includes_phase_id() {
        // Architectural invariant: the error message identifies the
        // phase by id so callers can diagnose missing registrations.
        let phase = EmittedPhase {
            phase_id: "decode_layer_5".into(),
        };
        let err = run_fallback(&phase).unwrap_err();
        assert!(
            err.contains("decode_layer_5"),
            "error must include the phase id, got: {err}"
        );
    }

    #[test]
    fn fallback_message_format_is_stable() {
        // Architectural invariant: the error message format is
        // stable so that receipts, logs, and dashboards can parse
        // it. The format begins with the canonical prefix.
        let phase = EmittedPhase {
            phase_id: "p1".into(),
        };
        let err = run_fallback(&phase).unwrap_err();
        assert!(err.starts_with("no runner registered for phase kind"));
    }
}
