//! Fallback handling for phases with no registered runner.
//!
//! When [`PhaseRunnerRegistry::dispatch`] cannot find a concrete runner for a
//! [`PhaseKind`], control falls through here to produce a meaningful diagnostic.

use crate::compute_image::phase_dag::EmittedPhase;

/// Handle a phase that has no registered runner.
///
/// Returns a clear error message identifying the unhandled phase kind and
/// phase id so callers can diagnose missing registrations.
pub fn run_fallback(phase: &EmittedPhase) -> Result<(), String> {
    Err(format!(
        "no runner registered for phase kind {:?} (phase_id={})",
        phase.kind, phase.phase_id
    ))
}
