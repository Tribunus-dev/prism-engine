//! Dispatch selection system (constitutional home).
//!
//! Placeholder for the engine's `phase_runner/dispatch.rs` (65 LOC).
//! The full algorithm migrates in step 17. The engine file is the
//! legacy duplicate and is deleted in step 58.
//!
//! The dispatch-selection system chooses which dispatch path a
//! phase takes — kernel, fallback, or another route.

use crate::scheduling::state::phase::PhaseId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchRoute {
    /// Normal lane dispatch.
    Lane,
    /// Fallback path (slower alternative).
    Fallback,
    /// Cached / pre-computed.
    Cached,
}

/// Select a dispatch route for a phase. Placeholder: always returns
/// `Lane`. The full algorithm (readiness checks, route-origin
/// hints, fallback logic) arrives when phase_runner/dispatch.rs
/// migrates.
pub fn select_dispatch_route(_phase_id: &PhaseId) -> DispatchRoute {
    DispatchRoute::Lane
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_dispatch_route_placeholder_returns_lane() {
        // Architectural invariant: the placeholder returns Lane.
        // The full algorithm arrives with the engine migration.
        let pid = PhaseId("p1".into());
        assert_eq!(select_dispatch_route(&pid), DispatchRoute::Lane);
    }

    #[test]
    fn dispatch_route_variants_are_distinct() {
        let routes = [
            DispatchRoute::Lane,
            DispatchRoute::Fallback,
            DispatchRoute::Cached,
        ];
        for r in routes {
            let count = routes.iter().filter(|&&v| v == r).count();
            assert_eq!(count, 1, "every variant must be self-equal exactly once");
        }
    }
}
