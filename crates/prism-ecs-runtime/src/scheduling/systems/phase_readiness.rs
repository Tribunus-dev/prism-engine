//! Phase readiness system (constitutional home).
//!
//! Computes the ready set of phases given completed phases. A phase is
//! ready when all its predecessor edges have been satisfied (the
//! source phases are in the completed set).
//!
//! # Authority
//!
//! The `ReadinessChecker` is a system (S bucket). It reads the
//! committed phase DAG and the completed-phase set, and produces a
//! ready-phase list. The dispatch-selection system consumes the
//! ready-phase list; the result is non-authoritative until the
//! dispatch commits.

use std::collections::HashSet;

use crate::scheduling::state::phase::PhaseId;

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::EmittedPhase`
/// (V1 graph). Replaced when phase_dag migrates.
#[derive(Debug, Clone, Default)]
pub struct EmittedPhase {
    pub phase_id: String,
}

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::EmittedPhaseGraph` (V1).
#[derive(Debug, Clone, Default)]
pub struct EmittedPhaseGraph {
    pub phases: Vec<EmittedPhase>,
    pub edges: Vec<EmittedPhaseEdge>,
}

#[derive(Debug, Clone, Default)]
pub struct EmittedPhaseEdge {
    pub from_phase: String,
    pub to_phase: String,
}

/// Placeholder for `compute-core::ecs::compute_image::phase_graph::EmittedPhaseGraphV2`.
#[derive(Debug, Clone, Default)]
pub struct EmittedPhaseGraphV2 {
    pub phases: Vec<EmittedPhaseV2>,
    pub edges: Vec<EmittedPhaseEdgeV2>,
}

#[derive(Debug, Clone, Default)]
pub struct EmittedPhaseV2 {
    pub id: PhaseId,
}

#[derive(Debug, Clone, Default)]
pub struct EmittedPhaseEdgeV2 {
    pub from_phase: PhaseId,
    pub to_phase: PhaseId,
}

// ---------------------------------------------------------------------------
// ReadinessChecker
// ---------------------------------------------------------------------------

/// Computes the ready set of phases given completed phases.
pub struct ReadinessChecker;

impl ReadinessChecker {
    pub fn new() -> Self {
        Self
    }

    /// Compute ready phases for a V1 graph.
    pub fn ready_phases(
        &self,
        graph: &EmittedPhaseGraph,
        completed: &HashSet<String>,
    ) -> Vec<String> {
        let mut ready = Vec::new();
        for phase in &graph.phases {
            if completed.contains(&phase.phase_id) {
                continue;
            }
            let blocked = graph
                .edges
                .iter()
                .any(|e| e.to_phase == phase.phase_id && !completed.contains(&e.from_phase));
            if !blocked {
                ready.push(phase.phase_id.clone());
            }
        }
        ready
    }

    /// Compute ready phases for a V2 graph.
    pub fn ready_phases_v2(
        &self,
        graph: &EmittedPhaseGraphV2,
        completed: &HashSet<PhaseId>,
    ) -> Vec<PhaseId> {
        let mut ready = Vec::new();
        for phase in &graph.phases {
            if completed.contains(&phase.id) {
                continue;
            }
            let blocked = graph
                .edges
                .iter()
                .any(|e| e.to_phase == phase.id && !completed.contains(&e.from_phase));
            if !blocked {
                ready.push(phase.id.clone());
            }
        }
        ready
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `phase_readiness` system.

    use super::*;

    fn make_v1_dag(phases: &[&str], edges: &[(&str, &str)]) -> EmittedPhaseGraph {
        EmittedPhaseGraph {
            phases: phases
                .iter()
                .map(|id| EmittedPhase {
                    phase_id: (*id).into(),
                })
                .collect(),
            edges: edges
                .iter()
                .map(|(from, to)| EmittedPhaseEdge {
                    from_phase: (*from).into(),
                    to_phase: (*to).into(),
                })
                .collect(),
        }
    }

    #[test]
    fn v1_ready_phases_with_no_deps() {
        // Architectural invariant: every phase with no predecessors
        // and not yet completed is ready.
        let dag = make_v1_dag(&["a", "b"], &[]);
        let r = ReadinessChecker::new();
        let completed = HashSet::new();
        let ready = r.ready_phases(&dag, &completed);
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn v1_ready_phases_respects_dependencies() {
        // Architectural invariant: a phase is ready iff all its
        // predecessors are completed (and the phase itself is not
        // completed).
        let dag = make_v1_dag(&["a", "b"], &[("a", "b")]);
        let r = ReadinessChecker::new();
        let mut completed = HashSet::new();
        let ready = r.ready_phases(&dag, &completed);
        assert_eq!(ready, vec!["a"]);
        completed.insert("a".into());
        let ready = r.ready_phases(&dag, &completed);
        assert_eq!(ready, vec!["b"]);
    }

    #[test]
    fn v1_ready_phases_excludes_completed() {
        // Architectural invariant: a phase in the completed set
        // is never returned.
        let dag = make_v1_dag(&["a", "b"], &[]);
        let r = ReadinessChecker::new();
        let mut completed = HashSet::new();
        completed.insert("a".into());
        let ready = r.ready_phases(&dag, &completed);
        assert_eq!(ready, vec!["b"]);
    }

    #[test]
    fn v1_blocked_phase_is_excluded() {
        // Architectural invariant: a phase whose predecessor is
        // missing is excluded even if other phases are ready.
        let dag = make_v1_dag(&["a", "b", "c"], &[("a", "b"), ("a", "c")]);
        let r = ReadinessChecker::new();
        let completed = HashSet::new();
        let ready = r.ready_phases(&dag, &completed);
        assert_eq!(ready, vec!["a"]);
    }

    #[test]
    fn v2_ready_phases_round_trip() {
        // The V2 path uses typed PhaseId; the constitutional PhaseId
        // is `PhaseId(String)`. The V2 path mirrors the V1 logic
        // but on the typed set.
        let p1 = PhaseId("a".into());
        let p2 = PhaseId("b".into());
        let dag = EmittedPhaseGraphV2 {
            phases: vec![
                EmittedPhaseV2 { id: p1.clone() },
                EmittedPhaseV2 { id: p2.clone() },
            ],
            edges: vec![EmittedPhaseEdgeV2 {
                from_phase: p1.clone(),
                to_phase: p2.clone(),
            }],
        };
        let r = ReadinessChecker::new();
        let completed = HashSet::new();
        let ready = r.ready_phases_v2(&dag, &completed);
        assert_eq!(ready, vec![p1]);
    }
}
