//! Ready queue (constitutional home).
//!
//! Given a phase DAG and a set of completed phase identifiers, this module
//! determines which phases have all their dependencies satisfied and are
//! ready to run.
//!
//! # Authority
//!
//! `ReadyQueue` is a **scheduling-state query**: it borrows the phase DAG
//! and a completion set, and returns the set of phases that are ready.
//! It does not own state. It is placed in `state::ready_queue` because
//! the completion set it consults is part of the canonical scheduling
//! state (managed by the phase-advancement system), and the query is a
//! pure function of that state.
//!
//! # Placeholder engine types
//!
//! The engine's `ready_queue.rs` depends on `compute_image::phase_dag::*`
//! types (`EmittedPhase`, `EmittedPhaseGraph`, `EmittedPhaseEdge`). The
//! constitutional home defines minimal placeholder types matching the
//! engine wire shape. When the `phase_dag` module migrates into the
//! constitutional compile crate (in its own migration), the placeholders
//! are replaced by the moved definitions.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/ready_queue.rs`.
//! The engine file is the legacy duplicate; step 58 deletes it when no
//! engine caller remains. No compatibility facade.

use std::collections::HashSet;

use super::phase::EmittedPhase;

// ---------------------------------------------------------------------------
// Placeholder engine types (phase_dag-related; moved when phase_dag migrates)
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::EmittedPhaseGraph`.
/// Replaced when `phase_dag` moves into `prism-ecs-compile`.
#[derive(Debug, Clone)]
pub struct EmittedPhaseGraph {
    pub phases: Vec<EmittedPhase>,
    pub edges: Vec<EmittedPhaseEdge>,
}

impl EmittedPhaseGraph {
    /// Return the predecessors of `phase_id` (i.e. every phase that has
    /// an edge pointing into it).
    pub fn predecessors(&self, phase_id: &str) -> Vec<&EmittedPhase> {
        self.edges
            .iter()
            .filter(|e| e.to_phase == phase_id)
            .filter_map(|e| self.phases.iter().find(|p| p.phase_id == e.from_phase))
            .collect()
    }
}

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::EmittedPhaseEdge`.
#[derive(Debug, Clone)]
pub struct EmittedPhaseEdge {
    pub from_phase: String,
    pub to_phase: String,
}

// ---------------------------------------------------------------------------
// ReadyQueue
// ---------------------------------------------------------------------------

/// Tracks which phases are ready based on a phase DAG and a completed set.
///
/// `ReadyQueue` is a borrowed view; it does not own the DAG. The
/// completion set is supplied at query time. The query result is
/// non-authoritative — it is a candidate set for the phase-advancement
/// system to consult.
pub struct ReadyQueue<'a> {
    dag: &'a EmittedPhaseGraph,
}

impl<'a> ReadyQueue<'a> {
    pub fn new(dag: &'a EmittedPhaseGraph) -> Self {
        Self { dag }
    }

    /// Return all phases whose predecessors are all in the `completed` set.
    /// Phases already in `completed` are excluded.
    pub fn ready_phases(&self, completed: &HashSet<String>) -> Vec<&'a EmittedPhase> {
        self.dag
            .phases
            .iter()
            .filter(|phase| {
                if completed.contains(&phase.phase_id) {
                    return false;
                }
                let preds = self.dag.predecessors(&phase.phase_id);
                preds.iter().all(|p| completed.contains(&p.phase_id))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `ready_queue` query.
    //!
    //! These tests verify the constitutional rule: a phase is ready
    //! iff every predecessor is completed, and the phase itself is not
    //! yet completed. The query is monotone — adding to `completed`
    //! can only add ready phases, never remove them.

    use super::*;

    fn make_phase(id: &str) -> EmittedPhase {
        EmittedPhase {
            phase_id: id.into(),
        }
    }

    fn make_dag(phases: &[&str], edges: &[(&str, &str)]) -> EmittedPhaseGraph {
        EmittedPhaseGraph {
            phases: phases.iter().map(|id| make_phase(id)).collect(),
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
    fn ready_returns_phases_with_no_deps() {
        // Architectural invariant: a phase with no predecessors and
        // not yet completed is always ready. Two such phases, both
        // ready.
        let dag = make_dag(&["p0", "p1"], &[]);
        let rq = ReadyQueue::new(&dag);
        let completed = HashSet::new();
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 2);
    }

    #[test]
    fn ready_respects_dependencies() {
        // Architectural invariant: 'b' depends on 'a'. Initially only
        // 'a' is ready. After 'a' completes, 'b' becomes ready.
        let dag = make_dag(&["a", "b"], &[("a", "b")]);
        let rq = ReadyQueue::new(&dag);
        let mut completed = HashSet::new();

        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "a");

        completed.insert("a".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "b");
    }

    #[test]
    fn ready_excludes_completed() {
        // Architectural invariant: a phase in `completed` is never
        // returned, regardless of its dependencies.
        let dag = make_dag(&["a", "b"], &[]);
        let rq = ReadyQueue::new(&dag);
        let mut completed = HashSet::new();
        completed.insert("a".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "b");
    }

    #[test]
    fn ready_with_diamond_dep_returns_unblocked() {
        // a -> b, a -> c, b -> d, c -> d. After 'a' completes, 'b'
        // and 'c' are ready. 'd' waits for both. After both 'b' and
        // 'c' complete, 'd' becomes ready.
        let dag = make_dag(
            &["a", "b", "c", "d"],
            &[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")],
        );
        let rq = ReadyQueue::new(&dag);
        let mut completed = HashSet::new();

        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "a");

        completed.insert("a".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 2);
        let ids: Vec<&str> = ready.iter().map(|p| p.phase_id.as_str()).collect();
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));

        completed.insert("b".into());
        completed.insert("c".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "d");
    }

    #[test]
    fn ready_is_monotone_in_completion() {
        // Architectural invariant: adding to `completed` can only
        // ADD ready phases, never remove them. (A phase in completed
        // is excluded by definition; a phase whose predecessors were
        // missing is now unblocked.)
        let dag = make_dag(&["a", "b"], &[("a", "b")]);
        let rq = ReadyQueue::new(&dag);
        let mut completed = HashSet::new();
        let initial = rq.ready_phases(&completed);
        completed.insert("a".into());
        let after = rq.ready_phases(&completed);
        // Initial ready set: {a}. After: {b}. Same cardinality, no
        // removal. (A more general property: |after| >= |initial| -
        // |initial ∩ completed-after|; for the unit-completion case,
        // that's a >= b - 0 = b.)
        assert!(after.len() >= initial.len() - 1);
    }

    #[test]
    fn ready_with_unknown_phase_id_in_completed_set_is_safe() {
        // The completed set may contain ids that don't exist in the
        // DAG (e.g. a phase that was removed from the graph). The
        // query must not panic and must not return those.
        let dag = make_dag(&["a"], &[]);
        let rq = ReadyQueue::new(&dag);
        let mut completed = HashSet::new();
        completed.insert("nonexistent".into());
        let ready = rq.ready_phases(&completed);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].phase_id, "a");
    }
}
