//! Phase invocation (constitutional home).
//!
//! A narrow typed invocation object passed to every phase runner. Runners
//! access the world only through this object — no reaching back into
//! session or step state directly.
//!
//! # Authority
//!
//! `PhaseInvocation` is a **borrowed view**: it holds references to
//! session, step, and phase state, but does not own them. The phase
//! runners that consume the invocation operate on the borrowed state
//! and stage their mutations through the underlying `WorldTxn` that
//! produced the references.
//!
//! Per the inventory v2.1, this type sits in `state::phase_invocation`.
//! The placeholder engine types it references move in their own
//! migrations.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/phase_invocation.rs`.
//! The engine file is the legacy duplicate; step 58 deletes it when
//! no engine caller remains. No compatibility facade.

use super::phase::{EmittedPhase, RuntimeWorkItemHandle};

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::inference::execution_image_state::ComputeImageState`.
/// Replaced when the inference state types move (separate migration).
#[derive(Debug, Clone)]
pub struct ComputeImageState {
    _placeholder: (),
}

/// Placeholder for `compute-core::inference::inference_session_state::InferenceSessionState`.
#[derive(Debug)]
pub struct InferenceSessionState {
    _placeholder: (),
}

/// Placeholder for `compute-core::inference::inference_step_state::InferenceStepState`.
#[derive(Debug)]
pub struct InferenceStepState {
    _placeholder: (),
}

/// Placeholder for `compute-core::ecs::compute_image::phase_graph::ResolvedPhaseBinding`.
#[derive(Debug, Clone)]
pub struct ResolvedPhaseBinding {
    _placeholder: (),
}

// ---------------------------------------------------------------------------
// PhaseInvocation
// ---------------------------------------------------------------------------

/// Narrow typed invocation object passed to every phase runner.
pub struct PhaseInvocation<'a> {
    pub image: &'a ComputeImageState,
    pub session: &'a mut InferenceSessionState,
    pub step: &'a mut InferenceStepState,
    pub phase: &'a EmittedPhase,
    pub resolved_binding: &'a ResolvedPhaseBinding,
    pub work_item: &'a RuntimeWorkItemHandle,
}

impl<'a> PhaseInvocation<'a> {
    pub fn new(
        image: &'a ComputeImageState,
        session: &'a mut InferenceSessionState,
        step: &'a mut InferenceStepState,
        phase: &'a EmittedPhase,
        resolved_binding: &'a ResolvedPhaseBinding,
        work_item: &'a RuntimeWorkItemHandle,
    ) -> Self {
        Self {
            image,
            session,
            step,
            phase,
            resolved_binding,
            work_item,
        }
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `phase_invocation` view.

    use super::*;

    #[test]
    fn invocation_stores_all_references() {
        // Architectural invariant: the invocation holds six references
        // — image, session, step, phase, binding, work item. A reader
        // can rely on all six being populated by `new()`.
        let image = ComputeImageState { _placeholder: () };
        let mut session = InferenceSessionState { _placeholder: () };
        let mut step = InferenceStepState { _placeholder: () };
        let phase = EmittedPhase::default();
        let binding = ResolvedPhaseBinding { _placeholder: () };
        let work_item = RuntimeWorkItemHandle::new(crate::scheduling::state::phase::PhaseId("p1".into()), 0);

        let inv = PhaseInvocation::new(
            &image,
            &mut session,
            &mut step,
            &phase,
            &binding,
            &work_item,
        );

        // Verify the references are stored (not moved).
        let _: &ComputeImageState = inv.image;
        let _: &mut InferenceSessionState = inv.session;
        let _: &mut InferenceStepState = inv.step;
        let _: &EmittedPhase = inv.phase;
        let _: &ResolvedPhaseBinding = inv.resolved_binding;
        let _: &RuntimeWorkItemHandle = inv.work_item;
    }
}
