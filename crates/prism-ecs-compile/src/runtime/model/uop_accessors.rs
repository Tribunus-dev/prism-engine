//! Runtime model — UOp capture and strategy-evidence accessors.
//!
//! This module owns the canonical authority for reading the embedded
//! UOp program metadata on a loaded [`RuntimeModel`]. The dispatch
//! layer in [`super::super::unified::dispatch`] consumes these
//! accessors when selecting and dispatching a UOp program; this
//! module is the read-side only.

use prism_spatial_ir::CapturePlan;
use prism_spatial_ir::WorkloadScenario;

use super::RuntimeModel;

impl RuntimeModel {
    /// Return sealed strategy evidence for one exact workload shape.
    pub fn uop_workload_evidence_for(
        &self,
        scenario: WorkloadScenario,
    ) -> Option<&crate::cimage::UOpWorkloadEvidence> {
        self.uop_workload_evidence
            .iter()
            .find(|entry| entry.scenario == scenario)
    }

    /// Return the embedded UOp capture after it has passed CImage admission.
    pub fn uop_capture(&self) -> Option<&CapturePlan> {
        self.uop_capture.as_ref()
    }
}
