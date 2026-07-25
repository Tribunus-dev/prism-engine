//! Runtime model — compiler-sealed evidence accessors.
//!
//! This module owns the canonical authority for reading the
//! compiler-sealed evidence envelopes on a loaded [`RuntimeModel`]:
//! the KV compression policy, the heterogeneous workload selection
//! evidence, the per-profile best-throughput evidence, and the
//! selected mixed-precision execution graph. These are metadata-only
//! views — the dispatch layer reads them to make admission decisions,
//! but the methods here never touch the file system or the backend.

use super::RuntimeModel;

impl RuntimeModel {
    /// Return the compiler-selected progressive KV policy, if this CImage
    /// contains measured KV compression evidence.
    pub fn kv_compression_policy(&self) -> Option<&str> {
        self.kv_compression_policy.as_deref()
    }

    pub fn heterogeneous_workload_evidence(
        &self,
    ) -> Option<&crate::search::HeterogeneousScheduleEvidence> {
        self.heterogeneous_workload_evidence.as_ref()
    }

    /// Return the best measured throughput evidence for one workload profile,
    /// preferring the highest measured tokens-per-second evidence. This is the
    /// metadata-only bridge between compilation and runtime telemetry.
    pub fn best_throughput_evidence_for_profile(
        &self,
        profile: &crate::workload_search::WorkloadProfile,
    ) -> Option<&crate::workload_search::WorkloadThroughputEvidence> {
        self.heterogeneous_workload_evidence
            .as_ref()
            .and_then(|evidence| {
                evidence
                    .throughput_evidence
                    .iter()
                    .filter(|candidate| {
                        candidate.profile.phase == profile.phase
                            && candidate.profile.batch_size == profile.batch_size
                            && candidate.profile.concurrency == profile.concurrency
                            && candidate.profile.service_class == profile.service_class
                    })
                    .max_by(|left, right| {
                        left.tokens_per_second
                            .partial_cmp(&right.tokens_per_second)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            })
    }

    /// Return the compact mixed-precision execution graph selected for the
    /// artifact-level heterogeneous search.
    pub fn selected_execution_graph(
        &self,
    ) -> Option<&crate::workload_search::SelectedExecutionGraph> {
        self.heterogeneous_workload_evidence
            .as_ref()
            .map(|evidence| &evidence.selected_execution_graph)
    }
}
