//! Workload profile selection and measured-strategy installation.
//!
//! This module owns the canonical authority for translating a runtime
//! dispatch context (execution mode + sequence length + batch size) into a
//! sealed workload profile, and for installing / reading the
//! runtime-override strategy that a caller may select for a measured
//! scenario. It is the orchestrator's *policy* layer — it never touches
//! tensors, kernels, or backends; it only reports what the orchestrator
//! *would* dispatch given the sealed evidence in the loaded CImage.
//!
//! All methods here are `impl UnifiedRuntime` blocks; the orchestrator
//! struct lives in [`super`].

use prism_spatial_ir::WorkloadScenario;

use super::UnifiedRuntime;

impl UnifiedRuntime {
    pub fn workload_profile_for_dispatch(
        &self,
        sequence_length: u32,
    ) -> Option<&crate::workload_search::WorkloadThroughputEvidence> {
        let evidence = self.model.heterogeneous_workload_evidence.as_ref()?;
        let phase = match self.mode {
            super::ExecutionMode::Batch => {
                crate::workload_search::InferenceWorkloadPhase::Decode
            }
            super::ExecutionMode::RealtimePrefill => {
                crate::workload_search::InferenceWorkloadPhase::Prefill
            }
            super::ExecutionMode::RealtimeDecode => {
                crate::workload_search::InferenceWorkloadPhase::Decode
            }
        };
        let service_class = match self.mode {
            super::ExecutionMode::Batch => crate::workload_search::ServiceClass::Batch,
            super::ExecutionMode::RealtimePrefill | super::ExecutionMode::RealtimeDecode => {
                crate::workload_search::ServiceClass::Realtime
            }
        };
        let batch_size = match self.mode {
            super::ExecutionMode::Batch => self
                .requested_batch_size
                .unwrap_or_else(|| sequence_length.max(1)),
            super::ExecutionMode::RealtimePrefill | super::ExecutionMode::RealtimeDecode => 1,
        };
        let selected_graph = self.model.selected_execution_graph();
        let matches_selected_graph =
            |sample: &&crate::workload_search::WorkloadThroughputEvidence| -> bool {
                let Some(graph) = selected_graph else {
                    return true;
                };
                if graph.route_sequence.is_empty() {
                    return true;
                }
                if !graph.route_sequence.contains(&sample.profile.primary_lane) {
                    return false;
                }
                if !graph
                    .route_sequence
                    .contains(&sample.profile.attention_lane)
                {
                    return false;
                }
                if graph.fused_interleaved_metal && !sample.profile.interleaved_metal {
                    return false;
                }
                if graph.stateless_ane
                    && sample.profile.primary_lane == crate::workload_search::ExecutionLane::Ane
                    && !sample.profile.stateless_shared_arena
                {
                    return false;
                }
                if graph.ane_int8_planar_boundaries
                    && sample.profile.primary_lane == crate::workload_search::ExecutionLane::Ane
                    && (!sample.profile.ane_compute_int8
                        || !sample.profile.planar_input_conversion
                        || !sample.profile.planar_output_conversion)
                {
                    return false;
                }
                true
            };

        let exact = evidence
            .throughput_evidence
            .iter()
            .filter(|sample| {
                sample.profile.phase == phase
                    && sample.profile.service_class == service_class
                    && sample.profile.batch_size == batch_size.max(1)
                    && sample.valid()
                    && matches_selected_graph(sample)
            })
            .max_by(|left, right| {
                left.tokens_per_second
                    .partial_cmp(&right.tokens_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if exact.is_some() {
            return exact;
        }

        evidence
            .throughput_evidence
            .iter()
            .filter(|sample| sample.profile.phase == phase && sample.valid())
            .filter(matches_selected_graph)
            .max_by(|left, right| {
                left.tokens_per_second
                    .partial_cmp(&right.tokens_per_second)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Install a measured strategy choice for one validated workload shape.
    /// The strategy must already exist in the sealed candidate set.
    pub fn install_measured_strategy(
        &mut self,
        scenario: WorkloadScenario,
        strategy_id: impl Into<String>,
    ) -> Result<(), String> {
        scenario.validate()?;
        let strategy_id = strategy_id.into();
        if !self.model.uop_strategy_programs.contains_key(&strategy_id) {
            return Err(format!(
                "UOp strategy '{strategy_id}' is not embedded in the model"
            ));
        }
        self.measured_strategy_overrides
            .insert(scenario, strategy_id);
        Ok(())
    }

    /// Select and install the best measured UOp candidate for one workload.
    /// Selection and installation are a single operation so a caller cannot
    /// accidentally publish a strategy ID that does not correspond to the
    /// measurement vector it just evaluated.
    pub fn install_measured_strategy_choice(
        &mut self,
        scenario: WorkloadScenario,
        strategies: &[prism_spatial_ir::FusionStrategy],
        measurements: &[prism_spatial_ir::FusionMeasurement],
    ) -> Result<String, String> {
        let (strategy_id, _) = crate::select_measured_uop_strategy(strategies, measurements)?;
        self.install_measured_strategy(scenario, strategy_id.clone())?;
        Ok(strategy_id)
    }

    /// Return the strategy currently selected for an exact workload shape.
    /// This exposes the policy decision separately from program dispatch so
    /// receipts and diagnostics can report why a workload took a path.
    pub fn selected_measured_strategy(&self, scenario: WorkloadScenario) -> Option<&str> {
        self.measured_strategy_for_scenario(scenario)
            .map(String::as_str)
    }

    /// Return the compiler-sealed mixed-precision strategy that best matches the
    /// requested runtime shape. This method does not alter execution policy; it
    /// exposes metadata for downstream diagnostics and dispatch orchestration.
    pub fn preferred_mixed_precision_profile(
        &self,
        profile: &crate::workload_search::WorkloadProfile,
    ) -> Option<&crate::workload_search::WorkloadThroughputEvidence> {
        self.model.best_throughput_evidence_for_profile(profile)
    }

    /// Return the canonical selected mixed-precision route for a concrete
    /// workload profile, if it exists.
    pub fn preferred_mixed_precision_graph_for_profile(
        &self,
        profile: &crate::workload_search::WorkloadProfile,
    ) -> Option<&str> {
        self.preferred_mixed_precision_profile(profile)
            .map(|evidence| evidence.mixed_precision_graph.as_str())
    }

    /// Return the mixed-precision graph selected for the active dispatch
    /// profile, if one was selected during evidence compilation.
    pub fn active_mixed_precision_graph(&self) -> Option<&str> {
        self.last_workload_selection
            .as_ref()
            .map(|selection| selection.mixed_precision_graph.as_str())
    }

    /// Return the canonical mixed-precision graph selected during compile-time
    /// scheduling. This can be used to report cross-lane routing decisions
    /// even when runtime workload selection falls back to nearest-shape
    /// matching.
    pub fn selected_execution_graph(
        &self,
    ) -> Option<&crate::workload_search::SelectedExecutionGraph> {
        self.model.selected_execution_graph()
    }

    pub(super) fn measured_strategy_for_scenario(
        &self,
        scenario: WorkloadScenario,
    ) -> Option<&String> {
        self.measured_strategy_overrides.get(&scenario).or_else(|| {
            self.measured_strategy_overrides
                .iter()
                .filter(|(candidate, _)| {
                    candidate.realtime == scenario.realtime
                        && candidate.batch_size == scenario.batch_size
                })
                .min_by_key(|(candidate, strategy)| {
                    (
                        candidate.sequence_length.abs_diff(scenario.sequence_length),
                        strategy.as_str(),
                    )
                })
                .map(|(_, strategy)| strategy)
        })
    }
}
