//! Builder for constructing a [`HeterogeneousExecutionImage`] step by step.
//!
//! The builder provides a convenient API for building the complex nested
//! structure of a heterogeneous execution image.  It is the primary way
//! the compiler constructs the image during pipeline stages 1–12.
//!
//! # Usage
//!
//! ```ignore
//! let mut builder = HeterogeneousImageBuilder::new(model_identity, graph_digest);
//! builder.add_phase_node(node);
//! builder.add_phase_edge(edge);
//! builder.add_slot(slot);
//! builder.add_metal_program(program);
//! // ...
//! let image = builder.build();
//! ```

use super::types::*;

/// Builder for constructing a [`HeterogeneousExecutionImage`].
#[derive(Debug, Clone)]
pub struct HeterogeneousImageBuilder {
    image_version: u32,
    model_identity: ModelIdentity,
    graph_digest: ContentHash,

    // Phase graph
    phase_nodes: Vec<CompiledPhaseNode>,
    phase_edges: Vec<CompiledPhaseEdge>,
    entrypoints: Vec<PhaseId>,
    terminal_nodes: Vec<PhaseId>,

    // Resource plan
    arenas: Vec<ArenaPlan>,
    slots: Vec<CompiledSlot>,
    slot_aliases: Vec<SlotAlias>,
    materializations: Vec<MaterializationNode>,
    lifetime_intervals: Vec<ResourceLifetime>,

    // Lane programs
    metal_programs: Vec<MetalProgram>,
    ane_programs: Vec<AneProgram>,
    accelerate_programs: Vec<AccelerateProgram>,

    // Concurrency
    ready_sets: Vec<ReadySetTemplate>,
    parallel_groups: Vec<ParallelGroup>,
    serialization_edges: Vec<SerializationEdge>,
    lane_caps: LaneCapacityRequirements,
    overlap_hints: Vec<OverlapHint>,

    // Admission
    hardware_requirements: HardwareRequirements,
    artifact_qualifications: Vec<ArtifactQualificationPlan>,
    route_admission_rules: Vec<RouteAdmissionRule>,

    // Fallback
    fallback_chains: Vec<FallbackChain>,
    transition_rules: Vec<FallbackTransitionRule>,

    // Execution policies
    execution_policies: CompiledExecutionPolicies,

    // Evidence contract
    evidence_contract: CompiledEvidenceContract,

    // Receipt
    receipt: CompilationReceipt,
}

impl HeterogeneousImageBuilder {
    /// Create a new builder with the given model identity and graph digest.
    pub fn new(model_identity: ModelIdentity, graph_digest: ContentHash) -> Self {
        Self {
            image_version: 1,
            model_identity,
            graph_digest,
            phase_nodes: Vec::new(),
            phase_edges: Vec::new(),
            entrypoints: Vec::new(),
            terminal_nodes: Vec::new(),
            arenas: Vec::new(),
            slots: Vec::new(),
            slot_aliases: Vec::new(),
            materializations: Vec::new(),
            lifetime_intervals: Vec::new(),
            metal_programs: Vec::new(),
            ane_programs: Vec::new(),
            accelerate_programs: Vec::new(),
            ready_sets: Vec::new(),
            parallel_groups: Vec::new(),
            serialization_edges: Vec::new(),
            lane_caps: LaneCapacityRequirements::default(),
            overlap_hints: Vec::new(),
            hardware_requirements: HardwareRequirements {
                min_soc_family: String::new(),
                min_macos_version: String::new(),
                min_coreml_version: String::new(),
                min_ane_count: 0,
                min_gpu_core_count: 0,
                required_features: Vec::new(),
            },
            artifact_qualifications: Vec::new(),
            route_admission_rules: Vec::new(),
            fallback_chains: Vec::new(),
            transition_rules: Vec::new(),
            execution_policies: CompiledExecutionPolicies::default(),
            evidence_contract: CompiledEvidenceContract::default(),
            receipt: CompilationReceipt {
                image_digest: graph_digest,
                phase_count: 0,
                variant_count: 0,
                metal_program_count: 0,
                ane_program_count: 0,
                accelerate_program_count: 0,
                parallel_group_count: 0,
                materialization_count: 0,
                rejected_variants: Vec::new(),
                emitted_fallback_count: 0,
            },
        }
    }

    // ── Phase graph ────────────────────────────────────────────────────

    /// Add a compiled phase node to the graph.
    pub fn add_phase_node(&mut self, node: CompiledPhaseNode) {
        self.entrypoints.retain(|&id| id != node.phase_id);
        self.terminal_nodes.retain(|&id| id != node.phase_id);
        self.phase_nodes.push(node);
    }

    /// Set the entrypoint phases (nodes with no incoming edges).
    pub fn set_entrypoints(&mut self, entrypoints: Vec<PhaseId>) {
        self.entrypoints = entrypoints;
    }

    /// Set the terminal phases (nodes with no outgoing edges).
    pub fn set_terminal_nodes(&mut self, terminals: Vec<PhaseId>) {
        self.terminal_nodes = terminals;
    }

    /// Add a dependency edge between two phases.
    pub fn add_phase_edge(&mut self, edge: CompiledPhaseEdge) {
        self.phase_edges.push(edge);
    }

    // ── Resource plan ──────────────────────────────────────────────────

    /// Add an arena plan.
    pub fn add_arena(&mut self, arena: ArenaPlan) {
        self.arenas.push(arena);
    }

    /// Add a compiled slot.
    pub fn add_slot(&mut self, slot: CompiledSlot) {
        self.slots.push(slot);
    }

    /// Add a slot alias.
    pub fn add_slot_alias(&mut self, alias: SlotAlias) {
        self.slot_aliases.push(alias);
    }

    /// Add a materialization node.
    pub fn add_materialization(&mut self, node: MaterializationNode) {
        self.materializations.push(node);
    }

    /// Add a resource lifetime interval.
    pub fn add_lifetime(&mut self, lifetime: ResourceLifetime) {
        self.lifetime_intervals.push(lifetime);
    }

    // ── Lane programs ──────────────────────────────────────────────────

    /// Add a compiled Metal GPU program.
    pub fn add_metal_program(&mut self, program: MetalProgram) {
        self.metal_programs.push(program);
    }

    /// Add a compiled Core ML / ANE program.
    pub fn add_ane_program(&mut self, program: AneProgram) {
        self.ane_programs.push(program);
    }

    /// Add a compiled Accelerate / CPU program.
    pub fn add_accelerate_program(&mut self, program: AccelerateProgram) {
        self.accelerate_programs.push(program);
    }

    // ── Concurrency ────────────────────────────────────────────────────

    /// Add a ready set template.
    pub fn add_ready_set(&mut self, ready_set: ReadySetTemplate) {
        self.ready_sets.push(ready_set);
    }

    /// Add a parallel group.
    pub fn add_parallel_group(&mut self, group: ParallelGroup) {
        self.parallel_groups.push(group);
    }

    /// Add a serialization edge.
    pub fn add_serialization_edge(&mut self, edge: SerializationEdge) {
        self.serialization_edges.push(edge);
    }

    /// Set the lane capacity requirements.
    pub fn set_lane_capacity(&mut self, caps: LaneCapacityRequirements) {
        self.lane_caps = caps;
    }

    /// Add an overlap hint.
    pub fn add_overlap_hint(&mut self, hint: OverlapHint) {
        self.overlap_hints.push(hint);
    }

    // ── Admission ──────────────────────────────────────────────────────

    /// Set hardware requirements for admission.
    pub fn set_hardware_requirements(&mut self, req: HardwareRequirements) {
        self.hardware_requirements = req;
    }

    /// Add an artifact qualification plan.
    pub fn add_artifact_qualification(&mut self, plan: ArtifactQualificationPlan) {
        self.artifact_qualifications.push(plan);
    }

    /// Add a route admission rule.
    pub fn add_route_admission_rule(&mut self, rule: RouteAdmissionRule) {
        self.route_admission_rules.push(rule);
    }

    // ── Fallback ───────────────────────────────────────────────────────

    /// Add a fallback chain.
    pub fn add_fallback_chain(&mut self, chain: FallbackChain) {
        self.fallback_chains.push(chain);
    }

    /// Add a fallback transition rule.
    pub fn add_fallback_transition(&mut self, rule: FallbackTransitionRule) {
        self.transition_rules.push(rule);
    }

    // ── Execution policies ─────────────────────────────────────────────

    /// Set the execution policies.
    pub fn set_execution_policies(&mut self, policies: CompiledExecutionPolicies) {
        self.execution_policies = policies;
    }

    // ── Evidence ───────────────────────────────────────────────────────

    /// Set the evidence contract.
    pub fn set_evidence_contract(&mut self, contract: CompiledEvidenceContract) {
        self.evidence_contract = contract;
    }

    // ── Receipt ────────────────────────────────────────────────────────

    /// Add a rejected variant record to the compilation receipt.
    pub fn add_rejected_variant(&mut self, rejected: RejectedVariant) {
        self.receipt.rejected_variants.push(rejected);
    }

    /// Finalize and build the [`HeterogeneousExecutionImage`].
    ///
    /// Computes auto-populated counters in the receipt and returns the
    /// sealed image.
    pub fn build(mut self) -> HeterogeneousExecutionImage {
        // Auto-populate receipt counters
        self.receipt.phase_count = self.phase_nodes.len();
        self.receipt.metal_program_count = self.metal_programs.len();
        self.receipt.ane_program_count = self.ane_programs.len();
        self.receipt.accelerate_program_count = self.accelerate_programs.len();
        self.receipt.parallel_group_count = self.parallel_groups.len();
        self.receipt.materialization_count = self.materializations.len();
        self.receipt.emitted_fallback_count = self.fallback_chains.len();
        // Variant count is best-effort (only tracks explicitly added rejected variants)
        self.receipt.variant_count = self
            .phase_nodes
            .iter()
            .map(|n| n.variant_set_id as usize)
            .max()
            .unwrap_or(0)
            + 1;

        // Auto-detect entrypoints and terminals if not explicitly set
        if self.entrypoints.is_empty() && !self.phase_nodes.is_empty() {
            let _all_from: std::collections::HashSet<PhaseId> =
                self.phase_edges.iter().map(|e| e.from).collect();
            let all_to: std::collections::HashSet<PhaseId> =
                self.phase_edges.iter().map(|e| e.to).collect();
            self.entrypoints = self
                .phase_nodes
                .iter()
                .map(|n| n.phase_id)
                .filter(|id| !all_to.contains(id))
                .collect();
        }
        if self.terminal_nodes.is_empty() && !self.phase_nodes.is_empty() {
            let all_from: std::collections::HashSet<PhaseId> =
                self.phase_edges.iter().map(|e| e.from).collect();
            let _all_to: std::collections::HashSet<PhaseId> =
                self.phase_edges.iter().map(|e| e.to).collect();
            self.terminal_nodes = self
                .phase_nodes
                .iter()
                .map(|n| n.phase_id)
                .filter(|id| !all_from.contains(id))
                .collect();
        }

        HeterogeneousExecutionImage {
            image_version: self.image_version,
            model_identity: self.model_identity,
            graph_digest: self.graph_digest,
            phase_graph: CompiledPhaseGraph {
                nodes: self.phase_nodes,
                edges: self.phase_edges,
                entrypoints: self.entrypoints,
                terminal_nodes: self.terminal_nodes,
            },
            resources: CompiledResourcePlan {
                arenas: self.arenas,
                slots: self.slots,
                aliases: self.slot_aliases,
                materializations: self.materializations,
                lifetime_intervals: self.lifetime_intervals,
            },
            lane_programs: CompiledLanePrograms {
                metal: self.metal_programs,
                ane: self.ane_programs,
                accelerate: self.accelerate_programs,
            },
            concurrency: CompiledConcurrencyPlan {
                ready_sets: self.ready_sets,
                parallel_groups: self.parallel_groups,
                serialization_edges: self.serialization_edges,
                lane_caps: self.lane_caps,
                overlap_hints: self.overlap_hints,
            },
            admission: CompiledAdmissionPlan {
                hardware_signature_requirements: self.hardware_requirements,
                artifact_qualification: self.artifact_qualifications,
                route_admission_rules: self.route_admission_rules,
            },
            fallback: CompiledFallbackPlan {
                fallback_chains: self.fallback_chains,
                transition_rules: self.transition_rules,
            },
            execution_policy: self.execution_policies,
            evidence_contract: self.evidence_contract,
        }
    }

    /// Return the current [`CompilationReceipt`].
    pub fn receipt(&self) -> &CompilationReceipt {
        &self.receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the builder produces a valid image with auto-inferred
    /// entrypoints and terminal nodes.
    #[test]
    fn test_builder_basic_graph() {
        let identity = ModelIdentity {
            model_name: "test".into(),
            model_family: "test".into(),
            model_variant: "1".into(),
            canonical_graph_hash: ContentHash(1),
            compile_timestamp: "now".into(),
            compiler_version: "0.1".into(),
        };
        let mut builder = HeterogeneousImageBuilder::new(identity, ContentHash(42));

        builder.add_phase_node(CompiledPhaseNode {
            phase_id: 0,
            variant_set_id: 0,
            ready_condition: ReadyCondition::AlwaysReady,
            parallel_group: None,
            priority_class: PriorityClass::Critical,
        });
        builder.add_phase_node(CompiledPhaseNode {
            phase_id: 1,
            variant_set_id: 0,
            ready_condition: ReadyCondition::AllDependenciesSatisfied,
            parallel_group: None,
            priority_class: PriorityClass::Critical,
        });
        builder.add_phase_edge(CompiledPhaseEdge {
            from: 0,
            to: 1,
            dependency: CompiledDependency::Control,
        });

        let image = builder.build();

        assert_eq!(image.phase_graph.nodes.len(), 2);
        assert_eq!(image.phase_graph.edges.len(), 1);
        // Entrypoint should be auto-detected as phase 0 (no incoming edges)
        assert_eq!(image.phase_graph.entrypoints, vec![0]);
        // Terminal should be auto-detected as phase 1 (no outgoing edges)
        assert_eq!(image.phase_graph.terminal_nodes, vec![1]);
        // Default execution policies should be present
        assert!(image
            .execution_policy
            .policies
            .contains_key(&image.execution_policy.latency_single_sequence));
        assert!(image
            .execution_policy
            .policies
            .contains_key(&image.execution_policy.throughput_multi_sequence));
    }

    /// Verify the builder with explicit entrypoints/terminals.
    #[test]
    fn test_builder_explicit_boundaries() {
        let identity = ModelIdentity {
            model_name: "test".into(),
            model_family: "test".into(),
            model_variant: "1".into(),
            canonical_graph_hash: ContentHash(1),
            compile_timestamp: "now".into(),
            compiler_version: "0.1".into(),
        };
        let mut builder = HeterogeneousImageBuilder::new(identity, ContentHash(42));

        builder.add_phase_node(CompiledPhaseNode {
            phase_id: 10,
            variant_set_id: 0,
            ready_condition: ReadyCondition::AlwaysReady,
            parallel_group: None,
            priority_class: PriorityClass::Critical,
        });
        builder.add_phase_node(CompiledPhaseNode {
            phase_id: 20,
            variant_set_id: 0,
            ready_condition: ReadyCondition::AllDependenciesSatisfied,
            parallel_group: None,
            priority_class: PriorityClass::Batch,
        });
        builder.add_phase_edge(CompiledPhaseEdge {
            from: 10,
            to: 20,
            dependency: CompiledDependency::Control,
        });
        builder.set_entrypoints(vec![10]);
        builder.set_terminal_nodes(vec![20]);

        let image = builder.build();

        assert_eq!(image.phase_graph.entrypoints, vec![10]);
        assert_eq!(image.phase_graph.terminal_nodes, vec![20]);
    }

    /// Verify builder adds slots correctly.
    #[test]
    fn test_builder_with_slots() {
        let identity = ModelIdentity {
            model_name: "test".into(),
            model_family: "test".into(),
            model_variant: "1".into(),
            canonical_graph_hash: ContentHash(1),
            compile_timestamp: "now".into(),
            compiler_version: "0.1".into(),
        };
        let mut builder = HeterogeneousImageBuilder::new(identity, ContentHash(42));

        builder.add_arena(ArenaPlan {
            arena_id: 0,
            byte_size: 65536,
            alignment: 4096,
            backing: ArenaBacking::IOSurface,
            ring_depth: 2,
        });
        builder.add_slot(CompiledSlot {
            slot_id: 0,
            arena_id: 0,
            activation_abi: ActivationAbi::MetalOnly(
                crate::compilation::activation_abi::MetalOnlyParams {
                    name: "metal_buffer".into(),
                    dtype: crate::compilation::phase_ir::TensorDtype::Float16,
                    byte_count: 4096,
                },
            ),
            byte_length: 4096,
            alignment: 256,
            backing: SlotBacking::IOSurface,
            producer_phase: 0,
            consumer_phases: vec![1],
            concurrency_class: ConcurrencyClass::Exclusive,
        });

        let image = builder.build();

        assert_eq!(image.resources.arenas.len(), 1);
        assert_eq!(image.resources.slots.len(), 1);
        assert_eq!(image.resources.slots[0].slot_id, 0);
    }
}
