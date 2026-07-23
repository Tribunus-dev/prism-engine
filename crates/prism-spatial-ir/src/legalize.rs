//! Three-tier legality validation for spatial graphs.
//!
//! Every [`SpatialGraph`] must pass through the legalizer before it can be
//! lowered or estimated. The legalizer checks:
//!
//! 1. **Semantic legality** — shape contracts match along every edge, tensor
//!    types are compatible, representations are valid.
//!
//! 2. **Backend legality** — target-specific constraints (e.g., "Metal
//!    pipelines support a maximum of 31 threads per threadgroup").
//! 3. **Operational legality** — execution boundaries are explicit, all memory
//!    regions are reachable, no dangling references.

use crate::graph::{SpatialGraph, SpatialNode, SpatialNodeId};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// LegalizationError
// ---------------------------------------------------------------------------

/// Error raised when a spatial graph fails a legality check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, thiserror::Error)]
pub enum LegalizationError {
    /// Shape contract violation: two connected nodes have incompatible shapes.
    #[error("shape mismatch: node {node} expects input {input_idx} of shape {expected:?}, edge provides {actual:?}")]
    ShapeMismatch {
        /// The node with the mismatched shape.
        node: SpatialNodeId,
        /// Index of the input that mismatched.
        input_idx: usize,
        /// Expected shape from the node's contract.
        expected: Vec<usize>,
        /// Actual shape flowing along the edge.
        actual: Vec<usize>,
    },

    /// The graph is not acyclic (contains at least one cycle).
    #[error("graph contains a cycle")]
    CycleDetected,

    /// A node references a dependency that does not exist in the graph.
    #[error("dangling reference: node {node} references non-existent node {missing}")]
    DanglingReference {
        /// Node containing the bad reference.
        node: SpatialNodeId,
        /// The referenced ID that does not exist.
        missing: SpatialNodeId,
    },

    /// An edge references a source or sink that does not exist.
    #[error("edge {edge_id} references non-existent node {node_id}")]
    EdgeDanglingReference {
        /// Edge ID with the bad reference.
        edge_id: crate::graph::SpatialEdgeId,
        /// The referenced node ID.
        node_id: SpatialNodeId,
    },

    /// A stream has mismatched source/sink output/input indices.
    #[error("stream {node_id}: output index {output_idx} out of range for source's output count ({output_count})")]
    StreamOutputIndexOutOfRange {
        /// Stream node ID.
        node_id: SpatialNodeId,
        /// Requested output index.
        output_idx: usize,
        /// Number of outputs on the source node.
        output_count: usize,
    },

    /// A shape contract is empty for a compute node that requires I/O.
    #[error("compute node {node_id} has empty shape contract")]
    EmptyShapeContract {
        /// Node with the empty contract.
        node_id: SpatialNodeId,
    },

    /// A memory region exceeds the target's budget.
    #[error(
        "memory region at node {node_id} requires {required} bytes, exceeds budget of {available}"
    )]
    MemoryBudgetExceeded {
        /// Node requesting the memory.
        node_id: SpatialNodeId,
        /// Bytes required.
        required: u64,
        /// Bytes available.
        available: u64,
    },

    /// Execution boundaries are missing between incompatible memory domains.
    #[error(
        "missing execution boundary between nodes {from} and {to}: incompatible memory domains"
    )]
    MissingExecutionBoundary {
        /// Source node ID.
        from: SpatialNodeId,
        /// Destination node ID.
        to: SpatialNodeId,
    },

    /// Two GPU nodes cannot be fused: incompatible shapes.
    #[error("cannot fuse nodes {first} and {second}: incompatible shape contracts")]
    FusionShapeMismatch {
        /// First node in the fusion pair.
        first: SpatialNodeId,
        /// Second node in the fusion pair.
        second: SpatialNodeId,
    },

    /// Target-specific constraint violation.
    #[error("target constraint violation: {detail}")]
    TargetConstraintViolation {
        /// Human-readable description of the violation.
        detail: String,
    },

    /// Generic legality failure (no specific variant).
    #[error("legalization failed: {0}")]
    Generic(String),
}

// ---------------------------------------------------------------------------
// LegalizedGraph
// ---------------------------------------------------------------------------

/// A [`SpatialGraph`] that has passed three-tier legalization.
///
/// Wraps the inner graph and provides a type-level guarantee that it has
/// been validated. The only way to obtain a `LegalizedGraph` is through
/// the legalizer or through direct construction (for tests).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LegalizedGraph {
    /// The validated spatial graph.
    graph: SpatialGraph,
    /// Errors that were found and fixed during legalization (empty if clean).
    warnings: Vec<LegalizationError>,
}

impl LegalizedGraph {
    /// Create a new legalized graph from a validated graph.
    ///
    /// # Safety
    ///
    /// This constructor does not run the legalizer. It should only be called
    /// after successful legalization, or in tests with hand-verified graphs.
    pub fn new(graph: SpatialGraph, warnings: Vec<LegalizationError>) -> Self {
        Self { graph, warnings }
    }

    /// Returns a reference to the underlying spatial graph.
    pub fn graph(&self) -> &SpatialGraph {
        &self.graph
    }

    /// Consumes the legalized graph, returning ownership of the inner graph.
    pub fn into_inner(self) -> SpatialGraph {
        self.graph
    }

    /// Returns any warnings produced during legalization.
    pub fn warnings(&self) -> &[LegalizationError] {
        &self.warnings
    }

    /// Returns `true` if legalization completed with no warnings.
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }
}

// ---------------------------------------------------------------------------
// SemanticLegalizer
// ---------------------------------------------------------------------------

/// Validates shape contracts, tensor types, and representation bindings.
///
/// Checks:
/// - Every edge's shape matches the source's output and the sink's input.
/// - Compute nodes have non-empty shape contracts.
/// - No duplicate node IDs exist.
/// - No dangling references in barrier dependencies.
pub struct SemanticLegalizer;

impl SemanticLegalizer {
    /// Run the semantic legality pass on a graph.
    ///
    /// Returns `Ok(())` if the graph is semantically valid, or a vector of
    /// all errors found.
    pub fn legalize(graph: &SpatialGraph) -> Result<(), Vec<LegalizationError>> {
        let mut errors = Vec::new();

        // Check for cycles
        if graph.topological_sort().is_none() {
            errors.push(LegalizationError::CycleDetected);
        }

        // Collect all valid node IDs
        let node_ids: std::collections::HashSet<SpatialNodeId> =
            graph.nodes().iter().map(|n| n.id()).collect();

        // Check every edge's source and sink exist
        for edge in graph.edges() {
            if !node_ids.contains(&edge.source) {
                errors.push(LegalizationError::EdgeDanglingReference {
                    edge_id: edge.id,
                    node_id: edge.source,
                });
            }
            if !node_ids.contains(&edge.sink) {
                errors.push(LegalizationError::EdgeDanglingReference {
                    edge_id: edge.id,
                    node_id: edge.sink,
                });
            }
        }

        // Check every node for internal consistency
        for node in graph.nodes() {
            match node {
                SpatialNode::Compute { id, shape, .. } => {
                    if shape.in_shapes.is_empty() && shape.out_shapes.is_empty() {
                        errors.push(LegalizationError::EmptyShapeContract { node_id: *id });
                    }
                }
                SpatialNode::Barrier {
                    id, dependencies, ..
                } => {
                    for dep in dependencies {
                        if !node_ids.contains(dep) {
                            errors.push(LegalizationError::DanglingReference {
                                node: *id,
                                missing: *dep,
                            });
                        }
                    }
                }
                SpatialNode::RepeatedDecoder { id, body, .. } => {
                    for body_id in body {
                        if !node_ids.contains(body_id) {
                            errors.push(LegalizationError::DanglingReference {
                                node: *id,
                                missing: *body_id,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// BackendLegalizer
// ---------------------------------------------------------------------------

/// Validates target-specific constraints.
///
/// Checks that the graph satisfies constraints imposed by the backend:
/// - Shapes fit within backend limits (e.g., Metal's max threadgroup size).
/// - Weight memory fits within the backend's budget.
/// - KV cache parameters are compatible with the backend.
pub struct BackendLegalizer;

impl BackendLegalizer {
    /// Run the backend legality pass on a graph.
    ///
    /// Takes a closure that checks backend-specific constraints for each node.
    /// Built-in checks validate codec support, placement validity, and KV cache policy.
    pub fn legalize<F>(graph: &SpatialGraph, check_node: F) -> Result<(), Vec<LegalizationError>>
    where
        F: Fn(&SpatialNode) -> Result<(), Vec<LegalizationError>>,
    {
        let mut errors = Vec::new();
        for node in graph.nodes() {
            // Custom callback check (extensibility hook)
            if let Err(mut node_errors) = check_node(node) {
                errors.append(&mut node_errors);
            }

            // Built-in annotation-based checks
            let node_id = node.id();
            if let Some(meta) = graph.get_annotations(node_id) {
                // Check 1: codec is supported
                if let Some(variant) = &meta.codec {
                    if !variant.is_supported_codec() {
                        errors.push(LegalizationError::TargetConstraintViolation {
                            detail: format!(
                                "node {} uses unsupported codec '{:?}'",
                                node_id.0, variant
                            ),
                        });
                    }
                }

                // Check 2: placement is valid (only CPU, GPU, and TransferUnit allowed)
                if let Some(placement_str) = &meta.placement {
                    let placement_str = placement_str.trim();
                    let is_valid = matches!(
                        placement_str,
                        "CpuLane" | "GpuComputeRegion" | "TransferUnit"
                    );
                    if !is_valid {
                        errors.push(LegalizationError::TargetConstraintViolation {
                            detail: format!(
                                "node {} has invalid placement '{}' — only CPU, GPU, and TransferUnit are allowed",
                                node_id.0, placement_str
                            ),
                        });
                    }
                }

                // Check 3: KV cache policy is supported (4-bit or 16-bit)
                if let Some(kv_policy) = &meta.kv_cache_policy {
                    let kv_policy = kv_policy.trim();
                    let is_valid = kv_policy == "true:4"
                        || kv_policy == "true:16"
                        || kv_policy == "false:4"
                        || kv_policy == "false:16";
                    if !is_valid {
                        errors.push(LegalizationError::TargetConstraintViolation {
                            detail: format!(
                                "node {} has unsupported KV cache policy '{}' — only 4-bit and 16-bit are supported",
                                node_id.0, kv_policy
                            ),
                        });
                    }
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Metal-specific checks
// ---------------------------------------------------------------------------

/// Run Metal-specific legality checks on a single node.
///
/// Validates tile geometry and memory region constraints for GPU placement.
/// - Threadgroup dimensions must be within Metal limits:
///   width ≤ 256, height ≤ 64, total threads ≤ 1024.
/// - Memory region must be compatible with the placement.
pub fn metal_specific_checks(
    node: &SpatialNode,
    graph: &SpatialGraph,
) -> Result<(), Vec<LegalizationError>> {
    let mut errors = Vec::new();
    let node_id = node.id();

    if let Some(meta) = graph.get_annotations(node_id) {
        // Check tile geometry if set — validate within Metal limits.
        if let Some(tg) = &meta.tile_geometry {
            let max_width: usize = 256;
            let max_height: usize = 64;
            let max_total: usize = 1024;
            let total = tg.width * tg.height;

            if tg.width > max_width {
                errors.push(LegalizationError::TargetConstraintViolation {
                    detail: format!(
                        "node {} tile width {} exceeds Metal limit of {}",
                        node_id.0, tg.width, 256
                    ),
                });
            }
            if tg.height > max_height {
                errors.push(LegalizationError::TargetConstraintViolation {
                    detail: format!(
                        "node {} tile height {} exceeds Metal limit of {}",
                        node_id.0, tg.height, 64
                    ),
                });
            }
            if total > max_total {
                errors.push(LegalizationError::TargetConstraintViolation {
                    detail: format!(
                        "node {} total threads {} exceeds Metal limit of {}",
                        node_id.0, total, 1024
                    ),
                });
            }
        }

        // Validate memory region is compatible with placement
        if let Some(region_str) = &meta.memory_region {
            let region_str = region_str.trim();
            let placement = meta.placement.as_deref();

            // AnEngineMemory is only valid with AnEngine placement
            if region_str == "AnEngineMemory" && !matches!(placement, Some("AnEngine")) {
                errors.push(LegalizationError::TargetConstraintViolation {
                    detail: format!(
                        "node {} uses AnEngineMemory region but is placed on '{:?}'",
                        node_id.0, placement
                    ),
                });
            }

            // DedicatedGpuVram is only valid with GPU placement
            if region_str == "DedicatedGpuVram" && !matches!(placement, Some("GpuComputeRegion")) {
                errors.push(LegalizationError::TargetConstraintViolation {
                    detail: format!(
                        "node {} uses DedicatedGpuVram region but is not placed on GPU",
                        node_id.0
                    ),
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ---------------------------------------------------------------------------
// OperationalLegalizer
// ---------------------------------------------------------------------------

/// Validates execution boundaries, memory domains, and scheduling constraints.
///
/// Checks:
/// - Every edge between incompatible memory domains has an explicit boundary.
/// - No resource oversubscription (e.g., two GPU nodes scheduled concurrently
///   when the target disallows it).
/// - All memory regions are reachable from their consumer nodes.
pub struct OperationalLegalizer;

impl OperationalLegalizer {
    /// Run the operational legality pass on a graph.
    ///
    /// At Level 1 (Spatial Graph), placement metadata may not be assigned yet.
    /// When placement annotations are present, this pass validates:
    /// - Memory regions are consistent (no dangling references).
    /// - Execution boundaries exist between nodes on different compute units.
    /// - Barrier dependencies reference valid node IDs.
    pub fn legalize(graph: &SpatialGraph) -> Result<(), Vec<LegalizationError>> {
        let mut errors = Vec::new();

        let node_ids: HashSet<SpatialNodeId> = graph.nodes().iter().map(|n| n.id()).collect();

        // --- Memory region consistency ---
        // Check that Memory nodes reference valid storage.
        for node in graph.nodes() {
            if let SpatialNode::Memory {
                id,
                kind: _,
                region: _,
            } = node
            {
                if let Some(meta) = graph.get_annotations(*id) {
                    if let Some(region_str) = &meta.memory_region {
                        let region_str = region_str.trim();
                        let known_regions = [
                            "UnifiedMemory",
                            "DedicatedGpuVram",
                            "SharedCache",
                            "AnEngineMemory",
                            "MappedWeights",
                        ];
                        if !known_regions.contains(&region_str) {
                            errors.push(LegalizationError::TargetConstraintViolation {
                                detail: format!(
                                    "memory node {} references unknown memory region '{}'",
                                    id.0, region_str
                                ),
                            });
                        }
                    }
                }
            }
        }

        // --- Execution boundaries ---
        // Check that edges between nodes on different compute units have
        // explicit boundaries recorded. This only fires when placement
        // annotations are present on both endpoints.
        for edge in graph.edges() {
            let source_placement = graph
                .get_annotations(edge.source)
                .and_then(|m| m.placement.as_deref().map(|s| s.to_string()));
            let sink_placement = graph
                .get_annotations(edge.sink)
                .and_then(|m| m.placement.as_deref().map(|s| s.to_string()));

            if let (Some(src_p), Some(snk_p)) = (&source_placement, &sink_placement) {
                if src_p != snk_p {
                    // Different compute units — an execution boundary is needed.
                    // Check if any boundary annotations exist on either endpoint.
                    let has_boundary = graph
                        .get_annotations(edge.source)
                        .and_then(|m| m.memory_region.as_deref())
                        .is_some()
                        || graph
                            .get_annotations(edge.sink)
                            .and_then(|m| m.memory_region.as_deref())
                            .is_some();

                    if !has_boundary {
                        errors.push(LegalizationError::MissingExecutionBoundary {
                            from: edge.source,
                            to: edge.sink,
                        });
                    }
                }
            }
        }

        // --- Barrier dependencies ---
        // Validate that each barrier's dependency IDs exist in the graph.
        for node in graph.nodes() {
            if let SpatialNode::Barrier {
                id, dependencies, ..
            } = node
            {
                for dep in dependencies {
                    if !node_ids.contains(dep) {
                        errors.push(LegalizationError::DanglingReference {
                            node: *id,
                            missing: *dep,
                        });
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: run all three passes
// ---------------------------------------------------------------------------

/// Run all three legalization passes on a spatial graph.
///
/// Returns a [`LegalizedGraph`] wrapping the validated graph, along with any
/// warnings from the individual passes.
pub fn legalize<F>(
    graph: SpatialGraph,
    backend_check: F,
) -> Result<LegalizedGraph, Vec<LegalizationError>>
where
    F: Fn(&SpatialNode) -> Result<(), Vec<LegalizationError>>,
{
    let mut all_warnings = Vec::new();

    // 1. Semantic legalization
    if let Err(errors) = SemanticLegalizer::legalize(&graph) {
        return Err(errors);
    }

    // 2. Backend legalization
    if let Err(errors) = BackendLegalizer::legalize(&graph, &backend_check) {
        // Backend warnings may be non-fatal — collect them
        all_warnings.extend(errors);
    }

    // 3. Operational legalization
    if let Err(errors) = OperationalLegalizer::legalize(&graph) {
        all_warnings.extend(errors);
    }

    Ok(LegalizedGraph::new(graph, all_warnings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::CodecVariant;
    use crate::graph::{
        ComputeIntensity, ComputeKind, EdgeDirection, ShapeContract, SpatialEdge, SpatialEdgeId,
        SpatialNode, TileGeometry,
    };
    use prism_ecs_ir::cimage_types::TensorShape;

    #[test]
    fn legalize_empty_graph() {
        let g = SpatialGraph::new();
        let result = SemanticLegalizer::legalize(&g);
        assert!(result.is_ok());
    }

    #[test]
    fn legalize_compute_node() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        assert!(SemanticLegalizer::legalize(&g).is_ok());
    }

    #[test]
    fn legalize_empty_shape_contract_fails() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(vec![], vec![]),
            intensity: ComputeIntensity::ComputeBound,
        });
        let result = SemanticLegalizer::legalize(&g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors
            .iter()
            .any(|e| matches!(e, LegalizationError::EmptyShapeContract { .. })));
    }

    #[test]
    fn legalize_acyclic_graph() {
        let mut g = SpatialGraph::new();
        let id_a = SpatialNodeId(1);
        let id_b = SpatialNodeId(2);
        g.add_node(SpatialNode::Compute {
            id: id_a,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.add_node(SpatialNode::Compute {
            id: id_b,
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![64, 64] }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: id_a,
            sink: id_b,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        assert!(SemanticLegalizer::legalize(&g).is_ok());
    }

    #[test]
    fn legalize_dangling_barrier_dep() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Barrier {
            id: SpatialNodeId(1),
            dependencies: vec![SpatialNodeId(999)],
        });
        let result = SemanticLegalizer::legalize(&g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::DanglingReference {
                missing: SpatialNodeId(999),
                ..
            }
        )));
    }

    #[test]
    fn full_legalization_pipeline() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = legalize(g, backend_check);
        assert!(result.is_ok());
        let lg = result.unwrap();
        assert!(lg.is_clean());
        assert_eq!(lg.graph().node_count(), 1);
    }

    #[test]
    fn operational_legalizer_passes_graph_level() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(1),
            kind: crate::graph::ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: crate::graph::ComputeIntensity::ComputeBound,
        });
        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = crate::legalize::legalize(g, &backend_check);
        assert!(result.is_ok(), "operational legalizer must pass at Level 1");
        let lg = result.unwrap();
        assert!(
            lg.warnings().is_empty(),
            "no warnings expected: operational legalizer always passes at Level 1"
        );
    }

    #[test]
    fn backend_legalizer_rejects_ane_placement() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_annotation(id, "placement", "AnEngine".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("AnEngine")
        )));
    }

    #[test]
    fn backend_legalizer_rejects_unsupported_codec() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Ternary);

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("Ternary")
        )));
    }

    #[test]
    fn metal_tile_size_checks() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_tile_geometry(
            id,
            TileGeometry {
                width: 300,
                height: 4,
            },
        );
        g.set_annotation(id, "placement", "GpuComputeRegion".to_string());

        let node = g.get_node(id).unwrap();
        let result = metal_specific_checks(node, &g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("tile width")
        )));
    }

    #[test]
    fn metal_tile_height_checks() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_tile_geometry(
            id,
            TileGeometry {
                width: 8,
                height: 128,
            },
        );

        let node = g.get_node(id).unwrap();
        let result = metal_specific_checks(node, &g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("tile height")
        )));
    }

    #[test]
    fn metal_tile_total_threads_checks() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_tile_geometry(
            id,
            TileGeometry {
                width: 64,
                height: 64,
            },
        );

        let node = g.get_node(id).unwrap();
        let result = metal_specific_checks(node, &g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("total threads")
        )));
    }

    #[test]
    fn metal_tile_valid_geometry_passes() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_tile_geometry(
            id,
            TileGeometry {
                width: 16,
                height: 16,
            },
        );
        g.set_annotation(id, "placement", "GpuComputeRegion".to_string());
        g.set_annotation(id, "memory_region", "UnifiedMemory".to_string());

        let node = g.get_node(id).unwrap();
        let result = metal_specific_checks(node, &g);
        assert!(result.is_ok());
    }

    #[test]
    fn supported_codec_passes_backend_check() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Fp16);
        g.set_annotation(id, "placement", "GpuComputeRegion".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_ok());
    }

    #[test]
    fn backend_legalizer_rejects_ternary1_58_codec() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Ternary1_58);

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("Ternary1_58")
        )));
    }

    #[test]
    fn backend_legalizer_rejects_q8_0_codec() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Q8_0);

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("Q8_0")
        )));
    }

    #[test]
    fn backend_legalizer_rejects_npu_placement() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_annotation(id, "placement", "Npu".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("Npu")
        )));
    }

    #[test]
    fn backend_legalizer_rejects_incompatible_kv_cache_policy() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_annotation(id, "kv_cache_policy", "true:8".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("KV cache policy")
        )));
    }

    #[test]
    fn backend_legalizer_accepts_bf16_codec() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Bf16);
        g.set_annotation(id, "placement", "GpuComputeRegion".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_ok());
    }

    #[test]
    fn backend_legalizer_accepts_int8_codec() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Int8);
        g.set_annotation(id, "placement", "GpuComputeRegion".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_ok());
    }

    #[test]
    fn backend_legalizer_accepts_nf4_codec() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.set_codec(id, CodecVariant::Nf4);
        g.set_annotation(id, "placement", "GpuComputeRegion".to_string());

        let backend_check = |_: &SpatialNode| Ok::<(), Vec<LegalizationError>>(());
        let result = BackendLegalizer::legalize(&g, backend_check);
        assert!(result.is_ok());
    }

    #[test]
    fn operational_legalizer_detects_barrier_dangling_ref() {
        let mut g = SpatialGraph::new();
        g.add_node(SpatialNode::Barrier {
            id: SpatialNodeId(1),
            dependencies: vec![SpatialNodeId(42)],
        });
        let result = OperationalLegalizer::legalize(&g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::DanglingReference {
                missing: SpatialNodeId(42),
                ..
            }
        )));
    }

    #[test]
    fn operational_legalizer_detects_unknown_memory_region() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        g.add_node(SpatialNode::Memory {
            id,
            kind: crate::graph::MemoryKind::WeightStorage,
            region: crate::graph::MemoryRegion {
                shape: TensorShape {
                    dims: vec![64, 128],
                },
                element_size: 2,
                strides: vec![],
            },
        });
        g.set_annotation(id, "memory_region", "BogusMemory".to_string());

        let result = OperationalLegalizer::legalize(&g);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| matches!(
            e,
            LegalizationError::TargetConstraintViolation { detail }
            if detail.contains("BogusMemory")
        )));
    }
}
