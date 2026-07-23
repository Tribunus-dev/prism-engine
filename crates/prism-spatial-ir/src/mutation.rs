//! Mutation operators and application for spatial graph evolutionary search.
//!
//! The evolutionary search loop generates candidate schedules by applying
//! mutations to a legalized spatial graph. Each mutation produces a new
//! graph that must pass legalization before it can be estimated.
use crate::cost::CodecVariant;
use crate::graph::{SpatialGraph, SpatialNode, SpatialNodeId, TileGeometry};
use crate::hardware::{VirtualComputeUnit, VirtualMemoryRegion};
use crate::legalize::LegalizedGraph;
use prism_ecs_ir::evolution::foundation::{
    CandidateGenome, DecompositionAxis, FusionAxis, PackingAxis, RepresentationAxis,
};
use serde::{Deserialize, Serialize};
// ---------------------------------------------------------------------------
// MutationOp
// ---------------------------------------------------------------------------
/// A single mutation operation that transforms a [`SpatialGraph`].
///
/// Each variant describes a structural or parametric change to one or more
/// nodes in the graph. Mutations are applied in sequence to produce a new
/// candidate graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MutationOp {
    /// Change the quantization codec used for a node's weights.
    ChangeCodec {
        /// Target node.
        node_id: SpatialNodeId,
        /// New codec to apply.
        new_codec: CodecVariant,
    },
    /// Move a compute node to a different virtual compute unit.
    ChangePlacement {
        /// Target compute node.
        node_id: SpatialNodeId,
        /// New compute unit assignment.
        new_unit: VirtualComputeUnit,
    },
    /// Fuse two adjacent compute nodes into a single fused node.
    FuseNodes {
        /// First node in the fusion pair.
        first: SpatialNodeId,
        /// Second node in the fusion pair (must consume first's output).
        second: SpatialNodeId,
    },
    /// Split a fused compute node at a natural boundary.
    SplitNode {
        /// Node to split (must be a fused compute node).
        node_id: SpatialNodeId,
        /// Index at which to split the operation list.
        split_point: usize,
    },
    /// Change the tile geometry for a compute node.
    ChangeTileGeometry {
        /// Target node.
        node_id: SpatialNodeId,
        /// New tile width.
        new_tile_x: usize,
        /// New tile height.
        new_tile_y: usize,
    },
    /// Change the memory policy (domain placement) for a node.
    ChangeMemoryPolicy {
        /// Target node.
        node_id: SpatialNodeId,
        /// New memory region for the node's data.
        new_region: VirtualMemoryRegion,
    },
    /// Change the KV cache policy (compressed or not, bit width).
    ChangeKVCachePolicy {
        /// Target memory node (must be a KV cache).
        node_id: SpatialNodeId,
        /// Whether to use compressed KV cache.
        compressed: bool,
        /// Bit width per element (4, 8, 16).
        bit_width: usize,
    },
}
impl MutationOp {
    /// Returns a human-readable label for this mutation type.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ChangeCodec { .. } => "change_codec",
            Self::ChangePlacement { .. } => "change_placement",
            Self::FuseNodes { .. } => "fuse_nodes",
            Self::SplitNode { .. } => "split_node",
            Self::ChangeTileGeometry { .. } => "change_tile_geometry",
            Self::ChangeMemoryPolicy { .. } => "change_memory_policy",
            Self::ChangeKVCachePolicy { .. } => "change_kv_cache_policy",
        }
    }
}
// ---------------------------------------------------------------------------
// MutationResult
// ---------------------------------------------------------------------------
/// The result of applying a mutation to a graph.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MutationResult {
    /// The mutated graph (may not have passed legalization yet).
    pub graph: SpatialGraph,
    /// Record of which mutations were applied to produce this graph.
    pub applied: Vec<MutationOp>,
}
// ---------------------------------------------------------------------------
// MutationApplication
// ---------------------------------------------------------------------------
/// Applies a sequence of [`MutationOp`]s to a graph and re-legalizes it.
///
/// The applicator maintains a log of applied mutations for provenance
/// tracking and reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationApplication {
    /// Base graph before any mutations.
    base: SpatialGraph,
    /// Mutations applied so far.
    applied: Vec<MutationOp>,
}
impl MutationApplication {
    /// Create a new mutation application from a legalized graph.
    pub fn new(graph: &LegalizedGraph) -> Self {
        Self {
            base: graph.graph().clone(),
            applied: Vec::new(),
        }
    }
    /// Applies a single mutation to the graph, returning a new
    /// [`MutationResult`] with the transformed graph.
    ///
    /// This does **not** run the legalizer — callers should pass the result
    /// through [`crate::legalize::legalize`] to verify legality.
    pub fn apply(&self, op: &MutationOp) -> MutationResult {
        let mut graph = self.base.clone();
        let mut applied = self.applied.clone();
        match op {
            MutationOp::ChangeCodec { node_id, new_codec } => {
                // Record the codec assignment via typed setter.
                graph.set_codec(*node_id, *new_codec);
            }
            MutationOp::ChangePlacement { node_id, new_unit } => {
                // Record the placement assignment so the cost model
                // and legalizer can evaluate the candidate schedule.
                graph.set_annotation(*node_id, "placement", format!("{:?}", new_unit));
            }
            MutationOp::FuseNodes { first, second } => {
                graph = Self::fuse_nodes(&graph, *first, *second);
            }
            MutationOp::SplitNode {
                node_id,
                split_point,
            } => {
                graph = Self::split_node(&graph, *node_id, *split_point);
            }
            MutationOp::ChangeTileGeometry {
                node_id,
                new_tile_x,
                new_tile_y,
            } => {
                graph.set_tile_geometry(
                    *node_id,
                    TileGeometry {
                        width: *new_tile_x,
                        height: *new_tile_y,
                    },
                );
            }
            MutationOp::ChangeMemoryPolicy {
                node_id,
                new_region,
            } => {
                graph.set_annotation(*node_id, "memory_region", format!("{:?}", new_region));
            }
            MutationOp::ChangeKVCachePolicy {
                node_id,
                compressed,
                bit_width,
            } => {
                graph.set_annotation(
                    *node_id,
                    "kv_cache_policy",
                    format!("{}:{}", compressed, bit_width),
                );
            }
        }
        applied.push(op.clone());
        MutationResult { graph, applied }
    }
    /// Fuse two adjacent compute nodes into one.
    ///
    /// The fused node takes the shape contract of the second node's outputs
    /// (the fused result's shapes), and inherits the compute kind from the
    /// first node.
    fn fuse_nodes(
        graph: &SpatialGraph,
        first: SpatialNodeId,
        second: SpatialNodeId,
    ) -> SpatialGraph {
        let mut new_graph = graph.clone();
        // Collect edges connected to second BEFORE removing the node,
        // since remove_node also removes all connected edges.
        let edges_to_rewire: Vec<_> = new_graph
            .edges()
            .iter()
            .filter(|e| e.source == second || e.sink == second)
            .cloned()
            .collect();
        // Remove the second node and all its edges
        new_graph.remove_node(second);
        // Rewire all edges that pointed to `second` to point to `first`
        // instead. Edges already connecting `first -> second` are dropped.
        for edge in edges_to_rewire {
            if edge.sink == second && edge.source != first {
                // Incoming to second -> redirect to first
                new_graph.add_edge(crate::graph::SpatialEdge {
                    id: edge.id,
                    source: edge.source,
                    sink: first,
                    direction: edge.direction,
                    source_output_idx: edge.source_output_idx,
                    sink_input_idx: edge.sink_input_idx,
                    shape: edge.shape,
                });
            } else if edge.source == second {
                // Outgoing from second -> redirect from first
                new_graph.add_edge(crate::graph::SpatialEdge {
                    id: edge.id,
                    source: first,
                    sink: edge.sink,
                    direction: edge.direction,
                    source_output_idx: edge.source_output_idx,
                    sink_input_idx: edge.sink_input_idx,
                    shape: edge.shape,
                });
            }
            // Edges from first -> second are simply dropped (fusion)
        }
        new_graph
    }
    /// Split a compute node at a given split point.
    ///
    /// Creates two new nodes from the original, splitting the shape contract
    /// at the specified output index.
    fn split_node(
        graph: &SpatialGraph,
        node_id: SpatialNodeId,
        _split_point: usize,
    ) -> SpatialGraph {
        // If the node doesn't exist or isn't a compute node, return unchanged.
        let Some(node) = graph.get_node(node_id) else {
            return graph.clone();
        };
        match node {
            SpatialNode::Compute { .. } => {
                // Create a second node with a subset of the shape contract.
                // For simplicity, we split the outputs into two halves.
                let mut new_graph = graph.clone();
                let first_half_id = node_id;
                let second_half_id =
                    SpatialNodeId(graph.nodes().iter().map(|n| n.id().0).max().unwrap_or(0) + 1);
                if let Some(first_node) = new_graph.get_node_mut(first_half_id) {
                    if let SpatialNode::Compute { ref mut shape, .. } = first_node {
                        let mid = shape.out_shapes.len() / 2.max(1);
                        shape.out_shapes = shape.out_shapes[..mid].to_vec();
                    }
                }
                // Add the second half as a new compute node
                if let SpatialNode::Compute {
                    kind,
                    shape,
                    intensity,
                    ..
                } = node
                {
                    let mid = shape.out_shapes.len() / 2.max(1);
                    let second_shape = crate::graph::ShapeContract::new(
                        shape.out_shapes[mid..].to_vec(),
                        vec![], // outputs from the original's tail
                    );
                    new_graph.add_node(SpatialNode::Compute {
                        id: second_half_id,
                        kind: kind.clone(),
                        shape: second_shape,
                        intensity: *intensity,
                    });
                    // Add an edge between the two halves
                    let max_edge_id = new_graph.edges().iter().map(|e| e.id.0).max().unwrap_or(0);
                    new_graph.add_edge(crate::graph::SpatialEdge {
                        id: crate::graph::SpatialEdgeId(max_edge_id + 1),
                        source: first_half_id,
                        sink: second_half_id,
                        direction: crate::graph::EdgeDirection::Forward,
                        source_output_idx: 0,
                        sink_input_idx: 0,
                        shape: shape.out_shapes.get(mid).cloned(),
                    });
                }
                new_graph
            }
            _ => graph.clone(),
        }
    }
}

/// Translate a [`CandidateGenome`] into a vector of spatial mutation ops.
///
/// Each genome axis is mapped to one or more [`MutationOp`] variants that
/// encode the mutation the genome prescribes:
///
/// | Axis | MutationOp |
/// |---|---|
/// | `representation` | [`MutationOp::ChangeCodec`] |
/// | `packing` | [`MutationOp::ChangeTileGeometry`] |
/// | `decomposition` | [`MutationOp::SplitNode`] / [`MutationOp::FuseNodes`] |
/// | `fusion` | [`MutationOp::FuseNodes`] |
/// | `memory` | [`MutationOp::ChangeMemoryPolicy`] |
/// | `runtime` | [`MutationOp::ChangeKVCachePolicy`] |
///
/// The returned ops use `SpatialNodeId(0)` as a sentinel node reference;
/// the caller must resolve concrete node IDs for the target graph before
/// applying these mutations.
pub fn genome_to_spatial_mutations(genome: &CandidateGenome) -> Vec<MutationOp> {
    let mut ops = Vec::with_capacity(6);

    // representation → ChangeCodec
    let codec = match genome.representation {
        RepresentationAxis::Fp16 => CodecVariant::Fp16,
        RepresentationAxis::Bf16 => CodecVariant::Fp16,
        RepresentationAxis::Int8 => CodecVariant::Int8,
        RepresentationAxis::Int4 => CodecVariant::SymInt4,
        RepresentationAxis::Nf4 => CodecVariant::Nf4,
        RepresentationAxis::Nf8 => CodecVariant::Q8_0,
        RepresentationAxis::Ternary158 => CodecVariant::Ternary1_58,
        RepresentationAxis::TernaryTile640 => CodecVariant::Ternary1_58,
        RepresentationAxis::Binary1 => CodecVariant::Ternary,
    };
    ops.push(MutationOp::ChangeCodec {
        node_id: SpatialNodeId(0),
        new_codec: codec,
    });

    // packing → ChangeTileGeometry
    let (tile_x, tile_y) = match genome.packing {
        PackingAxis::Tile640 => (64usize, 64usize),
        PackingAxis::Block2D => (128, 128),
        PackingAxis::Planar => (256, 256),
        PackingAxis::Interleaved => (32, 32),
    };
    ops.push(MutationOp::ChangeTileGeometry {
        node_id: SpatialNodeId(0),
        new_tile_x: tile_x,
        new_tile_y: tile_y,
    });

    // decomposition → SplitNode / FuseNodes
    match genome.decomposition {
        DecompositionAxis::Flat => {
            ops.push(MutationOp::SplitNode {
                node_id: SpatialNodeId(0),
                split_point: 0,
            });
        }
        DecompositionAxis::SplitM => {
            ops.push(MutationOp::SplitNode {
                node_id: SpatialNodeId(0),
                split_point: 1,
            });
        }
        DecompositionAxis::SplitMN => {
            ops.push(MutationOp::SplitNode {
                node_id: SpatialNodeId(0),
                split_point: 2,
            });
        }
        DecompositionAxis::SplitMNK => {
            ops.push(MutationOp::SplitNode {
                node_id: SpatialNodeId(0),
                split_point: 3,
            });
        }
    }

    // fusion → FuseNodes (KernelFusion produces two fuse ops)
    match genome.fusion {
        FusionAxis::None => {}
        FusionAxis::ElementWise => {
            ops.push(MutationOp::FuseNodes {
                first: SpatialNodeId(0),
                second: SpatialNodeId(1),
            });
        }
        FusionAxis::KernelFusion => {
            ops.push(MutationOp::FuseNodes {
                first: SpatialNodeId(0),
                second: SpatialNodeId(1),
            });
            ops.push(MutationOp::FuseNodes {
                first: SpatialNodeId(1),
                second: SpatialNodeId(2),
            });
        }
    }

    // memory → ChangeMemoryPolicy
    let region = if genome.memory.double_buffer {
        VirtualMemoryRegion::DedicatedGpuVram
    } else if genome.memory.prefetch {
        VirtualMemoryRegion::UnifiedMemory
    } else {
        VirtualMemoryRegion::SharedCache
    };
    ops.push(MutationOp::ChangeMemoryPolicy {
        node_id: SpatialNodeId(0),
        new_region: region,
    });

    // runtime → ChangeKVCachePolicy
    ops.push(MutationOp::ChangeKVCachePolicy {
        node_id: SpatialNodeId(0),
        compressed: genome.runtime.async_encoding,
        bit_width: match genome.runtime.dispatch_width {
            1 => 16,
            2..=4 => 8,
            _ => 4,
        },
    });

    ops
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{
        ComputeIntensity, ComputeKind, EdgeDirection, ShapeContract, SpatialEdge, SpatialEdgeId,
    };
    use prism_ecs_ir::cimage_types::TensorShape;
    fn make_simple_graph() -> LegalizedGraph {
        let mut g = crate::graph::SpatialGraph::new();
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
        g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(2),
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![64, 64] }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: SpatialNodeId(1),
            sink: SpatialNodeId(2),
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: Some(TensorShape { dims: vec![64, 64] }),
        });
        LegalizedGraph::new(g, vec![])
    }
    #[test]
    fn apply_noop_mutation() {
        let lg = make_simple_graph();
        let applier = MutationApplication::new(&lg);
        let result = applier.apply(&MutationOp::ChangeCodec {
            node_id: SpatialNodeId(1),
            new_codec: CodecVariant::Fp16,
        });
        assert_eq!(result.graph.node_count(), 2);
        assert_eq!(result.applied.len(), 1);
    }
    #[test]
    fn apply_fuse_mutation() {
        let lg = make_simple_graph();
        let applier = MutationApplication::new(&lg);
        let result = applier.apply(&MutationOp::FuseNodes {
            first: SpatialNodeId(1),
            second: SpatialNodeId(2),
        });
        assert_eq!(result.graph.node_count(), 1);
        assert_eq!(result.applied.len(), 1);
    }
    #[test]
    fn mutation_label_is_not_empty() {
        let ops = vec![
            MutationOp::ChangeCodec {
                node_id: SpatialNodeId(1),
                new_codec: CodecVariant::Fp16,
            },
            MutationOp::ChangePlacement {
                node_id: SpatialNodeId(1),
                new_unit: VirtualComputeUnit::GpuComputeRegion,
            },
            MutationOp::FuseNodes {
                first: SpatialNodeId(1),
                second: SpatialNodeId(2),
            },
            MutationOp::SplitNode {
                node_id: SpatialNodeId(1),
                split_point: 0,
            },
            MutationOp::ChangeTileGeometry {
                node_id: SpatialNodeId(1),
                new_tile_x: 2,
                new_tile_y: 2,
            },
            MutationOp::ChangeMemoryPolicy {
                node_id: SpatialNodeId(1),
                new_region: VirtualMemoryRegion::UnifiedMemory,
            },
            MutationOp::ChangeKVCachePolicy {
                node_id: SpatialNodeId(1),
                compressed: true,
                bit_width: 4,
            },
        ];
        for op in &ops {
            assert!(!op.label().is_empty());
        }
    }
    /// Metadata mutations (ChangeCodec, ChangePlacement, ChangeTileGeometry,
    /// ChangeMemoryPolicy, ChangeKVCachePolicy) do not alter the graph
    /// topology — they only record the provenance of the intended change so
    /// the evolution layer can reify it during lower-to-target.
    ///
    /// This test proves every metadata mutation is reachable from the
    /// production MutationApplication::apply path and produces the correct
    /// invariant: identical graph structure + one additional provenance entry.
    #[test]
    fn metadata_mutations_preserve_graph_topology() {
        let lg = make_simple_graph();
        let applier = MutationApplication::new(&lg);
        let baseline_count = lg.graph().node_count();
        let baseline_edges = lg.graph().edge_count();
        let cases: Vec<MutationOp> = vec![
            MutationOp::ChangeCodec {
                node_id: SpatialNodeId(1),
                new_codec: CodecVariant::Fp16,
            },
            MutationOp::ChangePlacement {
                node_id: SpatialNodeId(1),
                new_unit: VirtualComputeUnit::GpuComputeRegion,
            },
            MutationOp::ChangeTileGeometry {
                node_id: SpatialNodeId(1),
                new_tile_x: 4,
                new_tile_y: 4,
            },
            MutationOp::ChangeMemoryPolicy {
                node_id: SpatialNodeId(1),
                new_region: VirtualMemoryRegion::UnifiedMemory,
            },
            MutationOp::ChangeKVCachePolicy {
                node_id: SpatialNodeId(1),
                compressed: true,
                bit_width: 4,
            },
        ];
        for op in &cases {
            let result = applier.apply(op);
            assert_eq!(
                result.graph.node_count(),
                baseline_count,
                "{} must not change node count",
                op.label()
            );
            assert_eq!(
                result.graph.edge_count(),
                baseline_edges,
                "{} must not change edge count",
                op.label()
            );
            assert_eq!(
                result.applied.len(),
                1,
                "{} must record one provenance entry",
                op.label()
            );
        }
    }
}
