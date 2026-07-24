//! Core spatial graph types — Level 1 of the SpatialIR representation.
//!
//! A [`SpatialGraph`] is an explicit, serializable dataflow graph with no
//! mention of specific hardware. Nodes represent compute, memory, streaming,
//! barriers, and repeated decoder regions. Edges carry tensor stream
//! contracts between nodes.

use crate::cost::CodecVariant;
use crate::fused_ops::{enumerate_fusion_candidates, FusableOp, FusedPermutation};
use prism_ecs_ir::cimage_types::TensorShape;
use serde::{Deserialize, Serialize};
// ---------------------------------------------------------------------------
// Node identifiers
// ---------------------------------------------------------------------------

/// Unique node identifier within a [`SpatialGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpatialNodeId(pub usize);

impl std::fmt::Display for SpatialNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique edge identifier within a [`SpatialGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpatialEdgeId(pub usize);

impl std::fmt::Display for SpatialEdgeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Shape contract
// ---------------------------------------------------------------------------

/// Describes the tensor shape contract between nodes in the spatial graph.
///
/// Every compute node declares its expected input and output shapes so
/// the legalizer can validate compatibility along every edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ShapeContract {
    /// Expected input tensor shapes.
    pub in_shapes: Vec<TensorShape>,
    /// Expected output tensor shapes.
    pub out_shapes: Vec<TensorShape>,
}

impl ShapeContract {
    /// Create a new shape contract.
    pub fn new(in_shapes: Vec<TensorShape>, out_shapes: Vec<TensorShape>) -> Self {
        Self {
            in_shapes,
            out_shapes,
        }
    }

    /// Returns `true` if both input and output shapes match another contract.
    pub fn is_compatible_with(&self, other: &ShapeContract) -> bool {
        self.in_shapes == other.in_shapes && self.out_shapes == other.out_shapes
    }
}

// ---------------------------------------------------------------------------
// Compute intensity classification
// ---------------------------------------------------------------------------

/// Classification of the compute intensity of a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeIntensity {
    /// Compute-bound: dominated by arithmetic (e.g., matmul, convolution).
    ComputeBound,
    /// Memory-bound: dominated by data movement (e.g., element-wise, softmax).
    MemoryBound,
    /// Hybrid: significant contributions from both compute and memory.
    Hybrid,
}

// ---------------------------------------------------------------------------
// Compute kind
// ---------------------------------------------------------------------------

/// Classifies the kind of computation a node performs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComputeKind {
    /// Matrix multiplication: C = A @ B.
    MatMul,
    /// Convolution (2D or 1D).
    Convolution,
    /// Elementwise operation (add, mul, relu, etc.).
    Elementwise,
    /// Normalization (layer norm, rms norm, batch norm).
    Normalization,
    /// Softmax or log-softmax.
    Softmax,
    /// Attention (QKV projection, score, reduce).
    Attention,
    /// RoPE (rotary position embedding).
    RoPE,
    /// SSM (state space model) step.
    SSM,
    /// Reshape / transpose / permute.
    Reshape,
    /// Gather / scatter.
    Gather,
    /// Custom user-defined operation.
    Custom(String),
}

// ---------------------------------------------------------------------------
// Memory kind
// ---------------------------------------------------------------------------

/// Classifies the role of a memory node in the spatial graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryKind {
    /// Persistent weight storage (model parameters).
    WeightStorage,
    /// KV cache storage.
    KVCache,
    /// Intermediate activation buffer.
    ActivationBuffer,
    /// Scratch / temporary workspace.
    Scratch,
    /// Input tensor buffer.
    InputBuffer,
    /// Output tensor buffer.
    OutputBuffer,
}

// ---------------------------------------------------------------------------
// Memory region
// ---------------------------------------------------------------------------

/// Describes the shape and layout of a memory region.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryRegion {
    /// Shape of the tensor stored in this region.
    pub shape: TensorShape,
    /// Size in bytes of each element.
    pub element_size: usize,
    /// Stride for each dimension (empty = contiguous).
    pub strides: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// A node in the spatial graph.
///
/// Each variant places the node within one of the spatial graph's categories:
/// compute, memory, streaming, barrier, or repeated decoder.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SpatialNode {
    /// A computation node — an operation or fused operation group.
    Compute {
        /// Unique node identifier.
        id: SpatialNodeId,
        /// Classification of the computation.
        kind: ComputeKind,
        /// Shape contract for inputs and outputs.
        shape: ShapeContract,
        /// Intensity classification (for placement heuristics).
        intensity: ComputeIntensity,
    },
    /// A memory node representing a tensor storage allocation.
    Memory {
        /// Unique node identifier.
        id: SpatialNodeId,
        /// Role of this memory allocation.
        kind: MemoryKind,
        /// Shape and layout of the tensor.
        region: MemoryRegion,
    },
    /// A streaming edge representing data movement between two nodes.
    Stream {
        /// Unique node identifier.
        id: SpatialNodeId,
        /// Source node producing the data.
        source: SpatialNodeId,
        /// Sink node consuming the data.
        sink: SpatialNodeId,
        /// Width of the stream in bytes per cycle (for bandwidth modelling).
        width_bytes: usize,
        /// Expected total bytes transferred.
        total_bytes: u64,
    },
    /// A synchronization barrier that all dependencies must clear before
    /// any dependent node may execute.
    Barrier {
        /// Unique node identifier.
        id: SpatialNodeId,
        /// Nodes that must complete before this barrier is satisfied.
        dependencies: Vec<SpatialNodeId>,
    },
    /// A repeated decoder region — a body of nodes that is unrolled or
    /// loop-scheduled a fixed number of times.
    RepeatedDecoder {
        /// Unique node identifier.
        id: SpatialNodeId,
        /// Nodes forming the body of the repeated region.
        body: Vec<SpatialNodeId>,
        /// How many times the body repeats.
        count: usize,
    },
}

impl SpatialNode {
    /// Returns the node's [`SpatialNodeId`].
    pub fn id(&self) -> SpatialNodeId {
        match self {
            Self::Compute { id, .. }
            | Self::Memory { id, .. }
            | Self::Stream { id, .. }
            | Self::Barrier { id, .. }
            | Self::RepeatedDecoder { id, .. } => *id,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// Direction of a spatial edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeDirection {
    /// Source produces, sink consumes.
    Forward,
    /// Backward connection (e.g., gradient flow).
    Backward,
}

/// An edge in the spatial graph connecting two nodes via a tensor stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpatialEdge {
    /// Unique edge identifier.
    pub id: SpatialEdgeId,
    /// Source (producer) node.
    pub source: SpatialNodeId,
    /// Sink (consumer) node.
    pub sink: SpatialNodeId,
    /// Direction of the edge.
    pub direction: EdgeDirection,
    /// Index of the output stream on the source node.
    pub source_output_idx: usize,
    /// Index of the input stream on the sink node.
    pub sink_input_idx: usize,
    /// Shape of the tensor flowing across this edge.
    pub shape: Option<TensorShape>,
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Tile geometry, fusion policy, and per-node mutation metadata
// ---------------------------------------------------------------------------

/// Tile geometry dimensions.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct TileGeometry {
    pub width: usize,
    pub height: usize,
}

/// Fusion operation policy.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FusionPolicy {
    /// No fusion — nodes remain unfused.
    #[default]
    Unfused,
    /// Nodes are fused into a single kernel.
    Fused,
    /// Fusion with a maximum depth limit.
    MaxDepth(u32),
}

/// Per-node metadata set by mutation operations during evolutionary search.
///
/// Codec, placement, tile geometry, memory policy, KV-cache policy, and
/// fusion mutations record their effect here so the cost model and legalizer
/// can read the assigned values.  Fields with dedicated types (codec,
/// tile_geometry, fusion) use their enum/struct types; placement,
/// memory_region, and kv_cache_policy remain as strings for flexibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct NodeMeta {
    /// The quantization codec variant assigned to this node's weights.
    pub codec: Option<CodecVariant>,
    /// Virtual compute unit this node is placed on.
    pub placement: Option<String>,
    /// Tile geometry dimensions (width x height).
    pub tile_geometry: Option<TileGeometry>,
    /// Memory region name.
    pub memory_region: Option<String>,
    /// KV-cache policy as "compressed:bit_width".
    pub kv_cache_policy: Option<String>,
    /// Fusion policy.
    pub fusion: Option<FusionPolicy>,
    /// Threadgroup size hint for batch-mode execution.
    pub batch_threadgroup_size: Option<u32>,
    /// Threadgroup size hint for realtime (autoregressive) execution.
    pub realtime_threadgroup_size: Option<u32>,
    /// Tensor key for format-plan lookup.
    pub tensor_key: Option<String>,
    /// Semantic operation for a generic `ComputeKind::Elementwise` node.
    ///
    /// Kept as a string for serialized graph compatibility; compiler
    /// frontends should use canonical names such as `add`, `mul`, or `relu`.
    pub elementwise_op: Option<String>,
    /// Compile-time scalar exponent for a candidate `pow` operation. The
    /// operation remains unadmitted until every frontend lowering path uses
    /// this field consistently.
    #[serde(default)]
    pub pow_exponent: Option<f32>,
    /// Axis permutation for layout-transforming custom operations.
    #[serde(default)]
    pub permutation: Option<Vec<usize>>,
    /// Semantic normalization operation such as `rms_norm` or `layer_norm`.
    pub normalization_op: Option<String>,
    /// Conv2D stride in both spatial dimensions.
    pub convolution_stride: Option<usize>,
    /// Conv2D zero-padding in both spatial dimensions.
    pub convolution_padding: Option<usize>,
}

// SpatialGraph
// ---------------------------------------------------------------------------

/// The central spatial dataflow graph — Level 1 of SpatialIR.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpatialGraph {
    nodes: Vec<SpatialNode>,
    edges: Vec<SpatialEdge>,
    entry_points: Vec<SpatialNodeId>,
    exit_points: Vec<SpatialNodeId>,
    /// Per-node mutation metadata.
    annotations: std::collections::HashMap<SpatialNodeId, NodeMeta>,
    /// Threadgroup size for batch-mode execution.
    ///
    /// When set, the lowering pass uses this as the total thread count
    /// for batch-mode dispatches. The tile geometry is derived by taking
    /// the square root of this value and clamping to valid Metal dimensions.
    pub batch_threadgroup_size: Option<u32>,
    /// Threadgroup size for realtime (autoregressive) execution.
    ///
    /// When set, the lowering pass uses this as the total thread count
    /// for realtime-mode dispatches, but the height is pinned to 1 for
    /// GEMV-oriented execution.
    pub realtime_threadgroup_size: Option<u32>,
}

impl SpatialGraph {
    /// Creates an empty spatial graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            entry_points: Vec::new(),
            exit_points: Vec::new(),
            annotations: std::collections::HashMap::new(),
            batch_threadgroup_size: None,
            realtime_threadgroup_size: None,
        }
    }
    pub fn nodes(&self) -> &[SpatialNode] {
        &self.nodes
    }

    /// Returns a mutable reference to all nodes.
    pub fn nodes_mut(&mut self) -> &mut Vec<SpatialNode> {
        &mut self.nodes
    }

    /// Returns a reference to all edges.
    pub fn edges(&self) -> &[SpatialEdge] {
        &self.edges
    }

    /// Returns a reference to entry point nodes.
    pub fn entry_points(&self) -> &[SpatialNodeId] {
        &self.entry_points
    }

    /// Returns a reference to exit point nodes.
    pub fn exit_points(&self) -> &[SpatialNodeId] {
        &self.exit_points
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns a reference to a node by its ID, or `None` if not found.
    pub fn get_node(&self, id: SpatialNodeId) -> Option<&SpatialNode> {
        self.nodes.iter().find(|n| n.id() == id)
    }

    /// Returns a mutable reference to a node by its ID.
    pub fn get_node_mut(&mut self, id: SpatialNodeId) -> Option<&mut SpatialNode> {
        self.nodes.iter_mut().find(|n| n.id() == id)
    }

    /// Adds a node to the graph, returning its ID.
    ///
    /// If this is the first node, it is automatically marked as both an entry
    /// and exit point.
    pub fn add_node(&mut self, node: SpatialNode) -> SpatialNodeId {
        let id = node.id();
        let is_first = self.nodes.is_empty();
        self.nodes.push(node);
        if is_first {
            self.entry_points.push(id);
            self.exit_points.push(id);
        }
        id
    }

    /// Adds an edge to the graph.
    pub fn add_edge(&mut self, edge: SpatialEdge) {
        // Update entry/exit points: the source is no longer an exit
        // (it now has outgoing data), and the sink is no longer an entry
        // (it now has incoming data).
        self.exit_points.retain(|e| *e != edge.source);
        self.entry_points.retain(|e| *e != edge.sink);
        // If source has no incoming edges and isn't already an entry, add it.
        if !self.entry_points.contains(&edge.source)
            && !self.edges.iter().any(|e| e.sink == edge.source)
        {
            self.entry_points.push(edge.source);
        }
        // If sink has no outgoing edges and isn't already an exit, add it.
        if !self.exit_points.contains(&edge.sink)
            && !self.edges.iter().any(|e| e.source == edge.sink)
        {
            self.exit_points.push(edge.sink);
        }
        self.edges.push(edge);
    }

    /// Returns edges connected to a given node (both incoming and outgoing).
    pub fn edges_for_node(&self, id: SpatialNodeId) -> Vec<&SpatialEdge> {
        self.edges
            .iter()
            .filter(|e| e.source == id || e.sink == id)
            .collect()
    }

    /// Returns a mutable reference to the edges vector.
    pub fn edges_mut(&mut self) -> &mut Vec<SpatialEdge> {
        &mut self.edges
    }

    /// Returns edges where the given node is the source (outgoing).
    pub fn outgoing_edges(&self, id: SpatialNodeId) -> Vec<&SpatialEdge> {
        self.edges.iter().filter(|e| e.source == id).collect()
    }

    /// Returns edges where the given node is the sink (incoming).
    pub fn incoming_edges(&self, id: SpatialNodeId) -> Vec<&SpatialEdge> {
        self.edges.iter().filter(|e| e.sink == id).collect()
    }

    /// Removes a node and all its connected edges from the graph.
    ///
    /// Returns the removed node if found.
    pub fn remove_node(&mut self, id: SpatialNodeId) -> Option<SpatialNode> {
        let pos = self.nodes.iter().position(|n| n.id() == id)?;
        self.edges.retain(|e| e.source != id && e.sink != id);
        self.entry_points.retain(|e| *e != id);
        self.exit_points.retain(|e| *e != id);
        Some(self.nodes.remove(pos))
    }

    /// Performs a simple topological sort of the node IDs.
    ///
    /// Uses Kahn's algorithm. Returns `None` if the graph has a cycle.
    pub fn topological_sort(&self) -> Option<Vec<SpatialNodeId>> {
        use std::collections::{HashMap, VecDeque};

        let mut in_degree: HashMap<SpatialNodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<SpatialNodeId, Vec<SpatialNodeId>> = HashMap::new();

        for node in &self.nodes {
            let id = node.id();
            in_degree.entry(id).or_insert(0);
            adjacency.entry(id).or_default();
        }
        for edge in &self.edges {
            adjacency.entry(edge.source).or_default().push(edge.sink);
            *in_degree.entry(edge.sink).or_insert(0) += 1;
        }

        let mut queue: VecDeque<SpatialNodeId> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut sorted = Vec::with_capacity(self.nodes.len());
        while let Some(id) = queue.pop_front() {
            sorted.push(id);
            if let Some(neighbors) = adjacency.get(&id) {
                for &next in neighbors {
                    if let Some(deg) = in_degree.get_mut(&next) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(next);
                        }
                    }
                }
            }
        }

        if sorted.len() == self.nodes.len() {
            Some(sorted)
        } else {
            None // cycle detected
        }
    }

    /// Serializes the graph to a JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn annotations(&self) -> &std::collections::HashMap<SpatialNodeId, NodeMeta> {
        &self.annotations
    }

    /// Get annotations for a specific node.
    pub fn get_annotations(&self, id: SpatialNodeId) -> Option<&NodeMeta> {
        self.annotations.get(&id)
    }

    /// Set a metadata value on a node.
    pub fn set_annotation(&mut self, id: SpatialNodeId, key: &str, value: String) {
        let meta = self.annotations.entry(id).or_default();
        match key {
            "placement" => meta.placement = Some(value),
            "memory_region" => meta.memory_region = Some(value),
            "kv_cache_policy" => meta.kv_cache_policy = Some(value),
            "batch_threadgroup_size" => {
                self.batch_threadgroup_size = value.parse::<u32>().ok();
            }
            "realtime_threadgroup_size" => {
                self.realtime_threadgroup_size = value.parse::<u32>().ok();
            }
            "tensor_key" => meta.tensor_key = Some(value),
            "elementwise_op" => meta.elementwise_op = Some(value),
            "normalization_op" => meta.normalization_op = Some(value),
            "convolution_stride" => meta.convolution_stride = value.parse().ok(),
            "convolution_padding" => meta.convolution_padding = value.parse().ok(),
            _ => {}
        }
    }

    /// Set the codec variant annotation for a node.
    pub fn set_codec(&mut self, id: SpatialNodeId, codec: CodecVariant) {
        self.annotations.entry(id).or_default().codec = Some(codec);
    }

    /// Set the tile geometry annotation for a node.
    pub fn set_tile_geometry(&mut self, id: SpatialNodeId, tg: TileGeometry) {
        self.annotations.entry(id).or_default().tile_geometry = Some(tg);
    }

    /// Set the fusion policy annotation for a node.
    pub fn set_fusion(&mut self, id: SpatialNodeId, fusion: FusionPolicy) {
        self.annotations.entry(id).or_default().fusion = Some(fusion);
    }

    /// Deserializes a graph from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Enumerate legal fusion candidates for each matmul-anchored region
    /// of this graph.
    ///
    /// Walks the graph in topological order and for every compute node that
    /// performs a matrix-vector multiply, produces every legal fused
    /// permutation reachable via its element-wise successors.
    ///
    /// A "region" is a matmul followed by zero or more element-wise nodes
    /// along forward dataflow edges. Returns all fusion candidates that are
    /// structurally valid for the graph, independent of tile geometry or
    /// backend constraints (those are enforced by the legalizer later).
    ///
    /// Returns a vector of `(matmul_node_id, candidates)` pairs — one entry
    /// per matmul node with at least one valid fusion. Empty means no
    /// matmul nodes were found or the graph has a cycle.
    pub fn available_fusions(&self) -> Vec<(SpatialNodeId, Vec<FusedPermutation>)> {
        use crate::graph::ComputeKind;

        let topo = match self.topological_sort() {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut results: Vec<(SpatialNodeId, Vec<FusedPermutation>)> = Vec::new();

        for &node_id in &topo {
            let node = match self.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };

            // Only compute nodes can anchor a fusion.
            let kind = match node {
                SpatialNode::Compute { kind, .. } => kind,
                _ => continue,
            };

            // Map the node's ComputeKind to a FusableOp matmul variant.
            let matmul_op = match kind {
                ComputeKind::MatMul => {
                    // Check for ternary codec annotation.
                    if let Some(meta) = self.get_annotations(node_id) {
                        match meta.codec {
                            Some(crate::cost::CodecVariant::Ternary)
                            | Some(crate::cost::CodecVariant::Ternary1_58) => {
                                FusableOp::TernaryGemv
                            }
                            _ => FusableOp::FpGemv,
                        }
                    } else {
                        FusableOp::FpGemv
                    }
                }
                _ => continue,
            };

            // Collect unique element-wise ops among direct successors of
            // this matmul via forward edges.
            let mut successor_ops: Vec<FusableOp> = Vec::new();
            for edge in self.outgoing_edges(node_id) {
                let sink_node = match self.get_node(edge.sink) {
                    Some(n) => n,
                    None => continue,
                };
                let ew_op = match sink_node {
                    SpatialNode::Compute { kind, .. } => match kind {
                        ComputeKind::Elementwise => {
                            // Generic element-wise could be SiLU (activation)
                            // or a residual add.  Emit both; the fusion
                            // enumerator will produce candidates that use
                            // either, and the topological filter below
                            // keeps only those ops the graph actually
                            // supports.
                            if !successor_ops.contains(&FusableOp::Silu) {
                                successor_ops.push(FusableOp::Silu);
                            }
                            if !successor_ops.contains(&FusableOp::ElementWiseAdd) {
                                successor_ops.push(FusableOp::ElementWiseAdd);
                            }
                            continue;
                        }
                        ComputeKind::Normalization => FusableOp::RmsNorm,
                        ComputeKind::RoPE => FusableOp::Rope,
                        _ => continue,
                    },
                    _ => continue,
                };
                if !successor_ops.contains(&ew_op) {
                    successor_ops.push(ew_op);
                }
            }

            // Enumerate all fusion candidates and filter by graph topology.
            let all_candidates = enumerate_fusion_candidates(matmul_op);
            let mut filtered: Vec<FusedPermutation> = Vec::new();
            for perm in &all_candidates {
                let post_matmul = &perm.ops[1..];
                if post_matmul.is_empty() {
                    // Degenerate single-op kernel — always available.
                    filtered.push(perm.clone());
                    continue;
                }
                // Every post-matmul op must appear in the successor set,
                // preserving relative order.
                let mut si = 0;
                let mut matched = true;
                for &op in post_matmul {
                    while si < successor_ops.len() && successor_ops[si] != op {
                        si += 1;
                    }
                    if si >= successor_ops.len() {
                        matched = false;
                        break;
                    }
                    si += 1;
                }
                if matched {
                    filtered.push(perm.clone());
                }
            }

            if !filtered.is_empty() {
                results.push((node_id, filtered));
            }
        }

        results
    }

    /// Evaluate every legal fusion region using the output size of its
    /// anchor node. Runtime measurements can be supplied later through
    /// [`crate::fused_ops::evaluate_fusion_strategies_with_measurements`].
    pub fn evaluate_available_fusions(
        &self,
    ) -> Vec<(
        SpatialNodeId,
        Vec<crate::fused_ops::FusionStrategyEvaluation>,
    )> {
        self.evaluate_available_fusions_with_generation(0)
    }

    /// Evaluate graph fusion regions while preserving the evolutionary
    /// generation that produced the current graph candidate.
    pub fn evaluate_available_fusions_with_generation(
        &self,
        search_generation: u32,
    ) -> Vec<(
        SpatialNodeId,
        Vec<crate::fused_ops::FusionStrategyEvaluation>,
    )> {
        self.available_fusions()
            .into_iter()
            .map(|(node_id, permutations)| {
                let element_count = self
                    .get_node(node_id)
                    .and_then(|node| match node {
                        SpatialNode::Compute { shape, .. } => shape.out_shapes.first(),
                        _ => None,
                    })
                    .map(|shape| {
                        shape
                            .dims
                            .iter()
                            .try_fold(1u64, |count, dimension| {
                                count.checked_mul(*dimension as u64)
                            })
                            .unwrap_or(0)
                    })
                    .unwrap_or(0);
                (
                    node_id,
                    permutations
                        .iter()
                        .map(|permutation| {
                            crate::fused_ops::evaluate_fusion_strategies_with_generation(
                                permutation,
                                element_count,
                                search_generation,
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }
}

impl Default for SpatialGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Node location within graph
// ---------------------------------------------------------------------------

/// Location of a node within the graph, used for mutation targeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeLocation {
    /// The node's ID.
    pub node_id: SpatialNodeId,
    /// Index in the node vector (for stable access).
    pub index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_roundtrips_serde() {
        let g = SpatialGraph::new();
        let json = g.to_json().expect("serialize");
        let restored = SpatialGraph::from_json(&json).expect("deserialize");
        assert_eq!(g, restored);
    }

    #[test]
    fn add_compute_node() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(1);
        let node = SpatialNode::Compute {
            id,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![TensorShape {
                    dims: vec![64, 128],
                }],
                vec![TensorShape { dims: vec![64, 64] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        };
        let added_id = g.add_node(node);
        assert_eq!(added_id, id);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.entry_points(), &[id]);
        assert_eq!(g.exit_points(), &[id]);
    }

    #[test]
    fn add_compute_with_edge() {
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
            shape: Some(TensorShape { dims: vec![64, 64] }),
        });
        assert_eq!(g.edge_count(), 1);
        assert_eq!(g.entry_points(), &[id_a]);
        assert_eq!(g.exit_points(), &[id_b]);
    }

    #[test]
    fn topological_sort_simple() {
        let mut g = SpatialGraph::new();
        let id_a = SpatialNodeId(1);
        let id_b = SpatialNodeId(2);
        let id_c = SpatialNodeId(3);
        for &id in &[id_a, id_b, id_c] {
            g.add_node(SpatialNode::Compute {
                id,
                kind: ComputeKind::Elementwise,
                shape: ShapeContract::new(
                    vec![TensorShape { dims: vec![64, 64] }],
                    vec![TensorShape { dims: vec![64, 64] }],
                ),
                intensity: ComputeIntensity::ComputeBound,
            });
        }
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(1),
            source: id_a,
            sink: id_b,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(2),
            source: id_b,
            sink: id_c,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        let sorted = g.topological_sort().expect("no cycles");
        assert_eq!(sorted.len(), 3);
        // a before b before c
        let pos_a = sorted.iter().position(|&x| x == id_a).unwrap();
        let pos_b = sorted.iter().position(|&x| x == id_b).unwrap();
        let pos_c = sorted.iter().position(|&x| x == id_c).unwrap();
        assert!(pos_a < pos_b && pos_b < pos_c);
    }

    #[test]
    fn serde_roundtrip_with_graph() {
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
        g.add_node(SpatialNode::Memory {
            id: id_b,
            kind: MemoryKind::WeightStorage,
            region: MemoryRegion {
                shape: TensorShape {
                    dims: vec![64, 128],
                },
                element_size: 2,
                strides: vec![],
            },
        });
        let json = g.to_json().expect("serialize");
        let restored = SpatialGraph::from_json(&json).expect("deserialize");
        assert_eq!(g, restored);
    }

    #[test]
    fn normalization_annotation_roundtrips() {
        let mut g = SpatialGraph::new();
        let id = SpatialNodeId(7);
        g.add_node(SpatialNode::Memory {
            id,
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![1, 2] },
                element_size: 4,
                strides: vec![],
            },
        });
        g.set_annotation(id, "normalization_op", "layer_norm".into());
        let restored = SpatialGraph::from_json(&g.to_json().unwrap()).unwrap();
        assert_eq!(
            restored
                .get_annotations(id)
                .unwrap()
                .normalization_op
                .as_deref(),
            Some("layer_norm")
        );
    }

    #[test]
    fn remove_node_clears_edges() {
        let mut g = SpatialGraph::new();
        let id_a = SpatialNodeId(1);
        let id_b = SpatialNodeId(2);
        g.add_node(SpatialNode::Compute {
            id: id_a,
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(vec![], vec![]),
            intensity: ComputeIntensity::ComputeBound,
        });
        g.add_node(SpatialNode::Compute {
            id: id_b,
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(vec![], vec![]),
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
        assert!(g.remove_node(id_a).is_some());
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn graph_fusion_discovery_produces_workload_evaluations() {
        let mut g = SpatialGraph::new();
        let lhs = g.add_node(SpatialNode::Memory {
            id: SpatialNodeId(10),
            kind: MemoryKind::InputBuffer,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![2, 2] },
                element_size: 4,
                strides: vec![],
            },
        });
        let rhs = g.add_node(SpatialNode::Memory {
            id: SpatialNodeId(11),
            kind: MemoryKind::WeightStorage,
            region: MemoryRegion {
                shape: TensorShape { dims: vec![2, 2] },
                element_size: 4,
                strides: vec![],
            },
        });
        let matmul = g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(12),
            kind: ComputeKind::MatMul,
            shape: ShapeContract::new(
                vec![
                    TensorShape { dims: vec![2, 2] },
                    TensorShape { dims: vec![2, 2] },
                ],
                vec![TensorShape { dims: vec![2, 2] }],
            ),
            intensity: ComputeIntensity::ComputeBound,
        });
        let activation = g.add_node(SpatialNode::Compute {
            id: SpatialNodeId(13),
            kind: ComputeKind::Elementwise,
            shape: ShapeContract::new(
                vec![TensorShape { dims: vec![2, 2] }],
                vec![TensorShape { dims: vec![2, 2] }],
            ),
            intensity: ComputeIntensity::MemoryBound,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(10),
            source: lhs,
            sink: matmul,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(11),
            source: rhs,
            sink: matmul,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 1,
            shape: None,
        });
        g.add_edge(SpatialEdge {
            id: SpatialEdgeId(12),
            source: matmul,
            sink: activation,
            direction: EdgeDirection::Forward,
            source_output_idx: 0,
            sink_input_idx: 0,
            shape: None,
        });
        let evaluations = g.evaluate_available_fusions();
        assert_eq!(evaluations.len(), 1);
        assert_eq!(evaluations[0].0, matmul);
        assert!(!evaluations[0].1.is_empty());
        assert!(evaluations[0]
            .1
            .iter()
            .all(|evaluation| evaluation.candidates.len() == 4));
        let generated = g.evaluate_available_fusions_with_generation(11);
        assert!(generated[0].1.iter().any(|evaluation| {
            evaluation.candidates.iter().any(|candidate| {
                matches!(
                    candidate.strategy,
                    crate::fused_ops::FusionStrategy::PersistentMegakernel {
                        search_generation: 11
                    }
                )
            })
        }));
    }
}
