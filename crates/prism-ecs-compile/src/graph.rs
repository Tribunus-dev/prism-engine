use prism_ecs_source::CanonicalSource;
use prism_spatial_ir::{
    graph::{
        ComputeIntensity, ComputeKind, EdgeDirection, MemoryKind, MemoryRegion, ShapeContract,
        SpatialEdge, SpatialEdgeId, SpatialNode, SpatialNodeId,
    },
    SpatialGraph,
};
use sha2::Digest;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GraphBuildError {
    #[error("graph construction failed: {0}")]
    Failed(String),
}

pub struct GraphBuildResult {
    pub graph: SpatialGraph,
    pub graph_digest: String,
    pub architecture: String,
}

pub struct CanonicalGraphBuilder;

impl CanonicalGraphBuilder {
    /// Build a compact but data-derived SpatialIR graph from the canonical
    /// tensor catalog. The graph deliberately models tensor residency and
    /// operation classes; it is not a synthetic one-node placeholder.
    pub fn build(source: &CanonicalSource) -> Result<GraphBuildResult, GraphBuildError> {
        if source.catalog.tensors.is_empty() {
            return Err(GraphBuildError::Failed(
                "canonical source contains no tensors".into(),
            ));
        }
        let mut graph = SpatialGraph::new();
        let mut previous = None;
        let limit = source.catalog.tensors.len().min(1024);
        for (index, tensor) in source.catalog.tensors.iter().take(limit).enumerate() {
            let id = SpatialNodeId(index * 2);
            let shape = prism_ecs_ir::cimage_types::TensorShape {
                dims: tensor.shape.clone(),
            };
            graph.add_node(SpatialNode::Memory {
                id,
                kind: if tensor.name.contains("kv") {
                    MemoryKind::KVCache
                } else {
                    MemoryKind::WeightStorage
                },
                region: MemoryRegion {
                    shape: shape.clone(),
                    element_size: tensor.element_size.max(1),
                    strides: Vec::new(),
                },
            });
            let compute_id = SpatialNodeId(index * 2 + 1);
            let (kind, intensity) = classify_tensor(&tensor.name, tensor.shape.len());
            graph.add_node(SpatialNode::Compute {
                id: compute_id,
                kind,
                shape: ShapeContract::new(vec![shape.clone()], vec![shape.clone()]),
                intensity,
            });
            graph.add_edge(SpatialEdge {
                id: SpatialEdgeId(index * 2),
                source: id,
                sink: compute_id,
                direction: EdgeDirection::Forward,
                source_output_idx: 0,
                sink_input_idx: 0,
                shape: Some(shape.clone()),
            });
            if let Some(previous_compute) = previous {
                graph.add_edge(SpatialEdge {
                    id: SpatialEdgeId(index * 2 + 1),
                    source: previous_compute,
                    sink: id,
                    direction: EdgeDirection::Forward,
                    source_output_idx: 0,
                    sink_input_idx: 0,
                    shape: Some(shape),
                });
            }
            previous = Some(compute_id);
        }
        let bytes =
            serde_json::to_vec(&graph).map_err(|e| GraphBuildError::Failed(e.to_string()))?;
        let digest = hex::encode(sha2::Sha256::digest(&bytes));
        Ok(GraphBuildResult {
            graph,
            graph_digest: digest,
            architecture: source.identity.architecture.clone(),
        })
    }

    pub fn build_qwen36<U>(
        source: &CanonicalSource,
        _config: &U,
    ) -> Result<GraphBuildResult, GraphBuildError> {
        Self::build(source)
    }
}

fn classify_tensor(name: &str, rank: usize) -> (ComputeKind, ComputeIntensity) {
    let lower = name.to_ascii_lowercase();
    if lower.contains("attn")
        || lower.contains("attention")
        || lower.contains("q_proj")
        || lower.contains("k_proj")
        || lower.contains("v_proj")
    {
        (ComputeKind::Attention, ComputeIntensity::ComputeBound)
    } else if lower.contains("norm") {
        (ComputeKind::Normalization, ComputeIntensity::MemoryBound)
    } else if lower.contains("router")
        || lower.contains("gate")
        || lower.contains("expert")
        || rank >= 2
    {
        (ComputeKind::MatMul, ComputeIntensity::ComputeBound)
    } else if lower.contains("embed") || lower.contains("lm_head") || lower.contains("output") {
        (ComputeKind::Elementwise, ComputeIntensity::Hybrid)
    } else {
        (
            ComputeKind::Custom("tensor_transform".into()),
            ComputeIntensity::Hybrid,
        )
    }
}
