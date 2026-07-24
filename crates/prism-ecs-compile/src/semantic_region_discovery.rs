//! Static semantic-region discovery for graph- and architecture-derived partitions.

use prism_ecs_ir::evolution::foundation::LogicalTensorId;
use prism_ecs_ir::semantic_region::{
    RegionConstraints, RegionOrigin, RegionRole, RegionSelector, SemanticRegionDescriptor,
    SemanticRegionId, SemanticRegionPartition,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicalTensorDescriptor {
    pub id: LogicalTensorId,
    pub name: String,
    pub shape: Vec<u64>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticModelConfig {
    pub model_family: String,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub head_dim: Option<u32>,
    pub intermediate_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphRegionHint {
    AxisSplit {
        operation: String,
        source_value: String,
        axis: u32,
        spans: Vec<(u64, u64, RegionRole)>,
    },
}

#[derive(Debug, Error)]
pub enum SemanticDiscoveryError {
    #[error("tensor shape is empty")]
    EmptyShape,
    #[error("semantic discovery produced no regions")]
    NoRegions,
    #[error("invalid fused QKV dimensions")]
    InvalidQkvDimensions,
    #[error("invalid gate/up dimensions")]
    InvalidGateUpDimensions,
    #[error(transparent)]
    InvalidPartition(#[from] prism_ecs_ir::semantic_region::SemanticRegionError),
}

pub trait SemanticRegionDiscoverer: Send + Sync {
    fn discover(
        &self,
        tensor: &LogicalTensorDescriptor,
        model: &SemanticModelConfig,
        graph_hints: &[GraphRegionHint],
    ) -> Result<Vec<SemanticRegionDescriptor>, SemanticDiscoveryError>;
}

#[derive(Debug, Default)]
pub struct GraphExplicitDiscoverer;

impl SemanticRegionDiscoverer for GraphExplicitDiscoverer {
    fn discover(
        &self,
        tensor: &LogicalTensorDescriptor,
        _model: &SemanticModelConfig,
        graph_hints: &[GraphRegionHint],
    ) -> Result<Vec<SemanticRegionDescriptor>, SemanticDiscoveryError> {
        let mut regions = Vec::new();
        for hint in graph_hints {
            match hint {
                GraphRegionHint::AxisSplit {
                    operation,
                    source_value,
                    axis,
                    spans,
                } => {
                    for (start, end, role) in spans {
                        regions.push(descriptor(
                            tensor,
                            *axis,
                            *start,
                            *end,
                            role.clone(),
                            RegionOrigin::GraphDerived {
                                operation: operation.clone(),
                                source_value: source_value.clone(),
                            },
                            vec![
                                format!("graph-op:{operation}"),
                                format!("source:{source_value}"),
                            ],
                        ));
                    }
                }
            }
        }
        Ok(regions)
    }
}

#[derive(Debug, Default)]
pub struct ArchitectureDiscoverer;

impl SemanticRegionDiscoverer for ArchitectureDiscoverer {
    fn discover(
        &self,
        tensor: &LogicalTensorDescriptor,
        model: &SemanticModelConfig,
        _graph_hints: &[GraphRegionHint],
    ) -> Result<Vec<SemanticRegionDescriptor>, SemanticDiscoveryError> {
        if tensor.shape.is_empty() {
            return Err(SemanticDiscoveryError::EmptyShape);
        }
        let lower = tensor.name.to_ascii_lowercase();
        if lower.contains("qkv") || lower.contains("query_key_value") {
            return discover_qkv(tensor, model);
        }
        if lower.contains("gate_up") || lower.contains("gateup") {
            return discover_gate_up(tensor, model);
        }
        Ok(Vec::new())
    }
}

pub fn discover_semantic_partition(
    tensor: &LogicalTensorDescriptor,
    model: &SemanticModelConfig,
    graph_hints: &[GraphRegionHint],
) -> Result<SemanticRegionPartition, SemanticDiscoveryError> {
    if tensor.shape.is_empty() {
        return Err(SemanticDiscoveryError::EmptyShape);
    }
    let graph = GraphExplicitDiscoverer.discover(tensor, model, graph_hints)?;
    let regions = if graph.is_empty() {
        ArchitectureDiscoverer.discover(tensor, model, graph_hints)?
    } else {
        graph
    };
    let regions = if regions.is_empty() {
        vec![descriptor(
            tensor,
            0,
            0,
            tensor.shape[0],
            RegionRole::Generic {
                label: "whole_tensor".into(),
            },
            RegionOrigin::ArchitectureDerived {
                model_family: model.model_family.clone(),
                rule: "whole-tensor-fallback".into(),
            },
            vec!["fallback:whole-tensor".into()],
        )]
    } else {
        regions
    };
    Ok(SemanticRegionPartition {
        parent: tensor.id.clone(),
        parent_shape: tensor.shape.clone(),
        regions,
        exhaustive: true,
        disjoint: true,
        digest: String::new(),
    }
    .seal()?)
}

fn discover_qkv(
    tensor: &LogicalTensorDescriptor,
    model: &SemanticModelConfig,
) -> Result<Vec<SemanticRegionDescriptor>, SemanticDiscoveryError> {
    let heads = model
        .num_attention_heads
        .ok_or(SemanticDiscoveryError::InvalidQkvDimensions)? as u64;
    let kv_heads = model.num_key_value_heads.unwrap_or(heads as u32) as u64;
    let head_dim = model
        .head_dim
        .ok_or(SemanticDiscoveryError::InvalidQkvDimensions)? as u64;
    let q = heads.saturating_mul(head_dim);
    let k = kv_heads.saturating_mul(head_dim);
    let v = k;
    if q == 0 || k == 0 || q + k + v != tensor.shape[0] {
        return Err(SemanticDiscoveryError::InvalidQkvDimensions);
    }
    let origin = |rule: &str| RegionOrigin::ArchitectureDerived {
        model_family: model.model_family.clone(),
        rule: rule.into(),
    };
    Ok(vec![
        descriptor(
            tensor,
            0,
            0,
            q,
            RegionRole::QueryProjection,
            origin("fused-qkv-gqa"),
            vec![
                "config:num_attention_heads".into(),
                "config:head_dim".into(),
            ],
        ),
        descriptor(
            tensor,
            0,
            q,
            q + k,
            RegionRole::KeyProjection,
            origin("fused-qkv-gqa"),
            vec![
                "config:num_key_value_heads".into(),
                "config:head_dim".into(),
            ],
        ),
        descriptor(
            tensor,
            0,
            q + k,
            q + k + v,
            RegionRole::ValueProjection,
            origin("fused-qkv-gqa"),
            vec![
                "config:num_key_value_heads".into(),
                "config:head_dim".into(),
            ],
        ),
    ])
}

fn discover_gate_up(
    tensor: &LogicalTensorDescriptor,
    model: &SemanticModelConfig,
) -> Result<Vec<SemanticRegionDescriptor>, SemanticDiscoveryError> {
    let intermediate = model
        .intermediate_size
        .ok_or(SemanticDiscoveryError::InvalidGateUpDimensions)?;
    if intermediate == 0 || intermediate.saturating_mul(2) != tensor.shape[0] {
        return Err(SemanticDiscoveryError::InvalidGateUpDimensions);
    }
    let origin = |rule: &str| RegionOrigin::ArchitectureDerived {
        model_family: model.model_family.clone(),
        rule: rule.into(),
    };
    Ok(vec![
        descriptor(
            tensor,
            0,
            0,
            intermediate,
            RegionRole::GateProjection,
            origin("fused-gate-up"),
            vec!["config:intermediate_size".into()],
        ),
        descriptor(
            tensor,
            0,
            intermediate,
            intermediate * 2,
            RegionRole::UpProjection,
            origin("fused-gate-up"),
            vec!["config:intermediate_size".into()],
        ),
    ])
}

fn descriptor(
    tensor: &LogicalTensorDescriptor,
    axis: u32,
    start: u64,
    end: u64,
    role: RegionRole,
    origin: RegionOrigin,
    provenance_refs: Vec<String>,
) -> SemanticRegionDescriptor {
    let role_label = format!("{role:?}");
    let mut hasher = Sha256::new();
    hasher.update(tensor.id.0.as_bytes());
    hasher.update(axis.to_le_bytes());
    hasher.update(start.to_le_bytes());
    hasher.update(end.to_le_bytes());
    hasher.update(role_label.as_bytes());
    let id = format!(
        "sr:{}:axis:{axis}:{start}:{end}:{:x}",
        short_digest(&tensor.id.0),
        hasher.finalize()
    );
    SemanticRegionDescriptor {
        id: SemanticRegionId(id),
        parent: tensor.id.clone(),
        selector: RegionSelector::AxisSpan { axis, start, end },
        role,
        origin,
        constraints: RegionConstraints {
            allowed_formats: vec![
                "fp16".into(),
                "bf16".into(),
                "int8".into(),
                "nf4".into(),
                "ternary158".into(),
            ],
            ..RegionConstraints::default()
        },
        provenance_refs,
    }
}

fn short_digest(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    format!("{:x}", hasher.finalize())[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, rows: u64) -> LogicalTensorDescriptor {
        LogicalTensorDescriptor {
            id: LogicalTensorId(name.into()),
            name: name.into(),
            shape: vec![rows, 2048],
        }
    }

    #[test]
    fn discovers_gqa_qkv_without_equal_thirds() {
        let model = SemanticModelConfig {
            model_family: "qwen".into(),
            num_attention_heads: Some(32),
            num_key_value_heads: Some(8),
            head_dim: Some(128),
            intermediate_size: None,
        };
        let p = discover_semantic_partition(&tensor("layer.qkv_proj.weight", 6144), &model, &[])
            .unwrap();
        assert_eq!(p.regions.len(), 3);
        assert!(matches!(
            p.regions[1].selector,
            RegionSelector::AxisSpan {
                start: 4096,
                end: 5120,
                ..
            }
        ));
    }

    #[test]
    fn discovers_gate_up_split() {
        let model = SemanticModelConfig {
            model_family: "generic".into(),
            intermediate_size: Some(4096),
            ..Default::default()
        };
        let p = discover_semantic_partition(&tensor("mlp.gate_up_proj.weight", 8192), &model, &[])
            .unwrap();
        assert!(matches!(p.regions[1].role, RegionRole::UpProjection));
    }

    #[test]
    fn graph_hint_precedes_architecture_rule() {
        let hints = vec![GraphRegionHint::AxisSplit {
            operation: "split-17".into(),
            source_value: "v42".into(),
            axis: 0,
            spans: vec![
                (0, 2, RegionRole::QueryProjection),
                (2, 4, RegionRole::KeyProjection),
            ],
        }];
        let p = discover_semantic_partition(
            &tensor("custom.weight", 4),
            &SemanticModelConfig::default(),
            &hints,
        )
        .unwrap();
        assert!(matches!(
            p.regions[0].origin,
            RegionOrigin::GraphDerived { .. }
        ));
    }

    #[test]
    fn unsupported_tensor_falls_back_to_whole_tensor() {
        let p = discover_semantic_partition(
            &tensor("norm.weight", 2048),
            &SemanticModelConfig::default(),
            &[],
        )
        .unwrap();
        assert_eq!(p.regions.len(), 1);
    }
}
