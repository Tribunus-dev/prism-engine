//! Versioned explicit semantic-region specifications and compile-verified receipts.

use prism_ecs_ir::evolution::foundation::LogicalTensorId;
use prism_ecs_ir::semantic_region::{
    RegionConstraints, RegionOrigin, RegionRole, RegionSelector, SemanticRegionDescriptor,
    SemanticRegionError, SemanticRegionId, SemanticRegionPartition,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

pub const SEMANTIC_REGION_SPEC_V1: &str = "prism.semantic-regions.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionSpec {
    pub schema: String,
    pub tensor: String,
    pub shape: Vec<u64>,
    pub regions: Vec<SemanticRegionSpecEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionSpecEntry {
    pub role: String,
    pub axis: u32,
    pub start: u64,
    pub end: u64,
    #[serde(default)]
    pub allowed_formats: Vec<String>,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionDiscoveryReceipt {
    pub schema: String,
    pub tensor: String,
    pub tensor_shape: Vec<u64>,
    pub partition_digest: String,
    pub region_count: usize,
    pub coverage: String,
    pub overlap: String,
    pub boundary_evidence: String,
    pub legality: String,
    pub numerical_quality: String,
    pub execution_performance: String,
}

#[derive(Debug, Error)]
pub enum SemanticRegionSpecError {
    #[error("read semantic-region spec: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse semantic-region spec: {0}")]
    Json(#[from] serde_json::Error),
    #[error("unsupported semantic-region schema: {0}")]
    UnsupportedSchema(String),
    #[error("spec tensor {expected} does not match requested tensor {actual}")]
    TensorMismatch { expected: String, actual: String },
    #[error("spec shape {expected:?} does not match mapped tensor shape {actual:?}")]
    ShapeMismatch { expected: Vec<u64>, actual: Vec<u64> },
    #[error("unsupported semantic region role: {0}")]
    UnsupportedRole(String),
    #[error(transparent)]
    Region(#[from] SemanticRegionError),
}

impl SemanticRegionSpec {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SemanticRegionSpecError> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    pub fn into_partition(
        self,
        requested_tensor: &str,
        mapped_shape: &[u64],
        source_label: &str,
    ) -> Result<(SemanticRegionPartition, SemanticRegionDiscoveryReceipt), SemanticRegionSpecError> {
        if self.schema != SEMANTIC_REGION_SPEC_V1 {
            return Err(SemanticRegionSpecError::UnsupportedSchema(self.schema));
        }
        if self.tensor != requested_tensor {
            return Err(SemanticRegionSpecError::TensorMismatch {
                expected: self.tensor,
                actual: requested_tensor.to_string(),
            });
        }
        if self.shape != mapped_shape {
            return Err(SemanticRegionSpecError::ShapeMismatch {
                expected: self.shape,
                actual: mapped_shape.to_vec(),
            });
        }
        let parent = LogicalTensorId(requested_tensor.to_string());
        let mut regions = Vec::with_capacity(self.regions.len());
        for entry in self.regions {
            let role = parse_role(&entry.role)?;
            let formats = if entry.allowed_formats.is_empty() {
                vec!["fp16".into(), "int8".into()]
            } else {
                entry.allowed_formats
            };
            let id = SemanticRegionId(format!(
                "sr:{}:axis:{}:{}:{}:{}",
                short_digest(requested_tensor.as_bytes()),
                entry.axis,
                entry.start,
                entry.end,
                entry.role
            ));
            regions.push(SemanticRegionDescriptor {
                id,
                parent: parent.clone(),
                selector: RegionSelector::AxisSpan {
                    axis: entry.axis,
                    start: entry.start,
                    end: entry.end,
                },
                role,
                origin: RegionOrigin::Explicit {
                    source: entry.source.unwrap_or_else(|| source_label.to_string()),
                },
                constraints: RegionConstraints {
                    allowed_formats: formats,
                    ..RegionConstraints::default()
                },
                provenance_refs: vec![format!("explicit-spec:{source_label}")],
            });
        }
        let partition = SemanticRegionPartition {
            parent,
            parent_shape: mapped_shape.to_vec(),
            regions,
            exhaustive: true,
            disjoint: true,
            digest: String::new(),
        }
        .seal()?;
        let receipt = SemanticRegionDiscoveryReceipt {
            schema: "prism.semantic-region.discovery-receipt.v1".into(),
            tensor: requested_tensor.into(),
            tensor_shape: mapped_shape.to_vec(),
            partition_digest: partition.digest.clone(),
            region_count: partition.regions.len(),
            coverage: "100%".into(),
            overlap: "none".into(),
            boundary_evidence: "explicit architecture contract".into(),
            legality: "compile-verified".into(),
            numerical_quality: "unproven".into(),
            execution_performance: "unmeasured".into(),
        };
        Ok((partition, receipt))
    }
}

fn parse_role(value: &str) -> Result<RegionRole, SemanticRegionSpecError> {
    match value {
        "query_projection" => Ok(RegionRole::QueryProjection),
        "key_projection" => Ok(RegionRole::KeyProjection),
        "value_projection" => Ok(RegionRole::ValueProjection),
        "gate_projection" => Ok(RegionRole::GateProjection),
        "up_projection" => Ok(RegionRole::UpProjection),
        "down_projection" => Ok(RegionRole::DownProjection),
        "router" => Ok(RegionRole::Router),
        "shared_expert" => Ok(RegionRole::SharedExpert),
        other if other.starts_with("generic:") => Ok(RegionRole::Generic {
            label: other.trim_start_matches("generic:").to_string(),
        }),
        other => Err(SemanticRegionSpecError::UnsupportedRole(other.into())),
    }
}

fn short_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    hex::encode(&digest[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_verifies_qkv_spec() {
        let spec = SemanticRegionSpec {
            schema: SEMANTIC_REGION_SPEC_V1.into(),
            tensor: "qkv".into(),
            shape: vec![6, 2],
            regions: vec![
                SemanticRegionSpecEntry { role: "query_projection".into(), axis: 0, start: 0, end: 4, allowed_formats: vec!["fp16".into()], source: None },
                SemanticRegionSpecEntry { role: "key_projection".into(), axis: 0, start: 4, end: 5, allowed_formats: vec!["int8".into()], source: None },
                SemanticRegionSpecEntry { role: "value_projection".into(), axis: 0, start: 5, end: 6, allowed_formats: vec!["int8".into()], source: None },
            ],
        };
        let (partition, receipt) = spec.into_partition("qkv", &[6, 2], "test").unwrap();
        assert_eq!(partition.regions.len(), 3);
        assert_eq!(receipt.legality, "compile-verified");
    }
}
