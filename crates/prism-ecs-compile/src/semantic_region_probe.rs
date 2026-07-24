//! Region-scoped mapped tensor probes and evidence.

use prism_ecs_ir::evolution::foundation::LogicalTensorId;
use prism_ecs_ir::semantic_region::{RegionSelector, SemanticRegionId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedTensorRegionProbeContext {
    pub model_dir: PathBuf,
    pub tensor_name: String,
    pub tensor_id: LogicalTensorId,
    pub tensor_shape: Vec<u64>,
    pub selector: RegionSelector,
    pub partition_digest: String,
    pub region_id: SemanticRegionId,
    pub calibration_corpus_digest: String,
    pub probe_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionSensitivityReceipt {
    pub tensor_id: LogicalTensorId,
    pub region_id: SemanticRegionId,
    pub selector_digest: String,
    pub format_variance: f64,
    pub operation_variance: f64,
    pub geometry_variance: f64,
    pub memory_variance: f64,
    pub probe_valid: bool,
    pub evidence_source: String,
    pub materialized_bytes: u64,
    pub cache_key: String,
}

#[derive(Debug, Clone)]
pub struct RegionView {
    pub element_offset: u64,
    pub element_count: u64,
    pub contiguous: bool,
    pub materialized_bytes: u64,
}

#[derive(Debug, Error)]
pub enum RegionProbeError {
    #[error("tensor shape is empty")]
    EmptyShape,
    #[error("selector is out of bounds")]
    OutOfBounds,
    #[error("rectangular or strided selectors are not supported by v0 probes")]
    UnsupportedSelector,
}

impl MappedTensorRegionProbeContext {
    pub fn bounded_view(&self, element_size: u64) -> Result<RegionView, RegionProbeError> {
        if self.tensor_shape.is_empty() {
            return Err(RegionProbeError::EmptyShape);
        }
        match self.selector {
            RegionSelector::WholeTensor => Ok(RegionView {
                element_offset: 0,
                element_count: self.tensor_shape.iter().product(),
                contiguous: true,
                materialized_bytes: 0,
            }),
            RegionSelector::AxisSpan { axis, start, end } => {
                let axis = axis as usize;
                let Some(&axis_len) = self.tensor_shape.get(axis) else {
                    return Err(RegionProbeError::OutOfBounds);
                };
                if start >= end || end > axis_len {
                    return Err(RegionProbeError::OutOfBounds);
                }
                if axis == 0 {
                    let row_size: u64 = self.tensor_shape.iter().skip(1).product();
                    Ok(RegionView {
                        element_offset: start * row_size,
                        element_count: (end - start) * row_size,
                        contiguous: true,
                        materialized_bytes: 0,
                    })
                } else {
                    let selected = (end - start)
                        .saturating_mul(self.tensor_shape.iter().enumerate().filter(|(i, _)| *i != axis).map(|(_, value)| *value).product::<u64>());
                    Ok(RegionView {
                        element_offset: 0,
                        element_count: selected,
                        contiguous: false,
                        materialized_bytes: selected.saturating_mul(element_size),
                    })
                }
            }
            RegionSelector::Rect { .. } => Err(RegionProbeError::UnsupportedSelector),
        }
    }

    pub fn cache_key(&self, representation: &str) -> String {
        let mut hasher = Sha256::new();
        for value in [
            self.tensor_id.0.as_str(),
            self.region_id.0.as_str(),
            self.partition_digest.as_str(),
            representation,
            self.probe_version.as_str(),
            self.calibration_corpus_digest.as_str(),
        ] {
            hasher.update(value.as_bytes());
            hasher.update([0]);
        }
        hasher.update(selector_digest(&self.selector).as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn receipt(
        &self,
        representation: &str,
        element_size: u64,
        variances: [f64; 4],
        evidence_source: impl Into<String>,
    ) -> Result<RegionSensitivityReceipt, RegionProbeError> {
        let view = self.bounded_view(element_size)?;
        Ok(RegionSensitivityReceipt {
            tensor_id: self.tensor_id.clone(),
            region_id: self.region_id.clone(),
            selector_digest: selector_digest(&self.selector),
            format_variance: variances[0],
            operation_variance: variances[1],
            geometry_variance: variances[2],
            memory_variance: variances[3],
            probe_valid: variances.iter().all(|value| value.is_finite() && *value >= 0.0),
            evidence_source: evidence_source.into(),
            materialized_bytes: view.materialized_bytes,
            cache_key: self.cache_key(representation),
        })
    }
}

pub fn selector_digest(selector: &RegionSelector) -> String {
    let bytes = serde_json::to_vec(selector).expect("selector serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(selector: RegionSelector) -> MappedTensorRegionProbeContext {
        MappedTensorRegionProbeContext {
            model_dir: "/tmp/model".into(),
            tensor_name: "qkv.weight".into(),
            tensor_id: LogicalTensorId("qkv.weight".into()),
            tensor_shape: vec![6, 2],
            selector,
            partition_digest: "partition".into(),
            region_id: SemanticRegionId("q".into()),
            calibration_corpus_digest: "corpus".into(),
            probe_version: "v1".into(),
        }
    }

    #[test]
    fn contiguous_row_range_needs_no_materialization() {
        let view = context(RegionSelector::AxisSpan { axis: 0, start: 1, end: 3 }).bounded_view(2).unwrap();
        assert_eq!(view.element_offset, 2);
        assert_eq!(view.element_count, 4);
        assert_eq!(view.materialized_bytes, 0);
    }

    #[test]
    fn nonleading_axis_records_materialization() {
        let view = context(RegionSelector::AxisSpan { axis: 1, start: 0, end: 1 }).bounded_view(2).unwrap();
        assert!(!view.contiguous);
        assert_eq!(view.materialized_bytes, 12);
    }

    #[test]
    fn cache_key_separates_regions_and_calibration() {
        let a = context(RegionSelector::AxisSpan { axis: 0, start: 0, end: 4 });
        let mut b = a.clone();
        b.calibration_corpus_digest = "other".into();
        assert_ne!(a.cache_key("int8"), b.cache_key("int8"));
    }
}
