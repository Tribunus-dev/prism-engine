//! Quantization plan types — the structured result of running the per-tensor
//! codecs on a model.
//!
//! The previous compile pipeline silently wrote the source weights into the
//! CImage (or substituted one codec for another when a requested format was
//! not implemented). These types make the per-tensor decision observable
//! before the artifact is sealed, so:
//!
//!   * `format` is always the format that was actually applied — never a
//!     silent substitution.
//!   * the plan is content-addressed: same source + same plan = same
//!     `QuantizationResult` digest.
//!   * downstream code (the constitutional `system_emit_cimage`, the CLI,
//!     the dashboard) reads the plan instead of recomputing policy.
//!
//! These types are the chokepoint between per-tensor compilation and
//! artifact emission. They do not depend on the World, do not own a
//! filesystem handle, and do not know about runtime kernel lifecycle.

use crate::cimage::TensorType;
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A single tensor's quantization selection — the structured result of
/// applying one codec to one source weight tensor.
///
/// The `format` field is the format that was actually applied. It equals
/// the requested `TensorFormat` (or the default `Palettized4Bit` when no
/// plan is provided). It is a hard error to produce a selection whose
/// `format` does not match the codec that produced `payload`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizedTensorSelection {
    /// Source tensor key (e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub key: String,
    /// The format that was actually applied. Never silently substituted.
    pub format: TensorFormat,
    /// Payload bytes in the CImage-ready encoding for `tensor_type`.
    pub payload: Vec<u8>,
    /// Physical type tag for the CImage writer.
    pub tensor_type: TensorType,
    /// Output dimension (rows).
    pub dim_m: u32,
    /// Input dimension (cols).
    pub dim_n: u32,
    /// Effective bits per value as measured by the codec.
    pub effective_bpp: f32,
    /// Number of bytes actually written into the CImage.
    pub payload_bytes: u64,
}

impl QuantizedTensorSelection {
    /// Compute a stable digest of the selection. Same key + same payload +
    /// same format + same dimensions = same digest, regardless of struct
    /// layout.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.key.as_bytes());
        hasher.update([0u8]);
        hasher.update([self.format.discriminant_byte()]);
        hasher.update(self.tensor_type.discriminant_byte());
        hasher.update(self.dim_m.to_le_bytes());
        hasher.update(self.dim_n.to_le_bytes());
        hasher.update(self.effective_bpp.to_le_bytes());
        hasher.update(self.payload_bytes.to_le_bytes());
        hasher.update([0u8]);
        hasher.update(&self.payload);
        hasher.finalize().into()
    }
}

/// The complete per-tensor result of a compilation pass.
///
/// `selections` is ordered (source-graph iteration order). A CImage
/// emission pass consumes this plan in order; the bytes of a sealed
/// CImage are determined by `selections` plus the CImage page-alignment
/// invariant.
///
/// The plan itself is not the artifact. The artifact is whatever the
/// constitutional emission system produces from this plan and the
/// runtime's target profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationResult {
    /// Content digest of the source model. Used as a stable identifier.
    pub source_digest: String,
    /// Target hardware identifier from the compilation request.
    pub target_hardware: String,
    /// Per-tensor selections, ordered.
    pub selections: Vec<QuantizedTensorSelection>,
    /// Optional pre-rendered execution plan JSON, embedded into the
    /// CImage header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_plan_json: Option<String>,
    /// The format that was used for tensors that did not appear in the
    /// caller-supplied `FormatPlan`. Always `Palettized4Bit` for the
    /// legacy default path. Surfaced explicitly so receipts can prove
    /// the default policy was applied.
    pub default_format: TensorFormat,
}

impl QuantizationResult {
    /// Compute a content digest of the entire plan. Stable across runs.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.source_digest.as_bytes());
        hasher.update([0u8]);
        hasher.update(self.target_hardware.as_bytes());
        hasher.update([0u8]);
        for sel in &self.selections {
            hasher.update(sel.digest());
            hasher.update([0xffu8]);
        }
        if let Some(plan) = &self.execution_plan_json {
            hasher.update(plan.as_bytes());
        }
        hasher.update([0u8]);
        hasher.update([self.default_format.discriminant_byte()]);
        hasher.finalize().into()
    }

    /// Count of selections that were applied with the default format.
    pub fn default_format_count(&self) -> usize {
        self.selections
            .iter()
            .filter(|s| s.format == self.default_format)
            .count()
    }

    /// Count of selections that were applied with a non-default format.
    pub fn explicit_format_count(&self) -> usize {
        self.selections.len() - self.default_format_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_selection(key: &str, format: TensorFormat) -> QuantizedTensorSelection {
        QuantizedTensorSelection {
            key: key.into(),
            format,
            payload: vec![1, 2, 3, 4],
            tensor_type: TensorType::Blob,
            dim_m: 4,
            dim_n: 8,
            effective_bpp: 8.0,
            payload_bytes: 4,
        }
    }

    #[test]
    fn selection_digest_is_stable() {
        let a = fake_selection("k", TensorFormat::Nf4);
        let b = fake_selection("k", TensorFormat::Nf4);
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn selection_digest_changes_with_format() {
        let a = fake_selection("k", TensorFormat::Nf4);
        let b = fake_selection("k", TensorFormat::Int4);
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn selection_digest_changes_with_payload() {
        let a = QuantizedTensorSelection {
            payload: vec![1, 2, 3, 4],
            ..fake_selection("k", TensorFormat::Nf4)
        };
        let b = QuantizedTensorSelection {
            payload: vec![1, 2, 3, 5],
            ..fake_selection("k", TensorFormat::Nf4)
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn result_digest_is_stable() {
        let r1 = QuantizationResult {
            source_digest: "abc".into(),
            target_hardware: "apple-m1".into(),
            selections: vec![fake_selection("k1", TensorFormat::Nf4)],
            execution_plan_json: None,
            default_format: TensorFormat::Palettized4Bit,
        };
        let r2 = QuantizationResult {
            source_digest: "abc".into(),
            target_hardware: "apple-m1".into(),
            selections: vec![fake_selection("k1", TensorFormat::Nf4)],
            execution_plan_json: None,
            default_format: TensorFormat::Palettized4Bit,
        };
        assert_eq!(r1.digest(), r2.digest());
    }

    #[test]
    fn default_format_count_is_explicit() {
        let r = QuantizationResult {
            source_digest: "".into(),
            target_hardware: "".into(),
            selections: vec![
                fake_selection("a", TensorFormat::Palettized4Bit),
                fake_selection("b", TensorFormat::Nf4),
                fake_selection("c", TensorFormat::Palettized4Bit),
            ],
            execution_plan_json: None,
            default_format: TensorFormat::Palettized4Bit,
        };
        assert_eq!(r.default_format_count(), 2);
        assert_eq!(r.explicit_format_count(), 1);
    }
}
