//! PRISM-REGION-CATALOGUE-0001 — Static placement catalogue for Apple Silicon
//! transformer operations.
//!
//! Every transformer operation in the supported model architecture has a
//! statically-defined catalogue entry declaring its primary execution lane,
//! fallback lane, dtype contract, shape requirements, and evidence requirement.
//!
//! The catalogue is consumed by cimage compilation (PhaseIR -> region
//! partitioning) and by the runtime admission gate. No region may be
//! installed without a catalogue admission result.

use serde::{Deserialize, Serialize};

use crate::compilation::phase_ir::TensorDtype;
use RegionAdmission::*;
use EvidenceRequirement::*;

// ── Region admission ────────────────────────────────────────────────────

/// Whether a region class is admitted on a particular execution lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionAdmission {
    /// Production-qualified on Apple Neural Engine via Core ML.
    CoreMlProduction,
    /// Production-qualified on Metal GPU.
    MetalProduction,
    /// Production-qualified on CPU.
    CpuProduction,
    /// Only available as a fallback (not primary path).
    FallbackOnly,
    /// Not supported on this platform.
    Unsupported,
}

// ── Shape requirement ───────────────────────────────────────────────────

/// A dimension constraint for a region's input or output tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShapeRequirement {
    /// Dimension name (e.g. "batch", "sequence", "hidden", "heads").
    pub dimension: String,
    /// Minimum allowed value (inclusive).
    pub min: u32,
    /// Maximum allowed value (inclusive). None = unbounded.
    pub max: Option<u32>,
    /// Whether this dimension must be a multiple of some alignment.
    pub multiple_of: Option<u32>,
}

/// Memory layout constraint for a region's input or output tensor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LayoutContract {
    /// Accepted layout name (e.g. "NHWC", "NCWH", "linear").
    pub layout: String,
    /// Whether this layout is the preferred one for the primary lane.
    pub preferred: bool,
    /// Stride constraints for this layout, if any.
    pub stride_constraints: Vec<ShapeRequirement>,
}

// ── Numerical tolerance ─────────────────────────────────────────────────

/// Numerical tolerance contract for a region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalTolerance {
    /// Maximum absolute error per element.
    pub absolute_tolerance: f64,
    /// Maximum relative error.
    pub relative_tolerance: f64,
    /// Whether bit-exact results are required across lanes.
    pub require_bit_exact: bool,
    /// Number of validation samples required for admission.
    pub min_validation_samples: u32,
}

// ── Evidence requirement ────────────────────────────────────────────────

/// Required evidence tier for a region to be considered installed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvidenceRequirement {
    NotAttempted = 0,
    ConfiguredOnly = 1,
    Installed = 2,
    AllocationAttested = 3,
    Warmed = 4,
    PredictionValidated = 5,
    MetalConsumed = 6,
    TraceObserved = 7,
    Failed = 8,
}

// ── Region catalogue entry ──────────────────────────────────────────────

/// A single entry in the static region catalogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCatalogueEntry {
    /// Canonical operator family (e.g. "rms_norm", "q_projection").
    pub operator_family: String,
    /// Required input dtype.
    pub input_dtype: TensorDtype,
    /// Required output dtype.
    pub output_dtype: TensorDtype,
    /// Static shape requirements for all tensors.
    pub static_shape_requirements: Vec<ShapeRequirement>,
    /// Accepted memory layouts.
    pub admitted_layouts: Vec<LayoutContract>,
    /// Primary execution lane and admission status.
    pub primary_admission: RegionAdmission,
    /// Fallback execution lane (if any) and admission status.
    pub fallback_admission: Option<RegionAdmission>,
    /// Required evidence tier before the region may execute.
    pub evidence_requirement: EvidenceRequirement,
    /// Numerical tolerance bounds.
    pub numerical_tolerance: NumericalTolerance,
    /// Known exclusions or incompatibilities.
    pub known_exclusions: Vec<String>,
}

impl RegionCatalogueEntry {
    pub fn is_coreml_production(&self) -> bool {
        matches!(self.primary_admission, RegionAdmission::CoreMlProduction)
    }

    pub fn is_metal_production(&self) -> bool {
        matches!(self.primary_admission, RegionAdmission::MetalProduction)
    }

    pub fn has_fallback(&self) -> bool {
        self.fallback_admission
            .map(|a| !matches!(a, RegionAdmission::Unsupported))
            .unwrap_or(false)
    }
}

// ── The catalogue ───────────────────────────────────────────────────────

/// Static region catalogue for the first Apple Silicon FP16 alpha.
///
/// Every transformer operation in the supported model architecture has
/// exactly one catalogue entry. The compiler must not place any operation
/// without consulting this catalogue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionCatalogue {
    pub entries: Vec<RegionCatalogueEntry>,
}

impl RegionCatalogue {
    /// Build the production catalogue for the FP16 alpha.
    ///
    /// Placement policy:
    ///   Core ML — large static FP16 matmuls (Q, K, V, O, gate, up, down, logits)
    ///   Metal   — attention assembly, RoPE, RMSNorm, SiLU, softmax
    ///   CPU     — embedding lookup, token sampling, residual add
    pub fn fp16_alpha() -> Self {
        let fp16 = TensorDtype::Float16;
        let f32 = TensorDtype::Float32;
        let i32 = TensorDtype::Int32;

        RegionCatalogue {
            entries: vec![
                Self::make("embedding_lookup", i32, fp16, CpuProduction, None, ConfiguredOnly),
                Self::make("rms_norm", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("rotary_embedding", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("q_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("k_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("v_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("attention_score", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("attention_mask", fp16, fp16, MetalProduction, None, Warmed),
                Self::make("softmax", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("attention_value_aggregation", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("output_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("residual_add", fp16, fp16, CpuProduction, None, ConfiguredOnly),
                Self::make("gate_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("silu_activation", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("up_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("down_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("final_norm", fp16, fp16, MetalProduction, Some(CpuProduction), Warmed),
                Self::make("logits_projection", fp16, fp16, CoreMlProduction, Some(MetalProduction), PredictionValidated),
                Self::make("token_sampling", fp16, i32, CpuProduction, None, ConfiguredOnly),
                Self::make("kv_cache_append", fp16, fp16, MetalProduction, None, AllocationAttested),
                Self::make("kv_cache_view", fp16, fp16, MetalProduction, None, AllocationAttested),
            ],
        }
    }

    fn make(
        op: &str,
        input: TensorDtype,
        output: TensorDtype,
        primary: RegionAdmission,
        fallback: Option<RegionAdmission>,
        evidence: EvidenceRequirement,
    ) -> RegionCatalogueEntry {
        RegionCatalogueEntry {
            operator_family: op.to_string(),
            input_dtype: input,
            output_dtype: output,
            static_shape_requirements: vec![
                ShapeRequirement {
                    dimension: "batch".into(),
                    min: 1,
                    max: Some(1),
                    multiple_of: None,
                },
            ],
            admitted_layouts: vec![
                LayoutContract {
                    layout: "NHWC".into(),
                    preferred: true,
                    stride_constraints: vec![],
                },
            ],
            primary_admission: primary,
            fallback_admission: fallback,
            evidence_requirement: evidence,
            numerical_tolerance: NumericalTolerance {
                absolute_tolerance: 0.01,
                relative_tolerance: 0.01,
                require_bit_exact: false,
                min_validation_samples: 10,
            },
            known_exclusions: vec![],
        }
    }

    /// Look up a catalogue entry by operator family name.
    pub fn find(&self, operator_family: &str) -> Option<&RegionCatalogueEntry> {
        self.entries.iter().find(|e| e.operator_family == operator_family)
    }

    /// Returns all entries admitted for Core ML production.
    pub fn coreml_production_ops(&self) -> Vec<&RegionCatalogueEntry> {
        self.entries.iter().filter(|e| e.is_coreml_production()).collect()
    }

    /// Returns all entries admitted for Metal production.
    pub fn metal_production_ops(&self) -> Vec<&RegionCatalogueEntry> {
        self.entries.iter().filter(|e| e.is_metal_production()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalogue_has_all_required_ops() {
        let cat = RegionCatalogue::fp16_alpha();
        let required = [
            "embedding_lookup", "rms_norm", "rotary_embedding",
            "q_projection", "k_projection", "v_projection",
            "attention_score", "attention_mask", "softmax",
            "attention_value_aggregation", "output_projection",
            "residual_add", "gate_projection", "silu_activation",
            "up_projection", "down_projection", "final_norm",
            "logits_projection", "token_sampling",
            "kv_cache_append", "kv_cache_view",
        ];
        for op in &required {
            assert!(cat.find(op).is_some(), "missing entry for {op}");
        }
        assert_eq!(cat.entries.len(), required.len());
    }

    #[test]
    fn test_coreml_production_ops_use_fp16() {
        let cat = RegionCatalogue::fp16_alpha();
        for entry in cat.coreml_production_ops() {
            assert_eq!(entry.input_dtype, TensorDtype::Float16,
                "Core ML op {} must use FP16", entry.operator_family);
            assert_eq!(entry.output_dtype, TensorDtype::Float16,
                "Core ML op {} must output FP16", entry.operator_family);
        }
    }

    #[test]
    fn test_projection_ops_coreml() {
        let cat = RegionCatalogue::fp16_alpha();
        for op in &["q_projection", "k_projection", "v_projection",
                     "output_projection", "gate_projection",
                     "up_projection", "down_projection"] {
            let entry = cat.find(op).expect("op must exist");
            assert!(entry.is_coreml_production(),
                "{op} must be CoreMlProduction");
        }
    }

    #[test]
    fn test_attention_ops_metal() {
        let cat = RegionCatalogue::fp16_alpha();
        for op in &["attention_score", "softmax", "attention_value_aggregation"] {
            let entry = cat.find(op).expect("op must exist");
            assert!(entry.is_metal_production(),
                "{op} must be MetalProduction");
        }
    }

    #[test]
    fn test_every_coreml_op_has_metal_fallback() {
        let cat = RegionCatalogue::fp16_alpha();
        for entry in cat.coreml_production_ops() {
            assert!(entry.has_fallback(),
                "Core ML op {} must have fallback", entry.operator_family);
            if let Some(fb) = entry.fallback_admission {
                assert_eq!(fb, RegionAdmission::MetalProduction,
                    "Core ML op {} fallback must be MetalProduction", entry.operator_family);
            }
        }
    }

    #[test]
    fn test_evidence_requirements_are_set() {
        let cat = RegionCatalogue::fp16_alpha();
        for entry in &cat.entries {
            assert!(entry.evidence_requirement > EvidenceRequirement::NotAttempted,
                "op {} must have evidence requirement", entry.operator_family);
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let cat = RegionCatalogue::fp16_alpha();
        let json = serde_json::to_string(&cat).unwrap();
        let restored: RegionCatalogue = serde_json::from_str(&json).unwrap();
        assert_eq!(cat.entries.len(), restored.entries.len());
    }

    #[test]
    fn test_unknown_operator_returns_none() {
        let cat = RegionCatalogue::fp16_alpha();
        assert!(cat.find("nonexistent_op").is_none());
    }
}
