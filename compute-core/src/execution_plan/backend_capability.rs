//! Backend capability registry — describes what each lowering backend supports.
//!
//! The registry answers two questions:
//!   1. `supports()` — can this backend lower this group at all?
//!   2. `evaluate()` — how well would this backend handle this group for a given role?
//!
//! Every backend must declare its capabilities up front.  Unsupported combinations
//! must fail closed with a typed `UnsupportedFusionReason`.  No backend is allowed
//! to accept fusion it cannot actually execute.

/// Re-export of the CPU program op type — the spec declares this at the
/// backend capability layer for unified import paths.
pub use crate::cpu_runtime::capabilities::CpuProgramOp;

use crate::execution_plan::fusion::FusedGroup;
use crate::execution_plan::CodecFamily;
use crate::execution_plan::precision_plan::PrecisionScope;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── BackendLoweringTarget ─────────────────────────────────────────────────

/// The backend target for lowering a fused group.
///
/// Each variant names a concrete compilation target with its own lowering pass,
/// kernel library, and memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendLoweringTarget {
    /// Metal fused GPU kernels (tile640 NF4/INT8/Fp16 megakernels).
    MetalFusedGpu,
    /// Metal tensor API — per-op MPSGraph or MTLBuffer compute.
    MetalTensorApi,
    /// Apple Neural Engine — planar (non-fused) ANE operations.
    AnePlanarEngine,
    /// Core ML high-level graph — full MIL graph submission.
    CoreMlHighLevel,
    /// Accelerate + Rayon CPU backend — reference and fallback.
    AccelerateRayonCpu,
}

// ── BackendRole ───────────────────────────────────────────────────────────

/// The role a backend plays in the execution pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendRole {
    /// Default production path — fastest reliable execution.
    ProductionHotPath,
    /// Used when the primary backend is under memory/power pressure.
    PressureFallback,
    /// Bit-exact deterministic reference for equivalence checking.
    DeterministicReference,
    /// Validation probe — run alongside production to detect drift.
    ValidationProbe,
    /// Layout conversion — reformat tensors between backends.
    LayoutConversion,
}

// ── UnsupportedFusionReason ───────────────────────────────────────────────

/// Why a fusion candidate was rejected by a backend.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnsupportedFusionReason {
    /// The codec family is not supported (e.g. NF4 on ANE).
    UnsupportedCodec(CodecFamily),
    /// A specific operation is not supported.
    UnsupportedOp(String),
    /// The tensor layout is not supported.
    UnsupportedLayout(String),
    /// Cross-lane materialization is required but not supported.
    CrossLaneMaterialization,
    /// The fused group exceeds the backend's max op count.
    ExceedsMaxOps(usize),
    /// Quantization parameters are inconsistent across the group.
    QuantMismatch,
    /// Precision mismatch between ops in the group.
    PrecisionMismatch,
    /// Tile shape is incompatible with the backend.
    TileShapeMismatch,
    /// Nested parallelism introduces risk of GPU/ANE contention.
    NestedParallelismRisk,
    /// Huge dense materialization (size in bytes).
    HugeDenseMaterialization(u64),
}

// ── PowerClass ────────────────────────────────────────────────────────────

/// Estimated power envelope for a lowering choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerClass {
    /// Idle — no material GPU/ANE work, mostly CPU bookkeeping.
    Idle,
    /// Low — simple elementwise / small-tile arithmetic.
    Low,
    /// Medium — moderate matmul sizes, some fusion.
    Medium,
    /// High — large fused ops, many active execution units.
    High,
    /// Peak — full bandwidth utilization, all cores active.
    Peak,
}

impl Default for PowerClass {
    fn default() -> Self {
        Self::Medium
    }
}

// ── PrecisionClass ─────────────────────────────────────────────────────────

/// Precision class for a lowering choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionClass {
    F32,
    Fp16,
    Bf16,
    Int8,
    Nf4,
    Mixed,
}

impl Default for PrecisionClass {
    fn default() -> Self { Self::F32 }
}

// ── FusionSupport ─────────────────────────────────────────────────────────

/// The result of asking a backend whether it can lower a fused group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSupport {
    /// Whether the backend supports this group.
    pub supported: bool,
    /// Which backend target this evaluation is for.
    pub target: BackendLoweringTarget,
    /// If not supported, the reason why.
    pub reason: Option<UnsupportedFusionReason>,
    /// Estimated latency in microseconds.
    pub estimated_latency_us: Option<f64>,
    /// Estimated total memory footprint in bytes.
    pub estimated_memory_bytes: Option<u64>,
    /// Estimated scratch buffer size in bytes.
    pub estimated_scratch_bytes: Option<u64>,
    /// Estimated power class.
    pub estimated_power_class: PowerClass,
    /// Precision class for this lowering.
    pub precision_class: PrecisionClass,
    /// Whether a materialized intermediate is required.
    pub requires_materialization: bool,
    /// Whether the operation can run in-place.
    pub supports_in_place: bool,
    /// Whether the operation supports buffer aliasing.
    pub supports_aliasing: bool,
}

// ── BackendFusionRule ─────────────────────────────────────────────────────

/// A fusion rule describing what op patterns and constraints a backend supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendFusionRule {
    /// Sequence of DataflowOp variant names that can be fused, in order.
    pub pattern: Vec<String>,
    /// All ops in the fused group must use the same codec.
    pub requires_same_codec: bool,
    /// All ops must use the same precision / dtype.
    pub requires_same_precision: bool,
    /// Maximum tile elements for this rule (if constrained).
    pub max_tile_elements: Option<u64>,
    /// All ops must be on the same execution lane.
    pub requires_same_lane: bool,
}

// ── MixedPrecisionCapability ────────────────────────────────────────────

/// Describes whether and how a backend supports mixed-precision execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionCapability {
    /// Whether the backend can handle mixed-codec groups.
    pub supports_mixed_precision: bool,
    /// Which precision scopes the backend supports.
    pub supported_scopes: Vec<PrecisionScope>,
    /// Codec families that can serve as the base (default) precision.
    pub supported_base_codecs: Vec<CodecFamily>,
    /// Codec families that can serve as overrides.
    pub supported_override_codecs: Vec<CodecFamily>,
    /// Maximum fraction of weights that may use override codecs (None = unbounded).
    pub max_override_fraction: Option<f64>,
    /// Whether the backend requires separate sidecar buffers for override tiles.
    pub requires_sidecar: bool,
    /// Whether the backend supports inline mixed tiles (override data within
    /// the same buffer).
    pub supports_inline_mixed_tiles: bool,
}

// ── BackendCapability ─────────────────────────────────────────────────────

/// Full capability declaration for one backend target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapability {
    /// The backend target.
    pub target: BackendLoweringTarget,
    /// Codec families this backend supports.
    pub supported_codecs: Vec<CodecFamily>,
    /// Roles this backend can fulfil.
    pub supported_roles: Vec<BackendRole>,
    /// Maximum number of ops that can be fused into one group.
    pub max_ops_per_group: usize,
    /// Maximum tile elements (0 = unlimited).
    pub max_tile_elements: u64,
    /// Fusion rules describing supported op sequences.
    pub rules: Vec<BackendFusionRule>,
    /// Mixed-precision capability for this backend.
    pub mixed_precision: MixedPrecisionCapability,
}

// ── BackendCapabilityRegistry ─────────────────────────────────────────────

/// Registry of backend capabilities, used by the fusion scheduler to decide
/// which backend should lower each fused group.
#[derive(Debug, Clone)]
pub struct BackendCapabilityRegistry {
    entries: HashMap<BackendLoweringTarget, BackendCapability>,
}

impl BackendCapabilityRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Register a backend's capabilities.
    pub fn register(&mut self, cap: BackendCapability) {
        self.entries.insert(cap.target, cap);
    }

    /// Check whether a backend supports lowering a fused group.
    ///
    /// Returns a `FusionSupport` describing the level of support and optionally
    /// a reason for rejection.
    pub fn supports(
        &self,
        target: BackendLoweringTarget,
        group: &FusedGroup,
    ) -> FusionSupport {
        self.evaluate(target, group, BackendRole::ProductionHotPath)
    }

    /// Evaluate support for a specific role.
    ///
    /// Checks:
    /// 1. Backend is registered.
    /// 2. Derive group semantics; detect mixed-codec groups.
    /// 3. If mixed_codec: route through mixed-precision checks.
    /// 4. If not mixed: check codec support, max ops, role support.
    pub fn evaluate(
        &self,
        target: BackendLoweringTarget,
        group: &FusedGroup,
        role: BackendRole,
    ) -> FusionSupport {
        let some = |reason| FusionSupport {
            supported: false,
            target,
            reason: Some(reason),
            estimated_latency_us: None,
            estimated_memory_bytes: None,
            estimated_scratch_bytes: None,
            estimated_power_class: PowerClass::default(),
            precision_class: PrecisionClass::F32,
            requires_materialization: false,
            supports_in_place: false,
            supports_aliasing: false,
        };

        let cap = match self.entries.get(&target) {
            Some(c) => c,
            None => {
                return some(UnsupportedFusionReason::UnsupportedOp(
                    "backend not registered".into(),
                ));
            }
        };

        // Derive group semantics to detect mixed-codec groups.
        let semantics = match group.derive_semantics() {
            Ok(s) => s,
            Err(e) => return some(UnsupportedFusionReason::UnsupportedOp(
                format!("semantic error: {e:?}"),
            )),
        };

        // If the group has mixed codecs, route through mixed-precision checks.
        if semantics.mixed_codec {
            // Mixed-codec group must have a PrecisionPlan.
            let plan = match &semantics.precision_plan {
                Some(p) => p,
                None => return some(UnsupportedFusionReason::UnsupportedOp(
                    "MissingPrecisionPlan".into(),
                )),
            };

            let mp = &cap.mixed_precision;
            if !mp.supports_mixed_precision {
                return some(UnsupportedFusionReason::UnsupportedOp(
                    format!("mixed precision not supported by {target:?}"),
                ));
            }

            // Check scope support.
            if !mp.supported_scopes.contains(&plan.scope) {
                return some(UnsupportedFusionReason::UnsupportedOp(
                    format!("precision scope {:?} not supported by {target:?}", plan.scope),
                ));
            }

            // Check base codec support.
            if !mp.supported_base_codecs.contains(&plan.default_codec) {
                return some(UnsupportedFusionReason::UnsupportedCodec(plan.default_codec));
            }

            // Check each override codec and fraction.
            for override_entry in &plan.overrides {
                if !mp.supported_override_codecs.contains(&override_entry.codec) {
                    return some(UnsupportedFusionReason::UnsupportedCodec(override_entry.codec));
                }
                if let Some(max_frac) = mp.max_override_fraction {
                    let override_fraction = 1.0; // conservative: assume full override
                    if override_fraction > max_frac {
                        return some(UnsupportedFusionReason::UnsupportedOp(
                            format!("override fraction {override_fraction} exceeds max {max_frac}"),
                        ));
                    }
                }
            }

            // Check role support.
            if !cap.supported_roles.contains(&role) {
                return some(UnsupportedFusionReason::UnsupportedOp(
                    "role not supported".into(),
                ));
            }

            // Supported — return a positive assessment.
            return FusionSupport {
                supported: true,
                target,
                reason: None,
                estimated_latency_us: None,
                estimated_memory_bytes: None,
                estimated_scratch_bytes: None,
                estimated_power_class: PowerClass::Medium,
                precision_class: PrecisionClass::F32,
                requires_materialization: true,
                supports_in_place: true,
                supports_aliasing: true,
            };
        }

        // Non-mixed path: check codec support.
        let codec = group.codec_family;
        if !cap.supported_codecs.contains(&codec) {
            return some(UnsupportedFusionReason::UnsupportedCodec(codec));
        }

        // Check max ops.
        if group.body.len() > cap.max_ops_per_group {
            return some(UnsupportedFusionReason::ExceedsMaxOps(group.body.len()));
        }

        // Check role support.
        if !cap.supported_roles.contains(&role) {
            return some(UnsupportedFusionReason::UnsupportedOp(
                "role not supported".into(),
            ));
        }

        // Supported — return a positive assessment.
        FusionSupport {
            supported: true,
            target,
            reason: None,
            estimated_latency_us: None,
            estimated_memory_bytes: None,
            estimated_scratch_bytes: None,
            estimated_power_class: PowerClass::Medium,
            precision_class: PrecisionClass::F32,
            requires_materialization: false,
            supports_in_place: true,
            supports_aliasing: true,
        }
    }

    /// Return all registered backend targets.

    /// Return all registered backend targets.
    pub fn all_targets(&self) -> Vec<BackendLoweringTarget> {
        self.entries.keys().copied().collect()
    }
}

impl Default for BackendCapabilityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Pre-configured registries ─────────────────────────────────────────────

/// Build a registry pre-populated with all four standard backends.
pub fn default_registry() -> BackendCapabilityRegistry {
    use BackendLoweringTarget::*;
    use BackendRole::*;
    use CodecFamily::*;

    let mut reg = BackendCapabilityRegistry::new();

    // ── MetalFusedGpu ────────────────────────────────────────────────────
    reg.register(BackendCapability {
        target: MetalFusedGpu,
        supported_codecs: vec![RawF32, Fp16, Int8, Nf4],
        supported_roles: vec![ProductionHotPath, PressureFallback, ValidationProbe],
        max_ops_per_group: 4,
        max_tile_elements: 640,
        rules: vec![
            BackendFusionRule {
                pattern: vec!["MatMul".into(), "Add".into()],
                requires_same_codec: true,
                requires_same_precision: true,
                max_tile_elements: None,
                requires_same_lane: true,
            },
            BackendFusionRule {
                pattern: vec![
                    "MatMul".into(),
                    "MatMul".into(),
                    "SiLU".into(),
                    "Mul".into(),
                ],
                requires_same_codec: true,
                requires_same_precision: true,
                max_tile_elements: None,
                requires_same_lane: true,
            },
        ],
        mixed_precision: MixedPrecisionCapability {
            supports_mixed_precision: true,
            supported_scopes: vec![PrecisionScope::Tile, PrecisionScope::Group, PrecisionScope::InputAxisSlice, PrecisionScope::OutputAxisSlice],
            supported_base_codecs: vec![Nf4, Int8, Fp16],
            supported_override_codecs: vec![Int8, Fp16, RawF32],
            max_override_fraction: Some(0.10),
            requires_sidecar: true,
            supports_inline_mixed_tiles: true,
        },
    });

    // ── AnePlanarEngine ──────────────────────────────────────────────────
    reg.register(BackendCapability {
        target: AnePlanarEngine,
        supported_codecs: vec![Fp16, Int8],
        supported_roles: vec![ProductionHotPath, PressureFallback],
        max_ops_per_group: 4,
        max_tile_elements: 0,
        rules: vec![],
        mixed_precision: MixedPrecisionCapability {
            supports_mixed_precision: true,
            supported_scopes: vec![PrecisionScope::WholeTensor, PrecisionScope::FusedGroup],
            supported_base_codecs: vec![Fp16, Int8],
            supported_override_codecs: vec![Fp16],
            max_override_fraction: None,
            requires_sidecar: false,
            supports_inline_mixed_tiles: false,
        },
    });

    // ── CoreMlHighLevel ──────────────────────────────────────────────────
    reg.register(BackendCapability {
        target: CoreMlHighLevel,
        supported_codecs: vec![Fp16, Int8],
        supported_roles: vec![PressureFallback, LayoutConversion],
        max_ops_per_group: 1, // no fusion
        max_tile_elements: 0,
        rules: vec![],
        mixed_precision: MixedPrecisionCapability {
            supports_mixed_precision: false,
            supported_scopes: vec![],
            supported_base_codecs: vec![Fp16, Int8],
            supported_override_codecs: vec![],
            max_override_fraction: None,
            requires_sidecar: false,
            supports_inline_mixed_tiles: false,
        },
    });

    // ── AccelerateRayonCpu ───────────────────────────────────────────────
    reg.register(BackendCapability {
        target: AccelerateRayonCpu,
        supported_codecs: vec![RawF32, Fp16, Int8],
        supported_roles: vec![
            ProductionHotPath,
            PressureFallback,
            DeterministicReference,
            ValidationProbe,
            LayoutConversion,
        ],
        max_ops_per_group: 3,
        max_tile_elements: 0,
        rules: vec![],
        mixed_precision: MixedPrecisionCapability {
            supports_mixed_precision: true,
            supported_scopes: vec![PrecisionScope::Tile, PrecisionScope::Group, PrecisionScope::InputAxisSlice],
            supported_base_codecs: vec![RawF32, Fp16, Int8],
            supported_override_codecs: vec![RawF32, Fp16, Int8],
            max_override_fraction: Some(0.25),
            requires_sidecar: true,
            supports_inline_mixed_tiles: true,
        },
    });

    reg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::fusion::FusedGroup;
    use crate::execution_plan::CodecFamily;
    use crate::execution_plan::fusion::{DataflowNode, DataflowOp};
    use BackendLoweringTarget::*;
    use BackendRole::*;

    /// Helper: build a minimal FusedGroup with a given codec and body size.
    fn group_with_codec(codec: CodecFamily, body_len: usize) -> FusedGroup {
        FusedGroup {
            id: "g0".to_string(),
            body: (0..body_len)
                .map(|i| DataflowNode {
                    id: i,
                    op: DataflowOp::RmsNorm {
                        input: "in".into(),
                        weight: "w".into(),
                        output: "out".into(),
                        epsilon: 1e-6,
                    },
                    inputs: vec!["in".to_string()],
                    outputs: vec!["out".to_string()],
                })
                .collect(),
            inputs: vec![],
            outputs: vec![],
            internal_values: vec![],
            codec_family: codec,
            precision_plan: None,
        }
    }

    fn default_reg() -> BackendCapabilityRegistry {
        super::default_registry()
    }

    #[test]
    fn supports_metal_nf4_gate_up_silu_group() {
        let reg = default_reg();
        // 4-op gate/up/SiLU/Mul group with NF4 codec
        let group = group_with_codec(CodecFamily::Nf4, 4);
        let result = reg.supports(MetalFusedGpu, &group);
        assert!(
            result.supported,
            "Metal must support NF4 group with ≤4 ops: {:?}",
            result.reason
        );
        assert!(result.supports_in_place);
    }

    #[test]
    fn ane_rejects_nf4_with_unsupported_codec() {
        let reg = default_reg();
        let group = group_with_codec(CodecFamily::Nf4, 2);
        let result = reg.supports(AnePlanarEngine, &group);
        assert!(!result.supported, "ANE must reject Nf4");
        assert!(
            matches!(result.reason, Some(UnsupportedFusionReason::UnsupportedCodec(CodecFamily::Nf4))),
            "expected UnsupportedCodec(Nf4), got {:?}",
            result.reason
        );
    }

    #[test]
    fn ane_rejects_sym_int4_with_unsupported_codec() {
        let reg = default_reg();
        let group = group_with_codec(CodecFamily::SymInt4, 1);
        let result = reg.supports(AnePlanarEngine, &group);
        assert!(!result.supported, "ANE must reject SymInt4");
        assert!(
            matches!(result.reason, Some(UnsupportedFusionReason::UnsupportedCodec(CodecFamily::SymInt4))),
            "expected UnsupportedCodec(SymInt4), got {:?}",
            result.reason
        );
    }

    #[test]
    fn ane_rejects_ternary_with_unsupported_codec() {
        let reg = default_reg();
        let group = group_with_codec(CodecFamily::Ternary, 1);
        let result = reg.supports(AnePlanarEngine, &group);
        assert!(!result.supported, "ANE must reject Ternary");
        assert!(
            matches!(result.reason, Some(UnsupportedFusionReason::UnsupportedCodec(CodecFamily::Ternary))),
            "expected UnsupportedCodec(Ternary), got {:?}",
            result.reason
        );
    }

    #[test]
    fn metal_rejects_exceeds_max_ops() {
        let reg = default_reg();
        // 5 ops exceeds Metal's max_ops_per_group=4
        let group = group_with_codec(CodecFamily::Nf4, 5);
        let result = reg.supports(MetalFusedGpu, &group);
        assert!(!result.supported, "Metal must reject >4 ops");
        assert!(
            matches!(result.reason, Some(UnsupportedFusionReason::ExceedsMaxOps(5))),
            "expected ExceedsMaxOps(5), got {:?}",
            result.reason
        );
    }

    #[test]
    fn cpu_accepts_raw_f32_rms_norm() {
        let reg = default_reg();
        // Single-op group with RawF32 — should be accepted
        let group = group_with_codec(CodecFamily::RawF32, 1);
        let result = reg.supports(AccelerateRayonCpu, &group);
        assert!(
            result.supported,
            "CPU must accept RawF32 single-op group: {:?}",
            result.reason
        );
    }

    #[test]
    fn cpu_accepts_fp16_int8() {
        let reg = default_reg();
        for codec in &[CodecFamily::Fp16, CodecFamily::Int8] {
            let group = group_with_codec(*codec, 1);
            let result = reg.supports(AccelerateRayonCpu, &group);
            assert!(
                result.supported,
                "CPU must accept {:?}: {:?}",
                codec,
                result.reason
            );
        }
    }

    #[test]
    fn cpu_rejects_nf4_and_ternary() {
        let reg = default_reg();
        for codec in &[CodecFamily::Nf4, CodecFamily::Ternary, CodecFamily::SymInt4] {
            let group = group_with_codec(*codec, 1);
            let result = reg.supports(AccelerateRayonCpu, &group);
            assert!(!result.supported, "CPU must reject {:?}", codec);
        }
    }

    #[test]
    fn core_ml_no_fusion() {
        let reg = default_reg();
        // Core ML has max_ops_per_group=1, so a 2-op group must be rejected
        let group = group_with_codec(CodecFamily::Fp16, 2);
        let result = reg.supports(CoreMlHighLevel, &group);
        assert!(!result.supported, "CoreML must reject 2-op group");
        assert!(
            matches!(result.reason, Some(UnsupportedFusionReason::ExceedsMaxOps(2))),
            "expected ExceedsMaxOps(2), got {:?}",
            result.reason
        );
    }

    #[test]
    fn supports_method_correctness_all_backends() {
        let reg = default_reg();
        let targets = reg.all_targets();

        // Verify all 4 backends are registered
        assert_eq!(targets.len(), 4, "must have 4 backends");

        let required: std::collections::HashSet<BackendLoweringTarget> =
            vec![MetalFusedGpu, AnePlanarEngine, CoreMlHighLevel, AccelerateRayonCpu]
                .into_iter()
                .collect();
        let registered: std::collections::HashSet<BackendLoweringTarget> =
            targets.into_iter().collect();
        assert_eq!(
            required, registered,
            "all four expected backends must be registered"
        );

        // Metal: NF4 ≤4 ops supported, 5 ops rejected
        let nf4_4 = group_with_codec(CodecFamily::Nf4, 4);
        assert!(reg.supports(MetalFusedGpu, &nf4_4).supported);
        let nf4_5 = group_with_codec(CodecFamily::Nf4, 5);
        assert!(!reg.supports(MetalFusedGpu, &nf4_5).supported);

        // Metal: Fp16 ≤4 ops supported
        let fp16_2 = group_with_codec(CodecFamily::Fp16, 2);
        assert!(reg.supports(MetalFusedGpu, &fp16_2).supported);

        // ANE: Fp16/Int8 accepted, Nf4/SymInt4/Ternary rejected
        for accepted in &[CodecFamily::Fp16, CodecFamily::Int8] {
            let g = group_with_codec(*accepted, 1);
            assert!(reg.supports(AnePlanarEngine, &g).supported);
        }
        for rejected in &[CodecFamily::Nf4, CodecFamily::SymInt4, CodecFamily::Ternary] {
            let g = group_with_codec(*rejected, 1);
            assert!(!reg.supports(AnePlanarEngine, &g).supported);
        }

        // Core ML: single-ops accepted, multi-ops rejected
        let single = group_with_codec(CodecFamily::Fp16, 1);
        assert!(reg.supports(CoreMlHighLevel, &single).supported);
        let multi = group_with_codec(CodecFamily::Int8, 2);
        assert!(!reg.supports(CoreMlHighLevel, &multi).supported);

        // CPU: RawF32/Fp16/Int8 accepted, Nf4/Ternary/SymInt4 rejected
        for accepted in &[CodecFamily::RawF32, CodecFamily::Fp16, CodecFamily::Int8] {
            let g = group_with_codec(*accepted, 1);
            assert!(
                reg.supports(AccelerateRayonCpu, &g).supported,
                "CPU must accept {:?}",
                accepted
            );
        }
        for rejected in &[CodecFamily::Nf4, CodecFamily::Ternary, CodecFamily::SymInt4] {
            let g = group_with_codec(*rejected, 1);
            assert!(
                !reg.supports(AccelerateRayonCpu, &g).supported,
                "CPU must reject {:?}",
                rejected
            );
        }

        // CPU role: DeterministicReference is supported
        let rawf32_1 = group_with_codec(CodecFamily::RawF32, 1);
        let eval = reg.evaluate(AccelerateRayonCpu, &rawf32_1, DeterministicReference);
        assert!(eval.supported, "CPU must support DeterministicReference");

        // Metal role: ProductionHotPath supported
        let nf4_1 = group_with_codec(CodecFamily::Nf4, 1);
        let eval = reg.evaluate(MetalFusedGpu, &nf4_1, ProductionHotPath);
        assert!(eval.supported, "Metal must support ProductionHotPath");
    }
}
