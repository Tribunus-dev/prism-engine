/// Policy-to-spec resolver — generates `TrainingTargetSpec` instances from a
/// compiler policy configuration (`serde_json::Value`).
///
/// The resolver is stateless and deterministic: the same policy JSON always
/// produces the same spec bytes.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::execution_plan::CodecFamily;
use crate::execution_profile::PhysicalTileLayout;

use super::gates::{
    QuantTrainingMethod, RequiredEvidenceLevel,
    WeightTrainingGates,
};
use super::spec::{
    ActivationWeightedObjective, TrainingEvidenceGate, TrainingTargetPriority,
    TrainingTargetSpec, WeightTrainingTarget,
};

/// Stateless resolver that scans a compiler policy and produces training
/// target specifications.
#[derive(Debug, Clone, Default)]
pub struct TrainingTargetResolver;

/// Options controlling which experimental codec families produce targets.
///
/// All flags default to `false` — only production-ready codec families with
/// explicit QAT support (e.g., Ternary) generate targets by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTargetResolveOptions {
    /// Allow generation of training targets for RawF32 codec entries.
    pub experimental_raw_f32: bool,
    /// Allow generation of calibration targets for FP16 codec entries.
    pub experimental_fp16_calibration: bool,
    /// Allow generation of calibration targets for INT8 codec entries.
    pub experimental_int8_calibration: bool,
    /// Allow generation of training targets for NF4 codec entries.
    pub experimental_nf4_training: bool,
}

impl Default for TrainingTargetResolveOptions {
    fn default() -> Self {
        Self {
            experimental_raw_f32: false,
            experimental_fp16_calibration: false,
            experimental_int8_calibration: false,
            experimental_nf4_training: false,
        }
    }
}

/// Errors that can occur during policy resolution.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum TrainingTargetResolveError {
    /// The codec string in a policy entry is not recognized.
    #[error("Unsupported or unrecognized codec family: {0}")]
    UnsupportedCodec(String),
    /// A policy entry is missing the required `tensor_class` field.
    #[error("Policy entry is missing 'tensor_class' field")]
    MissingTensorClassName,
    /// The top-level policy structure is not valid (not an object, missing
    /// required fields, etc.).
    #[error("Invalid policy structure: {0}")]
    InvalidPolicy(String),
    /// An explicit layout field in the policy entry has an invalid value.
    #[error("Invalid layout field '{field}': {detail}")]
    InvalidLayoutField {
        /// The field name that contains the invalid value.
        field: String,
        /// Details about why the value is invalid.
        detail: String,
    },
}

// ── Digest helper ──────────────────────────────────────────────────────────

/// Compute the SHA-256 hex digest of a byte slice.
///
/// Used to stamp `TrainingTargetSpec.source_policy_digest` so each spec can
/// be traced back to the exact policy bytes that produced it.
pub fn compute_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()
}

// ── Codec parsing ──────────────────────────────────────────────────────────

/// Parse a codec family string from a policy entry.
fn parse_codec(s: &str) -> Result<CodecFamily, TrainingTargetResolveError> {
    match s.to_lowercase().as_str() {
        "rawf32" | "raw_f32" | "f32" => Ok(CodecFamily::RawF32),
        "fp16" | "f16" => Ok(CodecFamily::Fp16),
        "int8" | "i8" => Ok(CodecFamily::Int8),
        "nf4" => Ok(CodecFamily::Nf4),
        "symint4" | "sym_int4" | "i4" => Ok(CodecFamily::SymInt4),
        "ternary" | "t3" => Ok(CodecFamily::Ternary),
        other => Err(TrainingTargetResolveError::UnsupportedCodec(other.to_string())),
    }
}

/// Extract a string field from a JSON object.
fn get_str<'a>(obj: &'a Value, key: &str) -> Option<&'a str> {
    obj.get(key).and_then(|v| v.as_str())
}

/// Extract `f64` from a JSON object field, with default fallback.
fn get_f64(obj: &Value, key: &str, default: f64) -> f64 {
    obj.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
}

/// Extract `f32` from a JSON object field, with default fallback.
fn get_f32(obj: &Value, key: &str, default: f32) -> f32 {
    obj.get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

/// Extract `usize` from a JSON object field, with default fallback.
fn get_usize(obj: &Value, key: &str, default: usize) -> usize {
    obj.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(default)
}

/// Extract a boolean from a JSON object field, with default fallback.
fn get_bool(obj: &Value, key: &str, default: bool) -> bool {
    obj.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

// ── Physical layout construction ───────────────────────────────────────────

/// Build a `PhysicalTileLayout` from a policy entry, falling back to defaults
/// for missing fields.
fn build_layout(entry: &Value) -> Result<PhysicalTileLayout, TrainingTargetResolveError> {
    // Start with defaults, then override from explicit policy fields.
    let mut layout = PhysicalTileLayout::default();

    // ── tile_family ──────────────────────────────────────────────────────
    if let Some(val) = entry.get("tile_family").and_then(|v| v.as_str()) {
        match val {
            "Tile640" | "tile640" => {
                layout.tile_family = crate::execution_profile::TileFamily::tile640();
                layout.tile_shape = crate::execution_profile::TileShape::tile640();
            }
            _ => {
                return Err(TrainingTargetResolveError::InvalidLayoutField {
                    field: "tile_family".into(),
                    detail: format!("unrecognized tile family: '{val}'; expected 'Tile640'"),
                });
            }
        }
    }

    // ── group_size ───────────────────────────────────────────────────────
    if let Some(val) = entry.get("group_size").and_then(|v| v.as_u64()) {
        layout.group_size = val as u32;
    }

    // ── group_axis ───────────────────────────────────────────────────────
    if let Some(val) = entry.get("group_axis").and_then(|v| v.as_str()) {
        match val {
            "PackedContiguous" | "packed_contiguous" => {
                layout.group_axis = crate::execution_profile::GroupAxis::PackedContiguous;
            }
            "OutputAxis" | "output_axis" => {
                layout.group_axis = crate::execution_profile::GroupAxis::OutputAxis;
            }
            "InputAxis" | "input_axis" => {
                layout.group_axis = crate::execution_profile::GroupAxis::InputAxis;
            }
            "TileLocal" | "tile_local" => {
                layout.group_axis = crate::execution_profile::GroupAxis::TileLocal;
            }
            _ => {
                return Err(TrainingTargetResolveError::InvalidLayoutField {
                    field: "group_axis".into(),
                    detail: format!("unrecognized group_axis: '{val}'"),
                });
            }
        }
    }

    // ── metadata_layout ──────────────────────────────────────────────────
    if let Some(val) = entry.get("metadata_layout").and_then(|v| v.as_str()) {
        match val {
            "AdjacentTile" | "adjacent_tile" | "adjacent" => {
                layout.metadata_layout = crate::execution_profile::MetadataLayout::AdjacentTile;
            }
            "SeparatedManifest" | "separated_manifest" | "manifest" => {
                layout.metadata_layout = crate::execution_profile::MetadataLayout::SeparatedManifest;
            }
            "Interleaved" | "interleaved" => {
                layout.metadata_layout = crate::execution_profile::MetadataLayout::Interleaved;
            }
            _ => {
                return Err(TrainingTargetResolveError::InvalidLayoutField {
                    field: "metadata_layout".into(),
                    detail: format!("unrecognized metadata_layout: '{val}'"),
                });
            }
        }
    }

    // ── format ───────────────────────────────────────────────────────────
    if let Some(val) = entry.get("format").and_then(|v| v.as_str()) {
        layout.format = val.to_string();
    } else if let Some(val) = entry.get("codec").and_then(|v| v.as_str()) {
        // Derive format from codec as a default when not explicitly set.
        layout.format = val.to_string();
    }

    // ── alignment_bytes ──────────────────────────────────────────────────
    if let Some(val) = entry.get("alignment_bytes").and_then(|v| v.as_u64()) {
        layout.alignment_bytes = val as u32;
    }

    Ok(layout)
}

// ── Tensor key match generation ────────────────────────────────────────────

/// Generate glob-style tensor key match patterns from a tensor class name.
fn build_tensor_key_matches(tensor_class: &str) -> Vec<String> {
    let base = tensor_class.to_lowercase();
    let mut patterns = vec![format!("*{base}*")];

    if base.contains("qkv") {
        patterns.push("*qkv_proj*".to_string());
    } else if base.contains("attn") || base.contains("attention") {
        patterns.push("*attn*proj*".to_string());
    } else if base.contains("mlp") || base.contains("ffn") {
        patterns.push("*mlp*proj*".to_string());
        patterns.push("*ffn*proj*".to_string());
    } else if base.contains("embed") || base.contains("tok_embed") {
        patterns.push("*embed*weight*".to_string());
    } else if base.contains("lm_head") || base.contains("output") {
        patterns.push("*lm_head*".to_string());
        patterns.push("*output_proj*".to_string());
    }

    patterns
}

// ── Training method selection ──────────────────────────────────────────────

/// Select the appropriate `QuantTrainingMethod` based on the codec family and
/// any `training_method` override in the policy entry.
fn select_training_method(entry: &Value, codec: CodecFamily) -> QuantTrainingMethod {
    // Check for explicit training_method override in the entry.
    if let Some(method_str) = get_str(entry, "training_method") {
        return match method_str.to_lowercase().as_str() {
            "shadow_weights_ste" | "shadow-weights-ste" | "ste" => {
                QuantTrainingMethod::ShadowWeightsSte
            }
            "gradual_bit_transition" | "gradual-bit-transition" | "gbt" => {
                QuantTrainingMethod::GradualBitTransition {
                    start_bits: get_f32(entry, "start_bits", 8.0),
                    target_bits: get_f32(entry, "target_bits", 3.0),
                    schedule_steps: get_usize(entry, "schedule_steps", 1000),
                }
            }
            "soft_ternarization" | "soft-ternarization" => {
                QuantTrainingMethod::SoftTernarization {
                    temperature_start: get_f32(entry, "temperature_start", 1.0),
                    temperature_end: get_f32(entry, "temperature_end", 0.1),
                    learnable_modulation: get_bool(entry, "learnable_modulation", false),
                }
            }
            "activation_weighted" | "activation-weighted" | "aw" => {
                let profile_required = get_bool(entry, "profile_required", true);
                let profile_source = get_str(entry, "profile_source").unwrap_or("calibration");
                QuantTrainingMethod::ActivationWeighted {
                    profile_required,
                    objective: ActivationWeightedObjective {
                        profile_source: profile_source.to_string(),
                        activation_norm: get_str(entry, "activation_norm").unwrap_or("l2").to_string(),
                        percentile: get_f64(entry, "percentile", 95.0),
                        top_k_fraction: get_f64(entry, "top_k_fraction", 0.1),
                    },
                }
            }
            _ => QuantTrainingMethod::ShadowWeightsSte,
        };
    }

    // Default: pick based on codec family.
    match codec {
        CodecFamily::Ternary => QuantTrainingMethod::GradualBitTransition {
            start_bits: 8.0,
            target_bits: 2.0,
            schedule_steps: 2000,
        },
        CodecFamily::Nf4 => QuantTrainingMethod::ShadowWeightsSte,
        CodecFamily::Int8 => QuantTrainingMethod::ShadowWeightsSte,
        _ => QuantTrainingMethod::ShadowWeightsSte,
    }
}

// ── Gate construction ──────────────────────────────────────────────────────

/// Build `WeightTrainingGates` from a policy entry's `gates` sub-object.
fn build_gates(entry: &Value) -> WeightTrainingGates {
    let gates = entry.get("gates");

    let required_level = match get_str(entry, "required_evidence_level")
        .or_else(|| gates.and_then(|g| get_str(g, "required_evidence_level")))
    {
        Some("weight_space" | "WeightSpace") => RequiredEvidenceLevel::WeightSpace,
        Some("synthetic_operator" | "SyntheticOperator") => RequiredEvidenceLevel::SyntheticOperator,
        Some("hardware_operator" | "HardwareOperator") => RequiredEvidenceLevel::HardwareOperator,
        Some("model_quality" | "ModelQuality") => RequiredEvidenceLevel::ModelQuality,
        Some("runtime_profiled" | "RuntimeProfiled") => RequiredEvidenceLevel::RuntimeProfiled,
        Some("production_promoted" | "ProductionPromoted") => RequiredEvidenceLevel::ProductionPromoted,
        _ => RequiredEvidenceLevel::SyntheticOperator,
    };

    let extract_f64 = |key: &str| -> Option<f64> {
        gates
            .and_then(|g| g.get(key))
            .or_else(|| entry.get(key))
            .and_then(|v| v.as_f64())
    };

    WeightTrainingGates {
        max_weight_nrmse: extract_f64("max_weight_nrmse"),
        max_zero_collapse_ratio: extract_f64("max_zero_collapse_ratio"),
        max_operator_nrmse: extract_f64("max_operator_nrmse"),
        min_operator_cosine: extract_f64("min_operator_cosine"),
        max_operator_abs_error: extract_f64("max_operator_abs_error"),
        min_byte_savings_ratio: extract_f64("min_byte_savings_ratio"),
        required_evidence_level: required_level,
    }
}

// ── Priority extraction ────────────────────────────────────────────────────

/// Extract priority from a policy entry field.
fn extract_priority(entry: &Value) -> TrainingTargetPriority {
    match get_str(entry, "priority") {
        Some("required" | "Required") => TrainingTargetPriority::Required,
        Some("recommended" | "Recommended") => TrainingTargetPriority::Recommended,
        Some("experimental" | "Experimental") => TrainingTargetPriority::Experimental,
        Some("research" | "Research") => TrainingTargetPriority::Research,
        _ => TrainingTargetPriority::Recommended,
    }
}

// ── Resolver implementation ────────────────────────────────────────────────

impl TrainingTargetResolver {
    /// Resolve a compiler policy JSON value into a vector of training target
    /// specifications.
    ///
    /// Each entry in the policy's `entries` array is scanned. For codec
    /// families with explicit QAT support (Ternary), a target is always
    /// produced. Other families produce targets only when the corresponding
    /// experimental flag is enabled.
    pub fn resolve(
        &self,
        policy: &Value,
        options: &TrainingTargetResolveOptions,
    ) -> Result<Vec<TrainingTargetSpec>, TrainingTargetResolveError> {
        // Compute the policy digest once (deterministic — same JSON bytes
        // always produce the same hash).
        let policy_json_bytes = serde_json::to_vec(policy)
            .map_err(|e| TrainingTargetResolveError::InvalidPolicy(e.to_string()))?;
        let source_policy_digest = compute_digest(&policy_json_bytes);

        // Top-level metadata.
        let model_family = get_str(policy, "model_family")
            .unwrap_or("unknown")
            .to_string();
        let target_cimage_profile = get_str(policy, "target_cimage_profile")
            .unwrap_or("default")
            .to_string();

        // Locate the entries array.
        let entries = policy.get("entries").and_then(|v| v.as_array()).ok_or_else(
            || TrainingTargetResolveError::InvalidPolicy("policy must contain an 'entries' array".into()),
        )?;

        let mut weight_targets: Vec<WeightTrainingTarget> = Vec::new();
        let mut evidence_gates: Vec<TrainingEvidenceGate> = Vec::new();

        for entry in entries {
            let tensor_class = get_str(entry, "tensor_class")
                .ok_or(TrainingTargetResolveError::MissingTensorClassName)?
                .to_string();

            let codec_str = get_str(entry, "codec")
                .or_else(|| get_str(entry, "codec_family"))
                .or_else(|| get_str(entry, "compression"))
                .unwrap_or("RawF32");

            let codec = parse_codec(codec_str)?;

            // Determine whether to emit a target for this entry.
            let should_emit = codec == CodecFamily::Ternary
                || (codec == CodecFamily::RawF32 && options.experimental_raw_f32)
                || (codec == CodecFamily::Fp16 && options.experimental_fp16_calibration)
                || (codec == CodecFamily::Int8 && options.experimental_int8_calibration)
                || (codec == CodecFamily::Nf4 && options.experimental_nf4_training);

            if !should_emit {
                continue;
            }

            let tensor_key_match = build_tensor_key_matches(&tensor_class);
            let physical_layout = build_layout(entry)?;
            let training_method = select_training_method(entry, codec);
            let gates = build_gates(entry);
            let priority = extract_priority(entry);
            let target_id = format!("{model_family}-{tensor_class}-{codec_str}");

            weight_targets.push(WeightTrainingTarget {
                target_id,
                tensor_class: tensor_class.clone(),
                tensor_key_match,
                target_codec: codec,
                physical_layout,
                training_method,
                gates: gates.clone(),
                priority,
            });

            // Emit a standard evidence gate for this target.
            evidence_gates.push(TrainingEvidenceGate {
                gate_id: format!("gate::{tensor_class}::operator_nrmse"),
                gate_type: "OperatorNRMSE".to_string(),
                threshold: 0.05,
                weight: 1.0,
                required: priority == TrainingTargetPriority::Required,
            });
        }

        Ok(vec![TrainingTargetSpec {
            spec_version: 1,
            model_family,
            model_digest: None,
            source_policy_digest,
            target_cimage_profile,
            weight_targets,
            kv_cache_target: None,
            speculative_targets: Vec::new(),
            engram_targets: Vec::new(),
            attention_shape_targets: Vec::new(),
            evidence_gates,
        }])
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn resolver_generates_ternary_target() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "test-model",
            "target_cimage_profile": "test-profile",
            "entries": [
                {
                    "tensor_class": "attn_q",
                    "codec": "Ternary",
                    "priority": "required",
                    "gates": {
                        "max_operator_nrmse": 0.05,
                        "min_operator_cosine": 0.95
                    }
                }
            ]
        });

        let result = resolver.resolve(&policy, &options).unwrap();
        assert_eq!(result.len(), 1);
        let spec = &result[0];

        assert!(!spec.weight_targets.is_empty(), "expected at least one weight target for ternary entry");
        assert_eq!(spec.weight_targets[0].tensor_class, "attn_q");
        assert_eq!(spec.weight_targets[0].target_codec, CodecFamily::Ternary);
        assert!(spec.weight_targets[0].tensor_key_match.contains(&"*attn_q*".to_string()));

        assert!(!spec.evidence_gates.is_empty(), "expected evidence gates");
        assert_eq!(spec.evidence_gates[0].gate_id, "gate::attn_q::operator_nrmse");

        assert_eq!(spec.source_policy_digest.len(), 64);
    }

    #[test]
    fn resolver_skips_rawf32() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "test-model",
            "target_cimage_profile": "test-profile",
            "entries": [
                { "tensor_class": "attn_q", "codec": "RawF32" },
                { "tensor_class": "mlp_down", "codec": "RawF32" }
            ]
        });

        let result = resolver.resolve(&policy, &options).unwrap();
        assert_eq!(result.len(), 1);
        let spec = &result[0];

        assert!(
            spec.weight_targets.is_empty(),
            "expected no weight targets for RawF32-only policy, got {}",
            spec.weight_targets.len()
        );
    }

    #[test]
    fn resolver_is_deterministic() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "determinism-test",
            "target_cimage_profile": "deterministic",
            "entries": [
                { "tensor_class": "attn_q", "codec": "Ternary", "priority": "required" }
            ]
        });

        let result1 = resolver.resolve(&policy, &options).unwrap();
        let result2 = resolver.resolve(&policy, &options).unwrap();

        let bytes1 = serde_json::to_vec(&result1).unwrap();
        let bytes2 = serde_json::to_vec(&result2).unwrap();

        assert_eq!(bytes1, bytes2, "resolve() must be deterministic");
    }

    #[test]
    fn resolver_experimental_flags() {
        

        let resolver = TrainingTargetResolver;
        let default_opts = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "experimental-test",
            "target_cimage_profile": "experimental",
            "entries": [
                { "tensor_class": "w1", "codec": "Int8" },
                { "tensor_class": "w2", "codec": "NF4" },
                { "tensor_class": "w3", "codec": "FP16" }
            ]
        });

        let result = resolver.resolve(&policy, &default_opts).unwrap();
        assert_eq!(result[0].weight_targets.len(), 0, "expected no targets without experimental flags");

        let all_opts = TrainingTargetResolveOptions {
            experimental_int8_calibration: true,
            experimental_nf4_training: true,
            experimental_fp16_calibration: true,
            experimental_raw_f32: true,
        };

        let result = resolver.resolve(&policy, &all_opts).unwrap();
        assert_eq!(result[0].weight_targets.len(), 3, "expected 3 targets with all experimental flags");
    }

    #[test]
    fn resolver_missing_tensor_class_error() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "test",
            "target_cimage_profile": "test",
            "entries": [{ "codec": "Ternary" }]
        });

        let err = resolver.resolve(&policy, &options).unwrap_err();
        assert!(
            matches!(err, TrainingTargetResolveError::MissingTensorClassName),
            "expected MissingTensorClassName error, got {err:?}"
        );
    }

    #[test]
    fn resolver_unknown_codec_error() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "test",
            "target_cimage_profile": "test",
            "entries": [{ "tensor_class": "w1", "codec": "BogusCodec99" }]
        });

        let err = resolver.resolve(&policy, &options).unwrap_err();
        assert!(
            matches!(err, TrainingTargetResolveError::UnsupportedCodec(_)),
            "expected UnsupportedCodec error, got {err:?}"
        );
    }

    #[test]
    fn resolver_missing_entries_error() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions::default();

        let policy = json!({
            "model_family": "test",
            "target_cimage_profile": "test"
        });

        let err = resolver.resolve(&policy, &options).unwrap_err();
        assert!(
            matches!(err, TrainingTargetResolveError::InvalidPolicy(_)),
            "expected InvalidPolicy error, got {err:?}"
        );
    }

    #[test]
    fn test_compute_digest() {
        let input = b"hello world";
        let digest = compute_digest(input);
        assert_eq!(
            digest,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
            "sha256 of 'hello world' should match known value"
        );
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn training_layout_parses_from_policy() {
        let resolver = TrainingTargetResolver;
        let options = TrainingTargetResolveOptions {
            experimental_raw_f32: true,
            ..Default::default()
        };

        let policy = json!({
            "model_family": "layout-test",
            "target_cimage_profile": "test",
            "entries": [
                {
                    "tensor_class": "test_weight",
                    "codec": "RawF32",
                    "tile_family": "Tile640",
                    "group_size": 64,
                    "group_axis": "PackedContiguous",
                    "metadata_layout": "AdjacentTile",
                    "alignment_bytes": 512
                }
            ]
        });

        let result = resolver.resolve(&policy, &options).unwrap();
        let target = &result[0].weight_targets[0];

        assert_eq!(target.physical_layout.tile_family.name, "Tile640");
        assert_eq!(target.physical_layout.tile_shape.rows, 640);
        assert_eq!(target.physical_layout.group_size, 64);
        assert_eq!(
            target.physical_layout.group_axis,
            crate::execution_profile::GroupAxis::PackedContiguous
        );
        assert_eq!(
            target.physical_layout.metadata_layout,
            crate::execution_profile::MetadataLayout::AdjacentTile
        );
        assert_eq!(target.physical_layout.alignment_bytes, 512);
    }
}
