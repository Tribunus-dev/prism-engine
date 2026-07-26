//! Kernel generation — template selection, parameter resolution, and
//! template expansion.
//!
//! This module owns the canonical authority for the post-dispatch
//! kernel-generation step: given a `Dispatch` entity with a fusion
//! group, a codec, and a backend target, the module
//!
//! 1. **Selects** a kernel template by mapping the root op kind +
//!    codec to a `KernelTemplateId`.
//! 2. **Resolves** the `KernelParameters` from the dispatch's shape
//!    and the template's family.
//! 3. **Expands** the template source by substituting
//!    `{{PLACEHOLDER}}` markers with the parameter values, rejecting
//!    unknown placeholders and unexpanded remnants.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The kernel lowerer (the template's `from_source` lives in the
//!   AOT crate; this module consumes the existing surface).
//! - The dispatch entity lifecycle (owned by fusion scheduling).
//! - The tuning spec (owned by `prism-ecs-compile::hardware_tuning`).
//!
//! The module exposes pure-function entry points: `select_template`,
//! `resolve_parameters`, and `TemplateExpander::expand`. The schedule
//! is responsible for staging the result through a `WorldTxn`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Kernel family / template ID
// ---------------------------------------------------------------------------

/// Canonical kernel family. Each variant is a stable, vendor-neutral
/// name for a kernel pattern. The family is the bridge between the
/// compile-path fusion IR and the AOT template registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelFamily {
    /// Tiled GEMV with NF4 (4-bit) packed weights.
    GemvNf4Tile,
    /// Tiled GEMV with INT8 packed weights.
    GemvInt8Tile,
    /// Fused MLP projection (gate + up + down + activation).
    MlpFused,
    /// Fused attention score probe.
    AttentionScores,
    /// Gemma4-style staged decoder layer.
    DecoderLayerStaged,
}

/// Template ID — picks the AOT template that implements a given root
/// op / codec combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelTemplateId {
    Nf4Tile640Gemv,
    Int8Tile640Gemv,
    FusedGateUp,
    FusedGateUpActivation,
    FusedDownProjResidual,
    FusedOProjResidual,
    FusedRmsNormQkv,
    FusedAttentionScoreProbe,
    Gemma4FullInt4,
    RawF32Matmul,
    Fp16Matmul,
}

impl KernelTemplateId {
    /// Stable template name (matches the on-disk AOT template key).
    pub fn name(self) -> &'static str {
        match self {
            Self::Nf4Tile640Gemv => "nf4_tile640_gemv",
            Self::Int8Tile640Gemv => "int8_tile640_gemv",
            Self::FusedGateUp => "fused_gate_up",
            Self::FusedGateUpActivation => "fused_gate_up_activation",
            Self::FusedDownProjResidual => "fused_down_proj_residual",
            Self::FusedOProjResidual => "fused_o_proj_residual",
            Self::FusedRmsNormQkv => "fused_rms_norm_qkv",
            Self::FusedAttentionScoreProbe => "fused_attention_score_probe",
            Self::Gemma4FullInt4 => "gemma4_full_int4",
            Self::RawF32Matmul => "raw_f32_matmul",
            Self::Fp16Matmul => "fp16_matmul",
        }
    }

    /// Default entry point in the compiled kernel.
    pub fn default_entry_point(self) -> &'static str {
        self.name()
    }
}

/// Codec family — the dispatch's weight codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CodecFamily {
    Nf4,
    Int8,
    Fp16,
    RawF32,
}

// ---------------------------------------------------------------------------
// Kernel parameters
// ---------------------------------------------------------------------------

/// Accumulator / output dtype for the generated kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    Fp32,
    Fp16,
    Bf16,
    I8,
    I32,
}

/// Concrete parameters for one kernel instance. The values are
/// computed at generation time from the dispatch's shape and the
/// template's family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelParameters {
    pub kernel_family: KernelFamily,
    pub codec_family: CodecFamily,
    pub tile_width: u32,
    pub group_size: u32,
    pub threadgroup_size: u32,
    pub simdgroup_width: u32,
    pub groups_per_tile: u32,
    pub lane_values: u32,
    pub unroll_factor: u32,
    pub use_threadgroup_memory: bool,
    pub prefetch_distance: u32,
    pub accumulation_dtype: DType,
    pub output_dtype: DType,
}

impl KernelParameters {
    /// Map the parameters into a `BTreeMap<String, String>` suitable
    /// for template substitution. The BTreeMap keeps the iteration
    /// order deterministic across replays.
    pub fn to_placeholder_map(&self) -> BTreeMap<String, String> {
        let mut m: BTreeMap<String, String> = BTreeMap::new();
        m.insert("TILE_WIDTH".into(), self.tile_width.to_string());
        m.insert("GROUP_SIZE".into(), self.group_size.to_string());
        m.insert("THREADGROUP_SIZE".into(), self.threadgroup_size.to_string());
        m.insert("SIMDGROUP_WIDTH".into(), self.simdgroup_width.to_string());
        m.insert("GROUPS_PER_TILE".into(), self.groups_per_tile.to_string());
        m.insert("LANE_VALUES".into(), self.lane_values.to_string());
        m.insert("UNROLL_FACTOR".into(), self.unroll_factor.to_string());
        m.insert(
            "USE_THREADGROUP_MEMORY".into(),
            (u32::from(self.use_threadgroup_memory)).to_string(),
        );
        m.insert("PREFETCH_DISTANCE".into(), self.prefetch_distance.to_string());
        m.insert("ACCUMULATION_DTYPE".into(), dtype_label(self.accumulation_dtype).into());
        m.insert("OUTPUT_DTYPE".into(), dtype_label(self.output_dtype).into());
        m.insert(
            "KERNEL_FAMILY".into(),
            kernel_family_label(self.kernel_family).into(),
        );
        m
    }
}

fn dtype_label(d: DType) -> &'static str {
    match d {
        DType::Fp32 => "fp32",
        DType::Fp16 => "fp16",
        DType::Bf16 => "bf16",
        DType::I8 => "i8",
        DType::I32 => "i32",
    }
}

fn kernel_family_label(f: KernelFamily) -> &'static str {
    match f {
        KernelFamily::GemvNf4Tile => "gemv_nf4_tile",
        KernelFamily::GemvInt8Tile => "gemv_int8_tile",
        KernelFamily::MlpFused => "mlp_fused",
        KernelFamily::AttentionScores => "attention_scores",
        KernelFamily::DecoderLayerStaged => "decoder_layer_staged",
    }
}

// ---------------------------------------------------------------------------
// Kernel source
// ---------------------------------------------------------------------------

/// Source-language for the generated kernel. The shader language is
/// recorded alongside the source so downstream dispatchers can route
/// to the correct backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShaderLanguage {
    Msl,
    Glsl,
    Hlsl,
    OpenClC,
}

/// A kernel's source — language, expanded text, and entry point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSource {
    pub language: ShaderLanguage,
    pub source: String,
    pub entry_point: String,
}

impl prism_ecs_core::Component for KernelSource {}

// ---------------------------------------------------------------------------
// Template errors
// ---------------------------------------------------------------------------

/// Errors produced by template selection, parameter resolution, and
/// template expansion.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TemplateError {
    #[error("template `{template}` is missing a value for placeholder `{placeholder}`")]
    MissingValue {
        template: String,
        placeholder: String,
    },
    #[error("template `{template}` references unknown placeholder `{placeholder}`")]
    UnknownPlaceholder {
        template: String,
        placeholder: String,
    },
    #[error("template `{template}` has unexpanded placeholder remnant `{remnant}`")]
    UnexpandedPlaceholder {
        template: String,
        remnant: String,
    },
}

// ---------------------------------------------------------------------------
// Template selection
// ---------------------------------------------------------------------------

/// Select a kernel template for a root op kind and codec.
#[must_use]
pub fn select_template(root_op: &str, codec: CodecFamily) -> KernelTemplateId {
    match (root_op, codec) {
        ("mlp_gate_up", CodecFamily::Nf4) => KernelTemplateId::Nf4Tile640Gemv,
        ("mlp_gate_up", CodecFamily::Int8) => KernelTemplateId::Int8Tile640Gemv,
        ("mlp_down", CodecFamily::Nf4) => KernelTemplateId::FusedDownProjResidual,
        ("mlp_down", CodecFamily::Int8) => KernelTemplateId::Int8Tile640Gemv,
        ("fused_gate_up", _) => KernelTemplateId::FusedGateUp,
        ("fused_down_proj", _) => KernelTemplateId::FusedDownProjResidual,
        ("fused_o_proj", _) => KernelTemplateId::FusedOProjResidual,
        ("fused_rms_norm_qkv", _) => KernelTemplateId::FusedRmsNormQkv,
        ("attention_score", _) => KernelTemplateId::FusedAttentionScoreProbe,
        ("gemma4_full", _) => KernelTemplateId::Gemma4FullInt4,
        _ if codec == CodecFamily::RawF32 => KernelTemplateId::RawF32Matmul,
        _ if codec == CodecFamily::Fp16 => KernelTemplateId::Fp16Matmul,
        _ => KernelTemplateId::Nf4Tile640Gemv,
    }
}

// ---------------------------------------------------------------------------
// Parameter resolution
// ---------------------------------------------------------------------------

/// Resolve `KernelParameters` from a template name and an optional
/// tile width. The default tile width is 640 (the canonical
/// NF4-tile width) when the caller has no shape information.
#[must_use]
pub fn resolve_parameters(template: KernelTemplateId, tile_width: Option<u32>) -> KernelParameters {
    let tile_width = tile_width.unwrap_or(640);
    let family = template_name_to_family(template);
    KernelParameters {
        kernel_family: family,
        codec_family: codec_for_template(template),
        tile_width,
        group_size: 32,
        threadgroup_size: 256,
        simdgroup_width: 32,
        groups_per_tile: tile_width / 32,
        lane_values: 4,
        unroll_factor: 2,
        use_threadgroup_memory: true,
        prefetch_distance: 2,
        accumulation_dtype: DType::Fp32,
        output_dtype: DType::Fp16,
    }
}

fn codec_for_template(t: KernelTemplateId) -> CodecFamily {
    match t {
        KernelTemplateId::Nf4Tile640Gemv
        | KernelTemplateId::FusedGateUp
        | KernelTemplateId::FusedGateUpActivation
        | KernelTemplateId::FusedDownProjResidual
        | KernelTemplateId::FusedOProjResidual
        | KernelTemplateId::FusedRmsNormQkv
        | KernelTemplateId::FusedAttentionScoreProbe
        | KernelTemplateId::Gemma4FullInt4 => CodecFamily::Nf4,
        KernelTemplateId::Int8Tile640Gemv => CodecFamily::Int8,
        KernelTemplateId::RawF32Matmul => CodecFamily::RawF32,
        KernelTemplateId::Fp16Matmul => CodecFamily::Fp16,
    }
}

fn template_name_to_family(t: KernelTemplateId) -> KernelFamily {
    match t {
        KernelTemplateId::Nf4Tile640Gemv => KernelFamily::GemvNf4Tile,
        KernelTemplateId::Int8Tile640Gemv => KernelFamily::GemvInt8Tile,
        KernelTemplateId::FusedGateUp
        | KernelTemplateId::FusedGateUpActivation
        | KernelTemplateId::FusedDownProjResidual
        | KernelTemplateId::FusedOProjResidual
        | KernelTemplateId::FusedRmsNormQkv => KernelFamily::MlpFused,
        KernelTemplateId::FusedAttentionScoreProbe => KernelFamily::AttentionScores,
        KernelTemplateId::Gemma4FullInt4 => KernelFamily::DecoderLayerStaged,
        KernelTemplateId::RawF32Matmul => KernelFamily::GemvInt8Tile,
        KernelTemplateId::Fp16Matmul => KernelFamily::GemvNf4Tile,
    }
}

// ---------------------------------------------------------------------------
// Template expansion
// ---------------------------------------------------------------------------

/// Strict template expander. Rejects unknown placeholders and
/// unexpanded `{{...}}` patterns in the result.
#[derive(Debug, Default, Clone)]
pub struct TemplateExpander;

impl TemplateExpander {
    pub fn new() -> Self {
        Self
    }

    /// Expand `template.source` by substituting `{{KEY}}` markers
    /// with values from `params.to_placeholder_map()`.
    ///
    /// The expander is strict:
    /// - Unknown placeholders are rejected with
    ///   `TemplateError::UnknownPlaceholder`.
    /// - After expansion, any remaining `{{...}}` pattern is rejected
    ///   with `TemplateError::UnexpandedPlaceholder`.
    pub fn expand(
        &self,
        template: &str,
        template_id: &str,
        params: &KernelParameters,
    ) -> Result<String, TemplateError> {
        let entries = params.to_placeholder_map();
        let known: std::collections::BTreeSet<&str> = entries.keys().map(String::as_str).collect();

        let mut result = String::with_capacity(template.len());
        let mut chars = template.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'
                let mut placeholder = String::new();
                let mut terminated = false;
                loop {
                    match chars.next() {
                        None => {
                            return Err(TemplateError::MissingValue {
                                template: template_id.to_string(),
                                placeholder,
                            });
                        }
                        Some('}') if chars.peek() == Some(&'}') => {
                            chars.next();
                            terminated = true;
                            break;
                        }
                        Some('}') => placeholder.push('}'),
                        Some(ch) => placeholder.push(ch),
                    }
                }
                if !terminated {
                    return Err(TemplateError::UnexpandedPlaceholder {
                        template: template_id.to_string(),
                        remnant: placeholder,
                    });
                }                if let Some(value) = entries.get(placeholder.as_str()) {
                    result.push_str(value);
                } else if !known.contains(placeholder.as_str()) {
                    return Err(TemplateError::UnknownPlaceholder {
                        template: template_id.to_string(),
                        placeholder,
                    });
                } else {
                    // Known key but missing value (shouldn't happen if
                    // the parameter set is complete) — fall through to
                    // the unexpanded check.
                    result.push_str(&placeholder);
                }
            } else {
                result.push(c);
            }
        }

        // Post-expansion validation: any remaining `{{...}}` is an
        // unexpanded remnant.
        let mut chars = result.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next();
                let mut remnant = String::new();
                loop {
                    match chars.next() {
                        None => break,
                        Some('}') if chars.peek() == Some(&'}') => {
                            chars.next();
                            if !remnant.is_empty() {
                                return Err(TemplateError::UnexpandedPlaceholder {
                                    template: template_id.to_string(),
                                    remnant,
                                });
                            }
                            break;
                        }
                        Some(ch) => remnant.push(ch),
                    }
                }
            }
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params() -> KernelParameters {
        KernelParameters {
            kernel_family: KernelFamily::GemvNf4Tile,
            codec_family: CodecFamily::Nf4,
            tile_width: 640,
            group_size: 128,
            threadgroup_size: 32,
            simdgroup_width: 32,
            groups_per_tile: 5,
            lane_values: 4,
            unroll_factor: 4,
            use_threadgroup_memory: false,
            prefetch_distance: 2,
            accumulation_dtype: DType::Fp32,
            output_dtype: DType::Fp16,
        }
    }

    #[test]
    fn expander_rejects_unknown_placeholder() {
        let expander = TemplateExpander::new();
        let r = expander.expand(
            "const uint X = {{UNKNOWN_VAR}};",
            "test",
            &sample_params(),
        );
        assert!(matches!(
            r,
            Err(TemplateError::UnknownPlaceholder { ref placeholder, .. }) if placeholder == "UNKNOWN_VAR"
        ));
    }

    #[test]
    fn expander_substitutes_known_placeholders() {
        let expander = TemplateExpander::new();
        let r = expander.expand(
            "const uint TW = {{TILE_WIDTH}};\nconst uint GS = {{GROUP_SIZE}};",
            "test",
            &sample_params(),
        )
        .expect("expand succeeds");
        assert!(r.contains("TW = 640;"));
        assert!(r.contains("GS = 128;"));
    }

    #[test]
    fn expander_detects_unexpanded_remnant() {
        let expander = TemplateExpander::new();
        // TILE_WIDTH is in the placeholder map; NOT_IN_PARAMS is not.
        let r = expander.expand(
            "const uint TW = {{TILE_WIDTH}};\nconst uint BAD = {{NOT_IN_PARAMS}};",
            "test",
            &sample_params(),
        );
        assert!(matches!(
            r,
            Err(TemplateError::UnknownPlaceholder { ref placeholder, .. }) if placeholder == "NOT_IN_PARAMS"
        ));
    }

    #[test]
    fn expander_preserves_text_outside_placeholders() {
        let expander = TemplateExpander::new();
        let r = expander
            .expand(
                "kernel void {{ENTRY}}() { /* body */ }",
                "test",
                &sample_params(),
            )
            .expect("expand succeeds");
        assert!(r.contains("kernel void {{ENTRY}}()"));
    }

    #[test]
    fn select_template_picks_nf4_tile_for_mlp_gate_up_with_nf4() {
        let t = select_template("mlp_gate_up", CodecFamily::Nf4);
        assert_eq!(t, KernelTemplateId::Nf4Tile640Gemv);
    }

    #[test]
    fn select_template_picks_int8_tile_for_mlp_gate_up_with_int8() {
        let t = select_template("mlp_gate_up", CodecFamily::Int8);
        assert_eq!(t, KernelTemplateId::Int8Tile640Gemv);
    }

    #[test]
    fn select_template_picks_fused_for_fused_root() {
        let t = select_template("fused_o_proj", CodecFamily::RawF32);
        assert_eq!(t, KernelTemplateId::FusedOProjResidual);
    }

    #[test]
    fn select_template_falls_back_to_raw_f32_matmul() {
        let t = select_template("custom_op", CodecFamily::RawF32);
        assert_eq!(t, KernelTemplateId::RawF32Matmul);
    }

    #[test]
    fn select_template_falls_back_to_fp16_matmul() {
        let t = select_template("custom_op", CodecFamily::Fp16);
        assert_eq!(t, KernelTemplateId::Fp16Matmul);
    }

    #[test]
    fn select_template_unknown_codec_defaults_to_nf4() {
        let t = select_template("custom_op", CodecFamily::Nf4);
        assert_eq!(t, KernelTemplateId::Nf4Tile640Gemv);
    }

    #[test]
    fn resolve_parameters_uses_default_tile_width() {
        let p = resolve_parameters(KernelTemplateId::Nf4Tile640Gemv, None);
        assert_eq!(p.tile_width, 640);
        assert_eq!(p.kernel_family, KernelFamily::GemvNf4Tile);
        assert_eq!(p.codec_family, CodecFamily::Nf4);
        assert_eq!(p.groups_per_tile, 20);
    }

    #[test]
    fn resolve_parameters_uses_explicit_tile_width() {
        let p = resolve_parameters(KernelTemplateId::Int8Tile640Gemv, Some(1280));
        assert_eq!(p.tile_width, 1280);
        assert_eq!(p.groups_per_tile, 40);
    }

    #[test]
    fn template_id_names_are_stable() {
        assert_eq!(KernelTemplateId::Nf4Tile640Gemv.name(), "nf4_tile640_gemv");
        assert_eq!(
            KernelTemplateId::FusedGateUp.name(),
            "fused_gate_up"
        );
        assert_eq!(
            KernelTemplateId::FusedGateUp.default_entry_point(),
            "fused_gate_up"
        );
    }

    #[test]
    fn kernel_parameters_serialize_round_trip() {
        let p = sample_params();
        let s = serde_json::to_string(&p).expect("serialize");
        let back: KernelParameters = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(p, back);
    }

    #[test]
    fn kernel_parameters_placeholder_map_is_sorted() {
        let p = sample_params();
        let m = p.to_placeholder_map();
        let keys: Vec<&String> = m.keys().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
