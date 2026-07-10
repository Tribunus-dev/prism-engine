//! Kernel parameters and kernel family types for AOT kernel variant generation.
//!
//! `KernelParameters` describes one fully-resolved set of Metal kernel
//! compile-time constants for a specific kernel operation and hardware profile.

use serde::{Deserialize, Serialize};

// ── Kernel family ────────────────────────────────────────────────────────

/// Which operation this kernel implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum KernelFamily {
    GemvInt8Tile,
    GemvNf4Tile,
    GemvTernaryTile,
    /// Q8_0 block-f32 scale + int8 GEMV (32-element blocks)
    GemvQ8_0,
    #[allow(non_camel_case_types)]
    /// Q4_K K-quant 4-bit GEMV (256-element super-blocks)
    GemvQ4_K,
    #[allow(non_camel_case_types)]
    /// Q2_K K-quant 2-bit GEMV (256-element super-blocks)
    GemvQ2_K,
    #[allow(non_camel_case_types)]
    /// IQ2_XXS importance-weighted 2-bit GEMV with codebook
    GemvIQ2_XXS,
    /// MoE expert router (top-k gating)
    MoeRouter,
    /// MoE sparse matmul (selected expert forward)
    MoeSparseMatmul,
    /// Shared expert MLP (dense MLP shared across tokens)
    SharedExpertMlp,
    /// Compressed attention (grouped-query with latent compression)
    CompressedAttention,
    RmsNorm,
    Rope,
    AttentionScores,
    AttentionApply,
    MlpFused,
    DecoderLayerStaged,
}

impl KernelFamily {
    pub fn name(&self) -> &'static str {
        match self {
            Self::GemvInt8Tile => "gemv_int8_tile",
            Self::GemvNf4Tile => "gemv_nf4_tile",
            Self::GemvTernaryTile => "gemv_ternary_tile",
            Self::GemvQ8_0 => "gemv_q8_0",
            Self::GemvQ4_K => "gemv_q4_k",
            Self::GemvQ2_K => "gemv_q2_k",
            Self::GemvIQ2_XXS => "gemv_iq2_xxs",
            Self::MoeRouter => "moe_router",
            Self::MoeSparseMatmul => "moe_sparse_matmul",
            Self::SharedExpertMlp => "shared_expert_mlp",
            Self::CompressedAttention => "compressed_attention",
            Self::RmsNorm => "rms_norm",
            Self::Rope => "rope",
            Self::AttentionScores => "attention_scores",
            Self::AttentionApply => "attention_apply",
            Self::MlpFused => "mlp_fused",
            Self::DecoderLayerStaged => "decoder_layer_staged",
        }
    }
}

// ── Data type ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    Fp32,
    Fp16,
    Bf16,
    Int8,
}

// ── Kernel parameters ────────────────────────────────────────────────────

/// Fully-resolved compile-time constants for one kernel variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelParameters {
    pub kernel_family: KernelFamily,
    pub codec_family: crate::ecs::plan::CodecFamily,
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
    /// Placeholder names required by the template expansion system.
    pub fn required_placeholders() -> Vec<&'static str> {
        vec![
            "TILE_WIDTH",
            "GROUP_SIZE",
            "GROUPS_PER_TILE",
            "THREADGROUP_SIZE",
            "SIMDGROUP_WIDTH",
            "LANE_VALUES",
            "UNROLL_FACTOR",
            "PREFETCH_DISTANCE",
            "ACCUM_DTYPE",
            "OUTPUT_DTYPE",
            "USE_TGMEM",
        ]
    }

    /// Produce placeholder -> value pairs for template expansion.
    pub fn to_placeholder_map(&self) -> Vec<(&'static str, String)> {
        vec![
            ("TILE_WIDTH", self.tile_width.to_string()),
            ("GROUP_SIZE", self.group_size.to_string()),
            ("GROUPS_PER_TILE", self.groups_per_tile.to_string()),
            ("THREADGROUP_SIZE", self.threadgroup_size.to_string()),
            ("SIMDGROUP_WIDTH", self.simdgroup_width.to_string()),
            ("LANE_VALUES", self.lane_values.to_string()),
            ("UNROLL_FACTOR", self.unroll_factor.to_string()),
            ("PREFETCH_DISTANCE", self.prefetch_distance.to_string()),
            ("ACCUM_DTYPE", self.dtype_name(self.accumulation_dtype)),
            ("OUTPUT_DTYPE", self.dtype_name(self.output_dtype)),
            (
                "USE_TGMEM",
                if self.use_threadgroup_memory {
                    "1".into()
                } else {
                    "0".into()
                },
            ),
        ]
    }

    fn dtype_name(&self, dt: DType) -> String {
        match dt {
            DType::Fp32 => "float".into(),
            DType::Fp16 => "half".into(),
            DType::Bf16 => "bfloat".into(),
            DType::Int8 => "char".into(),
        }
    }
}
