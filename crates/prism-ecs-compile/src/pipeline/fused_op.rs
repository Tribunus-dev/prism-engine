//! `pipeline::fused_op` — fused operation kinds and precompiled kernel names.
//!
//! This file owns the canonical authority for the [`FusedOperation`] enum
//! that names precompiled fused kernels referenced from [`ScheduledRegion`]
//! fusion entries. The set of names is closed at compile time; the engine
//! does not own additional variants.

/// A named fused kernel executed as a single Metal/ANE launch.
///
/// Each variant corresponds to a precompiled kernel whose
/// [`FusedOperation::kernel_name`] is the Metal entry-point symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FusedOperation {
    /// Fused RMSNorm + Q projection.
    FusedNormQProj,
    /// Fused RMSNorm + K projection.
    FusedNormKProj,
    /// Fused RMSNorm + V projection.
    FusedNormVProj,
    /// Fused FFN gate + up + activation.
    FusedFfnActivation,
    /// Fused residual + RMSNorm.
    FusedResidualNorm,
    /// Fused flash attention.
    FusedFlashAttention,
    /// Fused MoE route (gating + dispatch).
    FusedMoERoute,
    /// Custom operator with caller-supplied name.
    Custom(String),
}

impl FusedOperation {
    /// Return the name of the precompiled Metal kernel.
    pub fn kernel_name(&self) -> &str {
        match self {
            Self::FusedNormQProj => "fused_norm_q_proj",
            Self::FusedNormKProj => "fused_norm_k_proj",
            Self::FusedNormVProj => "fused_norm_v_proj",
            Self::FusedFfnActivation => "fused_ffn_activation",
            Self::FusedResidualNorm => "fused_residual_norm",
            Self::FusedFlashAttention => "fused_flash_attention",
            Self::FusedMoERoute => "fused_moe_route",
            Self::Custom(name) => name.as_str(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_variants_have_distinct_kernel_names() {
        let names = [
            FusedOperation::FusedNormQProj,
            FusedOperation::FusedNormKProj,
            FusedOperation::FusedNormVProj,
            FusedOperation::FusedFfnActivation,
            FusedOperation::FusedResidualNorm,
            FusedOperation::FusedFlashAttention,
            FusedOperation::FusedMoERoute,
        ];
        let mut seen: Vec<&str> = Vec::with_capacity(names.len());
        for op in &names {
            let k = op.kernel_name();
            assert!(!seen.contains(&k), "duplicate kernel name {k}");
            seen.push(k);
        }
    }

    #[test]
    fn custom_passes_name_through() {
        let op = FusedOperation::Custom("my_special_kernel".into());
        assert_eq!(op.kernel_name(), "my_special_kernel");
    }
}
