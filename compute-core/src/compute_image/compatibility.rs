//! CompatibilityMatrix — model architecture vs hardware × quantization rules engine.
//!
//! Validates that a compiled image's configuration (quantization, segment layout,
//! batch size) is compatible with the model's architecture before compilation,
//! and provides a deterministic fallback chain when the preferred configuration
//! is incompatible.
//!
//! # Rules
//!
//! | Condition | Action |
//! |---|---|
//! | `head_dim % NF4_group_size != 0` | NF4 incompatible — packed attention weight can't be decoded |
//! | `num_key_value_heads != num_attention_heads` | GQA detected — warn, validate KV projection handling |
//! | `hidden_size != n_heads × head_dim` | Architecture inconsistency — hard error |

use crate::config::{CompileQuantMode, HardwareTarget, TextArchitecture};
use serde::Serialize;

// ---------------------------------------------------------------------------
// CompileDecision
// ---------------------------------------------------------------------------

/// The result of compatibility evaluation: a validated quantization choice
/// and a receipt documenting what was checked.
#[derive(Clone, Debug, Serialize)]
pub struct CompileDecision {
    /// The quantization mode to use (None = FP16, no quantization).
    #[serde(skip)]
    pub quant_mode: Option<CompileQuantMode>,
    /// Validation receipt documenting checks and any warnings.
    pub validation: ValidationReceipt,
}

/// Validation receipt documenting what was checked at compile-decision time.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ValidationReceipt {
    /// Whether the chosen configuration is fully compatible.
    pub valid: bool,
    /// Non-fatal warnings (e.g. GQA detected, suboptimal quant for hardware).
    pub warnings: Vec<String>,
    /// Incompatibilities that caused fallback (empty = no issues).
    pub incompatibilities: Vec<String>,
    /// Which family the model was identified as.
    pub model_family: String,
    /// Which quant modes were tried and why they failed.
    pub fallback_chain: Vec<FallbackAttempt>,
}

/// Record of one attempted fallback step.
#[derive(Clone, Debug, Serialize)]
pub struct FallbackAttempt {
    pub quant_label: String,
    pub compatible: bool,
    pub reason: Option<String>,
}

// ---------------------------------------------------------------------------
// CompatibilityMatrix
// ---------------------------------------------------------------------------

/// Rules engine that evaluates compilation options against a model architecture.
pub struct CompatibilityMatrix;

impl CompatibilityMatrix {
    /// Evaluate all quantization options and return the best compatible choice.
    ///
    /// `preferred_quant` is the user's explicit choice, or `None` for
    /// hardware-default.  The matrix validates it, and if incompatible,
    /// walks the fallback chain until a compatible option is found.
    pub fn evaluate(
        arch: &TextArchitecture,
        target: &HardwareTarget,
        preferred_quant: Option<CompileQuantMode>,
    ) -> CompileDecision {
        let family = Self::detect_family(arch);
        let mut receipt = ValidationReceipt {
            model_family: family.to_string(),
            ..Default::default()
        };

        // Basic architectural sanity.
        if let Err(issues) = Self::validate_architecture(arch) {
            return CompileDecision {
                quant_mode: None,
                validation: ValidationReceipt {
                    valid: false,
                    incompatibilities: issues,
                    model_family: family.to_string(),
                    ..Default::default()
                },
            };
        }

        // GQA detection.
        if arch.num_key_value_heads != arch.num_attention_heads {
            receipt.warnings.push(format!(
                "GQA detected: {} query heads, {} KV heads. KV projection shapes must use kv_heads, not attn_heads.",
                arch.num_attention_heads, arch.num_key_value_heads
            ));
        }

        // Build the fallback chain: preferred first, then hardware default, then increasingly safe.
        let chain = Self::build_fallback_chain(preferred_quant, target);

        for (i, candidate) in chain.iter().enumerate() {
            let label = candidate
                .as_ref()
                .map(|q| q.name().to_string())
                .unwrap_or_else(|| "none (FP16)".to_string());

            match Self::is_quant_compatible(arch, *candidate) {
                Ok(()) => {
                    if i > 0 {
                        receipt.warnings.push(format!(
                            "preferred quant incompatible, falling back to {}",
                            label
                        ));
                    }
                    receipt.fallback_chain.push(FallbackAttempt {
                        quant_label: label.clone(),
                        compatible: true,
                        reason: None,
                    });
                    receipt.valid = true;
                    return CompileDecision {
                        quant_mode: *candidate,
                        validation: receipt,
                    };
                }
                Err(reasons) => {
                    receipt.fallback_chain.push(FallbackAttempt {
                        quant_label: label.clone(),
                        compatible: false,
                        reason: Some(reasons.join("; ")),
                    });
                    receipt
                        .incompatibilities
                        .push(format!("{}: {}", label, reasons.join(", ")));
                }
            }
        }

        // Fallthrough: nothing compatible (shouldn't happen since "none" is always safe).
        receipt.valid = true;
        receipt
            .warnings
            .push("all quantization options exhausted, using FP16".to_string());
        CompileDecision {
            quant_mode: None,
            validation: receipt,
        }
    }

    /// Check whether a specific quantization mode is compatible with the
    /// model architecture. Returns `Ok(())` if compatible, or `Err` with
    /// reasons for incompatibility.
    pub fn is_quant_compatible(
        arch: &TextArchitecture,
        qmode: Option<CompileQuantMode>,
    ) -> Result<(), Vec<String>> {
        let Some(qmode) = qmode else {
            return Ok(()); // FP16 / no quantization is always safe
        };

        let mut issues = Vec::new();

        match qmode {
            CompileQuantMode::Nf4 { group_size } => {
                // NF4 packing changes the weight tensor's in_dim.
                // The attention runtime must be able to decode the packed
                // layout back into the original head structure.
                //
                // Rule: head_dim must be divisible by group_size for NF4
                // to produce packed shapes the runtime can decode.
                if arch.head_dim % group_size != 0 {
                    issues.push(format!(
                        "NF4 group_size={} requires head_dim ({}) to be divisible by group_size",
                        group_size, arch.head_dim
                    ));
                }

                // For GQA, also check that the KV head dimension is
                // compatible (k_proj and v_proj shapes).
                if arch.num_key_value_heads != arch.num_attention_heads {
                    let kv_hidden = arch.num_key_value_heads * arch.head_dim;
                    if kv_hidden % group_size != 0 {
                        issues.push(format!(
                            "NF4 group_size={} requires KV hidden dim ({}) to be divisible by group_size",
                            group_size, kv_hidden
                        ));
                    }
                }

                // Check intermediate_size for MLP projections.
                if arch.intermediate_size % group_size != 0 {
                    issues.push(format!(
                        "NF4 group_size={} requires intermediate_size ({}) to be divisible by group_size",
                        group_size, arch.intermediate_size
                    ));
                }

                // ── Model-family-specific NF4 checks ────────────────
                // Qwen2 with GQA: the packed KV projection shapes are
                // incompatible with the attention runtime's reshape logic.
                if arch.model_type == "qwen2"
                    && arch.num_key_value_heads != arch.num_attention_heads
                {
                    issues.push(format!(
                        "NF4 is incompatible with Qwen2 GQA attention ({} attn heads, {} KV heads). \
                         Use --quantize 8bit or --quantize none.",
                        arch.num_attention_heads, arch.num_key_value_heads
                    ));
                }
            }
            CompileQuantMode::Af8 { group_size } => {
                // 8-bit affine quantization changes byte layout but preserves
                // logical element count. Always compatible.
            }
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Detect the model family from architecture parameters.
    fn detect_family(arch: &TextArchitecture) -> &'static str {
        match arch.model_type.as_str() {
            "qwen2" => "qwen2",
            "llama" => "llama",
            "gemma" => "gemma",
            "gemma2" => "gemma2",
            "mistral" => "mistral",
            "phi3" | "phi-3" => "phi3",
            _ => "unknown",
        }
    }

    /// Basic architectural sanity checks.
    fn validate_architecture(arch: &TextArchitecture) -> Result<(), Vec<String>> {
        let mut issues = Vec::new();

        let expected_hidden = arch.num_attention_heads * arch.head_dim;
        if arch.hidden_size != expected_hidden {
            issues.push(format!(
                "hidden_size mismatch: config says {}, but n_heads × head_dim = {} × {} = {}",
                arch.hidden_size, arch.num_attention_heads, arch.head_dim, expected_hidden
            ));
        }

        if arch.num_hidden_layers == 0 {
            issues.push("num_hidden_layers is 0".to_string());
        }
        if arch.vocab_size == 0 {
            issues.push("vocab_size is 0".to_string());
        }

        if issues.is_empty() {
            Ok(())
        } else {
            Err(issues)
        }
    }

    /// Build the fallback chain: [preferred, hw_default, nf4-64, nf4-128, 8bit, none].
    fn build_fallback_chain(
        preferred: Option<CompileQuantMode>,
        target: &HardwareTarget,
    ) -> Vec<Option<CompileQuantMode>> {
        let hw_default = CompileQuantMode::from_name(target.recommended_quant());
        let mut chain = Vec::new();

        // Start with explicit user preference.
        if let Some(p) = preferred {
            chain.push(Some(p));
        }

        // Add hardware default if different from preference.
        if let Some(hw) = hw_default {
            if Some(hw) != preferred {
                chain.push(Some(hw));
            }
        }

        // Add increasingly safe fallbacks.
        let fallbacks = [
            CompileQuantMode::Nf4 { group_size: 64 },  // NF4-64
            CompileQuantMode::Nf4 { group_size: 128 }, // NF4-128
            CompileQuantMode::Af8 { group_size: 64 },  // 8-bit
        ];

        for fb in &fallbacks {
            if !chain.iter().any(|c| c.as_ref() == Some(fb)) {
                chain.push(Some(*fb));
            }
        }

        // Always end with FP16 as last resort.
        chain.push(None);

        chain
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qwen2_arch() -> TextArchitecture {
        TextArchitecture {
            hidden_size: 896,
            intermediate_size: 4864,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            head_dim: 64,
            global_head_dim: None,
            num_global_key_value_heads: None,
            num_hidden_layers: 24,
            vocab_size: 151936,
            sliding_window: 0,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: true,
            attention_k_eq_v: false,
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 896,
            layer_types: vec![],
            rope_local: crate::config::RopeSpec {
                theta: 1_000_000.0,
                rope_type: "default".to_string(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            model_type: "qwen2".to_string(),
            moe_config: None,
            diffusion_config: None,
        }
    }

    #[test]
    fn test_qwen2_nf4_64_compatible() {
        // Qwen2.5-0.5B: head_dim=64, group_size=64, but has GQA → NF4 incompatible
        let arch = make_qwen2_arch();
        assert!(
            CompatibilityMatrix::is_quant_compatible(
                &arch,
                Some(CompileQuantMode::Nf4 { group_size: 64 })
            )
            .is_err(),
            "Qwen2 GQA should be incompatible with NF4"
        );
    }

    #[test]
    fn test_qwen2_nf4_128_compatible() {
        // Qwen2.5-0.5B: head_dim=64, group_size=128 → 64 % 128 != 0 → incompatible
        let arch = make_qwen2_arch();
        assert!(CompatibilityMatrix::is_quant_compatible(
            &arch,
            Some(CompileQuantMode::Nf4 { group_size: 128 })
        )
        .is_err());
    }

    #[test]
    fn test_qwen2_af8_always_compatible() {
        let arch = make_qwen2_arch();
        assert!(CompatibilityMatrix::is_quant_compatible(
            &arch,
            Some(CompileQuantMode::Af8 { group_size: 64 })
        )
        .is_ok());
    }

    #[test]
    fn test_qwen2_none_always_compatible() {
        let arch = make_qwen2_arch();
        assert!(CompatibilityMatrix::is_quant_compatible(&arch, None).is_ok());
    }

    #[test]
    fn test_evaluate_preferred_compatible() {
        // Preferred NF4-64 is incompatible for Qwen2 GQA → should fall back to 8bit.
        let arch = make_qwen2_arch();
        let target = HardwareTarget::M1;
        let decision = CompatibilityMatrix::evaluate(
            &arch,
            &target,
            Some(CompileQuantMode::Nf4 { group_size: 64 }),
        );
        assert!(decision.validation.valid);
        // Qwen2 GQA → NF4 blocked → should fall to 8bit
        assert_eq!(
            decision.quant_mode,
            Some(CompileQuantMode::Af8 { group_size: 64 })
        );
    }

    #[test]
    fn test_evaluate_fallback_on_incompatible() {
        // Qwen2 GQA with NF4-128 → falls through all NF4 variants to 8bit.
        let arch = make_qwen2_arch();
        let target = HardwareTarget::M1;
        let decision = CompatibilityMatrix::evaluate(
            &arch,
            &target,
            Some(CompileQuantMode::Nf4 { group_size: 128 }),
        );
        assert!(decision.validation.valid);
        // Both NF4 variants blocked by Qwen GQA rule → should fall to 8bit
        assert_eq!(
            decision.quant_mode,
            Some(CompileQuantMode::Af8 { group_size: 64 })
        );
        assert!(
            decision.validation.fallback_chain.len() >= 3,
            "expected at least 3 fallback attempts (nf4-128, nf4-64, 8bit)"
        );
        // All NF4 attempts should be recorded as failed.
        for attempt in &decision.validation.fallback_chain {
            if attempt.quant_label.starts_with("nf4") {
                assert!(
                    !attempt.compatible,
                    "{} should be incompatible",
                    attempt.quant_label
                );
            }
        }
    }

    #[test]
    fn test_llama_arch_with_gqa() {
        // Llama 3.1-8B: n_heads=32, n_kv_heads=8, head_dim=128
        let arch = TextArchitecture {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            global_head_dim: None,
            num_global_key_value_heads: None,
            num_hidden_layers: 32,
            vocab_size: 128256,
            sliding_window: 0,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            attention_k_eq_v: false,
            final_logit_softcapping: None,
            hidden_size_per_layer_input: 4096,
            layer_types: vec![],
            rope_local: crate::config::RopeSpec {
                theta: 500_000.0,
                rope_type: "default".to_string(),
                partial_rotary_factor: None,
            },
            rope_global: None,
            model_type: "llama".to_string(),
            moe_config: None,
            diffusion_config: None,
        };
        // head_dim=128, group_size=64 → compatible
        assert!(CompatibilityMatrix::is_quant_compatible(
            &arch,
            Some(CompileQuantMode::Nf4 { group_size: 64 })
        )
        .is_ok());
        // head_dim=128, group_size=128 → compatible
        assert!(CompatibilityMatrix::is_quant_compatible(
            &arch,
            Some(CompileQuantMode::Nf4 { group_size: 128 })
        )
        .is_ok());
        // GQA warning should be present in evaluate
        let target = HardwareTarget::M2Ultra;
        let decision = CompatibilityMatrix::evaluate(&arch, &target, None);
        assert!(decision
            .validation
            .warnings
            .iter()
            .any(|w| w.contains("GQA")));
    }
}
