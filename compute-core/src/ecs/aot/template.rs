//! Metal kernel template system — strict placeholder substitution
//! with reject-on-unknown validation.
//!
//! Each `.metal` template source uses `{{PLACEHOLDER}}` syntax for
//! compile-time constants. The expander replaces known placeholders,
//! rejects unknown ones, and validates that no `{{...}}` remain after
//! expansion.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::parameters::KernelParameters;

/// A parameterized Metal kernel template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalKernelTemplate {
    pub template_id: String,
    pub source: String,
    pub required_placeholders: Vec<String>,
}

impl MetalKernelTemplate {
    /// Parse raw source to discover all `{{...}}` placeholders.
    pub fn from_source(template_id: &str, source: &str) -> Self {
        let mut required = Vec::new();
        let mut chars = source.chars().peekable();

        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next(); // consume second '{'
                let mut ph = String::new();
                loop {
                    match chars.next() {
                        None => break,
                        Some('}') if chars.peek() == Some(&'}') => {
                            chars.next(); // consume second '}'
                            if !ph.is_empty() {
                                required.push(ph);
                            }
                            break;
                        }
                        Some(c) => ph.push(c),
                    }
                }
            }
        }

        MetalKernelTemplate {
            template_id: template_id.to_string(),
            source: source.to_string(),
            required_placeholders: required,
        }
    }

    /// Validate that all required placeholders have corresponding parameters.
    pub fn validate_params(&self, params: &KernelParameters) -> Result<(), TemplateError> {
        let known: HashSet<&str> = params
            .to_placeholder_map()
            .iter()
            .map(|(k, _)| *k)
            .collect();

        for required in &self.required_placeholders {
            if !known.contains(required.as_str()) {
                return Err(TemplateError::MissingValue {
                    template: self.template_id.clone(),
                    placeholder: required.clone(),
                });
            }
        }
        Ok(())
    }
}

// ── Error types ────────────────────────────────────────────────────

#[derive(Debug, Clone, thiserror::Error)]
pub enum TemplateError {
    #[error("template {template}: missing value for {{{{{placeholder}}}}}")]
    MissingValue {
        template: String,
        placeholder: String,
    },
    #[error(
        "template {template}: unknown placeholder {{{{{placeholder}}}}} (not in parameter set)"
    )]
    UnknownPlaceholder {
        template: String,
        placeholder: String,
    },
    #[error(
        "template {template}: after expansion, unexpanded placeholder remains: {{{{{remnant}}}}}"
    )]
    UnexpandedPlaceholder { template: String, remnant: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::aot::parameters::{DType, KernelFamily, KernelParameters};

    fn sample_params() -> KernelParameters {
        KernelParameters {
            kernel_family: KernelFamily::GemvNf4Tile,
            codec_family: crate::ecs::plan::CodecFamily::Nf4,
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
    fn template_expander_rejects_missing_placeholder() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint X = {{MISSING}};".into(),
            required_placeholders: vec!["MISSING".into()],
        };
        let params = sample_params();
        // validate_params should catch it
        assert!(template.validate_params(&params).is_err());
    }

    #[test]
    fn from_source_discovers_placeholders() {
        let source = "{{A}} hello {{B}} world {{C}}".to_string();
        let tpl = MetalKernelTemplate::from_source("test", &source);
        assert_eq!(tpl.required_placeholders, vec!["A", "B", "C"]);
    }
}
