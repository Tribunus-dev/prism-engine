//! Metal kernel template system — strict placeholder substitution
//! with reject-on-unknown validation.
//!
//! Each `.metal` template source uses `{{PLACEHOLDER}}` syntax for
//! compile-time constants. The expander replaces known placeholders,
//! rejects unknown ones, and validates that no `{{...}}` remain after
//! expansion.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

// ── Template expander ────────────────────────────────────────────────────

/// Strict template expander. Rejects unknown placeholders and unexpanded
/// `{{...}}` patterns in the result.
pub struct KernelTemplateExpander;

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

impl KernelTemplateExpander {
    pub fn expand(
        template: &MetalKernelTemplate,
        params: &KernelParameters,
    ) -> Result<String, TemplateError> {
        let entries = params.to_placeholder_map();
        let mut map: HashMap<&str, &str> = HashMap::with_capacity(entries.len());
        for (k, v) in &entries {
            map.insert(k, v.as_str());
        }
        let known_keys: HashSet<&str> = map.keys().copied().collect();

        let mut result = String::with_capacity(template.source.len());
        let mut chars = template.source.chars().peekable();

        loop {
            match chars.next() {
                None => break,
                Some('{') => {
                    if chars.peek() == Some(&'{') {
                        chars.next(); // consume second '{'
                        let mut ph = String::new();
                        loop {
                            match chars.next() {
                                None => {
                                    return Err(TemplateError::MissingValue {
                                        template: template.template_id.clone(),
                                        placeholder: ph,
                                    });
                                }
                                Some('}') if chars.peek() == Some(&'}') => {
                                    chars.next(); // consume second '}'
                                    break;
                                }
                                Some(c) if c == '}' => {
                                    ph.push('}');
                                }
                                Some(c) => ph.push(c),
                            }
                        }

                        if let Some(value) = map.get(ph.as_str()) {
                            result.push_str(value);
                        } else if !known_keys.contains(ph.as_str()) {
                            return Err(TemplateError::UnknownPlaceholder {
                                template: template.template_id.clone(),
                                placeholder: ph,
                            });
                        } else {
                            // Known key but no value: shouldn't happen after validate_params
                            result.push_str(&ph);
                        }
                    } else {
                        result.push('{');
                    }
                }
                Some(c) => result.push(c),
            }
        }

        // Post-expansion validation: check for unexpanded {{...}} patterns.
        Self::check_unexpanded(&template.template_id, &result)?;

        Ok(result)
    }

    fn check_unexpanded(template_id: &str, source: &str) -> Result<(), TemplateError> {
        let mut chars = source.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '{' && chars.peek() == Some(&'{') {
                chars.next();
                let mut ph = String::new();
                loop {
                    match chars.next() {
                        None => break,
                        Some('}') if chars.peek() == Some(&'}') => {
                            chars.next();
                            if !ph.is_empty() {
                                return Err(TemplateError::UnexpandedPlaceholder {
                                    template: template_id.to_string(),
                                    remnant: ph,
                                });
                            }
                            break;
                        }
                        Some(c) => ph.push(c),
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aot_kernels::parameters::{DType, KernelFamily, KernelParameters};

    fn sample_params() -> KernelParameters {
        KernelParameters {
            kernel_family: KernelFamily::GemvNf4Tile,
            codec_family: crate::execution_plan::CodecFamily::Nf4,
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
    fn template_expander_rejects_unknown_placeholder() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint X = {{UNKNOWN_VAR}};".into(),
            required_placeholders: vec![],
        };
        let result = KernelTemplateExpander::expand(&template, &sample_params());
        assert!(result.is_err());
        match result.unwrap_err() {
            TemplateError::UnknownPlaceholder { placeholder, .. } => {
                assert_eq!(placeholder, "UNKNOWN_VAR");
            }
            _ => panic!("expected UnknownPlaceholder"),
        }
    }

    #[test]
    fn generated_source_contains_expected_constexprs() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint TW = {{TILE_WIDTH}};\nconst uint GS = {{GROUP_SIZE}};\nconst uint LV = {{LANE_VALUES}};".into(),
            required_placeholders: vec!["TILE_WIDTH".into(), "GROUP_SIZE".into(), "LANE_VALUES".into()],
        };
        let result = KernelTemplateExpander::expand(&template, &sample_params()).unwrap();
        assert!(result.contains("TW = 640;"), "result: {}", result);
        assert!(result.contains("GS = 128;"), "result: {}", result);
        assert!(result.contains("LV = 4;"), "result: {}", result);
    }

    #[test]
    fn detects_unexpanded_placeholder() {
        let template = MetalKernelTemplate {
            template_id: "test".into(),
            source: "const uint TW = {{TILE_WIDTH}};\nconst uint BAD = {{NOT_IN_PARAMS}};".into(),
            required_placeholders: vec!["TILE_WIDTH".into()],
        };
        let result = KernelTemplateExpander::expand(&template, &sample_params());
        assert!(result.is_err());
    }

    #[test]
    fn from_source_discovers_placeholders() {
        let source = "{{A}} hello {{B}} world {{C}}".to_string();
        let tpl = MetalKernelTemplate::from_source("test", &source);
        assert_eq!(tpl.required_placeholders, vec!["A", "B", "C"]);
    }
}
