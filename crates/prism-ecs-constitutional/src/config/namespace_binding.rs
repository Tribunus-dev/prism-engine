//! Selected namespace root for text model tensors.
//!
//! Authority: the canonical [`NamespaceBinding`] data type that
//! captures the discovered safetensors namespace root and the
//! location of the LM head. The discovery routine
//! ([`resolve_namespace`]) is engine-internal because it depends on
//! the engine's safetensors machinery; the data type itself is
//! platform-neutral and lives here.

use serde::Serialize;

/// Selected namespace root for text model tensors.
#[derive(Clone, Debug, Default, Serialize)]
pub struct NamespaceBinding {
    pub root: String,
    /// How the root was discovered.
    pub discovery: String,
    /// Where lm_head.weight lives (may alias embed_tokens if tied).
    pub lm_head_key: String,
    pub lm_head_aliased: bool,
}

/// Resolve a namespace from the given tensor names.
///
/// Anchor tensors that must exist under the text model root:
///
/// - `embed_tokens.weight`
/// - `norm.weight`
/// - `layers.0.input_layernorm.weight`
/// - either `layers.0.self_attn.q_proj.weight` (full attention) or
///   `layers.0.linear_attn.in_proj_qkv.weight` (linear attention).
///
/// Candidates are checked in order; the first that matches all
/// anchors wins. Returns `None` if no candidate matches.
///
/// This is a pure-Rust function over the tensor-name list; the
/// engine-internal safetensors I/O is the caller's responsibility.
pub fn resolve_namespace(tensor_names: &[String]) -> Option<NamespaceBinding> {
    const CANDIDATES: &[&str] = &["model.language_model", "language_model.model", "model"];

    for &candidate in CANDIDATES {
        if !matches_anchors(candidate, tensor_names) {
            continue;
        }
        let lm_head_key = format!("{}.lm_head.weight", candidate);
        let embed_key = format!("{}.embed_tokens.weight", candidate);
        let lm_head_exists = tensor_names.iter().any(|n| n == &lm_head_key);
        return Some(NamespaceBinding {
            root: candidate.to_string(),
            discovery: format!("matched anchors under '{}'", candidate),
            lm_head_key: if lm_head_exists {
                lm_head_key
            } else {
                embed_key
            },
            lm_head_aliased: !lm_head_exists,
        });
    }
    None
}

fn matches_anchors(candidate: &str, tensor_names: &[String]) -> bool {
    let required = [
        format!("{candidate}.embed_tokens.weight"),
        format!("{candidate}.norm.weight"),
        format!("{candidate}.layers.0.input_layernorm.weight"),
    ];
    for key in &required {
        if !tensor_names.iter().any(|n| n == key) {
            return false;
        }
    }
    // Either full-attention or linear-attention layer 0 is acceptable.
    let full_attn = format!("{candidate}.layers.0.self_attn.q_proj.weight");
    let linear_attn = format!("{candidate}.layers.0.linear_attn.in_proj_qkv.weight");
    tensor_names
        .iter()
        .any(|n| n == &full_attn || n == &linear_attn)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_namespace() -> Vec<String> {
        let root = "model.language_model";
        vec![
            format!("{root}.embed_tokens.weight"),
            format!("{root}.norm.weight"),
            format!("{root}.layers.0.input_layernorm.weight"),
            format!("{root}.layers.0.self_attn.q_proj.weight"),
            format!("{root}.lm_head.weight"),
        ]
    }

    #[test]
    fn resolve_picks_first_matching_candidate() {
        let names = standard_namespace();
        let ns = resolve_namespace(&names).unwrap();
        assert_eq!(ns.root, "model.language_model");
        assert!(!ns.lm_head_aliased);
        assert_eq!(ns.lm_head_key, "model.language_model.lm_head.weight");
    }

    #[test]
    fn resolve_aliases_lm_head_to_embed_tokens_when_missing() {
        let mut names = standard_namespace();
        names.retain(|n| !n.ends_with("lm_head.weight"));
        let ns = resolve_namespace(&names).unwrap();
        assert!(ns.lm_head_aliased);
        assert_eq!(ns.lm_head_key, "model.language_model.embed_tokens.weight");
    }

    #[test]
    fn resolve_returns_none_when_no_anchors_match() {
        let names = vec!["some.other.tensor".to_string()];
        assert!(resolve_namespace(&names).is_none());
    }

    #[test]
    fn resolve_accepts_linear_attention_layer_zero() {
        let mut names = standard_namespace();
        names.retain(|n| !n.ends_with("q_proj.weight"));
        names.push("model.language_model.layers.0.linear_attn.in_proj_qkv.weight".into());
        let ns = resolve_namespace(&names).unwrap();
        assert_eq!(ns.root, "model.language_model");
    }
}
