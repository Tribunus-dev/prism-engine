//! Namespace resolution for text model tensors.
//!
//! Discovers the text model namespace by probing candidate prefixes
//! against the tensor names in a safetensors file. Candidates are
//! checked in order; the first that matches all anchor tensors wins.

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

/// Anchor tensors that must exist under the text model root.
const ANCHORS: &[&str] = &[
    "embed_tokens.weight",
    "norm.weight",
    "layers.0.input_layernorm.weight",
    // Hybrid architectures (Qwen3.5) may use linear attention on layer 0.
    // Check both standard and linear attention tensor names.
];

/// Check whether a candidate prefix matches all required anchor tensors.
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
    tensor_names.iter().any(|n| n == &full_attn || n == &linear_attn)
}

/// Discover the text model namespace by probing candidate prefixes.
/// Candidates are checked in order; first to match all anchors wins.
pub fn resolve_namespace(tensor_names: &[String]) -> Option<NamespaceBinding> {
    let candidates = &["model.language_model", "language_model.model", "model"];

    for &candidate in candidates {
        let all_found = matches_anchors(candidate, tensor_names);
        if all_found {
            let lm_head_key = format!("{}.lm_head.weight", candidate);
            let embed_key = format!("{}.embed_tokens.weight", candidate);
            let lm_head_exists = tensor_names.iter().any(|n| n == &lm_head_key);
            return Some(NamespaceBinding {
                root: candidate.to_string(),
                discovery: format!("matched {} anchors under '{}'", ANCHORS.len(), candidate),
                lm_head_key: if lm_head_exists {
                    lm_head_key
                } else {
                    embed_key
                },
                lm_head_aliased: !lm_head_exists,
            });
        }
    }
    None
}
