//! Engine-internal `retry_with_error` wrapper (mlx-backend-gated).
//!
//! The original `compute-core/src/ecs/tools/parse.rs::retry_with_error`
//! took a `&mut crate::profiled_executor::ProfiledInferenceSession`
//! and a `&crate::profiled_executor::LoadedProfiledModel` — types
//! that live in the engine's `profiled_executor` module, not in
//! the constitutional surface. The constitutional surface stops at
//! the parse step; this module is the engine-internal home for the
//! MLX retry path.
//!
//! # Authority boundary
//!
//! This wrapper re-uses the constitutional
//! [`prism_ecs_server::tools::parse::parse_and_repair`] for the
//! final repair step. The MLX chat / sampler configuration is
//! engine-internal. The retry call is an effect of an authenticated
//! model call; the JSON result carrier is what the model receives.

use prism_ecs_server::tools::parse::parse_and_repair;
use prism_ecs_server::tools::{ToolCallResult, ToolDefinition};

/// Retry generation after a failed tool call by appending an error
/// description as a system message and regenerating.
///
/// The `messages` slice should match the original request's messages
/// array (serde_json::Value objects with `role` and `content`).
/// Returns the parsed-and-repaired result, or an error string on
/// generation failure.
#[cfg(feature = "mlx-backend")]
pub fn retry_with_error(
    sess: &mut crate::profiled_executor::ProfiledInferenceSession,
    model: &crate::profiled_executor::LoadedProfiledModel,
    messages: &[serde_json::Value],
    error: &str,
    tool: &ToolDefinition,
    max_tokens: u64,
) -> Result<ToolCallResult, String> {
    let correction_msg = format!(
        "The previous function call was invalid: {}\n\
         Please call the `{}` function with valid JSON matching this schema:\n{}\n\
         Required parameters: {}",
        error,
        tool.name,
        serde_json::to_string_pretty(&tool.parameters).unwrap_or_default(),
        tool.required.join(", "),
    );

    // Append a system message with the correction.
    let mut new_messages: Vec<serde_json::Value> = messages.to_vec();
    new_messages.push(serde_json::json!({
        "role": "system",
        "content": correction_msg
    }));

    // Build a chat prompt from the augmented messages.
    let prompt = build_chat_prompt(&new_messages);

    let sampler_config = crate::session::SamplerConfig {
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: None,
        repetition_penalty: None,
        seed: None,
        stop_token_ids: Vec::new(),
        grammar: None,
        grammar_tokenizer: None,
    };

    let output_text = sess
        .chat_with_sampler(&prompt, max_tokens, &sampler_config, model)
        .map_err(|e| format!("retry inference failed: {e}"))?;

    Ok(parse_and_repair(&output_text, tool))
}

/// Build a chat prompt string from a messages array (serde_json::Value).
/// Each message should have `role` and `content` fields.
#[cfg(feature = "mlx-backend")]
#[allow(dead_code)]
fn build_chat_prompt(messages: &[serde_json::Value]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = msg.get("content").and_then(|c| c.as_str()).unwrap_or("");
        prompt.push_str(&format!("<|{}|>\n{}\n", role, content));
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}
