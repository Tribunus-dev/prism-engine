//! Deterministic tool call parsing & repair for OpenAI-compatible function calling.
//!
//! When the model generates a malformed function call (broken JSON, missing params,
//! wrong function name), this module detects it, repairs it deterministically, and
//! only retries generation if repair is impossible.
//!
//! # Pipeline
//!
//! 1. [`parse_and_repair`] — try up to 4 strategies to extract JSON from raw text
//! 2. [`validate_and_fix`] — check required fields, fix type mismatches, correct names
//! 3. [`retry_with_error`] — if unrepairable, regenerate with error context

use serde::{Deserialize, Serialize};

pub mod parse;
pub mod ast_guard;
pub mod sandbox;
pub mod dispatch;
pub mod js_runtime;
pub mod xray;

/// A tool definition parsed from the OpenAI API request body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub required: Vec<String>,
}

/// A function call emitted by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

/// Result of attempting to parse and repair a tool call.
#[derive(Debug, Clone)]
pub enum ToolCallResult {
    /// Parsed successfully with the given (name, arguments).
    Valid(String, serde_json::Value),
    /// Parsed but repaired (fixed JSON, type mismatches, etc.).
    Repaired(String, serde_json::Value, Vec<String>),
    /// Generation must be retried with this error context.
    Unrepairable(String),
}

pub use parse::*;
pub use dispatch::*;
pub use sandbox::*;
pub use ast_guard::*;
pub use xray::*;
