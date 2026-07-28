//! OpenAI-compatible function-call tool surface (constitutional home).
//!
//! This module owns the canonical authority for the runtime-facing tool
//! surface that drives model-issued function calls in Prism: tool
//! definitions, function-call payloads, parse/repair pipelines, sandboxed
//! file-system tools, the AST-level JavaScript security guard, the
//! V8-backed JavaScript sandbox, the X-Ray HTML sanitization proxy, and
//! the tool-call dispatch layer. The legacy home was
//! `compute-core/src/ecs/tools/` (8 files, 2,911 LOC), absorbed during
//! the engine-subsystem deletion pass (see
//! `changelogs/2026-07-27-engine-subsystem-deletion-tools.md`).
//!
//! # Authority boundary
//!
//! These types are *server-side tool carriers* — they describe what
//! tools the model can call and how the runtime executes them. They are
//! not the canonical world; the world is the ECS and the constitutional
//! commands mutate it. Tool execution here is a side effect of an
//! external model call, executed against the runtime's sandbox root, and
//! the JSON result is the evidence returned to the model.
//!
//! # Engine-coupled extensions
//!
//! Two pieces of the original engine's `tools/` module depend on
//! engine-internal state and are *not* re-implemented in this
//! constitutional surface:
//!
//! - `list_devices` (engine reads from `crate::ecs::device::global_registry()`).
//!   The constitutional surface exposes a `DeviceInfo` data type and a
//!   pure `tool_list_devices` that takes a `&[DeviceInfo]` slice; the
//!   engine-internal wrapper at
//!   `compute-core/src/ecs/legacy_tools/list_devices.rs` queries the
//!   registry and forwards to the constitutional function.
//!
//! - `retry_with_error` (engine-internal `profiled_executor` types, gated
//!   by the `mlx-backend` feature). The engine-internal wrapper at
//!   `compute-core/src/ecs/legacy_tools/retry_with_error.rs` re-implements
//!   this against the engine's MLX executor and re-uses the constitutional
//!   `parse_and_repair` for the final repair step.
//!
//! Every state-bearing change in the underlying sandbox (write/edit) is
//! an effect of an authenticated model call; the tool result is a JSON
//! evidence carrier, not a canonical world mutation.

use serde::{Deserialize, Serialize};

#[cfg(feature = "deno_core")]
pub mod js_runtime;
pub mod ast_guard;
pub mod dispatch;
pub mod list_devices;
pub mod parse;
pub mod sandbox;
pub mod xray;

// Re-export the inner types so the surface mirrors the original
// `compute-core/src/ecs/tools/mod.rs` shape: callers do
// `use prism_ecs_server::tools::ToolDefinition;`.
pub use ast_guard::*;
pub use dispatch::*;
pub use list_devices::*;
pub use parse::*;
pub use sandbox::*;
pub use xray::*;

/// A tool definition parsed from the OpenAI API request body.
///
/// This is the typed shape the constitutional tool surface passes
/// around; it carries the function name, description, JSON-Schema
/// parameters, and the list of required parameter names. The schema
/// is *not* validated by this struct — it is the model's
/// responsibility to provide a well-formed JSON Schema, and the
/// `parse_and_repair` step is what repairs common malformations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool/function name.
    pub name: String,
    /// Human-readable description shown to the model.
    pub description: String,
    /// JSON-Schema describing the tool's parameters.
    pub parameters: serde_json::Value,
    /// Required parameter names (denormalized from
    /// `parameters["required"]` for fast access during parse).
    pub required: Vec<String>,
}

/// A function call emitted by the model.
///
/// The `arguments` field is the post-`parse_and_repair` JSON value:
/// the constitutional parse step normalizes the OpenAI tool-calls
/// wrapper into this flat shape before any execution happens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// The function name.
    pub name: String,
    /// The (possibly-repaired) JSON arguments.
    pub arguments: serde_json::Value,
}

/// Result of attempting to parse and repair a tool call.
///
/// The three outcomes describe the parse pipeline's certainty:
/// `Valid` for parsed-and-untouched, `Repaired` for parsed-with-fixes
/// (a fixes log is returned so the caller can audit the repair),
/// and `Unrepairable` for failures that need model-side retry.
#[derive(Debug, Clone)]
pub enum ToolCallResult {
    /// Parsed successfully with the given (name, arguments).
    Valid(String, serde_json::Value),
    /// Parsed but repaired (fixed JSON, type mismatches, etc.).
    Repaired(String, serde_json::Value, Vec<String>),
    /// Generation must be retried with this error context.
    Unrepairable(String),
}
