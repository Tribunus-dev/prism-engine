//! Engine-internal tool dispatch wrapper.
//!
//! Re-implements the engine's prior
//! `compute-core/src/ecs/tools/dispatch.rs` surface (`execute_tool_call`,
//! `sandbox_execute`, `default_sandbox_tools`) on top of the
//! constitutional `prism_ecs_server::tools` surface. The
//! engine-coupled `list_devices` tool is routed through the
//! engine-side wrapper in [`super::list_devices::tool_list_devices`]
//! so the engine's device registry stays the single source of
//! truth for device data.
//!
//! # Authority boundary
//!
//! These are the engine-internal entry points for the model-facing
//! tool surface. They do not mutate ECS world state; the JSON
//! result they return is the evidence carrier the model receives
//! about the side effect. Callers wrap the dispatch in their own
//! `WorldTxn` if the tool call is part of a constitutional command.

use crate::ecs::device;
use prism_ecs_server::tools::list_devices::DeviceInfo as ConstitutionalDeviceInfo;
use prism_ecs_server::tools::sandbox::sandbox_root;
use prism_ecs_server::tools::sandbox::{tool_edit_file, tool_file_info, tool_glob_files, tool_list_directory, tool_read_file, tool_read_file_lines, tool_search_files, tool_write_file};
use prism_ecs_server::tools::{FunctionCall, ToolDefinition};
use std::path::Path;

#[cfg(feature = "deno_core")]
use serde_json::json;

/// Collect a constitutional `DeviceInfo` slice from the engine's
/// device registry. This is the bridge between the engine's
/// `device::DeviceInfo` and the constitutional `DeviceInfo` slice
/// the dispatch layer expects.
fn engine_devices() -> Vec<ConstitutionalDeviceInfo> {
    device::global_registry()
        .enumerate()
        .iter()
        .map(|d| ConstitutionalDeviceInfo {
            id: d.id.0.to_string(),
            name: d.name.clone(),
            kind: d.kind.label().to_string(),
            backend: d.backend.label().to_string(),
            vendor: d.vendor.clone(),
            memory_bytes: d.memory.total_bytes,
            compute_units: d.compute_units,
            clock_mhz: d.clock_mhz,
        })
        .collect()
}

/// Execute a tool call and return the result as a JSON value.
/// Routes to sandbox tools by name.
///
/// The engine-side `list_devices` is wired through
/// [`super::list_devices::tool_list_devices`] so the engine's
/// device registry is the canonical source for device data.
pub fn execute_tool_call(call: &FunctionCall) -> Result<serde_json::Value, String> {
    sandbox_execute(call, None)
}

/// Execute a sandbox tool call with an explicit sandbox root.
pub fn sandbox_execute(
    call: &FunctionCall,
    root: Option<&Path>,
) -> Result<serde_json::Value, String> {
    let r = sandbox_root(root);
    let result = match call.name.as_str() {
        "read_file" => tool_read_file(&r, &call.arguments),
        "read_file_lines" => tool_read_file_lines(&r, &call.arguments),
        "write_file" => tool_write_file(&r, &call.arguments),
        "edit_file" => tool_edit_file(&r, &call.arguments),
        "list_directory" => tool_list_directory(&r, &call.arguments),
        "glob_files" => tool_glob_files(&r, &call.arguments),
        "search_files" => tool_search_files(&r, &call.arguments),
        "file_info" => tool_file_info(&r, &call.arguments),
        #[cfg(feature = "deno_core")]
        "run_javascript" => tool_javascript(&r, &call.arguments),
        "list_devices" => super::list_devices::tool_list_devices(&r, &call.arguments),
        _ => return Err(format!("unknown tool '{}'", call.name)),
    };
    Ok(result)
}

/// Return the set of built-in sandbox tool definitions (OpenAI Tool format).
pub fn default_sandbox_tools() -> Vec<ToolDefinition> {
    // Build the constitutional default tool set, then append the
    // engine's `list_devices` tool definition so the model can
    // request device listings. This matches the prior engine
    // behavior where `list_devices` was in the default set.
    let mut tools = prism_ecs_server::tools::dispatch::default_sandbox_tools();
    tools.push(
        prism_ecs_server::tools::dispatch::list_devices_tool_def(),
    );
    tools
}

#[cfg(feature = "deno_core")]
fn tool_javascript(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let code = match args.get("code").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => {
            return json!({"ok": false, "error": "missing 'code' argument", "code": "MISSING_ARG"})
        }
    };
    let timeout_ms = args.get("timeout_ms").and_then(|v| v.as_u64());
    let result = prism_ecs_server::tools::js_runtime::run_javascript(code, Some(root), timeout_ms);
    json!({
        "ok": result.ok,
        "output": result.output,
        "error": result.error,
        "duration_ms": result.duration_ms,
    })
}

// Suppress the unused-helper warning when `deno_core` is off
// (the helper is only invoked from the cfg-gated `run_javascript`
// arm above).
#[allow(dead_code)]
fn _force_link() {
    let _ = engine_devices;
}
