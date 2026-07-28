//! Tool-call dispatch layer (constitutional home).
//!
//! Routes a parsed [`FunctionCall`] to the appropriate tool
//! implementation and returns a JSON result. The constitutional
//! surface is engine-agnostic; the `list_devices` tool needs a
//! device list, so the dispatch layer takes the device slice as a
//! parameter and the engine-internal caller fills it in from
//! `device::global_registry()`.
//!
//! # Authority boundary
//!
//! Dispatch is an effect router: the JSON result it returns is the
//! evidence the model receives about the side effect. It does not
//! mutate ECS world state directly. The runtime that calls dispatch
//! is responsible for the surrounding `WorldTxn` semantics if the
//! tool call is part of a constitutional command.

use crate::tools::list_devices::{tool_def as list_devices_def, tool_list_devices, DeviceInfo};
use crate::tools::sandbox::{
    sandbox_root, tool_edit_file, tool_file_info, tool_glob_files, tool_list_directory,
    tool_read_file, tool_read_file_lines, tool_search_files, tool_write_file,
};
use crate::tools::{FunctionCall, ToolDefinition};
#[cfg(feature = "deno_core")]
use serde_json::json;
use std::path::Path;

/// Execute a tool call and return the result as a JSON value.
/// Routes to sandbox tools by name.
pub fn execute_tool_call(
    call: &FunctionCall,
    devices: &[DeviceInfo],
) -> Result<serde_json::Value, String> {
    sandbox_execute(call, devices, None)
}

/// Execute a sandbox tool call with an explicit sandbox root.
pub fn sandbox_execute(
    call: &FunctionCall,
    devices: &[DeviceInfo],
    root: Option<&Path>,
) -> Result<serde_json::Value, String> {
    let root = sandbox_root(root);
    let result = match call.name.as_str() {
        "read_file" => tool_read_file(&root, &call.arguments),
        "read_file_lines" => tool_read_file_lines(&root, &call.arguments),
        "write_file" => tool_write_file(&root, &call.arguments),
        "edit_file" => tool_edit_file(&root, &call.arguments),
        "list_directory" => tool_list_directory(&root, &call.arguments),
        "glob_files" => tool_glob_files(&root, &call.arguments),
        "search_files" => tool_search_files(&root, &call.arguments),
        "file_info" => tool_file_info(&root, &call.arguments),
        #[cfg(feature = "deno_core")]
        "run_javascript" => tool_javascript(&root, &call.arguments),
        "list_devices" => tool_list_devices(devices, &root, &call.arguments),
        _ => return Err(format!("unknown tool '{}'", call.name)),
    };
    Ok(result)
}

/// Return the set of built-in sandbox tool definitions (OpenAI Tool format).
///
/// This is the engine-agnostic default; the engine-internal wrapper
/// at `compute-core/src/ecs/legacy_tools/dispatch.rs` appends the
/// `list_devices` tool definition (which the engine wants to expose
/// to the model) so the engine retains the prior public shape
/// without re-implementing the rest.
pub fn default_sandbox_tools() -> Vec<ToolDefinition> {
    default_sandbox_tools_with_extras(&[])
}

/// Return the set of built-in sandbox tool definitions, optionally
/// including engine-specific extras (e.g. `list_devices`).
pub fn default_sandbox_tools_with_extras(extras: &[ToolDefinition]) -> Vec<ToolDefinition> {
    let mut tools = vec![
        ToolDefinition {
            name: "read_file".into(),
            description: "Read the full contents of a text file within the sandbox.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from sandbox root"}
                },
                "required": ["path"]
            }),
            required: vec!["path".into()],
        },
        ToolDefinition {
            name: "read_file_lines".into(),
            description: "Read a specific range of lines from a text file. Lines are 1-indexed.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"}
                },
                "required": ["path"]
            }),
            required: vec!["path".into()],
        },
        ToolDefinition {
            name: "write_file".into(),
            description: "Write content to a file within the sandbox. Atomic write.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
            required: vec!["path".into(), "content".into()],
        },
        ToolDefinition {
            name: "edit_file".into(),
            description: "Find and replace in a file. Reports affected lines.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"}
                },
                "required": ["path", "old_text"]
            }),
            required: vec!["path".into(), "old_text".into()],
        },
        ToolDefinition {
            name: "list_directory".into(),
            description: "List files and directories at the given path, sorted by name.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "include_hidden": {"type": "boolean"}
                }
            }),
            required: vec![],
        },
        ToolDefinition {
            name: "glob_files".into(),
            description: "Recursively find files by extension.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "extension": {"type": "string"},
                    "max_results": {"type": "integer"}
                }
            }),
            required: vec![],
        },
        ToolDefinition {
            name: "search_files".into(),
            description: "Search for a substring in files within the sandbox.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "pattern": {"type": "string"},
                    "extension": {"type": "string"}
                },
                "required": ["pattern"]
            }),
            required: vec!["pattern".into()],
        },
        ToolDefinition {
            name: "file_info".into(),
            description: "Get metadata about a file or directory.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
            required: vec!["path".into()],
        },
        #[cfg(feature = "deno_core")]
        ToolDefinition {
            name: "run_javascript".into(),
            description: "Run JavaScript code in a sandboxed V8 isolate.  Has access to readFile(path), writeFile(path, content), listDirectory(path), and console.log.  No network, no subprocess, no env access.  Use for automation, testing, and web dev tasks.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "JavaScript code to execute"},
                    "timeout_ms": {"type": "integer", "description": "Max execution time in ms (default: 30000)"}
                },
                "required": ["code"]
            }),
            required: vec!["code".into()],
        },
    ];
    tools.extend_from_slice(extras);
    tools
}

/// Convenience: return the canonical `list_devices` tool definition.
///
/// Engine-internal callers use this in their `default_sandbox_tools`
/// extras so the model can request device listings without
/// re-implementing the schema.
pub fn list_devices_tool_def() -> ToolDefinition {
    list_devices_def()
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
    let result = crate::tools::js_runtime::run_javascript(code, Some(root), timeout_ms);
    json!({
        "ok": result.ok,
        "output": result.output,
        "error": result.error,
        "duration_ms": result.duration_ms,
    })
}
