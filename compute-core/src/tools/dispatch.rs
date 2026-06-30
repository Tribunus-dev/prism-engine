use crate::tools::{FunctionCall, ToolDefinition};
use std::path::Path;

/// Execute a tool call and return the result as a JSON value.
/// Routes to sandbox tools by name.
pub fn execute_tool_call(call: &FunctionCall) -> Result<serde_json::Value, String> {
    sandbox_execute(call, None)
}

/// Execute a sandbox tool call with an explicit sandbox root.
pub fn sandbox_execute(call: &FunctionCall, root: Option<&Path>) -> Result<serde_json::Value, String> {
    let root = sandbox_root(root);
    let result = match call.name.as_str() {
        "read_file" => crate::tools::sandbox::tool_read_file(&root, &call.arguments),
        "read_file_lines" => crate::tools::sandbox::tool_read_file_lines(&root, &call.arguments),
        "write_file" => crate::tools::sandbox::tool_write_file(&root, &call.arguments),
        "edit_file" => crate::tools::sandbox::tool_edit_file(&root, &call.arguments),
        "list_directory" => crate::tools::sandbox::tool_list_directory(&root, &call.arguments),
        "glob_files" => crate::tools::sandbox::tool_glob_files(&root, &call.arguments),
        "search_files" => crate::tools::sandbox::tool_search_files(&root, &call.arguments),
        "file_info" => crate::tools::sandbox::tool_file_info(&root, &call.arguments),
        _ => return Err(format!("unknown tool '{}'", call.name)),
    };
    Ok(result)
}

use crate::tools::sandbox::sandbox_root;

/// Return the set of built-in sandbox tool definitions (OpenAI Tool format).
pub fn default_sandbox_tools() -> Vec<ToolDefinition> {
    vec![
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
    ]
}
