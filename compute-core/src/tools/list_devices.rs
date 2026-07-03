//! List all available compute devices on this host.
//!
//! Reads from the global device registry and returns a JSON array
//! with device name, kind, backend, vendor, memory, compute units, and clock speed.

use crate::device;
use crate::tools::ToolDefinition;
use serde_json::json;
use std::path::Path;

/// Execute the list_devices tool — returns a JSON array of device info.
pub fn tool_list_devices(_root: &Path, _args: &serde_json::Value) -> serde_json::Value {
    let registry = device::global_registry();
    let entries: Vec<serde_json::Value> = registry
        .enumerate()
        .iter()
        .map(|d| {
            json!({
                "id": d.id.0,
                "name": d.name,
                "kind": d.kind.label(),
                "backend": d.backend.label(),
                "vendor": d.vendor,
                "memory_gb": d.memory.total_bytes as f64 / 1_000_000_000.0,
                "compute_units": d.compute_units,
                "clock_mhz": d.clock_mhz,
            })
        })
        .collect();
    json!({
        "ok": true,
        "devices": entries,
        "count": entries.len(),
    })
}

/// Tool definition metadata for the list_devices tool.
pub fn tool_def() -> ToolDefinition {
    ToolDefinition {
        name: "list_devices".into(),
        description: "List all available compute devices (GPU, CPU, NPU) on this host with name, kind, backend, vendor, memory, compute units, and clock speed.".into(),
        parameters: serde_json::json!({"type": "object", "properties": {}}),
        required: vec![],
    }
}
