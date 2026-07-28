//! `list_devices` tool (constitutional home).
//!
//! Returns a JSON array of compute device info, sourced from a slice
//! of [`DeviceInfo`] the caller provides. The constitutional surface
//! has no access to the engine's `device::global_registry()` (that
//! would create a circular dependency — the engine depends on the
//! constitutional crate, not the other way around). The
//! engine-internal wrapper at
//! `compute-core/src/ecs/legacy_tools/list_devices.rs` queries the
//! registry and forwards the entries to the constitutional function.
//!
//! # Authority boundary
//!
//! The tool returns the JSON the model receives; it does not mutate
//! ECS world state. The `DeviceInfo` struct is the constitutional
//! typed shape; engine-side device implementations convert their
//! types into `DeviceInfo` before calling [`tool_list_devices`].

use crate::tools::ToolDefinition;
use serde_json::json;
use std::path::Path;

/// Stable, content-addressed device identifier (per-call payload).
///
/// The engine's `device::DeviceId` is the canonical engine-side
/// identifier; engine-internal callers convert to/from
/// `DeviceInfo::id` (a `String`) at the boundary.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Stable identifier for the device (e.g. `"metal:0"`, `"ane:0"`).
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Device kind — GPU, CPU, NPU, ANE, etc.
    pub kind: String,
    /// Backend label — metal, cuda, rocm, cpu, ane, etc.
    pub backend: String,
    /// Vendor — apple, nvidia, amd, intel, etc.
    pub vendor: String,
    /// Total memory in bytes.
    pub memory_bytes: u64,
    /// Number of compute units.
    pub compute_units: u32,
    /// Clock speed in MHz.
    pub clock_mhz: u32,
}

/// Execute the `list_devices` tool — returns a JSON array of device info.
///
/// The caller is responsible for providing the device slice; this
/// keeps the constitutional surface free of engine-internal state.
pub fn tool_list_devices(
    devices: &[DeviceInfo],
    _root: &Path,
    _args: &serde_json::Value,
) -> serde_json::Value {
    let entries: Vec<serde_json::Value> = devices
        .iter()
        .map(|d| {
            json!({
                "id": d.id,
                "name": d.name,
                "kind": d.kind,
                "backend": d.backend,
                "vendor": d.vendor,
                "memory_gb": d.memory_bytes as f64 / 1_000_000_000.0,
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

/// Tool definition metadata for the `list_devices` tool.
///
/// Engine callers can include this in their `default_sandbox_tools`
/// list so the model sees the tool.
pub fn tool_def() -> ToolDefinition {
    ToolDefinition {
        name: "list_devices".into(),
        description: "List all available compute devices (GPU, CPU, NPU) on this host with name, kind, backend, vendor, memory, compute units, and clock speed.".into(),
        parameters: serde_json::json!({"type": "object", "properties": {}}),
        required: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn sample_devices() -> Vec<DeviceInfo> {
        vec![
            DeviceInfo {
                id: "metal:0".into(),
                name: "Apple M2 Pro GPU".into(),
                kind: "GPU".into(),
                backend: "metal".into(),
                vendor: "apple".into(),
                memory_bytes: 16 * 1_000_000_000,
                compute_units: 19,
                clock_mhz: 1300,
            },
            DeviceInfo {
                id: "ane:0".into(),
                name: "Apple M2 Pro ANE".into(),
                kind: "NPU".into(),
                backend: "ane".into(),
                vendor: "apple".into(),
                memory_bytes: 0,
                compute_units: 16,
                clock_mhz: 0,
            },
        ]
    }

    #[test]
    fn tool_list_devices_returns_expected_json_shape() {
        let root = PathBuf::from("/");
        let result = tool_list_devices(&sample_devices(), &root, &serde_json::json!({}));
        assert_eq!(result["ok"], serde_json::json!(true));
        let count = result["count"].as_u64().expect("count");
        assert_eq!(count, 2);
        let devices = result["devices"].as_array().expect("devices array");
        assert_eq!(devices[0]["id"], serde_json::json!("metal:0"));
        assert_eq!(devices[0]["kind"], serde_json::json!("GPU"));
        assert_eq!(devices[0]["backend"], serde_json::json!("metal"));
        assert_eq!(devices[1]["id"], serde_json::json!("ane:0"));
        assert_eq!(devices[1]["kind"], serde_json::json!("NPU"));
    }

    #[test]
    fn tool_list_devices_empty_slice() {
        let root = PathBuf::from("/");
        let result = tool_list_devices(&[], &root, &serde_json::json!({}));
        assert_eq!(result["ok"], serde_json::json!(true));
        assert_eq!(result["count"], serde_json::json!(0));
        assert!(result["devices"].as_array().expect("devices").is_empty());
    }

    #[test]
    fn tool_def_metadata_is_stable() {
        let def = tool_def();
        assert_eq!(def.name, "list_devices");
        assert!(def.description.contains("compute devices"));
        assert!(def.required.is_empty());
    }
}
