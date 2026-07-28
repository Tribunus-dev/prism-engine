//! Engine-internal `list_devices` tool wrapper.
//!
//! The original `compute-core/src/ecs/tools/list_devices.rs` queried
//! `crate::ecs::device::global_registry()` directly. The
//! constitutional surface at
//! `prism_ecs_server::tools::list_devices` is engine-agnostic: it
//! takes a `&[DeviceInfo]` slice and returns a JSON value. This
//! module is the engine-side bridge: it queries the engine's
//! device registry, converts each entry to the constitutional
//! `DeviceInfo` shape, and forwards the call.
//!
//! # Authority boundary
//!
//! This wrapper is the only engine file that owns the engine→con-
//! stitutional device type conversion. It does not mutate ECS
//! world state; the tool result is a JSON evidence carrier
//! returned to the model. Engine callers (the agent state machine
//! in `compute-core/src/ecs/agent/mod.rs`, the dispatch in this
//! same directory, and downstream `prism-bridge`) see a `list_devices`
//! tool with the same shape they had pre-migration.

use crate::ecs::device;
use prism_ecs_server::tools::list_devices::{
    tool_list_devices as constitutional_tool_list_devices, DeviceInfo as ConstitutionalDeviceInfo,
};
use std::path::Path;

/// Execute the `list_devices` tool — returns a JSON array of
/// engine-registered device info.
///
/// Mirrors the public signature of the original
/// `compute-core/src/ecs/tools/list_devices.rs::tool_list_devices`
/// so the engine's dispatch layer can route the call without
/// change. Internally, this queries the engine's
/// `device::global_registry()` and converts the entries to the
/// constitutional `DeviceInfo` slice the constitutional surface
/// expects.
pub fn tool_list_devices(root: &Path, args: &serde_json::Value) -> serde_json::Value {
    let registry = device::global_registry();
    let constitutional: Vec<ConstitutionalDeviceInfo> = registry
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
        .collect();
    constitutional_tool_list_devices(&constitutional, root, args)
}
