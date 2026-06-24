use crate::linux::backend::{BackendKind, VendorKind};

#[derive(Debug, Clone)]
pub enum BackendAvailability {
    Available,
    DriverMissing,
    RuntimeLibraryMissing,
    UnsupportedHardware,
    PermissionDenied,
    FeatureNotCompiled,
    ProbeFailed { reason: String },
}

#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub backend: BackendKind,
    pub vendor: VendorKind,
    pub device_name: String,
    pub driver_version: Option<String>,
    pub architecture: Option<String>,
    pub device_memory_bytes: u64,
    pub host_visible_memory: bool,
    pub unified_addressing: bool,
    pub managed_memory: bool,
    pub peer_access: bool,
    pub async_copy: bool,
    pub events: bool,
    pub command_graphs: bool,
    pub cooperative_launch: bool,
    pub fp16: bool,
    pub bf16: bool,
    pub int8: bool,
    pub int4: bool,
    pub subgroup_widths: Vec<u32>,
    pub max_workgroup_size: u32,
    pub max_shared_memory_bytes: u64,
    pub max_allocation_bytes: u64,
    pub supports_timestamps: bool,
    pub supports_profiling: bool,
    pub supports_external_memory: bool,
    pub availability: BackendAvailability,
}
