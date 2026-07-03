//! Device registry — runtime hardware enumeration and capability discovery.
//!
//! The [`DeviceRegistry`] is the single source of truth for all compute
//! devices available on the host. It runs a set of pluggable [`DeviceProbe`]
//! implementations at startup, each of which discovers devices of a specific
//! backend type (Metal GPU, CUDA GPU, CPU, ANE/NPU, etc.).
//!
//! The registry is backend-agnostic: every device carries a [`BackendKind`]
//! tag and a [`DeviceKind`] classification, plus memory and capability info.
//! Host apps query the registry via the CLI (`prism list-devices`), the C FFI,
//! or the NAPI bridge.
//!
//! # Usage
//!
//! ```rust,ignore
//! let registry = DeviceRegistry::discover();
//! for device in registry.enumerate() {
//!     println!("{}: {} ({} MB)", device.name, device.kind.label(), device.memory.total_bytes / 1_000_000);
//! }
//! let gpus = registry.by_backend(BackendKind::Metal);
//! let json = registry.to_json_pretty();
//! ```

pub mod probes;

use std::sync::LazyLock;
use serde::{Deserialize, Serialize};

// ── Core types ──────────────────────────────────────────────────────────────

/// Unique identifier for a discovered compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(pub u32);

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "device:{}", self.0)
    }
}

/// Broad category of a compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    /// General-purpose CPU.
    Cpu,
    /// Discrete GPU with dedicated VRAM (NVIDIA, AMD dGPU).
    GpuDiscrete,
    /// Integrated GPU sharing system RAM (Intel UHD, AMD iGPU).
    GpuIntegrated,
    /// Unified memory GPU (Apple Silicon).
    GpuUnified,
    /// Neural processing unit (Apple ANE, Intel NPU, AMD XDNA).
    Npu,
    /// Specialized accelerator (Tenstorrent Tensix, etc.).
    Accelerator,
}

impl DeviceKind {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::GpuDiscrete => "dGPU",
            Self::GpuIntegrated => "iGPU",
            Self::GpuUnified => "Unified GPU",
            Self::Npu => "NPU",
            Self::Accelerator => "Accelerator",
        }
    }
}

/// Specific compute backend that can execute operations on this device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Apple Metal GPU.
    Metal,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// AMD ROCm GPU.
    Rocm,
    /// Intel Level Zero GPU (integrated and discrete).
    LevelZero,
    /// Apple Core ML (ANE via IOSurface).
    CoreAi,
    /// Apple Neural Engine (direct ANE).
    Ane,
    /// Apple Accelerate framework (CPU BLAS/NEON).
    Accelerate,
    /// Candle CPU backend.
    CandleCpu,
    /// General CPU (no specific backend).
    Cpu,
    /// Tenstorrent Tensix accelerator.
    Tensix,
}

impl BackendKind {
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Metal => "Metal",
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::LevelZero => "Level Zero",
            Self::CoreAi => "Core ML",
            Self::Ane => "ANE",
            Self::Accelerate => "Accelerate",
            Self::CandleCpu => "Candle CPU",
            Self::Cpu => "CPU",
            Self::Tensix => "Tensix",
        }
    }
}

/// Memory properties of a compute device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemoryInfo {
    /// Total physical memory in bytes.
    pub total_bytes: u64,
    /// Free/available memory at probe time in bytes. 0 if unknown.
    pub free_bytes: u64,
    /// Estimated memory bandwidth in GB/s. 0.0 if unknown.
    pub bandwidth_gb_per_sec: f64,
    /// Whether the device shares its address space with the CPU.
    pub unified_with_cpu: bool,
}

/// PCIe link information for discrete devices.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct PcieLinkInfo {
    pub generation: u32,   // 3, 4, 5
    pub lanes: u32,        // 4, 8, 16
    pub max_speed_gb_per_sec: f64,
}

/// A single discovered compute device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique identifier within this registry instance.
    pub id: DeviceId,
    /// Broad device category.
    pub kind: DeviceKind,
    /// Specific compute backend.
    pub backend: BackendKind,
    /// Human-readable name ("Apple M3 GPU", "NVIDIA RTX 4090").
    pub name: String,
    /// Vendor string ("Apple", "NVIDIA", "AMD", "Intel", "generic").
    pub vendor: String,
    /// Driver version string if available.
    pub driver_version: String,
    /// Memory properties.
    pub memory: DeviceMemoryInfo,
    /// Number of compute units / cores / SIMD width.
    pub compute_units: u32,
    /// Core clock in MHz. 0 if unknown.
    pub clock_mhz: u32,
    /// Number of NPU/ANE cores. 0 for non-NPU devices.
    pub ane_cores: u32,
    /// Supported data formats.
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_int8: bool,
    pub supports_ternary: bool,
    /// PCIe link info for discrete devices. None for integrated/unified.
    pub pcie_link: Option<PcieLinkInfo>,
}

impl DeviceInfo {
    /// Short one-line summary.
    pub fn summary(&self) -> String {
        let kind_label = self.kind.label();
        let mem_gb = self.memory.total_bytes as f64 / 1_000_000_000.0;
        format!(
            "{}: {} {:.1} GB {} cores",
            kind_label, self.name, mem_gb, self.compute_units
        )
    }
}

// ── Registry ────────────────────────────────────────────────────────────────

/// Runtime device registry — the single source of truth for available hardware.
///
/// Created by [`DeviceRegistry::discover()`], which runs all registered
/// [`DeviceProbe`] implementations and collects their results.
#[derive(Debug, Clone)]
pub struct DeviceRegistry {
    devices: Vec<DeviceInfo>,
    default_device: Option<DeviceId>,
}

impl DeviceRegistry {
    /// Discover all devices by running every registered probe.
    ///
    /// Probes are defined in the [`probes`] module and registered via
    /// [`probes::all_probes()`]. The registry assigns sequential [`DeviceId`]s
    /// and selects a default device (the first GPU, or the first CPU if no GPU
    /// is found).
    pub fn discover() -> Self {
        let mut devices: Vec<DeviceInfo> = Vec::new();
        let mut next_id: u32 = 0;

        for mut probe_devices in probes::all_probes() {
            for device in probe_devices.iter_mut() {
                device.id = DeviceId(next_id);
                next_id += 1;
            }
            devices.extend(probe_devices);
        }

        // Default: first non-CPU device (GPU, NPU), otherwise first CPU.
        let default_device = devices
            .iter()
            .find(|d| !matches!(d.kind, DeviceKind::Cpu))
            .or_else(|| devices.first())
            .map(|d| d.id);

        Self {
            devices,
            default_device,
        }
    }

    /// All discovered devices.
    pub fn enumerate(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Number of discovered devices.
    pub fn count(&self) -> usize {
        self.devices.len()
    }

    /// Get device info by id.
    pub fn get(&self, id: DeviceId) -> Option<&DeviceInfo> {
        self.devices.iter().find(|d| d.id == id)
    }

    /// Find all devices matching a backend kind.
    pub fn by_backend(&self, kind: BackendKind) -> Vec<&DeviceInfo> {
        self.devices.iter().filter(|d| d.backend == kind).collect()
    }

    /// Find all devices matching a device kind.
    pub fn by_kind(&self, kind: DeviceKind) -> Vec<&DeviceInfo> {
        self.devices.iter().filter(|d| d.kind == kind).collect()
    }

    /// The default device for inference (first GPU, or first CPU).
    pub fn default(&self) -> Option<&DeviceInfo> {
        self.default_device.and_then(|id| self.get(id))
    }

    /// Serialize the registry to JSON for headless/config output.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Serialize the registry to pretty-printed JSON.
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Re-discover all devices (re-runs all probes).
    pub fn rediscover(&self) -> Self {
        Self::discover()
    }
}

// ── JSON output for host apps ──────────────────────────────────────────────

impl Serialize for DeviceRegistry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("DeviceRegistry", 3)?;
        state.serialize_field("device_count", &self.devices.len())?;
        state.serialize_field("devices", &self.devices)?;
        state.serialize_field(
            "default_device",
            &self.default_device.map(|id| id.0),
        )?;
        state.end()
    }
}

// ── Lazy global registry ───────────────────────────────────────────────────

/// Global device registry, lazily initialized on first access.
pub static GLOBAL_REGISTRY: LazyLock<DeviceRegistry> = LazyLock::new(DeviceRegistry::discover);

/// Convenience: return a reference to the global device registry.
pub fn global_registry() -> &'static DeviceRegistry {
    &GLOBAL_REGISTRY
}
