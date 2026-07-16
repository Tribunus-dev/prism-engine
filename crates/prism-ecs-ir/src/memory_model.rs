//! Memory model and hardware descriptor types for the compiler & scheduler.
//!
//! `MemoryModel` distinguishes unified (CPU/GPU DDR) from discrete (PCIe-attached)
//! memory topologies. `HardwareDescriptor` is an ECS component carried by device
//! entities; the scheduler reads it to insert explicit copy ops at discrete-memory
//! boundaries.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

// ── MemoryModel ──────────────────────────────────────────────────────────────

/// Whether a device's memory is unified with the host or discrete.
///
/// - `Unified` — the device shares address space with the host (typical of
///   integrated GPUs, NPUs, and on‑package accelerators). No explicit copy is
///   needed; the scheduler may elide transfer barriers.
/// - `Discrete` — the device has its own private VRAM reachable via a
///   memory‑mapped aperture over a PCIe (or similar) bus. The scheduler MUST
///   insert explicit copy operations at every discrete‑memory boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MemoryModel {
    /// Unified (shared) memory — no explicit copies required.
    Unified,
    /// Discrete (private) memory accessible over a bus.
    Discrete {
        /// PCIe generation (e.g. 5 for Gen5).
        pcie_gen: u8,
        /// Number of PCIe lanes (e.g. 16 for x16).
        pcie_lanes: u8,
        /// Achievable bandwidth in GB/s.
        bandwidth_gbs: f32,
    },
}

// ── HardwareKind ─────────────────────────────────────────────────────────────

/// Broad classification of a hardware device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareKind {
    /// Graphics processing unit.
    Gpu,
    /// Neural processing unit (e.g. Apple Neural Engine).
    Npu,
    /// Central processing unit.
    Cpu,
    /// Apple Neural Engine (ANE — distinct from NPU for scheduling purposes).
    Ane,
}

// ── HardwareDescriptor ───────────────────────────────────────────────────────

/// ECS component describing a device's memory topology and identity.
///
/// Every hardware device entity in the ECS world carries this component.
/// The scheduler and compiler use it to decide whether copies are necessary
/// and how to model memory constraints during compilation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareDescriptor {
    /// Human‑readable device name (e.g. "Apple M1 GPU", "NVIDIA RTX 4090").
    pub name: String,
    /// Kind of hardware (GPU, NPU, CPU, ANE).
    pub kind: HardwareKind,
    /// Memory model describing the device's memory topology.
    pub memory_model: MemoryModel,
    /// Total device‑local memory in bytes (VRAM capacity).
    pub memory_size_bytes: u64,
}

impl Component for HardwareDescriptor {}

// ── Helper predicates ────────────────────────────────────────────────────────

/// Returns `true` when the device uses discrete memory and therefore requires
/// explicit copy operations at memory boundaries.
///
/// For `Unified` devices this is always `false`; the scheduler may elide
/// transfer barriers and assume direct pointer access.
pub fn needs_explicit_copy(descriptor: &HardwareDescriptor) -> bool {
    matches!(descriptor.memory_model, MemoryModel::Discrete { .. })
}

// ── Constructors ─────────────────────────────────────────────────────────────

/// Build a [`HardwareDescriptor`] for a device with unified memory.
pub fn unified_memory_system(name: &str, kind: HardwareKind, size: u64) -> HardwareDescriptor {
    HardwareDescriptor {
        name: name.to_string(),
        kind,
        memory_model: MemoryModel::Unified,
        memory_size_bytes: size,
    }
}

/// Build a [`HardwareDescriptor`] for a discrete GPU with a PCIe link.
pub fn discrete_gpu(
    name: &str,
    kind: HardwareKind,
    size: u64,
    gen: u8,
    lanes: u8,
) -> HardwareDescriptor {
    HardwareDescriptor {
        name: name.to_string(),
        kind,
        memory_model: MemoryModel::Discrete {
            pcie_gen: gen,
            pcie_lanes: lanes,
            bandwidth_gbs: 0.0, // caller can overwrite if known
        },
        memory_size_bytes: size,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_device_does_not_need_explicit_copy() {
        let desc = unified_memory_system("Apple M1", HardwareKind::Gpu, 16 << 30);
        assert!(!needs_explicit_copy(&desc));
    }

    #[test]
    fn discrete_device_needs_explicit_copy() {
        let desc = discrete_gpu("RTX 4090", HardwareKind::Gpu, 24 << 30, 5, 16);
        assert!(needs_explicit_copy(&desc));
    }
}
