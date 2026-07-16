//! TTCore component — the ECS component representing a Tenstorrent RISCV-32 core.
//!
//! Each core on the Wormhole/Grayskull mesh is identified by its (x, y) coordinate
//! on the 2D grid and carries the kernel it executes. The per-tensor evolution model
//! maps naturally: each core runs one (format, operation) kernel over the DRAM
//! buffers it is connected to via the NoC.

use serde::{Deserialize, Serialize};

use prism_ecs_core::Component;

/// A Tenstorrent compute core on the Wormhole/Grayskull mesh.
///
/// Coordinates follow the TT-Metalium convention:
/// - x: column index on the 2D mesh (east-west)
/// - y: row index on the 2D mesh (north-south)
/// - kernel: name or path of the compiled RISCV-32 ELF kernel to execute
///
/// The combination (x, y, kernel) uniquely identifies what work a given
/// core performs. A core may hold multiple overlays but only one active
/// kernel at a time per the hardware SRAM constraints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct TTCore {
    /// Column (east-west) coordinate on the 2D mesh.
    pub x: u16,
    /// Row (north-south) coordinate on the 2D mesh.
    pub y: u16,
    /// Kernel name or compiled ELF identifier to load onto this core.
    pub kernel: String,
}

impl TTCore {
    /// Create a new TTCore descriptor.
    pub fn new(x: u16, y: u16, kernel: impl Into<String>) -> Self {
        Self {
            x,
            y,
            kernel: kernel.into(),
        }
    }

    /// Placeholder — a Wormhole device has a 12 × 8 grid of compute cores
    /// (96 cores) with additional neighbour routing cores.
    pub fn wormhole_grid_size() -> (u16, u16) {
        (12, 8)
    }

    /// Placeholder — a Grayskull device has a 10 × 12 grid (120 cores).
    pub fn grayskull_grid_size() -> (u16, u16) {
        (10, 12)
    }

    /// Check whether this core coordinate is valid for a Wormhole device.
    pub fn is_valid_wormhole(&self) -> bool {
        let (max_x, max_y) = Self::wormhole_grid_size();
        self.x < max_x && self.y < max_y
    }

    /// Check whether this core coordinate is valid for a Grayskull device.
    pub fn is_valid_grayskull(&self) -> bool {
        let (max_x, max_y) = Self::grayskull_grid_size();
        self.x < max_x && self.y < max_y
    }
}

impl Component for TTCore {}

/// A logical rectangle of core coordinates — the typical dispatch unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreRange {
    /// Inclusive start coordinate.
    pub start: (u16, u16),
    /// Inclusive end coordinate.
    pub end: (u16, u16),
}

impl CoreRange {
    /// Build a new core range. No validation — the caller is responsible
    /// for ensuring start ≤ end in both dimensions.
    pub const fn new(start: (u16, u16), end: (u16, u16)) -> Self {
        Self { start, end }
    }

    /// Iterate over all (x, y) coordinates in row-major order.
    pub fn coords(&self) -> impl Iterator<Item = (u16, u16)> + '_ {
        let (sx, sy) = self.start;
        let (ex, ey) = self.end;
        (sy..=ey).flat_map(move |y| (sx..=ex).map(move |x| (x, y)))
    }

    /// Number of cores covered by this range.
    pub fn count(&self) -> usize {
        let (sx, sy) = self.start;
        let (ex, ey) = self.end;
        ((ex - sx + 1) * (ey - sy + 1)) as usize
    }
}

/// Ethernet link descriptor for multi-device synchronization.
///
/// Tenstorrent devices communicate chip-to-chip via ethernet links.
/// Each link connects a core on the local device to a core on a remote device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EthernetLink {
    /// Local core coordinate.
    pub local: (u16, u16),
    /// Remote device index (NOC target).
    pub remote_device: u8,
    /// Remote core coordinate.
    pub remote: (u16, u16),
}

impl EthernetLink {
    pub fn new(local: (u16, u16), remote_device: u8, remote: (u16, u16)) -> Self {
        Self {
            local,
            remote_device,
            remote,
        }
    }
}
/// Tenstorrent hardware configuration …
/// Tenstorrent hardware configuration for a card or mesh.
///
/// See also `inter_card_bandwidth()` for the known per-card-link ethernet
/// bandwidth — the Wormhole inter-card ethernet links run at ~400 GB/s,
/// substantially faster than PCIe Gen 5 (64 GB/s). This means cross-node
/// Tenstorrent transfers can be cheaper than same-node CPU↔GPU PCIe transfers.
#[derive(Debug, Clone)]
pub struct TtHardwareConfig {
    /// Number of compute cores on the device.
    pub core_count: u32,
    /// DRAM capacity in bytes.
    pub dram_bytes: u64,
    /// Inter-card ethernet bandwidth in GB/s (Wormhole: ~400 GB/s).
    pub inter_card_bandwidth_gbs: f32,
    /// Whether this device has ethernet links to other cards.
    pub has_ethernet_links: bool,
}

impl Default for TtHardwareConfig {
    fn default() -> Self {
        Self {
            core_count: 96,                       // Wormhole default
            dram_bytes: 128 * 1024 * 1024 * 1024, // 128 GB
            inter_card_bandwidth_gbs: 400.0,      // Wormhole ethernet ~400 GB/s
            has_ethernet_links: true,
        }
    }
}

/// Known per-card-link inter-card ethernet bandwidth.
///
/// Tenstorrent Wormhole inter-card ethernet links run at approximately
/// **400 GB/s** — substantially faster than PCIe Gen 5 (64 GB/s). This
/// means cross-node Tenstorrent transfers can be cheaper than same-node
/// CPU↔GPU PCIe transfers, directly impacting scheduling decisions in
/// the NTB (network topology bridge) scheduler.
///
/// The NTB scheduler MUST treat Tenstorrent nodes as "effectively unified"
/// for scheduling purposes, not like discrete NVIDIA dGPUs, because the
/// inter-card bandwidth dwarfs host PCIe bottlenecks.
///
/// # Return value
///
/// Returns 400.0 GB/s for Wormhole B0. Grayskull has no ethernet links
/// (returns 0.0).
pub fn inter_card_bandwidth() -> f32 {
    400.0
}

#[cfg(test)]
mod tests {
    #[test]
    fn inter_card_bandwidth_constant() {
        let bw = inter_card_bandwidth();
        assert!(
            bw > 0.0,
            "inter-card bandwidth must be positive for Wormhole"
        );
        assert!((bw - 400.0).abs() < 0.001, "expected 400 GB/s, got {bw}");
    }

    #[test]
    fn tt_hardware_config_default() {
        let cfg = TtHardwareConfig::default();
        assert_eq!(cfg.core_count, 96);
        assert!((cfg.inter_card_bandwidth_gbs - 400.0).abs() < 0.001);
        assert!(cfg.has_ethernet_links);
    }

    #[test]
    fn tt_hardware_config_custom() {
        let cfg = TtHardwareConfig {
            core_count: 120,
            dram_bytes: 64 * 1024 * 1024 * 1024,
            inter_card_bandwidth_gbs: 0.0,
            has_ethernet_links: false,
        };
        assert_eq!(cfg.core_count, 120);
    }

    #[test]
    fn tt_core_creation() {
        let core = TTCore::new(1, 2, "matmul_kernel");
        assert_eq!(core.x, 1);
        assert_eq!(core.y, 2);
        assert_eq!(core.kernel, "matmul_kernel");
    }

    #[test]
    fn tt_core_component_trait() {
        // Verifies that TTCore implements Component (and thus Debug + Send + Sync + 'static).
        fn assert_component<T: Component>() {}
        assert_component::<TTCore>();
    }

    #[test]
    fn tt_core_is_component() {
        let core = TTCore::new(4, 3, "gemm_kernel");
        // Use as a component in an ECS context by checking debug output.
        assert!(format!("{:?}", core).contains("gemm_kernel"));
    }

    #[test]
    fn wormhole_validation() {
        let valid = TTCore::new(8, 5, "test");
        assert!(valid.is_valid_wormhole());

        let invalid = TTCore::new(12, 8, "test");
        assert!(!invalid.is_valid_wormhole());
    }

    #[test]
    fn grayskull_validation() {
        let valid = TTCore::new(8, 10, "test");
        assert!(valid.is_valid_grayskull());

        let invalid = TTCore::new(10, 12, "test");
        assert!(!invalid.is_valid_grayskull());
    }

    #[test]
    fn core_range_coords() {
        let range = CoreRange::new((2, 3), (4, 5));
        let coords: Vec<_> = range.coords().collect();
        // 3 columns × 3 rows = 9 cores
        assert_eq!(coords.len(), 9);
        assert!(coords.contains(&(2, 3)));
        assert!(coords.contains(&(4, 5)));
        assert!(coords.contains(&(3, 4)));
    }

    #[test]
    fn core_range_count() {
        assert_eq!(CoreRange::new((0, 0), (0, 0)).count(), 1);
        assert_eq!(CoreRange::new((0, 0), (11, 7)).count(), 96); // full Wormhole
        assert_eq!(CoreRange::new((0, 0), (9, 11)).count(), 120); // full Grayskull
    }

    #[test]
    fn ethernet_link_creation() {
        let link = EthernetLink::new((1, 1), 2, (3, 4));
        assert_eq!(link.local, (1, 1));
        assert_eq!(link.remote_device, 2);
        assert_eq!(link.remote, (3, 4));
    }
}
