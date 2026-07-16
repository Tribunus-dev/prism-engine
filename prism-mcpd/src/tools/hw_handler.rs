use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde::Serialize;
use serde_json::json;
use sysinfo::{Disks, System};

/// Hardware profile collected from the host.
#[derive(Serialize)]
pub struct HardwareProfile {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub free_disk_gb: f64,
    pub metal_gpu_family: String,
    pub metal_gpu_cores: u32,
    pub ane_present: bool,
    pub cpu_cores: usize,
    pub cpu_model: String,
}

pub struct HwProbeHandler;

impl HwProbeHandler {
    pub fn new() -> Self {
        Self
    }
}

impl McpHandler for HwProbeHandler {
    fn name(&self) -> &'static str {
        "hw_probe"
    }

    fn description(&self) -> &'static str {
        "Probe hardware capabilities: RAM, disk, CPU, Metal GPU, and ANE."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({ "type": "object", "properties": {} })
    }

    fn call(
        &self,
        _request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let profile = ProbeResult::collect();
        Ok(ToolResult::text(serde_json::to_string_pretty(&profile)?))
    }
}

#[derive(Serialize)]
struct ProbeResult {
    total_ram_gb: f64,
    available_ram_gb: f64,
    free_disk_gb: f64,
    metal_gpu_family: String,
    metal_gpu_cores: u32,
    ane_present: bool,
    cpu_cores: usize,
    cpu_model: String,
}

impl ProbeResult {
    fn collect() -> Self {
        let mut sys = System::new();

        // -- RAM --
        sys.refresh_memory();
        let total_ram_gb = sys.total_memory() as f64 / 1_073_741_824.0;
        let available_ram_gb = sys.available_memory() as f64 / 1_073_741_824.0;

        // -- Disk (first non-removable mount) --
        let disks = Disks::new_with_refreshed_list();
        let free_disk_gb = disks
            .iter()
            .find(|d| !d.is_removable())
            .map(|d| d.available_space() as f64 / 1_073_741_824.0)
            .unwrap_or(0.0);

        // -- CPU --
        sys.refresh_cpu_all();
        let cpu_cores = sys.physical_core_count().unwrap_or(1);
        let cpu_brand = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown".into());
        let cpu_model = cpu_brand.trim().to_string();

        // -- Metal GPU family / cores / ANE from chip name --
        let (metal_gpu_family, metal_gpu_cores, ane_present) = gpu_defaults_from_cpu(&cpu_model);

        Self {
            total_ram_gb: (total_ram_gb * 10.0).round() / 10.0,
            available_ram_gb: (available_ram_gb * 10.0).round() / 10.0,
            free_disk_gb: (free_disk_gb * 10.0).round() / 10.0,
            metal_gpu_family,
            metal_gpu_cores,
            ane_present,
            cpu_cores,
            cpu_model,
        }
    }
}

/// Map a CPU model string to reasonable Metal GPU defaults.
///
/// The mapping is derived from published spec tables for Apple Silicon.
/// Family values correspond to `MTLGPUFamily`:
///   apple7 → M1, apple8 → M2, apple9 → M3/M4
fn gpu_defaults_from_cpu(model: &str) -> (String, u32, bool) {
    let lc = model.to_lowercase();

    if lc.contains("apple") {
        // ── M1 family ──────────────────────────────────────────
        if lc.contains("m1 ultra") {
            ("apple7".into(), 48, true)
        } else if lc.contains("m1 max") {
            ("apple7".into(), 24, true)
        } else if lc.contains("m1 pro") {
            ("apple7".into(), 14, true)
        } else if lc.starts_with("apple m1") {
            ("apple7".into(), 8, true)
        // ── M2 family ──────────────────────────────────────────
        } else if lc.contains("m2 ultra") {
            ("apple8".into(), 60, true)
        } else if lc.contains("m2 max") {
            ("apple8".into(), 30, true)
        } else if lc.contains("m2 pro") {
            ("apple8".into(), 16, true)
        } else if lc.starts_with("apple m2") {
            ("apple8".into(), 10, true)
        // ── M3 family ──────────────────────────────────────────
        } else if lc.contains("m3 max") {
            ("apple9".into(), 40, true)
        } else if lc.contains("m3 pro") {
            ("apple9".into(), 14, true)
        } else if lc.starts_with("apple m3") {
            ("apple9".into(), 10, true)
        // ── M4 family ──────────────────────────────────────────
        } else if lc.contains("m4 max") {
            ("apple9".into(), 40, true)
        } else if lc.contains("m4 pro") {
            ("apple9".into(), 16, true)
        } else if lc.starts_with("apple m4") {
            ("apple9".into(), 10, true)
        } else {
            // Generic Apple Silicon fallback
            ("apple_generic".into(), 8, true)
        }
    } else {
        ("unknown".into(), 0, false)
    }
}

/// Re-export for use by other handlers (e.g. hf_handler).
pub fn collect_hardware_profile() -> HardwareProfile {
    let r = ProbeResult::collect();
    HardwareProfile {
        total_ram_gb: r.total_ram_gb,
        available_ram_gb: r.available_ram_gb,
        free_disk_gb: r.free_disk_gb,
        metal_gpu_family: r.metal_gpu_family,
        metal_gpu_cores: r.metal_gpu_cores,
        ane_present: r.ane_present,
        cpu_cores: r.cpu_cores,
        cpu_model: r.cpu_model,
    }
}
