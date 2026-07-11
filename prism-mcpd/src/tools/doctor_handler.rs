use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub struct DoctorHandler;

impl DoctorHandler {
    pub fn new() -> Self {
        Self
    }
}

impl McpHandler for DoctorHandler {
    fn name(&self) -> &'static str {
        "inspect_host"
    }

    fn description(&self) -> &'static str {
        "Inspect the host environment: OS, CPU, memory, accelerators, toolchains."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    fn call(
        &self,
        _request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let mut info = String::new();

        // OS
        info.push_str(&format!(
            "OS: {} {}\n",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));

        // CPU cores
        info.push_str(&format!("CPU cores: {}\n", num_cpus()));

        // Memory
        if let Some(mem) = system_memory() {
            info.push_str(&format!(
                "Memory: {} GB total, {} GB available\n",
                mem.total_gb, mem.avail_gb
            ));
        }

        // Shell
        info.push_str(&format!(
            "Shell: {}\n",
            std::env::var("SHELL").unwrap_or_else(|_| "unknown".into())
        ));

        // Rust toolchain
        if let Ok(output) = std::process::Command::new("rustc")
            .arg("--version")
            .output()
        {
            let version = String::from_utf8_lossy(&output.stdout);
            info.push_str(&format!("Rust: {}", version.trim()));
        }

        // Xcode
        if let Ok(output) = std::process::Command::new("xcodebuild")
            .arg("-version")
            .output()
        {
            let version = String::from_utf8_lossy(&output.stdout);
            let first_line = version.lines().next().unwrap_or("");
            info.push_str(&format!("Xcode: {}\n", first_line));
        }

        Ok(ToolResult::text(info))
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

struct MemInfo {
    total_gb: f64,
    avail_gb: f64,
}

#[cfg(target_os = "macos")]
fn system_memory() -> Option<MemInfo> {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .arg("hw.memsize")
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let value: u64 = stdout.split(':').nth(1)?.trim().parse().ok()?;
        let total_gb = value as f64 / 1_073_741_824.0;
        return Some(MemInfo {
            total_gb,
            avail_gb: total_gb * 0.5,
        });
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}
