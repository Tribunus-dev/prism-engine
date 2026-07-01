// Hardware detection helpers for EXO cluster nodes.

/// Hardware capabilities detected on this node.
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub chip: String,
    pub ram_gb: u32,
    pub rdma_available: bool,
    pub ane_cores: u32,
}

/// Detect hardware capabilities of the current machine.
///
/// Reads sysctl for chip model, RAM, and ANE core count.  Checks for
/// Thunderbolt networking interfaces to determine RDMA availability.
pub fn detect_hardware() -> HardwareInfo {
    let chip = detect_chip();
    let ram_mb = crate::gpu_memory::total_physical_ram_mb();
    let ram_gb = (ram_mb + 512) / 1024; // round to nearest GB
    let rdma_available = detect_rdma();
    let ane_cores = detect_ane_cores();

    HardwareInfo {
        chip,
        ram_gb,
        rdma_available,
        ane_cores,
    }
}

/// Read the chip model via `sysctl -n machdep.cpu.brand_string`.
pub(crate) fn detect_chip() -> String {
    // Attempt sysctl query for Apple Silicon model.
    if let Ok(chip) = sysctl_value("machdep.cpu.brand_string") {
        if !chip.is_empty() {
            return chip;
        }
    }

    // Fallback: detect via sysctl for hw.model (e.g. "Mac15,7").
    if let Ok(model) = sysctl_value("hw.model") {
        return model;
    }

    // Last resort: uname.
    if let Ok(uname) = sysctl_value("kern.version") {
        let parts: Vec<&str> = uname.split_whitespace().collect();
        if !parts.is_empty() {
            return parts[0].to_string();
        }
    }

    "Apple Silicon".to_string()
}

/// Check for Thunderbolt networking interfaces as a proxy for RDMA
/// availability.  Thunderbolt 4/5 devices expose `en` interfaces
/// or `ap` interfaces, visible via `ifconfig` or the `IOThunderbolt`
/// IOKit registry.
pub(crate) fn detect_rdma() -> bool {
    // Check for Thunderbolt network interfaces by reading sysctl for
    // Thunderbolt-capable networking.
    if let Ok(thunderbolt) = sysctl_value("hw.thunderbolt") {
        if thunderbolt.contains("1") || thunderbolt.to_lowercase().contains("true") {
            return true;
        }
    }

    // Check for presence of Thunderbolt in IORegistry raw output.
    if let Ok(output) = std::process::Command::new("system_profiler")
        .args(["SPThunderboltDataType", "-json"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Any Thunderbolt device detected.
            if stdout.contains("Thunderbolt") || stdout.contains("thunderbolt") {
                return true;
            }
        }
    }

    // Check for IOThunderboltController in IORegistry (low-level check).
    if let Ok(output) = std::process::Command::new("ioreg")
        .args(["-rc", "IOThunderboltController"])
        .output()
    {
        if output.status.success() && !output.stdout.is_empty() {
            return true;
        }
    }

    false
}

/// Detect the number of ANE (Apple Neural Engine) cores.
pub(crate) fn detect_ane_cores() -> u32 {
    // The ANE is exposed via IOKit as `AppleNeuralEngine`.
    if let Ok(output) = std::process::Command::new("ioreg")
        .args(["-rc", "AppleNeuralEngine"])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Try to extract the number of cores from the ioreg output.
            for line in stdout.lines() {
                if line.contains("ANE.CoreCount") || line.contains("CoreCount") {
                    if let Some(val) = line.split('=').nth(1) {
                        if let Ok(n) = val.trim().parse::<u32>() {
                            return n;
                        }
                    }
                }
            }
            // If we found the AppleNeuralEngine, it has at least 16 cores.
            if stdout.contains("AppleNeuralEngine") {
                // Attempt to detect Ultra variants via chip string.
                let chip = detect_chip();
                if chip.to_lowercase().contains("ultra") {
                    return 32;
                }
                return 16;
            }
        }
    }

    // Conservative fallback: assume at least 16 ANE cores (all M-series chips).
    16
}

/// Read a sysctl value as a trimmed string.
pub(crate) fn sysctl_value(key: &str) -> Result<String, String> {
    let output = std::process::Command::new("sysctl")
        .args(["-n", key])
        .output()
        .map_err(|e| format!("sysctl {}: {}", key, e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(format!("sysctl {} returned non-zero", key))
    }
}

/// Format a chip name for display (short, readable).
pub(crate) fn format_chip_name(chip: &str) -> String {
    let lower = chip.to_lowercase();

    if lower.contains("m3 ultra") || lower.contains("m3-ultra") {
        return "Mac Studio M3 Ultra".to_string();
    }
    if lower.contains("m3 max") {
        return "MacBook Pro M3 Max".to_string();
    }
    if lower.contains("m3 pro") {
        return "MacBook Pro M3 Pro".to_string();
    }
    if lower.contains("m3") {
        return "Mac M3".to_string();
    }
    if lower.contains("m2 ultra") || lower.contains("m2-ultra") {
        return "Mac Studio M2 Ultra".to_string();
    }
    if lower.contains("m2 max") {
        return "MacBook Pro M2 Max".to_string();
    }
    if lower.contains("m2 pro") {
        return "MacBook Pro M2 Pro".to_string();
    }
    if lower.contains("m2") {
        return "Mac M2".to_string();
    }
    if lower.contains("m1 ultra") || lower.contains("m1-ultra") {
        return "Mac Studio M1 Ultra".to_string();
    }
    if lower.contains("m1 max") {
        return "MacBook Pro M1 Max".to_string();
    }
    if lower.contains("m1 pro") {
        return "MacBook Pro M1 Pro".to_string();
    }
    if lower.contains("m1") {
        return "Mac M1".to_string();
    }

    // Fallback: return the raw string, truncated.
    if chip.len() > 28 {
        chip[..28].to_string()
    } else {
        chip.to_string()
    }
}
