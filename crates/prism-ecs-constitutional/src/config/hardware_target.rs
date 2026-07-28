//! Hardware target for a ComputeImage compilation.
//!
//! Authority: the canonical enumeration of Apple Silicon targets the
//! Constitutional Engine may select for a ComputeImage, plus detection
//! helpers and per-target tuning constants. The detection is platform-aware
//! on macOS via `sysctl`; other platforms fall back to a conservative
//! default so the surface stays usable in cross-platform builds.

use serde::{Deserialize, Serialize};

/// Target hardware for a ComputeImage compilation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// Apple M1 (16GB baseline) — max compression, streaming-friendly segments
    M1,
    /// Apple M1 Pro/Max (32-64GB) — moderate compression
    M1Pro,
    /// Apple M2/M2 Pro/Max (24-96GB) — balanced
    M2,
    /// Apple M2 Ultra/M3 Max (96-192GB) — high precision
    M2Ultra,
    /// Apple M3 Ultra (256-512GB) — maximum precision, batched layout
    M3Ultra,
}

impl HardwareTarget {
    /// Construct a [`HardwareTarget`] from observed RAM (MB) and CPU count.
    ///
    /// This is the pure-Rust constructor used by [`Self::detect`] and by
    /// tests; the engine may also call it directly when the host's RAM is
    /// already known from another source.
    pub fn from_observed(ram_mb: u32, cpu_count: u32) -> Self {
        match (ram_mb, cpu_count) {
            (r, c) if r >= 393_216 && c >= 24 => Self::M3Ultra,
            (r, c) if r >= 131_072 && c >= 20 => Self::M2Ultra,
            (r, c) if r >= 65_536 && c >= 12 => Self::M2,
            (r, _c) if r >= 32_768 => Self::M1Pro,
            _ => Self::M1,
        }
    }

    /// Auto-detect the current machine's target.
    ///
    /// Uses `sysctl -n hw.memsize` on macOS to read total physical RAM
    /// and falls back to a conservative 16 GB M1 default on other
    /// platforms or when the call fails. CPU count comes from
    /// `std::thread::available_parallelism` with a fallback of 8.
    pub fn detect() -> Self {
        let ram_mb = detect_physical_ram_mb();
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(8);
        Self::from_observed(ram_mb, cpu_count)
    }

    /// Optimal quantization for this hardware.
    pub fn recommended_quant(&self) -> &'static str {
        match self {
            Self::M1 => "nf4-128",
            Self::M1Pro => "nf4-64",
            Self::M2 => "nf4-64",
            Self::M2Ultra => "8bit",
            Self::M3Ultra => "none",
        }
    }

    /// Whether weight streaming is beneficial (low RAM systems).
    pub fn needs_weight_streaming(&self) -> bool {
        matches!(self, Self::M1 | Self::M1Pro)
    }

    /// Recommended batch size for prefill+decode.
    pub fn recommended_batch(&self) -> u32 {
        match self {
            Self::M1 => 4,
            Self::M1Pro => 8,
            Self::M2 => 12,
            Self::M2Ultra => 20,
            Self::M3Ultra => 32,
        }
    }

    /// Number of ANE cores available for speculation.
    pub fn ane_cores(&self) -> u32 {
        match self {
            Self::M1 | Self::M1Pro | Self::M2 => 16,
            Self::M2Ultra | Self::M3Ultra => 32,
        }
    }

    /// Segment layout: small + many for streaming, large + few for batched.
    pub fn segment_target_size_mb(&self) -> u32 {
        match self {
            Self::M1 => 64,
            Self::M1Pro => 128,
            Self::M2 => 256,
            Self::M2Ultra => 512,
            Self::M3Ultra => 1024,
        }
    }
}

/// Read total physical RAM in megabytes via `sysctl hw.memsize` on macOS.
/// On other platforms or when the call fails, returns a conservative
/// 16 GB M1 default so downstream selection logic remains bounded.
fn detect_physical_ram_mb() -> u32 {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok();
        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if let Ok(bytes) = s.parse::<u64>() {
                return (bytes / 1_048_576) as u32;
            }
        }
    }
    16_384
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_observed_classifies_by_tiers() {
        assert_eq!(HardwareTarget::from_observed(16_384, 8), HardwareTarget::M1);
        assert_eq!(HardwareTarget::from_observed(49_152, 12), HardwareTarget::M1Pro);
        assert_eq!(HardwareTarget::from_observed(98_304, 16), HardwareTarget::M2);
        assert_eq!(HardwareTarget::from_observed(196_608, 24), HardwareTarget::M2Ultra);
        assert_eq!(HardwareTarget::from_observed(524_288, 32), HardwareTarget::M3Ultra);
    }

    #[test]
    fn recommended_quant_progression() {
        assert_eq!(HardwareTarget::M1.recommended_quant(), "nf4-128");
        assert_eq!(HardwareTarget::M2Ultra.recommended_quant(), "8bit");
        assert_eq!(HardwareTarget::M3Ultra.recommended_quant(), "none");
    }

    #[test]
    fn streaming_predicate_matches_only_low_ram() {
        assert!(HardwareTarget::M1.needs_weight_streaming());
        assert!(HardwareTarget::M1Pro.needs_weight_streaming());
        assert!(!HardwareTarget::M2.needs_weight_streaming());
        assert!(!HardwareTarget::M3Ultra.needs_weight_streaming());
    }

    #[test]
    fn segment_target_scales_with_tier() {
        assert!(HardwareTarget::M1.segment_target_size_mb() < HardwareTarget::M3Ultra.segment_target_size_mb());
    }
}
