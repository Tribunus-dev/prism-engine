//! Profile identifiers for supported GPU families — Apple Silicon and AMD.
//!
//! Each family has its own enum of stable, coarse-grained profile keys
//! used for AOT kernel variant selection in CImage kernel catalogs.
//! Unknown variants use the conservative fallback for their family.

use serde::{Deserialize, Serialize};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Apple Silicon
// ═══════════════════════════════════════════════════════════════════════════

/// Stable profile identifier for Apple Silicon hardware.
///
/// Coarse enough for kernel variant selection, but not so coarse that
/// M1 and M4 Max collide on the same entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AppleSiliconProfileId {
    M1,
    M1Pro,
    M1Max,
    M1Ultra,
    M2,
    M2Pro,
    M2Max,
    M2Ultra,
    M3,
    M3Pro,
    M3Max,
    M3Ultra,
    M4,
    M4Pro,
    M4Max,
    M4Ultra,
    M5,
    M5Pro,
    M5Max,
    M5Ultra,
    /// Fallback for unrecognized Apple Silicon or non-Apple GPUs.
    UnknownAppleSilicon,
}

impl AppleSiliconProfileId {
    /// SOC generation: M1-family, M2-family, etc.
    pub fn soc_generation(self) -> u32 {
        match self {
            Self::M1 | Self::M1Pro | Self::M1Max | Self::M1Ultra => 1,
            Self::M2 | Self::M2Pro | Self::M2Max | Self::M2Ultra => 2,
            Self::M3 | Self::M3Pro | Self::M3Max | Self::M3Ultra => 3,
            Self::M4 | Self::M4Pro | Self::M4Max | Self::M4Ultra => 4,
            Self::M5 | Self::M5Pro | Self::M5Max | Self::M5Ultra => 5,
            Self::UnknownAppleSilicon => 0,
        }
    }

    /// GPU tier within a generation: base, Pro, Max, Ultra.
    pub fn gpu_tier(self) -> u32 {
        match self {
            Self::M1 | Self::M2 | Self::M3 | Self::M4 | Self::M5 => 1,
            Self::M1Pro | Self::M2Pro | Self::M3Pro | Self::M4Pro | Self::M5Pro => 2,
            Self::M1Max | Self::M2Max | Self::M3Max | Self::M4Max | Self::M5Max => 3,
            Self::M1Ultra | Self::M2Ultra | Self::M3Ultra | Self::M4Ultra | Self::M5Ultra => 4,
            Self::UnknownAppleSilicon => 0,
        }
    }

    /// Human-readable marketing name.
    pub fn marketing_name(self) -> &'static str {
        match self {
            Self::M1 => "Apple M1",
            Self::M1Pro => "Apple M1 Pro",
            Self::M1Max => "Apple M1 Max",
            Self::M1Ultra => "Apple M1 Ultra",
            Self::M2 => "Apple M2",
            Self::M2Pro => "Apple M2 Pro",
            Self::M2Max => "Apple M2 Max",
            Self::M2Ultra => "Apple M2 Ultra",
            Self::M3 => "Apple M3",
            Self::M3Pro => "Apple M3 Pro",
            Self::M3Max => "Apple M3 Max",
            Self::M3Ultra => "Apple M3 Ultra",
            Self::M4 => "Apple M4",
            Self::M4Pro => "Apple M4 Pro",
            Self::M4Max => "Apple M4 Max",
            Self::M4Ultra => "Apple M4 Ultra",
            Self::M5 => "Apple M5",
            Self::M5Pro => "Apple M5 Pro",
            Self::M5Max => "Apple M5 Max",
            Self::M5Ultra => "Apple M5 Ultra",
            Self::UnknownAppleSilicon => "Unknown Apple Silicon",
        }
    }
}

impl fmt::Display for AppleSiliconProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.marketing_name().to_lowercase().replace(' ', "_")
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AMD
// ═══════════════════════════════════════════════════════════════════════════

/// Stable profile identifier for AMD GPU hardware.
///
/// Coarse enough for kernel variant selection. Groups GPUs by architecture
/// generation and performance tier (compute unit count, memory bandwidth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AmdGpuProfileId {
    // ── CDNA 3 (Instinct MI300) ────────────────────────────────────────
    /// AMD Instinct MI300X — 304 CU, 192 GB HBM3, 5.2 TB/s
    InstinctMi300X,
    /// AMD Instinct MI300A — 228 CU, 128 GB HBM3, 5.2 TB/s (APU)
    InstinctMi300A,
    // ── CDNA 4 (Instinct MI350) ────────────────────────────────────────
    /// AMD Instinct MI350 — next-gen CDNA 4 (placeholder, TBD specs)
    InstinctMi350,
    // ── RDNA 3 (consumer) ──────────────────────────────────────────────
    /// AMD Radeon RX 7900 XTX — 96 CU, 24 GB GDDR6, 960 GB/s
    RadeonRx7900Xtx,
    /// AMD Radeon RX 7900 XT — 84 CU, 20 GB GDDR6, 800 GB/s
    RadeonRx7900Xt,
    /// AMD Radeon RX 7800 XT — 60 CU, 16 GB GDDR6, 624 GB/s
    RadeonRx7800Xt,
    // ── RDNA 3.5 (integrated / Strix Point) ────────────────────────────
    /// AMD Ryzen AI 9 HX 370 (RDNA 3.5 iGPU) — 16 CU
    RyzenAi9Hx370,
    /// Fallback for unrecognized AMD GPUs.
    UnknownAmd,
}

impl AmdGpuProfileId {
    /// Architecture generation (for fallback matching).
    pub fn arch_generation(self) -> u32 {
        match self {
            Self::InstinctMi300X | Self::InstinctMi300A => 3, // CDNA 3
            Self::InstinctMi350 => 4,                         // CDNA 4
            Self::RadeonRx7900Xtx | Self::RadeonRx7900Xt | Self::RadeonRx7800Xt => 3, // RDNA 3
            Self::RyzenAi9Hx370 => 35,                        // RDNA 3.5
            Self::UnknownAmd => 0,
        }
    }

    /// Performance tier within architecture: 1=entry, 2=mid, 3=high, 4/5=flagship.
    pub fn perf_tier(self) -> u32 {
        match self {
            Self::InstinctMi350 => 5, // datacenter flagship
            Self::InstinctMi300X => 4,
            Self::InstinctMi300A => 4,
            Self::RadeonRx7900Xtx => 3,
            Self::RadeonRx7900Xt => 2,
            Self::RadeonRx7800Xt => 2,
            Self::RyzenAi9Hx370 => 1,
            Self::UnknownAmd => 0,
        }
    }

    /// Human-readable marketing name.
    pub fn marketing_name(self) -> &'static str {
        match self {
            Self::InstinctMi300X => "AMD Instinct MI300X",
            Self::InstinctMi300A => "AMD Instinct MI300A",
            Self::InstinctMi350 => "AMD Instinct MI350",
            Self::RadeonRx7900Xtx => "AMD Radeon RX 7900 XTX",
            Self::RadeonRx7900Xt => "AMD Radeon RX 7900 XT",
            Self::RadeonRx7800Xt => "AMD Radeon RX 7800 XT",
            Self::RyzenAi9Hx370 => "AMD Ryzen AI 9 HX 370",
            Self::UnknownAmd => "Unknown AMD GPU",
        }
    }

    /// Number of compute units for this GPU.
    pub fn compute_units(self) -> u32 {
        match self {
            Self::InstinctMi300X => 304,
            Self::InstinctMi300A => 228,
            Self::InstinctMi350 => 344,
            Self::RadeonRx7900Xtx => 96,
            Self::RadeonRx7900Xt => 84,
            Self::RadeonRx7800Xt => 60,
            Self::RyzenAi9Hx370 => 16,
            Self::UnknownAmd => 0,
        }
    }

    /// Whether profile is a datacenter-class GPU (Instinct series).
    pub fn is_datacenter(self) -> bool {
        matches!(
            self,
            Self::InstinctMi300X | Self::InstinctMi300A | Self::InstinctMi350
        )
    }
}

impl fmt::Display for AmdGpuProfileId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.marketing_name().to_lowercase().replace(' ', "_")
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Shared types
// ═══════════════════════════════════════════════════════════════════════════

/// Evidence quality for profile and receipt data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileEvidenceStatus {
    /// Only static/declared specs are known (no measurement).
    StaticOnly,
    /// Measured on the local developer machine.
    MeasuredLocal,
    /// Measured in a controlled lab environment.
    MeasuredLab,
    /// Predicted from a sibling profile (same generation, different tier).
    PredictedFromSibling,
    /// Previously valid data that is now deprecated (superseded by newer profile).
    Deprecated,
}

impl ProfileEvidenceStatus {
    pub fn is_measured(self) -> bool {
        matches!(self, Self::MeasuredLocal | Self::MeasuredLab)
    }

    pub fn is_actionable_for_aot(self) -> bool {
        matches!(
            self,
            Self::MeasuredLocal | Self::MeasuredLab | Self::PredictedFromSibling
        )
    }
}
