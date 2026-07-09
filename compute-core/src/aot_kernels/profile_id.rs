//! Apple Silicon profile identifiers — stable, coarse-grained profile keys
//! used for AOT kernel variant selection in CImage kernel catalogs.
//!
//! Each enum variant corresponds to a known Apple Silicon generation + GPU tier.
//! Unknown variants use the conservative `UnknownAppleSilicon` fallback.

use serde::{Deserialize, Serialize};
use std::fmt;

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
