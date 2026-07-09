//! Device profile matching — maps a runtime Metal device to the
//! nearest known Apple Silicon profile ID.
//!
//! Uses name matching first, then falls back to compute-unit proximity,
//! then to the generic unknown fallback.

use serde::{Deserialize, Serialize};

use super::profile_db::AppleSiliconProfileDb;
use super::profile_id::AppleSiliconProfileId;

/// Device profile extracted from the live Metal device at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMetalDeviceProfile {
    /// Raw Metal GPU name (e.g. "Apple M4 Max").
    pub device_name: String,
    /// Registered name from Metal API.
    pub registry_name: String,
    /// Number of compute units.
    pub compute_units: u32,
    /// Max threads per threadgroup.
    pub max_threads_per_threadgroup: u32,
    /// Max threadgroup memory in bytes.
    pub max_threadgroup_memory_bytes: u32,
    /// Recommended max working set (may be None on older Metal).
    pub recommended_max_working_set: Option<u64>,
    /// Whether the device supports SIMD-group operations.
    pub supports_simdgroup: bool,
}

/// Match a runtime device against the profile DB and return the
/// best profile ID.
pub fn match_device_to_profile(
    device: &RuntimeMetalDeviceProfile,
    db: &AppleSiliconProfileDb,
) -> AppleSiliconProfileId {
    // 1. Try exact name match.
    if let Some(id) = try_match_by_name(device) {
        if db.by_id(id).is_some() {
            return id;
        }
    }

    // 2. Try profile with closest compute unit count.
    if let Some(id) = try_match_by_cu_count(device, db) {
        return id;
    }

    // 3. Fall back to unknown.
    AppleSiliconProfileId::UnknownAppleSilicon
}

fn try_match_by_name(device: &RuntimeMetalDeviceProfile) -> Option<AppleSiliconProfileId> {
    let name = device.device_name.to_lowercase();
    let reg = device.registry_name.to_lowercase();
    let combined = format!("{} {}", name, reg);

    // SOC generation
    let gen: u32;
    if combined.contains("m5") {
        gen = 5;
    } else if combined.contains("m4") {
        gen = 4;
    } else if combined.contains("m3") {
        gen = 3;
    } else if combined.contains("m2") {
        gen = 2;
    } else if combined.contains("m1") {
        gen = 1;
    } else {
        return None;
    }

    // GPU tier
    let tier: u32;
    if combined.contains("ultra") {
        tier = 4;
    } else if combined.contains("max") {
        tier = 3;
    } else if combined.contains("pro") {
        tier = 2;
    } else {
        tier = 1;
    }

    Some(AppleSiliconProfileId::from_gen_tier(gen, tier))
}

fn try_match_by_cu_count(
    device: &RuntimeMetalDeviceProfile,
    db: &AppleSiliconProfileDb,
) -> Option<AppleSiliconProfileId> {
    let cu = device.compute_units;
    let mut best_id = None;
    let mut best_diff = i64::MAX;

    for profile in &db.profiles {
        let diff = (profile.gpu.compute_units as i64 - cu as i64).abs();
        let threshold = (profile.gpu.compute_units as f64 * 0.2) as i64;
        if diff <= threshold && diff < best_diff {
            best_diff = diff;
            best_id = Some(profile.profile_id);
        }
    }

    best_id
}

impl AppleSiliconProfileId {
    fn from_gen_tier(generation: u32, tier: u32) -> Self {
        use AppleSiliconProfileId::*;
        match (generation, tier) {
            (1, 1) => M1,
            (1, 2) => M1Pro,
            (1, 3) => M1Max,
            (1, 4) => M1Ultra,
            (2, 1) => M2,
            (2, 2) => M2Pro,
            (2, 3) => M2Max,
            (2, 4) => M2Ultra,
            (3, 1) => M3,
            (3, 2) => M3Pro,
            (3, 3) => M3Max,
            (3, 4) => M3Ultra,
            (4, 1) => M4,
            (4, 2) => M4Pro,
            (4, 3) => M4Max,
            (4, 4) => M4Ultra,
            (5, 1) => M5,
            (5, 2) => M5Pro,
            (5, 3) => M5Max,
            (5, 4) => M5Ultra,
            _ => UnknownAppleSilicon,
        }
    }
}
