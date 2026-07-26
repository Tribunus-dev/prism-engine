//! Hardware-targeted kernel tuning — tile shape selection and AMD GPU
//! profile matching.
//!
//! This module owns the canonical authority for two related decisions
//! in the compile path:
//!
//! 1. **Tile shape selection** — for each eligible dispatch with a
//!    `FusionGroup` and a `GpuArch`, sample a small sweep of candidate
//!    tile shapes and pick the one with the highest score.
//! 2. **AMD GPU profile matching** — for each ROCm-targeted kernel
//!    entity, find the closest match in the inline AMD profile table
//!    using a 20% compute-unit proximity threshold, with a preference
//!    for datacenter-class Instinct GPUs when the match is close.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The kernel lowerer (owned by `prism-ecs-kernel`).
//! - The dispatch entity lifecycle (owned by fusion scheduling).
//! - Backend dispatch / hardware discovery (owned by `prism-ecs-kernel`).
//!
//! All output is a pure function of the inputs.

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AmdGpuProfileId {
    InstinctMi300X,
    InstinctMi300A,
    InstinctMi350,
    RadeonRx7900Xtx,
    RadeonRx7900Xt,
    RadeonRx7800Xt,
    RyzenAi9Hx370,
    UnknownAmd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuProfile {
    pub profile_id: AmdGpuProfileId,
    pub name: &'static str,
    pub compute_units: u32,
    pub memory_gb_x100: u32,
    pub is_datacenter: bool,
    pub wave_size: u32,
}

impl GpuProfile {
    pub fn memory_gb(&self) -> f32 {
        self.memory_gb_x100 as f32 / 100.0
    }
}

pub const AMD_PROFILES: &[GpuProfile] = &[
    GpuProfile {
        profile_id: AmdGpuProfileId::InstinctMi300X,
        name: "MI300X",
        compute_units: 304,
        memory_gb_x100: 19_200,
        is_datacenter: true,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::InstinctMi300A,
        name: "MI300A",
        compute_units: 228,
        memory_gb_x100: 12_800,
        is_datacenter: true,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::InstinctMi350,
        name: "MI350",
        compute_units: 344,
        memory_gb_x100: 28_800,
        is_datacenter: true,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RadeonRx7900Xtx,
        name: "RX 7900 XTX",
        compute_units: 96,
        memory_gb_x100: 2_400,
        is_datacenter: false,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RadeonRx7900Xt,
        name: "RX 7900 XT",
        compute_units: 84,
        memory_gb_x100: 2_000,
        is_datacenter: false,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RadeonRx7800Xt,
        name: "RX 7800 XT",
        compute_units: 60,
        memory_gb_x100: 1_600,
        is_datacenter: false,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RyzenAi9Hx370,
        name: "Ryzen AI 9 HX 370",
        compute_units: 16,
        memory_gb_x100: 3_200,
        is_datacenter: false,
        wave_size: 32,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::UnknownAmd,
        name: "Unknown AMD GPU",
        compute_units: 0,
        memory_gb_x100: 0,
        is_datacenter: false,
        wave_size: 64,
    },
];

pub type TileShape = [u32; 3];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningSpec {
    pub tile_shape: TileShape,
    pub vector_width: u32,
    pub unroll_factor: u32,
    pub lds_usage_bytes: u64,
    pub wave_limit: Option<u32>,
}

impl prism_ecs_core::Component for TuningSpec {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AotProfileMatch {
    pub profile_id: String,
    pub match_confidence: f32,
}

impl prism_ecs_core::Component for AotProfileMatch {}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TuningError {
    #[error("profile table is missing the `UnknownAmd` fallback entry")]
    MissingUnknownFallback,
    #[error("no candidate tile shape produced a non-zero score for compute_units={0}")]
    NoViableCandidate(u32),
}

pub fn score_tile_shape(tile: TileShape, compute_units: u32) -> f64 {
    let threads = tile[1] as f64;
    let unroll = tile[2] as f64;
    let batches = tile[0] as f64;

    let per_cu_threads = threads / compute_units.max(1) as f64;
    let occupancy_score = if per_cu_threads >= 8.0 && per_cu_threads <= 64.0 {
        1.0
    } else if per_cu_threads < 8.0 {
        per_cu_threads / 8.0
    } else {
        64.0 / per_cu_threads
    };

    let coalescing = (threads / 128.0).min(4.0) / 4.0;
    let arith = (unroll / 32.0).min(4.0) / 4.0;
    let batch = (batches / 4.0).min(1.0);

    0.4 * occupancy_score + 0.3 * coalescing + 0.2 * arith + 0.1 * batch
}

pub const DEFAULT_TILE_CANDIDATES: &[TileShape] = &[
    [1, 128, 64],
    [1, 256, 64],
    [1, 320, 64],
    [2, 128, 64],
    [1, 128, 128],
];

#[derive(Debug, Clone)]
pub struct TileSelector<'a> {
    pub candidates: &'a [TileShape],
    pub max_lds_bytes: u64,
}

impl<'a> Default for TileSelector<'a> {
    fn default() -> Self {
        Self {
            candidates: DEFAULT_TILE_CANDIDATES,
            max_lds_bytes: 32 * 1024,
        }
    }
}

impl<'a> TileSelector<'a> {
    pub fn new(candidates: &'a [TileShape], max_lds_bytes: u64) -> Self {
        Self {
            candidates,
            max_lds_bytes,
        }
    }

    pub fn select(
        &self,
        compute_units: u32,
        wave_size: u32,
    ) -> Result<TuningSpec, TuningError> {
        let best_tile = self
            .candidates
            .iter()
            .max_by(|a, b| {
                let sa = score_tile_shape(**a, compute_units);
                let sb = score_tile_shape(**b, compute_units);
                sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
            })
            .copied()
            .ok_or(TuningError::NoViableCandidate(compute_units))?;

        Ok(TuningSpec {
            tile_shape: best_tile,
            vector_width: if wave_size == 64 { 4 } else { 2 },
            unroll_factor: 2,
            lds_usage_bytes: self.max_lds_bytes.min(32 * 1024),
            wave_limit: Some(compute_units.saturating_mul(4).max(1)),
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct AmdGpuProfileMatcher {
    pub threshold_num: u32,
    pub threshold_den: u32,
}

impl AmdGpuProfileMatcher {
    pub fn new() -> Self {
        Self {
            threshold_num: 1,
            threshold_den: 5,
        }
    }

    pub fn match_profile(&self, compute_units: u32) -> Result<&'static GpuProfile, TuningError> {
        let unknown = AMD_PROFILES
            .iter()
            .find(|p| matches!(p.profile_id, AmdGpuProfileId::UnknownAmd))
            .ok_or(TuningError::MissingUnknownFallback)?;

        let threshold_den = self.threshold_den.max(1) as u64;
        let threshold_num = self.threshold_num as u64;
        let cu_i = compute_units as i64;

        let mut candidates: Vec<&'static GpuProfile> = AMD_PROFILES
            .iter()
            .filter(|p| {
                if matches!(p.profile_id, AmdGpuProfileId::UnknownAmd) {
                    return false;
                }
                let diff = (p.compute_units as i64 - cu_i).abs();
                let threshold =
                    (p.compute_units as u64 * threshold_num / threshold_den) as i64;
                diff <= threshold
            })
            .collect();

        candidates.sort_by(|a, b| {
            let a_key = (
                !a.is_datacenter,
                (a.compute_units as i64 - cu_i).abs(),
            );
            let b_key = (
                !b.is_datacenter,
                (b.compute_units as i64 - cu_i).abs(),
            );
            a_key.cmp(&b_key)
        });

        Ok(candidates.first().copied().unwrap_or(unknown))
    }

    pub fn build_match_and_spec(
        &self,
        compute_units: u32,
        max_lds_bytes: u64,
    ) -> Result<(AotProfileMatch, TuningSpec), TuningError> {
        let matched = self.match_profile(compute_units)?;
        let is_unknown = matches!(matched.profile_id, AmdGpuProfileId::UnknownAmd);
        let cu = matched.compute_units.max(1);
        let wave_size = matched.wave_size;

        let match_component = AotProfileMatch {
            profile_id: matched.name.to_string(),
            match_confidence: if is_unknown { 0.5 } else { 1.0 },
        };

        let spec = TuningSpec {
            tile_shape: if cu >= 200 {
                [4, 256, 64]
            } else if cu >= 64 {
                [2, 128, 64]
            } else {
                [1, 128, 64]
            },
            vector_width: if wave_size == 64 { 4 } else { 2 },
            unroll_factor: 4,
            lds_usage_bytes: max_lds_bytes.min(64 * 1024),
            wave_limit: Some(cu.saturating_mul(4).max(1)),
        };

        Ok((match_component, spec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_tile_shape_high_occupancy_wins() {
        let s1 = score_tile_shape([1, 1024, 32], 16);
        let s2 = score_tile_shape([1, 256, 32], 16);
        let s3 = score_tile_shape([1, 64, 32], 16);
        assert!(s1 > s2);
        assert!(s2 > s3);
    }

    #[test]
    fn score_tile_shape_handles_zero_compute_units() {
        let s = score_tile_shape([1, 128, 64], 0);
        assert!(s.is_finite());
    }

    #[test]
    fn default_tile_selector_returns_a_spec() {
        let sel = TileSelector::default();
        let spec = sel.select(60, 64).expect("select succeeds");
        assert!(DEFAULT_TILE_CANDIDATES.contains(&spec.tile_shape));
        assert_eq!(spec.vector_width, 4);
        assert_eq!(spec.unroll_factor, 2);
    }

    #[test]
    fn tile_selector_uses_custom_sweep() {
        let custom: &[TileShape] = &[[3, 256, 128]];
        let sel = TileSelector::new(custom, 16 * 1024);
        let spec = sel.select(64, 64).expect("select succeeds");
        assert_eq!(spec.tile_shape, [3, 256, 128]);
        assert_eq!(spec.lds_usage_bytes, 16 * 1024);
    }

    #[test]
    fn amd_profiles_table_is_consistent() {
        assert!(AMD_PROFILES
            .iter()
            .any(|p| matches!(p.profile_id, AmdGpuProfileId::UnknownAmd)));
        for p in AMD_PROFILES {
            if !matches!(p.profile_id, AmdGpuProfileId::UnknownAmd) {
                assert!(p.compute_units > 0);
            }
        }
    }

    #[test]
    fn matcher_prefers_closer_datacenter_when_in_range() {
        // 250 CU is between MI300A (228) and MI300X (304). Both are
        // within the 20% threshold; the closer datacenter wins.
        let m = AmdGpuProfileMatcher::new();
        let p = m.match_profile(250).expect("match succeeds");
        assert_eq!(p.profile_id, AmdGpuProfileId::InstinctMi300A);
    }

    #[test]
    fn matcher_falls_back_to_unknown_when_no_match() {
        let m = AmdGpuProfileMatcher::new();
        let p = m.match_profile(4).expect("match returns unknown");
        assert_eq!(p.profile_id, AmdGpuProfileId::UnknownAmd);
    }

    #[test]
    fn matcher_close_consumer_card_matches_radeon_7800() {
        let m = AmdGpuProfileMatcher::new();
        let p = m.match_profile(60).expect("match succeeds");
        assert_eq!(p.profile_id, AmdGpuProfileId::RadeonRx7800Xt);
    }

    #[test]
    fn build_match_and_spec_datacenter_uses_large_tile() {
        let m = AmdGpuProfileMatcher::new();
        let (mat, spec) = m
            .build_match_and_spec(304, 64 * 1024)
            .expect("build succeeds");
        assert_eq!(mat.profile_id, "MI300X");
        assert!((mat.match_confidence - 1.0).abs() < 1e-6);
        assert_eq!(spec.tile_shape, [4, 256, 64]);
        assert_eq!(spec.unroll_factor, 4);
        assert_eq!(spec.lds_usage_bytes, 64 * 1024);
    }

    #[test]
    fn build_match_and_spec_unknown_uses_safe_defaults() {
        let m = AmdGpuProfileMatcher::new();
        let (mat, spec) = m.build_match_and_spec(2, 32 * 1024).expect("build succeeds");
        assert_eq!(mat.profile_id, "Unknown AMD GPU");
        assert!((mat.match_confidence - 0.5).abs() < 1e-6);
        assert_eq!(spec.tile_shape, [1, 128, 64]);
    }

    #[test]
    fn tuning_spec_serializes_round_trip() {
        let spec = TuningSpec {
            tile_shape: [2, 128, 64],
            vector_width: 4,
            unroll_factor: 4,
            lds_usage_bytes: 16_384,
            wave_limit: Some(64),
        };
        let s = serde_json::to_string(&spec).expect("serialize");
        let back: TuningSpec = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(spec, back);
    }
}
