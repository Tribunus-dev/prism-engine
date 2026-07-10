use crate::ecs::component::backend::{BackendTarget, GPUArch, TuningSpec};
use serde::{Deserialize, Serialize};

/// Stable profile identifier for AMD GPU hardware.
///
/// Coarse enough for kernel variant selection. Groups GPUs by architecture
/// generation and performance tier (compute unit count, memory bandwidth).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AmdGpuProfileId {
    // \u2500\u2500 CDNA 3 (Instinct MI300) \u2500\u2500
    /// AMD Instinct MI300X \u2014 304 CU, 192 GB HBM3, 5.2 TB/s
    InstinctMi300X,
    /// AMD Instinct MI300A \u2014 228 CU, 128 GB HBM3, 5.2 TB/s (APU)
    InstinctMi300A,
    // \u2500\u2500 CDNA 4 (Instinct MI350) \u2500\u2500
    /// AMD Instinct MI350 \u2014 next-gen CDNA 4 (placeholder, TBD specs)
    InstinctMi350,
    // \u2500\u2500 RDNA 3 (consumer) \u2500\u2500
    /// AMD Radeon RX 7900 XTX \u2014 96 CU, 24 GB GDDR6, 960 GB/s
    RadeonRx7900Xtx,
    /// AMD Radeon RX 7900 XT \u2014 84 CU, 20 GB GDDR6, 800 GB/s
    RadeonRx7900Xt,
    /// AMD Radeon RX 7800 XT \u2014 60 CU, 16 GB GDDR6, 624 GB/s
    RadeonRx7800Xt,
    // \u2500\u2500 RDNA 3.5 (integrated / Strix Point) \u2500\u2500
    /// AMD Ryzen AI 9 HX 370 (RDNA 3.5 iGPU) \u2014 16 CU
    RyzenAi9Hx370,
    /// Fallback for unrecognized AMD GPUs.
    UnknownAmd,
}

/// Inline AMD GPU profile data for AOT profile matching.
/// Replaces the old external profile DB to keep the system self-contained.
#[allow(dead_code)]
struct GpuProfile {
    profile_id: AmdGpuProfileId,
    name: &'static str,
    compute_units: u32,
    memory_gb: f32,
    is_datacenter: bool,
    wave_size: u32,
}

const AMD_PROFILES: &[GpuProfile] = &[
    GpuProfile {
        profile_id: AmdGpuProfileId::InstinctMi300X,
        name: "MI300X",
        compute_units: 304,
        memory_gb: 192.0,
        is_datacenter: true,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::InstinctMi300A,
        name: "MI300A",
        compute_units: 228,
        memory_gb: 128.0,
        is_datacenter: true,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::InstinctMi350,
        name: "MI350",
        compute_units: 344,
        memory_gb: 288.0,
        is_datacenter: true,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RadeonRx7900Xtx,
        name: "RX 7900 XTX",
        compute_units: 96,
        memory_gb: 24.0,
        is_datacenter: false,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RadeonRx7900Xt,
        name: "RX 7900 XT",
        compute_units: 84,
        memory_gb: 20.0,
        is_datacenter: false,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RadeonRx7800Xt,
        name: "RX 7800 XT",
        compute_units: 60,
        memory_gb: 16.0,
        is_datacenter: false,
        wave_size: 64,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::RyzenAi9Hx370,
        name: "Ryzen AI 9 HX 370",
        compute_units: 16,
        memory_gb: 32.0,
        is_datacenter: false,
        wave_size: 32,
    },
    GpuProfile {
        profile_id: AmdGpuProfileId::UnknownAmd,
        name: "Unknown AMD GPU",
        compute_units: 0,
        memory_gb: 0.0,
        is_datacenter: false,
        wave_size: 64,
    },
];
use crate::ecs::component::fusion::FusionGroup;
use crate::ecs::component::quality::AOTProfileMatch;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

/// Match an AMD device by compute-unit proximity with a 20% threshold.
/// Datacenter-class GPUs (Instinct) are preferred when matches are close.
/// Returns the best-matching profile, or the unknown fallback.
fn match_amd_device(compute_units: u32) -> &'static GpuProfile {
    let mut candidates: Vec<&GpuProfile> = AMD_PROFILES
        .iter()
        .filter(|p| {
            // Skip the unknown fallback during matching.
            if matches!(p.profile_id, AmdGpuProfileId::UnknownAmd) {
                return false;
            }
            let diff = (p.compute_units as i64 - compute_units as i64).abs();
            let threshold = (p.compute_units as f64 * 0.2) as i64;
            diff <= threshold
        })
        .collect();

    // Sort: datacenter first, then by diff ascending.
    candidates.sort_by(|a, b| {
        (
            !a.is_datacenter,
            (a.compute_units as i64 - compute_units as i64).abs(),
        )
            .cmp(&(
                !b.is_datacenter,
                (b.compute_units as i64 - compute_units as i64).abs(),
            ))
    });

    // Fallback to unknown if no candidate is within threshold.
    // Find the unknown fallback entry robustly.
    let unknown = AMD_PROFILES
        .iter()
        .find(|p| matches!(p.profile_id, AmdGpuProfileId::UnknownAmd))
        .expect("AMD_PROFILES must contain an UnknownAmd entry");
    candidates.first().copied().unwrap_or(unknown)
}

/// Generates variant tile sizes for a subset of dispatches, benchmarks
/// them against the profile DB, and selects the optimal `TuningSpec`.
///
/// For each eligible dispatch entity with a `FusionGroup` this system
/// produces a small sweep of tile shapes and attaches the best-scoring
/// `TuningSpec` to the corresponding `KernelEntity`.
pub struct AutoTuningSystem;
impl CompilerSystem for AutoTuningSystem {
    fn name(&self) -> &str {
        "AutoTuningSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        // Collect dispatch entities that have both a FusionGroup and GPUArch.
        let dispatch_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Dispatch);
        let candidates: Vec<CompEntity> = dispatch_entities
            .into_iter()
            .filter(|e| {
                world.get_component::<FusionGroup>(*e).is_some()
                    && world.get_component::<GPUArch>(*e).is_some()
            })
            .collect();

        if candidates.is_empty() {
            return Ok(());
        }

        // Sample a subset (first 3 or 10%) to keep tuning overhead bounded.
        let sample_size = std::cmp::max(3, candidates.len() / 10).min(20);
        let sample: Vec<CompEntity> = candidates.into_iter().take(sample_size).collect();

        // Collect per-sample info under immutable borrow.
        struct SampleEntry {
            kernel: Option<CompEntity>,
            arch: GPUArch,
        }
        let entries: Vec<SampleEntry> = sample
            .iter()
            .map(|e| {
                let arch = world.get_component::<GPUArch>(*e).cloned().unwrap();
                // Find the kernel spawned for this dispatch (earliest kernel entity).
                let kernel_entities = world.entities_of_kind(EntityKind::Kernel);
                let kernel = kernel_entities.into_iter().next();
                SampleEntry { kernel, arch }
            })
            .collect();

        // For each entry, generate candidate tile shapes and pick the best.
        for entry in &entries {
            let Some(kernel) = entry.kernel else {
                continue;
            };

            let compute_units = entry.arch.compute_units;
            // Generate candidates based on compute unit count.
            let candidates_tiles: Vec<[u32; 3]> = vec![
                [1, 128, 64],
                [1, 256, 64],
                [1, 320, 64],
                [2, 128, 64],
                [1, 128, 128],
            ];

            // Score each candidate (higher is better).
            let scored: Vec<([u32; 3], f64)> = candidates_tiles
                .into_iter()
                .map(|tile| {
                    let score = score_tile_shape(tile, compute_units);
                    (tile, score)
                })
                .collect();

            // Pick the best.
            let best = scored
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(tile, _)| tile)
                .unwrap_or([1, 128, 64]);

            let spec = TuningSpec {
                tile_shape: best,
                vector_width: match entry.arch.wave_size {
                    64 => 4,
                    _ => 2,
                },
                unroll_factor: 2,
                lds_usage_bytes: entry.arch.max_lds_bytes.min(32 * 1024),
                wave_limit: Some(std::cmp::max(1, compute_units * 4)),
            };

            world.add_component(kernel, spec);
        }

        Ok(())
    }
}

/// Scores a tile shape against the hardware profile.
///
/// Factors considered:
/// - Occupancy: wider tiles use more threads/compute-unit
/// - LDS pressure: deeper tiles may exceed per-CU LDS limits
/// - Memory coalescing: wider tiles improve bandwidth utilization
fn score_tile_shape(tile: [u32; 3], compute_units: u32) -> f64 {
    let threads = tile[1] as f64; // tile_n
    let unroll = tile[2] as f64; // tile_k (or unroll)
    let batches = tile[0] as f64; // tile_m

    // Target ~64-512 threads per CU for good occupancy.
    let per_cu_threads = threads / compute_units.max(1) as f64;
    let occupancy_score = if per_cu_threads >= 8.0 && per_cu_threads <= 64.0 {
        1.0
    } else if per_cu_threads < 8.0 {
        per_cu_threads / 8.0
    } else {
        64.0 / per_cu_threads
    };

    // Coalescing: wider rows improve memory throughput.
    let coalescing = (threads / 128.0).min(4.0) / 4.0;

    // Arithmetic intensity: larger inner dim improves compute/byte ratio.
    let arith = (unroll / 32.0).min(4.0) / 4.0;

    // Batch parallelism (for tile_m > 1).
    let batch = (batches / 4.0).min(1.0);

    0.4 * occupancy_score + 0.3 * coalescing + 0.2 * arith + 0.1 * batch
}

/// Matches the target GPU to the AOT profile database and attaches the
/// resulting profile match as an `AOTProfileMatch` component on each
/// `KernelEntity` destined for an AMD (ROCm) backend.
pub struct AOTProfileMatchSystem;
impl CompilerSystem for AOTProfileMatchSystem {
    fn name(&self) -> &str {
        "AOTProfileMatchSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::KernelGeneration
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let kernel_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Kernel);

        // Collect AMD kernels with their GPU arch info.
        struct AmdMatch {
            kernel: CompEntity,
            arch: GPUArch,
        }
        let amd_kernels: Vec<AmdMatch> = kernel_entities
            .iter()
            .filter_map(|e| {
                let target = world.get_component::<BackendTarget>(*e)?;
                if !matches!(target, BackendTarget::ROCm) {
                    return None;
                }
                let arch = world.get_component::<GPUArch>(*e).cloned()?;
                Some(AmdMatch { kernel: *e, arch })
            })
            .collect();

        if amd_kernels.is_empty() {
            return Ok(());
        }

        for entry in &amd_kernels {
            // Match via compute-unit proximity with 20% threshold (from match_amd_device).
            let matched = match_amd_device(entry.arch.compute_units);
            let profile_id_str = matched.name.to_string();
            let match_confidence = if matches!(matched.profile_id, AmdGpuProfileId::UnknownAmd) {
                0.5
            } else {
                1.0
            };
            let cu = matched.compute_units.max(1);
            let wave_size = matched.wave_size;

            let match_component = AOTProfileMatch {
                profile_id: profile_id_str,
                match_confidence,
            };
            world.add_component(entry.kernel, match_component);

            // Also add a TuningSpec with profile-informed parameters.
            let spec = TuningSpec {
                tile_shape: if cu >= 200 {
                    [4, 256, 64] // datacenter: large batches
                } else if cu >= 64 {
                    [2, 128, 64] // high-end consumer
                } else {
                    [1, 128, 64] // entry-level
                },
                vector_width: if wave_size == 64 { 4 } else { 2 },
                unroll_factor: 4,
                lds_usage_bytes: entry.arch.max_lds_bytes.min(64 * 1024),
                wave_limit: Some(std::cmp::max(1, cu * 4)),
            };
            world.add_component(entry.kernel, spec);
        }

        Ok(())
    }
}
