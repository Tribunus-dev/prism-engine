//! Apple Silicon profile database — collection of known hardware profiles
//! with static specifications and optional measured receipts.
//!
//! Separates static/declared facts from empirically measured data.
//! Profiles are versioned and tagged with evidence status so callers
//! can distinguish "known from datasheet" vs "validated on real hardware."

use serde::{Deserialize, Serialize};

use super::profile_id::{AmdGpuProfileId, AppleSiliconProfileId, ProfileEvidenceStatus};

// ── Static capabilities ──────────────────────────────────────────────────

/// Statically declared Metal capabilities for a device profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticMetalCaps {
    pub supports_family: Vec<MetalGpuFamily>,
    pub max_threads_per_threadgroup: u32,
    pub max_threadgroup_memory_bytes: u32,
    pub recommended_max_working_set_bytes: Option<u64>,
    pub supports_simdgroup: bool,
    pub supports_function_constants: bool,
    pub supports_binary_archives: bool,
    pub supports_dynamic_libraries: bool,
    pub supports_webgpu_equivalent: bool,
}

/// Metal GPU family identifiers (maps to MTLGPUFamily).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalGpuFamily {
    Apple1,
    Apple2,
    Apple3,
    Apple4,
    Apple5,
    Apple6,
    Apple7,
    Apple8,
    Apple9,
}

// ── Memory profile ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryProfile {
    pub unified_memory_gb: f64,
    pub memory_bus_width_bits: u32,
    pub memory_bandwidth_gbs: f64,
    pub l1_cache_per_cu_kb: u32,
    pub l2_cache_mb: f32,
}

// ── GPU profile ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfile {
    pub compute_units: u32,
    pub max_threads_per_threadgroup: u32,
    pub simd_width: u32,
    pub max_threadgroup_memory_bytes: u32,
}

// ── ANE profile ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneProfile {
    pub ne_cores: u32,
    pub performance_tops_bf16: f64,
    pub available: bool,
}

// ── Measured kernel profile ─────────────────────────────────────────────

/// One microbenchmark receipt from a measured kernel run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelMicrobenchReceipt {
    pub kernel_name: String,
    pub tile_width: u32,
    pub threadgroup_size: u32,
    pub tokens_per_second: f64,
    pub memory_bandwidth_utilization: f64,
    pub command_buffer_ms: f64,
}

/// Aggregate measured kernel profile for a device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredKernelProfile {
    pub profile_receipt_id: String,
    pub measured_on_device: String,
    pub os_version: String,
    pub metal_version: String,
    pub measurements: Vec<KernelMicrobenchReceipt>,
}

// ── Source receipt ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSourceReceipt {
    pub source_id: String,
    pub source_type: String,
    pub timestamp: String,
    pub notes: String,
}

// ── Full profile ────────────────────────────────────────────────────────

/// Complete hardware profile for one Apple Silicon variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleSiliconProfile {
    pub profile_id: AppleSiliconProfileId,
    pub soc_family: u32,
    pub marketing_name: String,
    pub static_caps: StaticMetalCaps,
    pub memory: MemoryProfile,
    pub gpu: GpuProfile,
    pub ane: Option<AneProfile>,
    pub measured: Option<MeasuredKernelProfile>,
    pub evidence_status: ProfileEvidenceStatus,
}

// ── Profile database ────────────────────────────────────────────────────

/// Versioned collection of known Apple Silicon hardware profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleSiliconProfileDb {
    pub db_version: u32,
    pub generated_at: String,
    pub profiles: Vec<AppleSiliconProfile>,
    pub source_receipts: Vec<ProfileSourceReceipt>,
}

impl AppleSiliconProfileDb {
    /// Look up a profile by its ID.
    pub fn by_id(&self, id: AppleSiliconProfileId) -> Option<&AppleSiliconProfile> {
        self.profiles.iter().find(|p| p.profile_id == id)
    }

    /// Look up a profile matching the given generation and GPU tier.
    pub fn by_generation_and_tier(
        &self,
        generation: u32,
        tier: u32,
    ) -> Option<&AppleSiliconProfile> {
        self.profiles.iter().find(|p| {
            p.profile_id.soc_generation() == generation && p.profile_id.gpu_tier() == tier
        })
    }

    /// All profiles for a given SOC generation.
    pub fn by_generation(&self, generation: u32) -> Vec<&AppleSiliconProfile> {
        self.profiles
            .iter()
            .filter(|p| p.profile_id.soc_generation() == generation)
            .collect()
    }

    /// Return the fallback generic profile, if present.
    pub fn generic_fallback(&self) -> Option<&AppleSiliconProfile> {
        self.by_id(AppleSiliconProfileId::UnknownAppleSilicon)
    }

    /// Number of profiles with measured data.
    pub fn measured_count(&self) -> usize {
        self.profiles
            .iter()
            .filter(|p| p.evidence_status.is_measured())
            .count()
    }

    /// Number of profiles usable for AOT kernel compilation.
    pub fn aot_actionable_count(&self) -> usize {
        self.profiles
            .iter()
            .filter(|p| p.evidence_status.is_actionable_for_aot())
            .count()
    }
}

// ── Default profiles ────────────────────────────────────────────────────

impl AppleSiliconProfileDb {
    /// Build a default profile database with static specs for all known
    /// Apple Silicon variants.  Marked StaticOnly — measurements come
    /// from lab runs or local profiling.
    pub fn default_static() -> Self {
        Self {
            db_version: 1,
            generated_at: String::new(),
            profiles: vec![
                Self::m1_profile(),
                Self::m1_pro_profile(),
                Self::m1_max_profile(),
                Self::m1_ultra_profile(),
                Self::m2_profile(),
                Self::m2_pro_profile(),
                Self::m2_max_profile(),
                Self::m2_ultra_profile(),
                Self::m3_profile(),
                Self::m3_pro_profile(),
                Self::m3_max_profile(),
                Self::m3_ultra_profile(),
                Self::m4_profile(),
                Self::m4_pro_profile(),
                Self::m4_max_profile(),
                Self::m4_ultra_profile(),
                Self::m5_profile(),
                Self::m5_pro_profile(),
                Self::m5_max_profile(),
                Self::m5_ultra_profile(),
                Self::unknown_fallback(),
            ],
            source_receipts: vec![ProfileSourceReceipt {
                source_id: "builtin-default".into(),
                source_type: "static-spec".into(),
                timestamp: String::new(),
                notes: "Default static profile database — replace with lab-measured data for production AOT use.".into(),
            }],
        }
    }

    fn base_m1_caps() -> StaticMetalCaps {
        StaticMetalCaps {
            supports_family: vec![MetalGpuFamily::Apple7],
            max_threads_per_threadgroup: 1024,
            max_threadgroup_memory_bytes: 32 * 1024,
            recommended_max_working_set_bytes: None,
            supports_simdgroup: true,
            supports_function_constants: true,
            supports_binary_archives: true,
            supports_dynamic_libraries: false,
            supports_webgpu_equivalent: false,
        }
    }

    fn base_m2_caps() -> StaticMetalCaps {
        StaticMetalCaps {
            supports_family: vec![MetalGpuFamily::Apple8],
            ..Self::base_m1_caps()
        }
    }

    fn base_m3_caps() -> StaticMetalCaps {
        StaticMetalCaps {
            supports_family: vec![MetalGpuFamily::Apple9],
            max_threadgroup_memory_bytes: 64 * 1024,
            ..Self::base_m2_caps()
        }
    }

    fn base_m4_caps() -> StaticMetalCaps {
        StaticMetalCaps {
            supports_simdgroup: true,
            supports_function_constants: true,
            supports_binary_archives: true,
            supports_dynamic_libraries: true,
            supports_webgpu_equivalent: true,
            ..Self::base_m3_caps()
        }
    }

    fn base_m5_caps() -> StaticMetalCaps {
        Self::base_m4_caps()
    }

    fn profile(
        id: AppleSiliconProfileId,
        caps: StaticMetalCaps,
        memory: MemoryProfile,
        gpu: GpuProfile,
        ane: Option<AneProfile>,
    ) -> AppleSiliconProfile {
        AppleSiliconProfile {
            profile_id: id,
            soc_family: id.soc_generation(),
            marketing_name: id.marketing_name().to_string(),
            static_caps: caps,
            memory,
            gpu,
            ane,
            measured: None,
            evidence_status: ProfileEvidenceStatus::StaticOnly,
        }
    }

    fn m1_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M1,
            Self::base_m1_caps(),
            MemoryProfile {
                unified_memory_gb: 8.0,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbs: 66.7,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 8.0,
            },
            GpuProfile {
                compute_units: 7,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 11.0,
                available: true,
            }),
        )
    }

    fn m1_pro_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M1Pro,
            Self::base_m1_caps(),
            MemoryProfile {
                unified_memory_gb: 16.0,
                memory_bus_width_bits: 256,
                memory_bandwidth_gbs: 200.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 12.0,
            },
            GpuProfile {
                compute_units: 14,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 11.0,
                available: true,
            }),
        )
    }

    fn m1_max_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M1Max,
            Self::base_m1_caps(),
            MemoryProfile {
                unified_memory_gb: 32.0,
                memory_bus_width_bits: 512,
                memory_bandwidth_gbs: 400.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 16.0,
            },
            GpuProfile {
                compute_units: 24,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 11.0,
                available: true,
            }),
        )
    }

    fn m1_ultra_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M1Ultra,
            Self::base_m1_caps(),
            MemoryProfile {
                unified_memory_gb: 64.0,
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbs: 800.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 32.0,
            },
            GpuProfile {
                compute_units: 48,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 32,
                performance_tops_bf16: 22.0,
                available: true,
            }),
        )
    }

    fn m2_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M2,
            Self::base_m2_caps(),
            MemoryProfile {
                unified_memory_gb: 8.0,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbs: 100.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 8.0,
            },
            GpuProfile {
                compute_units: 8,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 15.8,
                available: true,
            }),
        )
    }

    fn m2_pro_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M2Pro,
            Self::base_m2_caps(),
            MemoryProfile {
                unified_memory_gb: 16.0,
                memory_bus_width_bits: 256,
                memory_bandwidth_gbs: 200.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 12.0,
            },
            GpuProfile {
                compute_units: 16,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 15.8,
                available: true,
            }),
        )
    }

    fn m2_max_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M2Max,
            Self::base_m2_caps(),
            MemoryProfile {
                unified_memory_gb: 32.0,
                memory_bus_width_bits: 512,
                memory_bandwidth_gbs: 400.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 16.0,
            },
            GpuProfile {
                compute_units: 30,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 15.8,
                available: true,
            }),
        )
    }

    fn m2_ultra_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M2Ultra,
            Self::base_m2_caps(),
            MemoryProfile {
                unified_memory_gb: 64.0,
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbs: 800.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 32.0,
            },
            GpuProfile {
                compute_units: 60,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            Some(AneProfile {
                ne_cores: 32,
                performance_tops_bf16: 31.6,
                available: true,
            }),
        )
    }

    fn m3_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M3,
            Self::base_m3_caps(),
            MemoryProfile {
                unified_memory_gb: 8.0,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbs: 100.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 8.0,
            },
            GpuProfile {
                compute_units: 8,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 18.0,
                available: true,
            }),
        )
    }

    fn m3_pro_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M3Pro,
            Self::base_m3_caps(),
            MemoryProfile {
                unified_memory_gb: 18.0,
                memory_bus_width_bits: 192,
                memory_bandwidth_gbs: 150.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 12.0,
            },
            GpuProfile {
                compute_units: 14,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 18.0,
                available: true,
            }),
        )
    }

    fn m3_max_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M3Max,
            Self::base_m3_caps(),
            MemoryProfile {
                unified_memory_gb: 48.0,
                memory_bus_width_bits: 512,
                memory_bandwidth_gbs: 400.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 16.0,
            },
            GpuProfile {
                compute_units: 30,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 18.0,
                available: true,
            }),
        )
    }

    fn m3_ultra_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M3Ultra,
            Self::base_m3_caps(),
            MemoryProfile {
                unified_memory_gb: 96.0,
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbs: 800.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 32.0,
            },
            GpuProfile {
                compute_units: 60,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 32,
                performance_tops_bf16: 36.0,
                available: true,
            }),
        )
    }

    fn m4_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M4,
            Self::base_m4_caps(),
            MemoryProfile {
                unified_memory_gb: 8.0,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbs: 120.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 8.0,
            },
            GpuProfile {
                compute_units: 10,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 22.0,
                available: true,
            }),
        )
    }

    fn m4_pro_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M4Pro,
            Self::base_m4_caps(),
            MemoryProfile {
                unified_memory_gb: 24.0,
                memory_bus_width_bits: 256,
                memory_bandwidth_gbs: 220.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 12.0,
            },
            GpuProfile {
                compute_units: 16,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 22.0,
                available: true,
            }),
        )
    }

    fn m4_max_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M4Max,
            Self::base_m4_caps(),
            MemoryProfile {
                unified_memory_gb: 48.0,
                memory_bus_width_bits: 512,
                memory_bandwidth_gbs: 480.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 16.0,
            },
            GpuProfile {
                compute_units: 40,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 22.0,
                available: true,
            }),
        )
    }

    fn m4_ultra_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M4Ultra,
            Self::base_m4_caps(),
            MemoryProfile {
                unified_memory_gb: 96.0,
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbs: 960.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 32.0,
            },
            GpuProfile {
                compute_units: 80,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 32,
                performance_tops_bf16: 44.0,
                available: true,
            }),
        )
    }

    /// M5 profiles are placeholders marked StaticOnly until lab-measured data is available.
    fn m5_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M5,
            Self::base_m5_caps(),
            MemoryProfile {
                unified_memory_gb: 12.0,
                memory_bus_width_bits: 128,
                memory_bandwidth_gbs: 150.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 10.0,
            },
            GpuProfile {
                compute_units: 12,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 30.0,
                available: true,
            }),
        )
    }

    fn m5_pro_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M5Pro,
            Self::base_m5_caps(),
            MemoryProfile {
                unified_memory_gb: 24.0,
                memory_bus_width_bits: 256,
                memory_bandwidth_gbs: 300.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 16.0,
            },
            GpuProfile {
                compute_units: 20,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 16,
                performance_tops_bf16: 30.0,
                available: true,
            }),
        )
    }

    fn m5_max_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M5Max,
            Self::base_m5_caps(),
            MemoryProfile {
                unified_memory_gb: 64.0,
                memory_bus_width_bits: 512,
                memory_bandwidth_gbs: 600.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 24.0,
            },
            GpuProfile {
                compute_units: 48,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            Some(AneProfile {
                ne_cores: 32,
                performance_tops_bf16: 60.0,
                available: true,
            }),
        )
    }

    fn m5_ultra_profile() -> AppleSiliconProfile {
        Self::profile(
            AppleSiliconProfileId::M5Ultra,
            Self::base_m5_caps(),
            MemoryProfile {
                unified_memory_gb: 128.0,
                memory_bus_width_bits: 1024,
                memory_bandwidth_gbs: 1200.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 48.0,
            },
            GpuProfile {
                compute_units: 96,
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
            None,
        )
    }

    fn unknown_fallback() -> AppleSiliconProfile {
        AppleSiliconProfile {
            profile_id: AppleSiliconProfileId::UnknownAppleSilicon,
            soc_family: 0,
            marketing_name: "Unknown Apple Silicon".into(),
            static_caps: StaticMetalCaps {
                supports_family: vec![],
                max_threads_per_threadgroup: 256,
                max_threadgroup_memory_bytes: 16 * 1024,
                recommended_max_working_set_bytes: None,
                supports_simdgroup: false,
                supports_function_constants: false,
                supports_binary_archives: false,
                supports_dynamic_libraries: false,
                supports_webgpu_equivalent: false,
            },
            memory: MemoryProfile {
                unified_memory_gb: 0.0,
                memory_bus_width_bits: 0,
                memory_bandwidth_gbs: 0.0,
                l1_cache_per_cu_kb: 64,
                l2_cache_mb: 2.0,
            },
            gpu: GpuProfile {
                compute_units: 2,
                max_threads_per_threadgroup: 256,
                simd_width: 16,
                max_threadgroup_memory_bytes: 16 * 1024,
            },
            ane: None,
            measured: None,
            evidence_status: ProfileEvidenceStatus::StaticOnly,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AMD
// ═══════════════════════════════════════════════════════════════════════════

/// Complete hardware profile for one AMD GPU variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmdGpuProfile {
    pub profile_id: AmdGpuProfileId,
    pub arch_name: String,
    pub marketing_name: String,
    pub gpu: GpuProfile,
    pub memory: MemoryProfile,
    pub measured: Option<MeasuredKernelProfile>,
    pub evidence_status: ProfileEvidenceStatus,
}

/// Versioned collection of known AMD GPU hardware profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmdProfileDb {
    pub db_version: u32,
    pub generated_at: String,
    pub profiles: Vec<AmdGpuProfile>,
    pub source_receipts: Vec<ProfileSourceReceipt>,
}

impl AmdProfileDb {
    pub fn by_id(&self, id: AmdGpuProfileId) -> Option<&AmdGpuProfile> {
        self.profiles.iter().find(|p| p.profile_id == id)
    }

    /// Build a default profile database with static specs for all known
    /// AMD GPU variants. Marked StaticOnly — measurements come from
    /// lab runs or local profiling.
    pub fn default_static() -> Self {
        Self {
            db_version: 1,
            generated_at: String::new(),
            profiles: vec![
                Self::mi300x_profile(),
                Self::mi300a_profile(),
                Self::mi350_profile(),
                Self::rx7900xtx_profile(),
                Self::rx7900xt_profile(),
                Self::rx7800xt_profile(),
                Self::ryzenai_hx370_profile(),
                Self::unknown_fallback(),
            ],
            source_receipts: vec![ProfileSourceReceipt {
                source_id: "builtin-amd-default".into(),
                source_type: "static-spec".into(),
                timestamp: String::new(),
                notes: "Default static AMD profile database — replace with lab-measured data for production AOT use.".into(),
            }],
        }
    }

    fn profile(id: AmdGpuProfileId, memory: MemoryProfile, gpu: GpuProfile) -> AmdGpuProfile {
        let arch_name: String = if id.is_datacenter() {
            format!("CDNA {}", id.arch_generation())
        } else if id == AmdGpuProfileId::RyzenAi9Hx370 {
            "RDNA 3.5".into()
        } else {
            format!("RDNA {}", id.arch_generation())
        };
        AmdGpuProfile {
            profile_id: id,
            arch_name,
            marketing_name: id.marketing_name().to_string(),
            memory,
            gpu,
            measured: None,
            evidence_status: ProfileEvidenceStatus::StaticOnly,
        }
    }

    fn mi300x_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::InstinctMi300X,
            MemoryProfile {
                unified_memory_gb: 192.0,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbs: 5300.0,
                l1_cache_per_cu_kb: 64,
                l2_cache_mb: 256.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::InstinctMi300X.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn mi300a_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::InstinctMi300A,
            MemoryProfile {
                unified_memory_gb: 128.0,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbs: 5300.0,
                l1_cache_per_cu_kb: 64,
                l2_cache_mb: 256.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::InstinctMi300A.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn mi350_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::InstinctMi350,
            MemoryProfile {
                unified_memory_gb: 288.0,
                memory_bus_width_bits: 8192,
                memory_bandwidth_gbs: 6500.0,
                l1_cache_per_cu_kb: 64,
                l2_cache_mb: 256.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::InstinctMi350.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn rx7900xtx_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::RadeonRx7900Xtx,
            MemoryProfile {
                unified_memory_gb: 24.0,
                memory_bus_width_bits: 384,
                memory_bandwidth_gbs: 960.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 6.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::RadeonRx7900Xtx.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn rx7900xt_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::RadeonRx7900Xt,
            MemoryProfile {
                unified_memory_gb: 20.0,
                memory_bus_width_bits: 320,
                memory_bandwidth_gbs: 800.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 6.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::RadeonRx7900Xt.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn rx7800xt_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::RadeonRx7800Xt,
            MemoryProfile {
                unified_memory_gb: 16.0,
                memory_bus_width_bits: 256,
                memory_bandwidth_gbs: 624.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 4.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::RadeonRx7800Xt.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 64,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn ryzenai_hx370_profile() -> AmdGpuProfile {
        Self::profile(
            AmdGpuProfileId::RyzenAi9Hx370,
            MemoryProfile {
                unified_memory_gb: 32.0,
                memory_bus_width_bits: 256,
                memory_bandwidth_gbs: 150.0,
                l1_cache_per_cu_kb: 128,
                l2_cache_mb: 4.0,
            },
            GpuProfile {
                compute_units: AmdGpuProfileId::RyzenAi9Hx370.compute_units(),
                max_threads_per_threadgroup: 1024,
                simd_width: 32,
                max_threadgroup_memory_bytes: 64 * 1024,
            },
        )
    }

    fn unknown_fallback() -> AmdGpuProfile {
        AmdGpuProfile {
            profile_id: AmdGpuProfileId::UnknownAmd,
            arch_name: "Unknown AMD".into(),
            marketing_name: "Unknown AMD GPU".into(),
            memory: MemoryProfile {
                unified_memory_gb: 0.0,
                memory_bus_width_bits: 0,
                memory_bandwidth_gbs: 0.0,
                l1_cache_per_cu_kb: 0,
                l2_cache_mb: 0.0,
            },
            gpu: GpuProfile {
                compute_units: 0,
                max_threads_per_threadgroup: 256,
                simd_width: 64,
                max_threadgroup_memory_bytes: 32 * 1024,
            },
            measured: None,
            evidence_status: ProfileEvidenceStatus::StaticOnly,
        }
    }
}
