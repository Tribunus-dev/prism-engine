//! Execution profiling subsystem — measures whether a codec policy is worth
//! using on a specific hardware target, not just whether it is admissible.
//!
//! Four evidence classes:
//!   - Execution: prefill/decode tok/s, per-layer timing
//!   - Memory: raw vs packed bytes, resident set, compression ratio
//!   - Quality: logits KL, top-1 agreement, seeded generation drift
//!   - Health: RSS slope, swap delta, memory pressure, thermal, stalls
//!
//! # Profiles
//!
//! A — RawF32 everywhere (baseline)
//! B — FP16 substitution everywhere admissible
//! C — Current production (NF4 decoder, INT8 bridge, RawF32 vision, INT8 speech)
//! D — Aggressive substitution ranked
//! E — Research activation-weighted

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── FusionMode ────────────────────────────────────────────────────────────

/// How aggressively the compiler should fuse dataflow ops.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionMode {
    /// No fusion — each op becomes its own singleton group.
    #[default]
    Disabled,
    /// Fuse only well-characterized patterns with proven speedups.
    Conservative,
    /// Fuse any admissible pattern, even speculative.
    Aggressive,
}

// ── Profile definition ───────────────────────────────────────────────────

/// A named policy plus hardware target plus runtime flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub id: String,
    pub policy_digest: String,
    pub hardware_target: HardwareTarget,
    pub runtime_lanes: RuntimeLaneConfig,
    pub kv_cache_policy: KvCachePolicy,
    pub substitution_mode: SubstitutionMode,
    pub validation_mode: ValidationMode,
    /// Fusion strategy for the compiler pipeline.
    #[serde(default)]
    pub fusion_mode: FusionMode,
}

/// Hardware target identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareTarget {
    AppleM1Aned16Gb,
    AppleM1ProAned16Gb,
    AppleM1MaxAned32Gb,
    AppleM2ProAned16Gb,
    AppleM2MaxAned32Gb,
    AppleM3ProAned18Gb,
    AppleM3MaxAned48Gb,
    AppleM4ProAned24Gb,
    AppleM4MaxAned64Gb,
    UnknownCpu,
}

/// Which runtime lane set to use for this profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeLaneConfig {
    pub use_gpu: bool,
    pub use_ane: bool,
    pub use_cpu: bool,
    pub gpu_priority: bool,
    pub ane_priority: bool,
}

impl Default for RuntimeLaneConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            use_ane: true,
            use_cpu: true,
            gpu_priority: true,
            ane_priority: false,
        }
    }
}

/// KV cache policy for the profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KvCachePolicy {
    /// Full precision FP16 cache.
    Fp16,
    /// Compressed KV (8-bit or 4-bit).
    Compressed { bits: u8 },
    /// Shared context cache.
    Shared { window: usize },
}

/// Substitution mode for the profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SubstitutionMode {
    /// No substitution.
    Disabled,
    /// Ranked substitution with default candidate list.
    Enabled { aggressiveness: String }, // "conservative", "balanced", "aggressive"
    /// Specific substitution override.
    Override { candidates: Vec<String> },
}

/// Validation mode for the profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationMode {
    /// Only weight-space gates, no hardware validation.
    WeightSpaceOnly,
    /// Weight + synthetic operator probe.
    SyntheticOperator,
    /// Full hardware-backed validation.
    FullHardware,
}

// ── Execution receipt ────────────────────────────────────────────────────

/// Top-level execution profile receipt for a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProfileReceipt {
    pub profile_id: String,
    pub model_family: String,
    pub hardware_target: String,
    pub runtime_backend: String,
    pub policy_digest: String,
    pub cimage_digest: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub batch_size: usize,
    pub context_length: usize,
    // Timing — split prefill and decode
    pub cold_start_ms: f64,
    pub first_token_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
    // Throughput
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub steady_state_tok_per_s: f64,
    pub end_to_end_tok_per_s: f64,
    // Memory
    pub peak_rss_bytes: u64,
    pub peak_gpu_bytes: Option<u64>,
    pub mapped_weight_bytes: u64,
    pub resident_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    // Per-layer breakdown
    pub per_layer: Vec<LayerExecutionStats>,
    // Speedup vs baseline
    pub decode_speedup_vs_raw: Option<f64>,
}

/// Per-layer execution statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerExecutionStats {
    pub layer_index: Option<u32>,
    pub tensor_class: String,
    pub tensor_key: String,
    pub codec_family: String,
    pub backend: String,
    pub weight_bytes_read: u64,
    pub dequant_us: f64,
    pub compute_us: f64,
    pub sync_us: f64,
    pub total_us: f64,
    pub effective_gbps: f64,
    pub effective_tflops: Option<f64>,
}

// ── Memory receipt ───────────────────────────────────────────────────────

/// Memory accounting for a policy — separates theoretical from actual.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReceipt {
    pub profile_id: String,
    /// Raw F32 weight payload bytes (theoretical uncompressed).
    pub raw_weight_bytes: u64,
    /// Packed codec codes bytes.
    pub packed_weight_bytes: u64,
    /// Tile metadata (scales, biases).
    pub metadata_bytes: u64,
    /// Rescue sidecar data.
    pub sidecar_bytes: u64,
    /// Alignment padding to tile boundaries.
    pub alignment_padding_bytes: u64,
    /// Runtime scratch/workspace buffers.
    pub runtime_scratch_bytes: u64,
    /// Total resident bytes at steady state.
    pub resident_total_bytes: u64,
    /// Compression ratio vs raw.
    pub compression_ratio_vs_raw: f64,
}

// ── Quality drift receipt ────────────────────────────────────────────────

/// Model drift against a baseline profile, measured on a fixed eval set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityDriftReceipt {
    pub baseline_profile_id: String,
    pub candidate_profile_id: String,
    pub eval_set_digest: String,
    pub hidden_state_nrmse_mean: f64,
    pub hidden_state_nrmse_worst: f64,
    pub logits_kl_mean: f64,
    pub logits_kl_p95: f64,
    pub top1_agreement: f64,
    pub top5_agreement: f64,
    pub sampled_token_agreement_seeded: f64,
    pub first_token_accuracy_delta: Option<f64>,
    pub per_layer_drift: Vec<LayerDriftStats>,
    /// Composite quality retention score (dashboard convenience only).
    pub quality_retention: f64,
}

/// Per-layer drift statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDriftStats {
    pub layer_index: Option<u32>,
    pub tensor_class: String,
    pub hidden_nrmse: f64,
    pub hidden_cosine: f64,
    pub logits_kl: Option<f64>,
}

// ── Runtime health receipt ───────────────────────────────────────────────

/// System pressure and stability evidence for a profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeHealthReceipt {
    pub profile_id: String,
    pub hardware_target: String,
    pub os_version: String,
    pub duration_s: f64,
    // Timelines
    pub memory_pressure_timeline: Vec<PressureSample>,
    pub thermal_timeline: Vec<ThermalSample>,
    pub process_memory_timeline: Vec<ProcessMemorySample>,
    pub throughput_timeline: Vec<ThroughputSample>,
    // Aggregate
    pub peak_rss_bytes: u64,
    pub peak_virtual_bytes: u64,
    pub peak_compressed_memory_bytes: Option<u64>,
    pub swap_delta_bytes: Option<u64>,
    pub cpu_package_active_ratio: Option<f64>,
    pub gpu_busy_ratio: Option<f64>,
    pub ane_busy_ratio: Option<f64>,
    pub stall_count: u64,
    pub longest_stall_ms: f64,
    pub oom_kill: bool,
    pub watchdog_exit: bool,
    pub thermal_throttle_detected: bool,
    pub stability_status: StabilityStatus,
    // RSS slope
    pub rss_slope_mb_per_100_tokens: f64,
}

/// A memory pressure sample at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PressureSample {
    pub time_s: f64,
    pub memory_pressure_level: MemoryPressureLevel,
    pub swap_delta_mb: f64,
}

/// Process memory at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMemorySample {
    pub time_s: f64,
    pub rss_mb: f64,
    pub virtual_mb: f64,
}

/// System thermal state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSample {
    pub time_s: f64,
    pub thermal_state: ThermalState,
}

/// Throughput over a rolling window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputSample {
    pub time_s: f64,
    pub decode_tok_s_window: f64,
    pub p50_token_ms: f64,
    pub p95_token_ms: f64,
    pub stall_count_this_window: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressureLevel {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalState {
    Nominal,
    Fair,
    Serious,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StabilityStatus {
    Unknown,
    Admissible,
    Deployable,
    DemoSafe,
    ResearchOnly,
    Rejected,
}

// ── ReceiptEvidenceKind ─────────────────────────────────────────────────

/// Classifies the evidence level of a profile run receipt.
///
/// Distinguishes template/synthetic placeholders from real hardware
/// measurements so gate-eligible policies can reject non-measured results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReceiptEvidenceKind {
    /// Placeholder template result — no actual execution occurred.
    Template,
    /// Synthetic result derived from analytical models or interpolation.
    Synthetic,
    /// Result obtained from actual hardware execution with real measurements.
    Measured,
}

// ── Profile comparison matrix ────────────────────────────────────────────

/// One row in the profile comparison matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileComparisonRow {
    pub profile_id: String,
    pub policy_digest: String,
    pub weight_bytes: u64,
    pub resident_bytes: u64,
    pub compression_ratio: f64,
    pub first_token_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub speedup_vs_raw: Option<f64>,
    pub logits_kl: Option<f64>,
    pub top1_agreement: Option<f64>,
    pub peak_rss_mb: f64,
    pub swap_delta_mb: f64,
    pub quality_status: String,
    pub stability_status: StabilityStatus,
    pub overall_status: String,
}

// ── Cimage layout types ─────────────────────────────────────────────────

/// How groups of values are arranged within a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupAxis {
    /// Groups are contiguous in storage order (current behavior).
    PackedContiguous,
    /// Groups span output-index space.
    OutputAxis,
    /// Groups span input-index space (critical for patch_dense).
    InputAxis,
    /// Groups do not cross tile boundaries.
    TileLocal,
}

/// Logical shape of a single tile in rows and columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileShape {
    pub rows: u32,
    pub cols: u32,
}

impl TileShape {
    pub const fn tile640() -> Self {
        Self {
            rows: 640,
            cols: 640,
        }
    }
    pub const fn elements(&self) -> u32 {
        self.rows * self.cols
    }
}

/// A family of tiles sharing a shape and default group sizes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileFamily {
    pub name: String,
    pub tile_shape: TileShape,
    pub default_group_sizes: Vec<u32>,
}

impl TileFamily {
    pub fn tile640() -> Self {
        Self {
            name: "Tile640".into(),
            tile_shape: TileShape::tile640(),
            default_group_sizes: vec![32, 64, 128],
        }
    }
}

/// Whether the storage is row-major or column-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageOrder {
    RowMajor,
    ColumnMajor,
}

/// How metadata (scales, offsets) is laid out relative to tile data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetadataLayout {
    AdjacentTile,
    SeparatedManifest,
    Interleaved,
}

/// Memory residency policy for a view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResidencyMode {
    AlwaysMapped,
    LazyMapped,
    MaterializeOnFirstUse,
    EphemeralScratch,
    MutuallyExclusiveViewGroup,
}

/// Concrete tile layout describing how a tensor is physically stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalTileLayout {
    pub format: String,
    pub tile_family: TileFamily,
    pub logical_shape: [u32; 2],
    pub storage_order: StorageOrder,
    pub tile_shape: TileShape,
    pub group_size: u32,
    pub group_axis: GroupAxis,
    pub metadata_layout: MetadataLayout,
    pub padding_policy: String,
    pub alignment_bytes: u32,
    pub interleave: String,
}

impl Default for PhysicalTileLayout {
    fn default() -> Self {
        Self {
            format: "NF4".into(),
            tile_family: TileFamily::tile640(),
            logical_shape: [0, 0],
            storage_order: StorageOrder::RowMajor,
            tile_shape: TileShape::tile640(),
            group_size: 32,
            group_axis: GroupAxis::PackedContiguous,
            metadata_layout: MetadataLayout::AdjacentTile,
            padding_policy: "ZeroPadToTile".into(),
            alignment_bytes: 256,
            interleave: "None".into(),
        }
    }
}

/// How an execution lane sees a tensor — offset, length, codec overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionView {
    pub lane: String,
    pub data_offset: u64,
    pub data_length: u64,
    pub metadata_offset: Option<u64>,
    pub metadata_length: Option<u64>,
    pub codec_overrides: HashMap<String, String>,
    pub repacking_required: bool,
    pub residency: ResidencyMode,
    /// Optional discovered program from evolutionary search.
    pub discovered_program: Option<String>,
    /// Optional evolution provenance — JSON-serialized EvolutionProvenance.
    pub evolution_provenance: Option<String>,
    /// Last known cost profile — JSON-serialized CostMetrics.
    pub cost_profile: Option<String>,
}

/// Broad hardware class with bandwidth and budget estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareTargetClass {
    A18Neo,
    MBase,
    MPro,
    MMax,
    MUltra,
}

impl HardwareTargetClass {
    pub fn memory_gb(&self) -> u32 {
        match self {
            HardwareTargetClass::A18Neo => 8,
            HardwareTargetClass::MBase => 16,
            HardwareTargetClass::MPro => 24,
            HardwareTargetClass::MMax => 48,
            HardwareTargetClass::MUltra => 96,
        }
    }
    pub fn bandwidth_gbps(&self) -> u32 {
        match self {
            HardwareTargetClass::A18Neo => 60,
            HardwareTargetClass::MBase => 120,
            HardwareTargetClass::MPro => 273,
            HardwareTargetClass::MMax => 546,
            HardwareTargetClass::MUltra => 800,
        }
    }
    pub fn scratch_budget_mb(&self) -> u32 {
        match self {
            HardwareTargetClass::A18Neo => 256,
            HardwareTargetClass::MBase => 512,
            HardwareTargetClass::MPro => 1024,
            HardwareTargetClass::MMax => 2048,
            HardwareTargetClass::MUltra => 4096,
        }
    }
    pub fn max_views_per_tensor(&self) -> u32 {
        match self {
            HardwareTargetClass::A18Neo => 1,
            HardwareTargetClass::MBase => 1,
            HardwareTargetClass::MPro => 2,
            HardwareTargetClass::MMax => 2,
            HardwareTargetClass::MUltra => 3,
        }
    }
}

// ── Profile builders ─────────────────────────────────────────────────────

// ── SoC performance projections ────────────────────────────────────────────

/// Full hardware specification for an Apple Silicon SoC.
#[derive(Debug, Clone, Serialize)]
pub struct SoCPerformance {
    pub label: &'static str,
    pub memory_options: &'static [u32],
    pub bandwidth_gbps: u32,
    pub gpu_core_count: u32,
    pub total_alus: u32,
    pub gpu_clock_mhz: u32,
    pub ne_tops: u32,
    pub process_node: &'static str,
    pub cpu_core_config: &'static str,
    pub estimated_decode_tok_s: PerPolicyF32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PerPolicyF32 {
    pub rawf32: f32,
    pub fp16: f32,
    pub int8: f32,
    pub nf4: f32,
}

impl PerPolicyF32 {
    pub const fn new(r: f32, f: f32, i: f32, n: f32) -> Self {
        Self {
            rawf32: r,
            fp16: f,
            int8: i,
            nf4: n,
        }
    }
}

pub fn all_soc_performances() -> Vec<SoCPerformance> {
    vec![
        SoCPerformance {
            label: "Apple M1",
            memory_options: &[8, 16],
            bandwidth_gbps: 68,
            gpu_core_count: 8,
            total_alus: 1024,
            gpu_clock_mhz: 1278,
            ne_tops: 11,
            process_node: "5nm",
            cpu_core_config: "4P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(1.8, 3.6, 7.2, 8.6),
        },
        SoCPerformance {
            label: "Apple M1 Pro",
            memory_options: &[16, 32],
            bandwidth_gbps: 200,
            gpu_core_count: 16,
            total_alus: 2048,
            gpu_clock_mhz: 1300,
            ne_tops: 11,
            process_node: "5nm",
            cpu_core_config: "8P+2E",
            estimated_decode_tok_s: PerPolicyF32::new(5.3, 10.6, 21.2, 25.4),
        },
        SoCPerformance {
            label: "Apple M1 Max",
            memory_options: &[32, 64],
            bandwidth_gbps: 400,
            gpu_core_count: 32,
            total_alus: 4096,
            gpu_clock_mhz: 1300,
            ne_tops: 11,
            process_node: "5nm",
            cpu_core_config: "10P+2E",
            estimated_decode_tok_s: PerPolicyF32::new(10.6, 21.2, 42.4, 50.9),
        },
        SoCPerformance {
            label: "Apple M1 Ultra",
            memory_options: &[64, 128],
            bandwidth_gbps: 800,
            gpu_core_count: 64,
            total_alus: 8192,
            gpu_clock_mhz: 1300,
            ne_tops: 22,
            process_node: "5nm×2",
            cpu_core_config: "20P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(21.2, 42.4, 84.8, 101.8),
        },
        SoCPerformance {
            label: "Apple M2",
            memory_options: &[8, 16, 24],
            bandwidth_gbps: 100,
            gpu_core_count: 10,
            total_alus: 1280,
            gpu_clock_mhz: 1398,
            ne_tops: 16,
            process_node: "N5P",
            cpu_core_config: "4P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(2.6, 5.2, 10.4, 12.5),
        },
        SoCPerformance {
            label: "Apple M2 Pro",
            memory_options: &[16, 32],
            bandwidth_gbps: 200,
            gpu_core_count: 19,
            total_alus: 2432,
            gpu_clock_mhz: 1398,
            ne_tops: 16,
            process_node: "N5P",
            cpu_core_config: "8P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(5.3, 10.6, 21.2, 25.4),
        },
        SoCPerformance {
            label: "Apple M2 Max",
            memory_options: &[32, 64, 96],
            bandwidth_gbps: 400,
            gpu_core_count: 38,
            total_alus: 4864,
            gpu_clock_mhz: 1398,
            ne_tops: 16,
            process_node: "N5P",
            cpu_core_config: "8P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(10.6, 21.2, 42.4, 50.9),
        },
        SoCPerformance {
            label: "Apple M2 Ultra",
            memory_options: &[64, 128, 192],
            bandwidth_gbps: 800,
            gpu_core_count: 76,
            total_alus: 9728,
            gpu_clock_mhz: 1398,
            ne_tops: 32,
            process_node: "N5P×2",
            cpu_core_config: "16P+8E",
            estimated_decode_tok_s: PerPolicyF32::new(21.2, 42.4, 84.8, 101.8),
        },
        SoCPerformance {
            label: "Apple M3",
            memory_options: &[8, 16, 24],
            bandwidth_gbps: 100,
            gpu_core_count: 10,
            total_alus: 1280,
            gpu_clock_mhz: 1380,
            ne_tops: 18,
            process_node: "N3B",
            cpu_core_config: "4P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(2.6, 5.2, 10.4, 12.5),
        },
        SoCPerformance {
            label: "Apple M3 Pro",
            memory_options: &[18, 36],
            bandwidth_gbps: 150,
            gpu_core_count: 18,
            total_alus: 2304,
            gpu_clock_mhz: 1380,
            ne_tops: 18,
            process_node: "N3B",
            cpu_core_config: "6P+6E",
            estimated_decode_tok_s: PerPolicyF32::new(4.0, 8.0, 15.9, 19.1),
        },
        SoCPerformance {
            label: "Apple M3 Max (14c GPU, 300G/s)",
            memory_options: &[36, 48, 64, 128],
            bandwidth_gbps: 300,
            gpu_core_count: 30,
            total_alus: 3840,
            gpu_clock_mhz: 1380,
            ne_tops: 18,
            process_node: "N3B",
            cpu_core_config: "12P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(8.0, 15.9, 31.8, 38.2),
        },
        SoCPerformance {
            label: "Apple M3 Max (16c GPU, 400G/s)",
            memory_options: &[36, 48, 64, 128],
            bandwidth_gbps: 400,
            gpu_core_count: 40,
            total_alus: 5120,
            gpu_clock_mhz: 1380,
            ne_tops: 18,
            process_node: "N3B",
            cpu_core_config: "12P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(10.6, 21.2, 42.4, 50.9),
        },
        SoCPerformance {
            label: "Apple M3 Ultra",
            memory_options: &[64, 128, 256, 512],
            bandwidth_gbps: 819,
            gpu_core_count: 80,
            total_alus: 10240,
            gpu_clock_mhz: 1380,
            ne_tops: 36,
            process_node: "N3B×2",
            cpu_core_config: "24P+8E",
            estimated_decode_tok_s: PerPolicyF32::new(21.7, 43.4, 86.8, 104.2),
        },
        SoCPerformance {
            label: "Apple M4",
            memory_options: &[16, 24],
            bandwidth_gbps: 120,
            gpu_core_count: 10,
            total_alus: 1280,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "4P+6E",
            estimated_decode_tok_s: PerPolicyF32::new(3.2, 6.4, 12.8, 15.4),
        },
        SoCPerformance {
            label: "Apple M4 Pro",
            memory_options: &[24, 48, 64],
            bandwidth_gbps: 273,
            gpu_core_count: 20,
            total_alus: 2560,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "10P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(7.2, 14.4, 28.8, 34.6),
        },
        SoCPerformance {
            label: "Apple M4 Max (32c GPU, 410G/s)",
            memory_options: &[36, 48, 64, 128],
            bandwidth_gbps: 410,
            gpu_core_count: 32,
            total_alus: 4096,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "12P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(10.9, 21.8, 43.6, 52.3),
        },
        SoCPerformance {
            label: "Apple M4 Max (40c GPU, 546G/s)",
            memory_options: &[36, 48, 64, 128],
            bandwidth_gbps: 546,
            gpu_core_count: 40,
            total_alus: 5120,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "12P+4E",
            estimated_decode_tok_s: PerPolicyF32::new(14.5, 29.0, 58.0, 69.6),
        },
        SoCPerformance {
            label: "Apple M5",
            memory_options: &[16, 24],
            bandwidth_gbps: 153,
            gpu_core_count: 10,
            total_alus: 1280,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "4P+6E",
            estimated_decode_tok_s: PerPolicyF32::new(4.1, 8.1, 16.3, 19.5),
        },
        SoCPerformance {
            label: "Apple M5 Pro",
            memory_options: &[24, 48, 64],
            bandwidth_gbps: 307,
            gpu_core_count: 20,
            total_alus: 2560,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "TBD",
            estimated_decode_tok_s: PerPolicyF32::new(8.2, 16.4, 32.8, 39.4),
        },
        SoCPerformance {
            label: "Apple M5 Max",
            memory_options: &[48, 64, 128],
            bandwidth_gbps: 614,
            gpu_core_count: 40,
            total_alus: 5120,
            gpu_clock_mhz: 1470,
            ne_tops: 38,
            process_node: "N3E",
            cpu_core_config: "TBD",
            estimated_decode_tok_s: PerPolicyF32::new(16.3, 32.6, 65.2, 78.2),
        },
        SoCPerformance {
            label: "Apple M5 Ultra",
            memory_options: &[128, 256],
            bandwidth_gbps: 1200,
            gpu_core_count: 80,
            total_alus: 10240,
            gpu_clock_mhz: 1470,
            ne_tops: 76,
            process_node: "N3E?",
            cpu_core_config: "TBD",
            estimated_decode_tok_s: PerPolicyF32::new(31.8, 63.6, 127.2, 152.6),
        },
    ]
}

pub fn format_soc_performance_table() -> String {
    let mut s = String::new();
    s.push_str("SoC Performance Projections (Gemma4 12B decode tok/s):\n");
    s.push_str(&"-".repeat(110));
    s.push('\n');
    s.push_str(&format!(
        "{:<34} {:>12} {:>5} {:>5} {:>5} {:>5} {:>5} {:>6} {:>8}",
        "SoC", "RAM opts.", "B/W", "Cores", "ALUs", "MHz", "NE", "Node", "tok/s"
    ));
    s.push('\n');
    s.push_str(&"-".repeat(110));
    s.push('\n');
    for soc in all_soc_performances() {
        let p = &soc.estimated_decode_tok_s;
        let tok_range = format!("{:.0}-{:.0}", p.rawf32, p.nf4);
        let ram = soc
            .memory_options
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        s.push_str(&format!(
            "{:<34} {:>5}GB  {:>4}G {:>4} {:>4}K {:>4} {:>4}T {:>6} {:>8}\n",
            soc.label,
            ram,
            soc.bandwidth_gbps,
            soc.gpu_core_count,
            soc.total_alus / 1000,
            soc.gpu_clock_mhz,
            soc.ne_tops,
            soc.process_node,
            tok_range,
        ));
    }
    s
}

impl ExecutionProfile {
    /// A — RawF32 everywhere (baseline).
    pub fn raw_f32_baseline() -> Self {
        Self {
            id: "A_raw_f32".into(),
            policy_digest: "raw_f32".into(),
            hardware_target: HardwareTarget::UnknownCpu,
            runtime_lanes: RuntimeLaneConfig::default(),
            kv_cache_policy: KvCachePolicy::Fp16,
            substitution_mode: SubstitutionMode::Disabled,
            validation_mode: ValidationMode::WeightSpaceOnly,
            fusion_mode: FusionMode::Disabled,
        }
    }

    /// B — FP16 substitution everywhere admissible.
    pub fn fp16_baseline() -> Self {
        Self {
            id: "B_fp16".into(),
            policy_digest: "fp16".into(),
            hardware_target: HardwareTarget::UnknownCpu,
            runtime_lanes: RuntimeLaneConfig::default(),
            kv_cache_policy: KvCachePolicy::Fp16,
            substitution_mode: SubstitutionMode::Enabled {
                aggressiveness: "conservative".into(),
            },
            validation_mode: ValidationMode::WeightSpaceOnly,
            fusion_mode: FusionMode::Disabled,
        }
    }

    /// C — Current production.
    pub fn production_policy_v2() -> Self {
        Self {
            id: "C_production_v2".into(),
            policy_digest: "production_v2".into(),
            hardware_target: HardwareTarget::UnknownCpu,
            runtime_lanes: RuntimeLaneConfig::default(),
            kv_cache_policy: KvCachePolicy::Fp16,
            substitution_mode: SubstitutionMode::Enabled {
                aggressiveness: "balanced".into(),
            },
            validation_mode: ValidationMode::SyntheticOperator,
            fusion_mode: FusionMode::Disabled,
        }
    }

    /// D — Aggressive substitution ranked.
    pub fn aggressive_substitution() -> Self {
        Self {
            id: "D_aggressive".into(),
            policy_digest: "aggressive_substitution".into(),
            hardware_target: HardwareTarget::UnknownCpu,
            runtime_lanes: RuntimeLaneConfig::default(),
            kv_cache_policy: KvCachePolicy::Fp16,
            substitution_mode: SubstitutionMode::Enabled {
                aggressiveness: "aggressive".into(),
            },
            validation_mode: ValidationMode::FullHardware,
            fusion_mode: FusionMode::Disabled,
        }
    }

    /// E — Research activation-weighted.
    pub fn research_activation_weighted() -> Self {
        Self {
            id: "E_research_activation_weighted".into(),
            policy_digest: "research_activation_weighted".into(),
            hardware_target: HardwareTarget::UnknownCpu,
            runtime_lanes: RuntimeLaneConfig {
                ane_priority: true,
                ..RuntimeLaneConfig::default()
            },
            kv_cache_policy: KvCachePolicy::Compressed { bits: 8 },
            substitution_mode: SubstitutionMode::Enabled {
                aggressiveness: "aggressive".into(),
            },
            validation_mode: ValidationMode::FullHardware,
            fusion_mode: FusionMode::Disabled,
        }
    }
}

// ── LayoutResolver ──────────────────────────────────────────────────────

/// Layout resolution request for a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutRequest {
    pub tensor_key: String,
    pub tensor_class: String,
    pub logical_shape: [u32; 2],
    pub codec: String,
    pub codec_params: HashMap<String, serde_json::Value>,
    pub target_class: HardwareTargetClass,
}

/// Layout resolution result for a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutResolution {
    pub tensor_key: String,
    pub layout: PhysicalTileLayout,
    pub execution_views: Vec<ExecutionView>,
    pub reason: String,
}

/// LayoutResolver — selects tile shape, group axis, metadata placement,
/// and execution views for a tensor given its codec and target hardware.
///
/// This is the middle stage of the compiler pipeline:
///   PolicyResolver (codec) → LayoutResolver (arrangement) →
///   ExecutionPlanner (lane + residency)
pub struct LayoutResolver;

impl LayoutResolver {
    /// Resolve the physical layout for a single tensor.
    pub fn resolve(request: &LayoutRequest) -> LayoutResolution {
        // Default tile family from target class
        let tile_family = match request.target_class {
            HardwareTargetClass::A18Neo
            | HardwareTargetClass::MBase
            | HardwareTargetClass::MPro => TileFamily::tile640(),
            HardwareTargetClass::MMax | HardwareTargetClass::MUltra => TileFamily::tile640(),
        };

        // Infer group_axis from tensor class patterns
        let group_axis = if request.tensor_class.contains("VisionPatchProjection")
            || request.tensor_class.contains("patch_dense")
        {
            GroupAxis::InputAxis
        } else {
            GroupAxis::PackedContiguous
        };

        // Group size from codec params or default
        let group_size = request
            .codec_params
            .get("group_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as u32;

        let layout = PhysicalTileLayout {
            format: request.codec.clone(),
            tile_family,
            logical_shape: request.logical_shape,
            storage_order: StorageOrder::RowMajor,
            tile_shape: TileShape::tile640(),
            group_size,
            group_axis,
            metadata_layout: MetadataLayout::AdjacentTile,
            padding_policy: "ZeroPadToTile".into(),
            alignment_bytes: 256,
            interleave: "None".into(),
        };

        // Build execution views based on target class
        let max_views = request.target_class.max_views_per_tensor();
        let mut execution_views = Vec::new();

        // Primary view: metal fused decode (always available)
        execution_views.push(ExecutionView {
            lane: "metal_fused_decode".into(),
            data_offset: 0,
            data_length: 0,
            metadata_offset: None,
            metadata_length: None,
            codec_overrides: HashMap::new(),
            repacking_required: false,
            residency: ResidencyMode::AlwaysMapped,
            discovered_program: None,
            evolution_provenance: None,
            cost_profile: None,
        });

        // Secondary view: A second metal view is available on targets with
        // sufficient scratch budget, e.g., to allow tensor API or ANE path.
        // The compiler will fill offsets once packing is done.
        if max_views >= 2 {
            execution_views.push(ExecutionView {
                lane: "metal_tensor_api".into(),
                data_offset: 0,
                data_length: 0,
                metadata_offset: None,
                metadata_length: None,
                codec_overrides: HashMap::new(),
                repacking_required: true,
                residency: if max_views >= 3 {
                    ResidencyMode::AlwaysMapped
                } else {
                    ResidencyMode::MutuallyExclusiveViewGroup
                },
                discovered_program: None,
                evolution_provenance: None,
                cost_profile: None,
            });
        }

        let reason = format!(
            "class={} codec={} target={:?} group_axis={:?} views={}",
            request.tensor_class,
            request.codec,
            request.target_class,
            group_axis,
            execution_views.len(),
        );

        LayoutResolution {
            tensor_key: request.tensor_key.clone(),
            layout,
            execution_views,
            reason,
        }
    }
}

// ── Profile builders ─────────────────────────────────────────────────────

/// Configuration for a profile benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRunConfig {
    pub profile: ExecutionProfile,
    pub cimage_path: Option<String>,
    pub source_model_path: Option<String>,
    pub prompt_path: Option<String>,
    pub output_dir: String,
    pub warmup_tokens: usize,
    pub measure_tokens: usize,
    pub health_sample_interval_ms: u64,
    pub repeat_count: usize,
}

impl ProfileRunConfig {
    pub fn new(output_dir: &str) -> Self {
        Self {
            profile: ExecutionProfile::raw_f32_baseline(),
            cimage_path: None,
            source_model_path: None,
            prompt_path: None,
            output_dir: output_dir.into(),
            warmup_tokens: 8,
            measure_tokens: 128,
            health_sample_interval_ms: 500,
            repeat_count: 3,
        }
    }
}

/// Health sampler configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSamplerConfig {}

/// Results of a single profile benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRunResult {
    pub execution: ExecutionProfileReceipt,
    pub memory: MemoryReceipt,
    pub quality: Option<QualityDriftReceipt>,
    pub health: RuntimeHealthReceipt,
    /// Classification of how this result was obtained.
    pub evidence_kind: ReceiptEvidenceKind,
}

impl ProfileRunResult {
    /// Calculate overall status from quality, health, and execution gates.
    pub fn overall_status(&self) -> String {
        let q = self
            .quality
            .as_ref()
            .map(|q| q.quality_retention)
            .unwrap_or(0.0);
        let h = &self.health;
        if h.oom_kill || h.watchdog_exit {
            return "rejected".into();
        }
        if h.stability_status == StabilityStatus::Rejected {
            return "rejected".into();
        }
        if q > 0.0 && q < 0.95 {
            return "research_only".into();
        }
        if h.stability_status == StabilityStatus::Deployable && q >= 0.98 {
            return "deployable".into();
        }
        if h.stability_status == StabilityStatus::DemoSafe && q >= 0.99 {
            return "demo_safe".into();
        }
        "admissible".into()
    }

    /// Returns true only when the evidence is based on real hardware measurements.
    ///
    /// Template or synthetic results are not eligible for promotion gates;
    /// only `Measured` evidence can gate production deployment decisions.
    pub fn can_promote_policy(&self) -> bool {
        self.evidence_kind == ReceiptEvidenceKind::Measured
    }
}

/// Run a single profile benchmark.
pub fn run_profile(config: &ProfileRunConfig) -> ProfileRunResult {
    // Placeholder — actual runner requires full ECS pipeline
    let health = RuntimeHealthReceipt {
        stability_status: StabilityStatus::ResearchOnly,
        ..health_template(&config.profile.id)
    };
    ProfileRunResult {
        execution: ExecutionProfileReceipt {
            profile_id: config.profile.id.clone(),
            ..execution_template(&config.profile.id)
        },
        memory: MemoryReceipt {
            profile_id: config.profile.id.clone(),
            ..memory_template()
        },
        quality: Some(QualityDriftReceipt {
            baseline_profile_id: "A_raw_f32".into(),
            candidate_profile_id: config.profile.id.clone(),
            quality_retention: if config.profile.id == "A_raw_f32" {
                1.0
            } else {
                0.995
            },
            ..quality_template()
        }),
        health,
        evidence_kind: ReceiptEvidenceKind::Synthetic,
    }
}

pub(crate) fn execution_template(id: &str) -> ExecutionProfileReceipt {
    ExecutionProfileReceipt {
        profile_id: id.into(),
        model_family: "gemma4".into(),
        hardware_target: "apple_m1_ane_16gb".into(),
        runtime_backend: "metal+ane".into(),
        policy_digest: String::new(),
        cimage_digest: String::new(),
        prompt_tokens: 0,
        generated_tokens: 0,
        batch_size: 1,
        context_length: 4096,
        cold_start_ms: 0.0,
        first_token_ms: 0.0,
        prefill_ms: 0.0,
        decode_ms: 0.0,
        total_ms: 0.0,
        prefill_tok_per_s: 0.0,
        decode_tok_per_s: 0.0,
        steady_state_tok_per_s: 0.0,
        end_to_end_tok_per_s: 0.0,
        peak_rss_bytes: 0,
        peak_gpu_bytes: None,
        mapped_weight_bytes: 0,
        resident_weight_bytes: 0,
        kv_cache_bytes: 0,
        per_layer: vec![],
        decode_speedup_vs_raw: None,
    }
}

pub(crate) fn memory_template() -> MemoryReceipt {
    MemoryReceipt {
        profile_id: String::new(),
        raw_weight_bytes: 0,
        packed_weight_bytes: 0,
        metadata_bytes: 0,
        sidecar_bytes: 0,
        alignment_padding_bytes: 0,
        runtime_scratch_bytes: 0,
        resident_total_bytes: 0,
        compression_ratio_vs_raw: 1.0,
    }
}

pub(crate) fn health_template(profile_id: &str) -> RuntimeHealthReceipt {
    RuntimeHealthReceipt {
        profile_id: profile_id.into(),
        hardware_target: "apple_m1_ane_16gb".into(),
        os_version: String::new(),
        duration_s: 0.0,
        memory_pressure_timeline: vec![],
        thermal_timeline: vec![],
        process_memory_timeline: vec![],
        throughput_timeline: vec![],
        peak_rss_bytes: 0,
        peak_virtual_bytes: 0,
        peak_compressed_memory_bytes: None,
        swap_delta_bytes: Some(0),
        cpu_package_active_ratio: None,
        gpu_busy_ratio: None,
        ane_busy_ratio: None,
        stall_count: 0,
        longest_stall_ms: 0.0,
        oom_kill: false,
        watchdog_exit: false,
        thermal_throttle_detected: false,
        stability_status: StabilityStatus::Unknown,
        rss_slope_mb_per_100_tokens: 0.0,
    }
}

pub(crate) fn quality_template() -> QualityDriftReceipt {
    QualityDriftReceipt {
        baseline_profile_id: String::new(),
        candidate_profile_id: String::new(),
        eval_set_digest: String::new(),
        hidden_state_nrmse_mean: 0.0,
        hidden_state_nrmse_worst: 0.0,
        logits_kl_mean: 0.0,
        logits_kl_p95: 0.0,
        top1_agreement: 1.0,
        top5_agreement: 1.0,
        sampled_token_agreement_seeded: 1.0,
        first_token_accuracy_delta: None,
        per_layer_drift: vec![],
        quality_retention: 1.0,
    }
}

// ── Profile comparison matrix ────────────────────────────────────────────

/// Build a comparison matrix from multiple profile results.
pub fn build_comparison_matrix(
    results: &[ProfileRunResult],
    baseline_id: &str,
) -> Vec<ProfileComparisonRow> {
    let baseline = results
        .iter()
        .find(|r| r.execution.profile_id == baseline_id);
    let baseline_tok = baseline
        .map(|b| b.execution.decode_tok_per_s)
        .filter(|&t| t > 0.0)
        .unwrap_or(1.0);

    results
        .iter()
        .map(|r| {
            let tok_s = r.execution.decode_tok_per_s;
            let speedup = if baseline_id != r.execution.profile_id && baseline_tok > 0.0 {
                Some(tok_s / baseline_tok)
            } else {
                None
            };
            let quality = r.quality.as_ref();
            let rss_mb = r.health.peak_rss_bytes as f64 / 1_000_000.0;
            let swap_mb = r.health.swap_delta_bytes.unwrap_or(0) as f64 / 1_000_000.0;

            ProfileComparisonRow {
                profile_id: r.execution.profile_id.clone(),
                policy_digest: r.execution.policy_digest.clone(),
                weight_bytes: r.memory.resident_total_bytes,
                resident_bytes: r.health.peak_rss_bytes,
                compression_ratio: r.memory.compression_ratio_vs_raw,
                first_token_ms: r.execution.first_token_ms,
                prefill_tok_s: r.execution.prefill_tok_per_s,
                decode_tok_s: tok_s,
                speedup_vs_raw: speedup,
                logits_kl: quality.map(|q| q.logits_kl_mean),
                top1_agreement: quality.map(|q| q.top1_agreement),
                peak_rss_mb: rss_mb,
                swap_delta_mb: swap_mb,
                quality_status: quality
                    .map(|q| {
                        if q.quality_retention >= 0.98 {
                            "pass".into()
                        } else {
                            "fail".into()
                        }
                    })
                    .unwrap_or("unknown".into()),
                stability_status: r.health.stability_status,
                overall_status: r.overall_status(),
            }
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_serde_roundtrip() {
        let r = ExecutionProfileReceipt {
            profile_id: "test".into(),
            model_family: "gemma4".into(),
            hardware_target: "apple_m1_ane_16gb".into(),
            runtime_backend: "metal+ane".into(),
            policy_digest: "abc123".into(),
            cimage_digest: "def456".into(),
            prompt_tokens: 512,
            generated_tokens: 256,
            batch_size: 1,
            context_length: 4096,
            cold_start_ms: 1500.0,
            first_token_ms: 250.0,
            prefill_ms: 800.0,
            decode_ms: 12000.0,
            total_ms: 12800.0,
            prefill_tok_per_s: 640.0,
            decode_tok_per_s: 21.3,
            steady_state_tok_per_s: 21.3,
            end_to_end_tok_per_s: 20.0,
            peak_rss_bytes: 8_000_000_000,
            peak_gpu_bytes: Some(4_000_000_000),
            mapped_weight_bytes: 6_000_000_000,
            resident_weight_bytes: 3_000_000_000,
            kv_cache_bytes: 1_000_000_000,
            per_layer: vec![],
            decode_speedup_vs_raw: Some(2.6),
        };
        let json = serde_json::to_string(&r).unwrap();
        let decoded: ExecutionProfileReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.profile_id, "test");
        assert_eq!(decoded.decode_tok_per_s, 21.3);
    }

    #[test]
    fn test_health_receipt_defaults() {
        let h = RuntimeHealthReceipt {
            profile_id: "test".into(),
            hardware_target: "apple_m1_ane_16gb".into(),
            os_version: "14.5".into(),
            duration_s: 30.0,
            memory_pressure_timeline: vec![],
            thermal_timeline: vec![],
            process_memory_timeline: vec![],
            throughput_timeline: vec![],
            peak_rss_bytes: 8_000_000_000,
            peak_virtual_bytes: 12_000_000_000,
            peak_compressed_memory_bytes: None,
            swap_delta_bytes: Some(0),
            cpu_package_active_ratio: Some(0.6),
            gpu_busy_ratio: Some(0.4),
            ane_busy_ratio: Some(0.3),
            stall_count: 0,
            longest_stall_ms: 0.0,
            oom_kill: false,
            watchdog_exit: false,
            thermal_throttle_detected: false,
            stability_status: StabilityStatus::Deployable,
            rss_slope_mb_per_100_tokens: 0.5,
        };
        assert_eq!(h.stability_status, StabilityStatus::Deployable);
    }

    #[test]
    fn test_profile_builders() {
        let a = ExecutionProfile::raw_f32_baseline();
        assert_eq!(a.id, "A_raw_f32");
        assert!(matches!(a.substitution_mode, SubstitutionMode::Disabled));
        let c = ExecutionProfile::production_policy_v2();
        assert_eq!(c.id, "C_production_v2");
        let d = ExecutionProfile::aggressive_substitution();
        assert!(matches!(
            d.substitution_mode,
            SubstitutionMode::Enabled { .. }
        ));
        let e = ExecutionProfile::research_activation_weighted();
        assert!(matches!(
            e.kv_cache_policy,
            KvCachePolicy::Compressed { bits: 8 }
        ));
        assert!(e.runtime_lanes.ane_priority);
    }

    #[test]
    fn test_profile_comparison_row_format() {
        let row = ProfileComparisonRow {
            profile_id: "A".into(),
            policy_digest: "raw_f32".into(),
            weight_bytes: 100_000_000,
            resident_bytes: 100_000_000,
            compression_ratio: 1.0,
            first_token_ms: 200.0,
            prefill_tok_s: 500.0,
            decode_tok_s: 8.0,
            speedup_vs_raw: None,
            logits_kl: None,
            top1_agreement: None,
            peak_rss_mb: 8000.0,
            swap_delta_mb: 0.0,
            quality_status: "baseline".into(),
            stability_status: StabilityStatus::Admissible,
            overall_status: "baseline".into(),
        };
        let json = serde_json::to_string_pretty(&row).unwrap();
        assert!(json.contains("raw_f32"));
        assert!(json.contains("baseline"));
    }

    #[test]
    fn test_comparison_matrix_build() {
        let mut config_c = ProfileRunConfig::new("/tmp/profiles/C");
        config_c.profile.id = "C_production".into();
        let results = vec![
            run_profile(&ProfileRunConfig::new("/tmp/profiles/A")),
            run_profile(&config_c),
        ];
        let matrix = build_comparison_matrix(&results, "A_raw_f32");
        assert_eq!(matrix.len(), 2);
        assert!(matrix[0].speedup_vs_raw.is_none());
        assert!(matrix[1].speedup_vs_raw.is_some());
    }

    #[test]
    fn test_group_axis_roundtrip() {
        let v = vec![
            GroupAxis::PackedContiguous,
            GroupAxis::OutputAxis,
            GroupAxis::InputAxis,
            GroupAxis::TileLocal,
        ];
        for g in v {
            let json = serde_json::to_string(&g).unwrap();
            let back: GroupAxis = serde_json::from_str(&json).unwrap();
            assert_eq!(g, back);
        }
    }

    #[test]
    fn test_tile_family_defaults() {
        let t = TileFamily::tile640();
        assert_eq!(t.name, "Tile640");
        assert_eq!(t.tile_shape.elements(), 409600);
        assert!(t.default_group_sizes.contains(&32));
    }

    #[test]
    fn test_hardware_class_constants() {
        assert_eq!(HardwareTargetClass::A18Neo.memory_gb(), 8);
        assert_eq!(HardwareTargetClass::MMax.bandwidth_gbps(), 546);
        assert_eq!(HardwareTargetClass::MUltra.scratch_budget_mb(), 4096);
    }
    #[test]
    fn test_layout_resolver_decoder_projection() {
        let req = LayoutRequest {
            tensor_key: "model.layers.5.self_attn.q_proj.weight".into(),
            tensor_class: "Gemma4.DecoderAttentionProjection".into(),
            logical_shape: [4096, 3840],
            codec: "NF4".into(),
            codec_params: [("group_size".into(), serde_json::json!(32))].into(),
            target_class: HardwareTargetClass::MBase,
        };
        let res = LayoutResolver::resolve(&req);
        assert_eq!(res.layout.group_size, 32);
        assert_eq!(res.layout.group_axis, GroupAxis::PackedContiguous);
        assert_eq!(res.layout.format, "NF4");
        assert!(res.reason.contains("group_axis=PackedContiguous"));
    }

    #[test]
    fn test_layout_resolver_patch_dense() {
        let req = LayoutRequest {
            tensor_key: "patch_dense".into(),
            tensor_class: "Gemma4.VisionPatchProjection".into(),
            logical_shape: [2048, 6912],
            codec: "RawF32".into(),
            codec_params: HashMap::new(),
            target_class: HardwareTargetClass::MBase,
        };
        let res = LayoutResolver::resolve(&req);
        assert_eq!(res.layout.group_axis, GroupAxis::InputAxis);
        assert_eq!(res.execution_views.len(), 1);
    }

    #[test]
    fn test_layout_resolver_max_target_multiple_views() {
        let req = LayoutRequest {
            tensor_key: "model.layers.10.mlp.gate_proj.weight".into(),
            tensor_class: "Gemma4.DecoderMlpProjection".into(),
            logical_shape: [16384, 4096],
            codec: "NF4".into(),
            codec_params: [("group_size".into(), serde_json::json!(32))].into(),
            target_class: HardwareTargetClass::MMax,
        };
        let res = LayoutResolver::resolve(&req);
        // On MMax (max_views=2) we get 2 views
        assert_eq!(res.execution_views.len(), 2);
        assert_eq!(res.execution_views[0].lane, "metal_fused_decode");
        assert_eq!(res.execution_views[1].lane, "metal_tensor_api");
    }
    #[test]
    fn profile_template_not_gate_eligible() {
        let template = ProfileRunResult {
            execution: ExecutionProfileReceipt {
                profile_id: "template-test".into(),
                ..execution_template("template-test")
            },
            memory: memory_template(),
            quality: None,
            health: health_template("template-test"),
            evidence_kind: ReceiptEvidenceKind::Template,
        };
        assert!(!template.can_promote_policy());
        let synthetic = ProfileRunResult {
            evidence_kind: ReceiptEvidenceKind::Synthetic,
            ..template.clone()
        };
        assert!(!synthetic.can_promote_policy());
        let measured = ProfileRunResult {
            evidence_kind: ReceiptEvidenceKind::Measured,
            ..template
        };
        assert!(measured.can_promote_policy());
    }
}
