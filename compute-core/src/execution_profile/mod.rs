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
        Self { use_gpu: true, use_ane: true, use_cpu: true, gpu_priority: true, ane_priority: false }
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

// ── Profile builders ─────────────────────────────────────────────────────

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
            substitution_mode: SubstitutionMode::Enabled { aggressiveness: "conservative".into() },
            validation_mode: ValidationMode::WeightSpaceOnly,
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
            substitution_mode: SubstitutionMode::Enabled { aggressiveness: "balanced".into() },
            validation_mode: ValidationMode::SyntheticOperator,
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
            substitution_mode: SubstitutionMode::Enabled { aggressiveness: "aggressive".into() },
            validation_mode: ValidationMode::FullHardware,
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
            substitution_mode: SubstitutionMode::Enabled { aggressiveness: "aggressive".into() },
            validation_mode: ValidationMode::FullHardware,
        }
    }
}

// ── Profile runner ──────────────────────────────────────────────────────

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
}

impl ProfileRunResult {
    /// Calculate overall status from quality, health, and execution gates.
    pub fn overall_status(&self) -> String {
        let q = self.quality.as_ref().map(|q| q.quality_retention).unwrap_or(0.0);
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
            quality_retention: if config.profile.id == "A_raw_f32" { 1.0 } else { 0.995 },
            ..quality_template()
        }),
        health,
    }
}

fn execution_template(id: &str) -> ExecutionProfileReceipt {
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

fn memory_template() -> MemoryReceipt {
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

fn health_template(profile_id: &str) -> RuntimeHealthReceipt {
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

fn quality_template() -> QualityDriftReceipt {
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
pub fn build_comparison_matrix(results: &[ProfileRunResult], baseline_id: &str) -> Vec<ProfileComparisonRow> {
    let baseline = results.iter().find(|r| r.execution.profile_id == baseline_id);
    let baseline_tok = baseline.map(|b| b.execution.decode_tok_per_s).filter(|&t| t > 0.0).unwrap_or(1.0);

    results.iter().map(|r| {
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
            quality_status: quality.map(|q| if q.quality_retention >= 0.98 { "pass".into() } else { "fail".into() }).unwrap_or("unknown".into()),
            stability_status: r.health.stability_status,
            overall_status: r.overall_status(),
        }
    }).collect()
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
        assert!(matches!(d.substitution_mode, SubstitutionMode::Enabled { .. }));
        let e = ExecutionProfile::research_activation_weighted();
        assert!(matches!(e.kv_cache_policy, KvCachePolicy::Compressed { bits: 8 }));
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
}
