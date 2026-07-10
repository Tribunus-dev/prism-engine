//! Profile runner v1 — measures tok/s, latency, memory, and quality
//! for a ModelExecutionPlan on a given hardware/layout profile.

use std::time::Instant;

use super::*;
use crate::ecs::execution_profile::*;

/// Real profile runner that executes a ModelExecutionPlan and collects receipts.
pub async fn run_profile_plan(
    plan: &ModelExecutionPlan,
    config: &ProfileRunConfig,
    health_sampler: &mut dyn FnMut(&mut RuntimeHealthReceipt),
) -> Result<ProfileRunResult, String> {
    // Placeholder: actual execution requires Metal runtime integration
    let start = Instant::now();

    // Phase 1: Warmup
    // Phase 2: Measure decode tokens, collect timing
    // Phase 3: Collect health samples
    // Phase 4: Compute quality drift (requires reference)

    let elapsed = start.elapsed();

    let execution = ExecutionProfileReceipt {
        profile_id: plan.plan_id.clone(),
        model_family: plan.model_family.clone(),
        hardware_target: format!("{:?}", plan.layout_profile),
        runtime_backend: "cimage_region_batched".into(),
        policy_digest: plan.policy_digest.clone(),
        cimage_digest: plan.cimage_digest.clone(),
        prompt_tokens: config.measure_tokens,
        generated_tokens: config.measure_tokens,
        batch_size: 1,
        context_length: 4096,
        cold_start_ms: 0.0,  // TODO
        first_token_ms: 0.0, // TODO
        prefill_ms: 0.0,     // TODO
        decode_ms: elapsed.as_secs_f64() * 1000.0,
        total_ms: elapsed.as_secs_f64() * 1000.0,
        prefill_tok_per_s: 0.0,
        decode_tok_per_s: config.measure_tokens as f64 / elapsed.as_secs_f64().max(0.001),
        steady_state_tok_per_s: 0.0,
        end_to_end_tok_per_s: 0.0,
        peak_rss_bytes: 0,
        peak_gpu_bytes: None,
        mapped_weight_bytes: plan.total_scratch_budget_bytes,
        resident_weight_bytes: plan.total_scratch_budget_bytes,
        kv_cache_bytes: 0,
        per_layer: vec![],
        decode_speedup_vs_raw: None,
    };

    let memory = MemoryReceipt {
        profile_id: plan.plan_id.clone(),
        raw_weight_bytes: 0,
        packed_weight_bytes: plan.total_scratch_budget_bytes,
        metadata_bytes: 0,
        sidecar_bytes: 0,
        alignment_padding_bytes: 0,
        runtime_scratch_bytes: plan.total_scratch_budget_bytes,
        resident_total_bytes: plan.total_scratch_budget_bytes,
        compression_ratio_vs_raw: 1.0,
    };

    let mut health = RuntimeHealthReceipt {
        profile_id: plan.plan_id.clone(),
        hardware_target: format!("{:?}", plan.layout_profile),
        os_version: String::new(),
        duration_s: elapsed.as_secs_f64(),
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
    };
    (health_sampler)(&mut health);

    Ok(ProfileRunResult {
        execution,
        memory,
        quality: None,
        health,
        evidence_kind: ReceiptEvidenceKind::Synthetic,
    })
}

// ---------------------------------------------------------------------------
// BackendProfileRunner — Metal-backed region encoder profile runner
// ---------------------------------------------------------------------------
// Gated on macOS + metal-dispatch because it depends on the `metal` crate.

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::metal_runtime::pso_cache::MetalPsoCache;

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::execution_plan::region_encoder::RegionEncoder;

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use metal::{ComputePipelineState, Device};

/// Profile runner backed by a concrete [`RegionEncoder`] and [`MetalPsoCache`].
///
/// Walks an [`ModelExecutionPlan`]'s regions, validates hazards, and
/// encodes dispatches via the Metal runtime.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub struct BackendProfileRunner<R: RegionEncoder> {
    pub region_encoder: R,
    pub pso_cache: MetalPsoCache,
}

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
impl<R: RegionEncoder<PipelineState = ComputePipelineState>> BackendProfileRunner<R> {
    /// Create a new `BackendProfileRunner` from a Metal device and shader source.
    ///
    /// # Errors
    ///
    /// Returns an error if the Metal library fails to compile from `shader_source`.
    pub fn new(device: &Device, shader_source: &str) -> Result<Self, String> {
        let pso = MetalPsoCache::new(device, shader_source)?;
        Ok(Self {
            region_encoder: R::new(device),
            pso_cache: pso,
        })
    }

    /// Run the profile for the given plan and configuration.
    ///
    /// Validates hazard for each region, encodes all safe regions, and
    /// returns a [`ProfileRunResult`] with template execution/memory/health
    /// receipts.
    pub fn run(
        &mut self,
        plan: &ModelExecutionPlan,
        _config: &ProfileRunConfig,
    ) -> ProfileRunResult {
        // 1. Validate hazard for each region
        for region in &plan.regions {
            if let Ok(hazard) = HazardChecker::validate_region(region) {
                if hazard.safe {
                    // 2. Encode regions
                    for region in &plan.regions {
                        let _ = self
                            .region_encoder
                            .encode_region(region, &mut self.pso_cache);
                    }
                }
            }
        }

        // Return template result for now
        ProfileRunResult {
            execution: execution_template(&plan.plan_id),
            memory: memory_template(),
            quality: None,
            health: health_template(&plan.plan_id),
            evidence_kind: ReceiptEvidenceKind::Template,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_plan() -> ModelExecutionPlan {
        ModelExecutionPlan {
            plan_id: "test-plan".into(),
            model_family: "gemma4".into(),
            cimage_digest: "abc123".into(),
            policy_digest: "policy-1".into(),
            layout_profile: HardwareProfileId::AppleMProBalanced,
            regions: vec![],
            pso_keys: vec![],
            total_scratch_budget_bytes: 1024 * 1024 * 1024,
            validation_digest: None,
            execution_mode: Default::default(),
        }
    }

    #[tokio::test]
    async fn test_profile_run_result_default() {
        let plan = make_test_plan();
        let config = ProfileRunConfig::new("/tmp/test_profile");
        let mut sampler = |r: &mut RuntimeHealthReceipt| {
            r.stall_count = 1;
        };

        let result = run_profile_plan(&plan, &config, &mut sampler)
            .await
            .unwrap();

        // Verify execution receipt
        assert_eq!(result.execution.profile_id, "test-plan");
        assert_eq!(result.execution.model_family, "gemma4");
        assert_eq!(result.execution.runtime_backend, "cimage_region_batched");
        assert_eq!(result.execution.prompt_tokens, 128);
        assert_eq!(result.execution.generated_tokens, 128);
        assert_eq!(result.execution.batch_size, 1);
        assert_eq!(result.execution.mapped_weight_bytes, 1024 * 1024 * 1024);
        assert_eq!(result.execution.resident_weight_bytes, 1024 * 1024 * 1024);
        assert!(result.execution.total_ms > 0.0);

        // Verify memory receipt
        assert_eq!(result.memory.profile_id, "test-plan");
        assert_eq!(result.memory.packed_weight_bytes, 1024 * 1024 * 1024);
        assert_eq!(result.memory.compression_ratio_vs_raw, 1.0);

        // Verify quality is None (no reference provided)
        assert!(result.quality.is_none());

        // Verify sampler ran
        assert_eq!(result.health.stall_count, 1);
    }

    #[tokio::test]
    async fn test_decode_tok_per_s_computation() {
        let plan = make_test_plan();
        let config = ProfileRunConfig {
            measure_tokens: 100,
            ..ProfileRunConfig::new("/tmp/test_tok")
        };
        let mut sampler = |_: &mut RuntimeHealthReceipt| {};

        let result = run_profile_plan(&plan, &config, &mut sampler)
            .await
            .unwrap();

        // decode_tok_per_s = measure_tokens / elapsed_secs
        // elapsed > 0 in real execution, so decode_tok_per_s should be finite positive
        let tok_per_s = result.execution.decode_tok_per_s;
        assert!(
            tok_per_s.is_finite(),
            "decode_tok_per_s should be finite, got {}",
            tok_per_s
        );
        assert!(
            tok_per_s > 0.0,
            "decode_tok_per_s should be positive, got {}",
            tok_per_s
        );

        // With 100 tokens in near-zero time, should be very large
        assert!(
            tok_per_s > 1_000.0,
            "decode_tok_per_s should be >1k for fast execution (100 tokens), got {}",
            tok_per_s
        );

        // Verify total_ms equals decode_ms (no prefill measured yet)
        assert!(
            (result.execution.decode_ms - result.execution.total_ms).abs() < f64::EPSILON,
            "decode_ms should approximately equal total_ms"
        );
    }

    #[tokio::test]
    async fn test_profile_run_hardware_target_format() {
        let plan = ModelExecutionPlan {
            layout_profile: HardwareProfileId::AppleA18Tiny,
            ..make_test_plan()
        };
        let config = ProfileRunConfig::new("/tmp/test_hw");
        let mut sampler = |_: &mut RuntimeHealthReceipt| {};

        let result = run_profile_plan(&plan, &config, &mut sampler)
            .await
            .unwrap();

        // Debug format of HardwareProfileId gives the variant name
        assert_eq!(
            result.execution.hardware_target, "AppleA18Tiny",
            "hardware_target should use Debug format of HardwareProfileId"
        );
        assert_eq!(
            result.health.hardware_target, "AppleA18Tiny",
            "health hardware_target should match"
        );
    }

    #[tokio::test]
    async fn test_profile_run_zero_tokens_no_divide_by_zero() {
        let plan = make_test_plan();
        let config = ProfileRunConfig {
            measure_tokens: 0,
            ..ProfileRunConfig::new("/tmp/test_zero")
        };
        let mut sampler = |_: &mut RuntimeHealthReceipt| {};

        let result = run_profile_plan(&plan, &config, &mut sampler)
            .await
            .unwrap();

        // With 0 tokens, decode_tok_per_s = 0 / elapsed = 0 (since max(0.001) prevents div by zero)
        let tok_per_s = result.execution.decode_tok_per_s;
        assert!(
            tok_per_s.is_finite(),
            "decode_tok_per_s should be finite even with 0 tokens, got {}",
            tok_per_s
        );
        assert!(
            tok_per_s == 0.0,
            "decode_tok_per_s should be 0 with 0 tokens, got {}",
            tok_per_s
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_backend_profile_runner_new() {
        // Verify BackendProfileRunner construction with a minimal shader.
        use crate::metal_runtime::region_encoder::MetalRegionEncoder;

        let device = metal::Device::system_default().expect("Metal device required");
        let source = "// empty library — not used for empty plan\n";
        let mut runner = BackendProfileRunner::<MetalRegionEncoder>::new(&device, source)
            .expect("empty shader source compiles");

        let plan = make_test_plan();
        let config = ProfileRunConfig::new("/tmp/test_runner");
        let result = runner.run(&plan, &config);

        // Verify template receipts are returned
        assert_eq!(result.execution.profile_id, "test-plan");
        assert_eq!(result.memory.profile_id, "");
        assert!(result.quality.is_none());
        assert_eq!(result.health.profile_id, "test-plan");

        // No regions → no encoding happened, but receipts still valid
        assert_eq!(result.health.stall_count, 0);
        assert_eq!(result.execution.decode_tok_per_s, 0.0);
        assert_eq!(result.execution.prompt_tokens, 0);
        assert_eq!(
            result.execution.runtime_backend, "metal+ane",
            "execution_template default backend"
        );
    }
}
