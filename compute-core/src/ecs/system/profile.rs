//! ProfileExecutionSystem — ECS-native profile plan execution.
//!
//! Port of `profile.rs::run_profile_plan()` logic: executes a profile plan on a
//! backend, produces a `ProfileRunResult` component containing execution,
//! memory, quality, and health receipts.

use std::time::Instant;

use crate::ecs::component::planning::{
    ModelExecutionPlanComp, ProfileRunConfigComp, ProfileRunResult,
};
use crate::ecs::execution_profile::{
    ExecutionProfileReceipt, MemoryReceipt, ProfileRunConfig, ReceiptEvidenceKind,
    RuntimeHealthReceipt, StabilityStatus,
};
use crate::ecs::plan::ModelExecutionPlan;
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

use crate::ecs::Entity;
use crate::ecs::{CompilerSystem, EntityKind, SchedulePhase, World};

/// ECS system that executes a profile plan on a backend and produces
/// `ProfileRunResult` components.
///
/// Iterates entities with `ModelExecutionPlanComp` and `ProfileRunConfigComp`
/// components, runs the profile (placeholder — requires Metal runtime
/// integration), and attaches a `ProfileRunResult` component.
pub struct ProfileExecutionSystem;

impl CompilerSystem for ProfileExecutionSystem {
    fn name(&self) -> &str {
        "ProfileExecutionSystem"
    }

    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }

    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let model_entities: Vec<Entity> = world.entities_of_kind(EntityKind::Model);

        // Stage every per-model `ProfileRunResult` insert on a single
        // `ConstitutionalWorldTxn` and commit atomically. Direct
        // `world.add_component` calls outside the WorldTxn seam are
        // forbidden.
        let mut txn = ConstitutionalWorldTxn::new();
        for entity in model_entities {
            // Skip if already profiled.
            if world.get_component::<ProfileRunResult>(entity).is_some() {
                continue;
            }

            let plan = match world.get_component::<ModelExecutionPlanComp>(entity) {
                Some(p) => p.0.clone(),
                None => continue,
            };

            let config = match world.get_component::<ProfileRunConfigComp>(entity) {
                Some(c) => c.0.clone(),
                None => continue,
            };

            let result = execute_profile(&plan, &config);

            if let Err(e) = txn.stage_insert(entity, ProfileRunResult(result)) {
                tracing::warn!(entity = ?entity, error = %e, "profile: stage_insert ProfileRunResult");
            }
        }
        let _ = txn.commit(world).map_err(|e| {
            tracing::error!(error = %e, "profile: ConstitutionalWorldTxn commit failed");
            anyhow::anyhow!("profile: ConstitutionalWorldTxn commit failed: {e}")
        })?;

        Ok(())
    }
}

/// Execute a single profile run against the plan (inline port of `run_profile_plan`).
///
/// Placeholder — actual Metal/runtime execution requires runtime integration.
/// This produces a synthetic result with timing from the wall clock and
/// information drawn from the plan and config.
fn execute_profile(
    plan: &ModelExecutionPlan,
    config: &ProfileRunConfig,
) -> crate::ecs::execution_profile::ProfileRunResult {
    let start = Instant::now();

    // Phase 1: Warmup (placeholder)
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
        cold_start_ms: 0.0,
        first_token_ms: 0.0,
        prefill_ms: 0.0,
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

    let health = RuntimeHealthReceipt {
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

    crate::ecs::execution_profile::ProfileRunResult {
        execution,
        memory,
        quality: None,
        health,
        evidence_kind: ReceiptEvidenceKind::Synthetic,
    }
}
