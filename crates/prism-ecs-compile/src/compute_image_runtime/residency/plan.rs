//! Compiled residency plan — pure data types and pure algorithms for
//! memory residency scheduling.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::{ContentHash, ExecutionShapeClass};

/// Opaque identifier for a compiled residency plan.
pub type ResidencyPlanId = String;

/// Identifier for a weight object that the runtime must make resident.
pub type RequiredWeightObjectId = String;

/// Compiled residency plan — the authoritative contract between the
/// compiler and the runtime for memory management during execution of
/// one program variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledResidencyPlan {
    /// Unique identifier for this plan.
    pub plan_id: ResidencyPlanId,
    /// Content hash covering the entire plan.
    pub plan_hash: ContentHash,
    /// Execution shape class this plan was compiled for.
    pub shape_class: ExecutionShapeClass,
    /// Weight objects that must be resident at some point during execution.
    pub required_weight_objects: Vec<RequiredWeightObject>,
    /// Ordered prefetch schedule.
    pub prefetch_schedule: Vec<PrefetchAction>,
    /// Weight objects the runtime may evict after a given phase.
    pub evictable_weight_objects: Vec<EvictableWeightObject>,
    /// Compiler-computed activation arena dimensions.
    pub activation_arena_requirements: ActivationArenaRequirements,
    /// Compiler-computed KV-cache reservation.
    pub kv_cache_requirements: KvCacheRequirements,
    /// Aggregate peak memory estimate across all categories.
    pub peak_memory_estimate: PeakMemoryEstimate,
    /// Admission contract — minimum and recommended memory budgets.
    pub memory_admission_contract: MemoryAdmissionContract,
}

/// A weight object that must be resident at some point during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequiredWeightObject {
    /// Stable identifier for this weight object within the compute image.
    pub object_id: RequiredWeightObjectId,
    /// Residency class that governs the load/lifecycle strategy.
    pub residency_class: ResidencyClass,
    /// Compiler-estimated byte size of this weight object.
    pub estimated_bytes: u64,
}

/// A single prefetch action in the compiled schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchAction {
    /// Weight object to prefetch.
    pub object_id: RequiredWeightObjectId,
    /// Identifier of the phase before which the prefetch must be initiated.
    pub prefetch_before_phase: String,
    /// Urgency of this prefetch action.
    pub priority: PrefetchPriority,
}

/// Urgency of a prefetch action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PrefetchPriority {
    /// Must be resident before the target phase starts.
    High,
    /// Prefetch opportunistically when I/O bandwidth is available.
    Low,
}

/// A weight object that the runtime may evict after a given phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictableWeightObject {
    /// Weight object eligible for eviction.
    pub object_id: RequiredWeightObjectId,
    /// Phase identifier after which the object may be evicted.
    pub evict_after_phase: String,
    /// Policy governing eviction behaviour.
    pub eviction_policy: EvictionPolicy,
}

/// Policy for evicting a weight object from device memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Release the device buffer but keep the mmap'd backing store.
    DiscardView,
    /// Release both the device buffer and the mmap'd backing store.
    DiscardAll,
    /// Keep the weight data on device in a compressed form.
    CompressInPlace,
}

/// Compiler-computed requirements for the activation memory arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationArenaRequirements {
    /// Total number of bytes required for activation storage.
    pub total_activation_bytes: u64,
    /// Number of distinct activation arena regions.
    pub arena_region_count: u32,
}

/// Compiler-computed requirements for the key-value cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheRequirements {
    /// Maximum number of context tokens this plan supports.
    pub max_context_tokens: u32,
    /// Number of bytes consumed per token in the KV cache.
    pub cache_bytes_per_token: u64,
    /// Total KV-cache byte requirement.
    pub total_cache_bytes: u64,
    /// Total KV-cache bytes (alias for [`Self::total_cache_bytes`]).
    #[serde(default)]
    pub total_kv_cache_bytes: u64,
    /// KV-cache bytes per layer.
    #[serde(default)]
    pub kv_cache_per_layer_bytes: u64,
    /// Number of transformer layers.
    #[serde(default)]
    pub n_layers: u32,
    /// Number of KV heads.
    #[serde(default)]
    pub n_kv_heads: u32,
    /// Head dimension.
    #[serde(default)]
    pub head_dim: u32,
    /// Maximum context length.
    #[serde(default)]
    pub max_context: u32,
}

/// Aggregate peak memory estimate across all memory categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakMemoryEstimate {
    /// Total estimated resident memory at peak.
    pub total_resident_bytes: u64,
    /// Peak activation arena memory.
    pub activation_peak_bytes: u64,
    /// Peak KV-cache memory.
    pub kv_cache_bytes: u64,
    /// Sum of bytes for all weight objects expected to be resident at peak.
    pub resident_weight_bytes: u64,
    /// Runtime overhead bytes.
    pub overhead_bytes: u64,
}

/// The compiled memory admission contract that gates execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAdmissionContract {
    /// Absolute lower bound: execution MUST NOT start unless this is available.
    pub minimum_required_bytes: u64,
    /// Recommended budget at which the plan operates at full quality.
    pub recommended_bytes: u64,
    /// If true, the runtime MAY degrade when memory is between min and recommended.
    pub graceful_degradation: bool,
}

/// Classifies the residency strategy for a weight object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResidencyClass {
    /// Must be resident from session start until the end of the session.
    MandatoryAtSessionStart,
    /// Must be resident before a specific phase begins.
    MandatoryBeforePhase,
    /// Strong candidate for prefetch but not required for correctness.
    PrefetchCandidate,
    /// May be evicted between reuse windows.
    ReusablePinned,
    /// May be evicted after the declared phase.
    EvictableAfterPhase,
    /// The weight data lives on disk only.
    DiskOnly,
}

/// Static peak-memory analyzer for compiled residency plans.
#[derive(Debug, Clone, Default)]
pub struct PeakMemoryAnalyzer;

impl PeakMemoryAnalyzer {
    /// Create a new analyzer.
    pub fn new() -> Self {
        Self
    }

    /// Compute the peak memory estimate from required weight objects
    /// and activation / KV-cache requirements.
    pub fn estimate_peak(
        &self,
        required_weights: &[RequiredWeightObject],
        activation_reqs: &ActivationArenaRequirements,
        kv_reqs: &KvCacheRequirements,
    ) -> PeakMemoryEstimate {
        let resident_weight_bytes: u64 = required_weights
            .iter()
            .filter(|w| {
                matches!(
                    w.residency_class,
                    ResidencyClass::MandatoryAtSessionStart
                        | ResidencyClass::MandatoryBeforePhase
                )
            })
            .map(|w| w.estimated_bytes)
            .sum();

        let activation_peak_bytes = activation_reqs.total_activation_bytes;
        let kv_cache_bytes = kv_reqs.total_cache_bytes;
        let overhead_bytes = (resident_weight_bytes + activation_peak_bytes + kv_cache_bytes) / 10;
        let total_resident_bytes =
            resident_weight_bytes + activation_peak_bytes + kv_cache_bytes + overhead_bytes;

        PeakMemoryEstimate {
            total_resident_bytes,
            activation_peak_bytes,
            kv_cache_bytes,
            resident_weight_bytes,
            overhead_bytes,
        }
    }

    /// Build the admission contract that gates execution within a given
    /// memory budget.
    pub fn check_admission(
        &self,
        plan: &CompiledResidencyPlan,
        available_bytes: u64,
    ) -> MemoryAdmissionContract {
        let peak_total = plan.peak_memory_estimate.total_resident_bytes;
        let minimum_required_bytes = peak_total;
        let recommended_bytes = peak_total;
        let graceful_degradation = peak_total as f64 >= (available_bytes as f64) * 0.9;

        MemoryAdmissionContract {
            minimum_required_bytes,
            recommended_bytes,
            graceful_degradation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weight(estimated_bytes: u64, class: ResidencyClass) -> RequiredWeightObject {
        RequiredWeightObject {
            object_id: format!("w_{}", estimated_bytes),
            residency_class: class,
            estimated_bytes,
        }
    }

    fn arena_reqs(total_activation_bytes: u64) -> ActivationArenaRequirements {
        ActivationArenaRequirements {
            total_activation_bytes,
            arena_region_count: 1,
        }
    }

    fn kv_reqs(total_cache_bytes: u64) -> KvCacheRequirements {
        KvCacheRequirements {
            max_context_tokens: 4096,
            cache_bytes_per_token: total_cache_bytes / 4096,
            total_cache_bytes,
            total_kv_cache_bytes: total_cache_bytes,
            kv_cache_per_layer_bytes: total_cache_bytes / 32,
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            max_context: 4096,
        }
    }

    fn make_plan(
        weights: Vec<RequiredWeightObject>,
        act: ActivationArenaRequirements,
        kv: KvCacheRequirements,
    ) -> CompiledResidencyPlan {
        let analyzer = PeakMemoryAnalyzer::new();
        let est = analyzer.estimate_peak(&weights, &act, &kv);
        CompiledResidencyPlan {
            plan_id: "test".into(),
            plan_hash: Default::default(),
            shape_class: ExecutionShapeClass::Decode1,
            required_weight_objects: weights,
            prefetch_schedule: Vec::new(),
            evictable_weight_objects: Vec::new(),
            activation_arena_requirements: act,
            kv_cache_requirements: kv,
            peak_memory_estimate: est,
            memory_admission_contract: MemoryAdmissionContract {
                minimum_required_bytes: 0,
                recommended_bytes: 0,
                graceful_degradation: false,
            },
        }
    }

    #[test]
    fn test_empty_weights_estimate() {
        let analyzer = PeakMemoryAnalyzer::new();
        let estimate = analyzer.estimate_peak(&[], &arena_reqs(0), &kv_reqs(0));
        assert_eq!(estimate.resident_weight_bytes, 0);
        assert_eq!(estimate.total_resident_bytes, 0);
    }

    #[test]
    fn test_single_weight_object() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(1024, ResidencyClass::MandatoryAtSessionStart)];
        let estimate = analyzer.estimate_peak(&weights, &arena_reqs(2048), &kv_reqs(4096));
        assert_eq!(estimate.resident_weight_bytes, 1024);
        assert_eq!(estimate.overhead_bytes, 716);
    }

    #[test]
    fn test_admission_passes_when_budget_exceeds_peak() {
        let analyzer = PeakMemoryAnalyzer::new();
        let weights = vec![weight(100, ResidencyClass::MandatoryAtSessionStart)];
        let plan = make_plan(weights, arena_reqs(100), kv_reqs(100));
        let contract = analyzer.check_admission(&plan, 500);
        assert_eq!(contract.minimum_required_bytes, 330);
        assert!(!contract.graceful_degradation);
    }
}
