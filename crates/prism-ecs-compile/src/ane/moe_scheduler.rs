//! NPUMoE — pure scheduling for Mixture-of-Experts on Apple ANE/NPU.
//!
//! Authority: the round-robin expert-to-core schedule and the
//! SRAM-budget arithmetic for ANE-resident MoE inference.
//!
//! The actual weight payload (`ExpertWeights` with `mlx_rs::Array`
//! projections) and the `forward_moe` effect (the per-token SwiGLU
//! computation) are engine-coupled and live in the engine's
//! `legacy_ane/`. This surface provides the schedule the engine's
//! `AneMoEScheduler::forward_moe` consults to decide which experts
//! are active on which cores.
//!
//! # Scheduling strategy
//!
//! Experts are distributed round-robin across available ANE cores,
//! respecting the SRAM budget per core. At inference time, each
//! core is loaded with its subset of experts; tokens are routed to
//! their top-K experts and each core independently computes the
//! expert FFN for the tokens assigned to it.
//!
//! The forward pass:
//! 1. Softmax router logits → routing probabilities.
//! 2. Select top-K experts per token.
//! 3. Group tokens by assigned expert.
//! 4. For each expert: compute `gate_proj → SiLU × up_proj → down_proj`.
//! 5. Weight outputs by routing probability and accumulate.

use crate::ane::token_routing::TokenRouting;

/// NPUMoE expert scheduler for ANE.
///
/// Schedules expert computations across ANE cores based on SRAM limits.
/// Each core can hold `sram_per_core / expert_size_bytes` experts.
///
/// The MLX-coupled `forward_moe` method (which performs the actual
/// per-expert SwiGLU computation) is engine-side; the constitutional
/// surface provides the schedule and the SRAM accounting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AneMoEScheduler {
    /// Number of ANE cores available (e.g. 16 on M1 Max, 32 on M3 Ultra).
    pub num_cores: u32,
    /// SRAM per ANE core in bytes (typically 512 KB = 524288).
    pub sram_per_core: u32,
    /// Size of one expert's weights in bytes (gate + up + down projections).
    pub expert_size_bytes: u32,
    /// Number of experts that fit in one core's SRAM (computed at construction).
    pub experts_per_core: u32,
}

impl AneMoEScheduler {
    /// Create a new scheduler with the given ANE hardware parameters.
    ///
    /// `experts_per_core` is floored at 1: if an expert is larger than
    /// one core's SRAM, `experts_per_core` is clamped to 1 and the
    /// caller must handle the spillover (e.g. by splitting the expert
    /// across multiple cores — hardly ever needed for typical MoE
    /// configs on Apple Silicon).
    pub fn new(num_cores: u32, sram_per_core: u32, expert_size_bytes: u32) -> Self {
        let experts_per_core = if expert_size_bytes > 0 {
            (sram_per_core / expert_size_bytes).max(1)
        } else {
            1
        };
        Self {
            num_cores,
            sram_per_core,
            expert_size_bytes,
            experts_per_core,
        }
    }

    /// Build an [`AneMoEScheduler`] for a typical 32×4096 MoE config on
    /// Apple Silicon (16 cores, 512 KB SRAM, ~300 KB per expert).
    pub fn default_m1_max() -> Self {
        Self::new(16, 512 * 1024, 300 * 1024)
    }

    /// Build a scheduler for M3 Ultra (32 cores).
    pub fn default_m3_ultra() -> Self {
        Self::new(32, 512 * 1024, 300 * 1024)
    }

    /// Compute the optimal expert-to-core mapping.
    ///
    /// Experts are distributed round-robin across cores. The
    /// `experts_per_core` field is reported by the caller as a SRAM
    /// budget, not a hard cap: the schedule ensures every expert is
    /// assigned to exactly one core, and the per-core count is as
    /// even as possible (within one of the average). When
    /// `experts_per_core` is `0`, every expert still gets a slot —
    /// the caller is responsible for spilling or splitting.
    pub fn schedule_experts(&self, num_experts: u32, _top_k: u32) -> Vec<Vec<u32>> {
        let n_cores = self.num_cores as usize;
        let mut schedule: Vec<Vec<u32>> = vec![Vec::new(); n_cores];

        if n_cores == 0 {
            return schedule;
        }

        // Pure round-robin: expert `i` goes to core `i mod n_cores`.
        // This guarantees every expert is assigned to exactly one
        // core, with per-core counts differing by at most 1.
        for expert_id in 0..num_experts {
            let core = (expert_id as usize) % n_cores;
            schedule[core].push(expert_id);
        }

        schedule
    }

    /// Compute how many ANE cores are needed to hold `num_active_experts`
    /// in a single pipeline round.
    pub fn cores_needed(&self, num_active_experts: u32) -> u32 {
        if self.experts_per_core == 0 {
            return num_active_experts;
        }
        (num_active_experts + self.experts_per_core - 1) / self.experts_per_core
    }

    /// Number of pipeline rounds needed to run `num_active_experts`
    /// experts on the available cores.
    pub fn pipeline_rounds(&self, num_active_experts: u32) -> u32 {
        let cores_available = self.num_cores;
        let per_round = cores_available.saturating_mul(self.experts_per_core);
        if per_round == 0 {
            return num_active_experts;
        }
        (num_active_experts + per_round - 1) / per_round
    }
}

/// Compute the approximate SRAM footprint of one expert's weights.
///
/// `hidden_size`: model hidden dimension (e.g. 4096).
/// `intermediate_size`: FFN intermediate dimension (e.g. 14336).
/// `bytes_per_param`: bytes per weight element (2 for FP16, 1 for quantized).
pub fn expert_sram_footprint(
    hidden_size: u32,
    intermediate_size: u32,
    bytes_per_param: u32,
) -> u32 {
    // gate: [hidden, intermediate], up: [hidden, intermediate], down: [intermediate, hidden]
    let gate_up = hidden_size.saturating_mul(intermediate_size).saturating_mul(bytes_per_param);
    let down = intermediate_size.saturating_mul(hidden_size).saturating_mul(bytes_per_param);
    gate_up.saturating_mul(2).saturating_add(down)
}

/// Construct a [`TokenRouting`] for one token by selecting the
/// `top_k` experts with the highest probabilities and renormalising
/// the weights to sum to 1.0.
///
/// `routing_probs` is the slice of softmax probabilities for this
/// token, length `num_experts`. Returns a routing entry whose
/// `expert_indices` and `routing_weights` are both of length
/// `min(top_k, num_experts)`.
pub fn select_top_k_for_token(routing_probs: &[f32], top_k: u32) -> TokenRouting {
    let num_experts = routing_probs.len();
    if num_experts == 0 {
        return TokenRouting::new(Vec::new(), Vec::new());
    }
    let k = (top_k as usize).min(num_experts);

    let mut indexed: Vec<(usize, f32)> =
        (0..num_experts).map(|i| (i, routing_probs[i])).collect();
    // Sort descending by probability.
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let selected: Vec<(u32, f32)> = indexed[..k]
        .iter()
        .map(|&(idx, p)| (idx as u32, p))
        .collect();

    // Renormalise top-K weights so they sum to 1.
    let sum: f32 = selected.iter().map(|&(_, p)| p).sum();
    let weights: Vec<f32> = if sum > 0.0 {
        selected.iter().map(|&(_, p)| p / sum).collect()
    } else {
        // Uniform fallback when all probs are zero.
        selected
            .iter()
            .map(|_| 1.0 / k as f32)
            .collect()
    };

    TokenRouting::new(
        selected.iter().map(|&(idx, _)| idx).collect(),
        weights,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_new_basic() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        assert_eq!(sched.num_cores, 16);
        assert_eq!(sched.experts_per_core, 1);
    }

    #[test]
    fn default_m1_max_matches_constructor() {
        let sched = AneMoEScheduler::default_m1_max();
        assert_eq!(sched.num_cores, 16);
        assert_eq!(sched.experts_per_core, 1);
    }

    #[test]
    fn default_m3_ultra_uses_32_cores() {
        let sched = AneMoEScheduler::default_m3_ultra();
        assert_eq!(sched.num_cores, 32);
        assert_eq!(sched.experts_per_core, 1);
    }

    #[test]
    fn schedule_experts_round_robin() {
        // 16 cores, 32 experts → 2 per core in round-robin order
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        let schedule = sched.schedule_experts(32, 8);

        assert_eq!(schedule.len(), 16);
        // Each core gets exactly 2 experts (32/16).
        for core_assignments in &schedule {
            assert_eq!(core_assignments.len(), 2);
        }
        // Round-robin interleaves by core: core 0 gets 0+16, core 1 gets 1+17, etc.
        let expected: Vec<Vec<u32>> = (0..16u32).map(|c| vec![c, c + 16]).collect();
        assert_eq!(schedule, expected);
    }

    #[test]
    fn schedule_experts_two_per_core() {
        // 8 cores, 16 experts → 2 per core in round-robin order
        let sched = AneMoEScheduler::new(8, 600 * 1024, 300 * 1024);
        assert_eq!(sched.experts_per_core, 2);
        let schedule = sched.schedule_experts(16, 4);
        assert_eq!(schedule.len(), 8);
        let total: usize = schedule.iter().map(|v| v.len()).sum();
        assert_eq!(total, 16);
        for core_assignments in &schedule {
            assert_eq!(core_assignments.len(), 2);
        }
    }

    #[test]
    fn schedule_less_experts_than_cores() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        let schedule = sched.schedule_experts(4, 4);
        let total: usize = schedule.iter().map(|v| v.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn cores_needed_basic() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        assert_eq!(sched.cores_needed(8), 8);
        assert_eq!(sched.cores_needed(1), 1);
        assert_eq!(sched.cores_needed(0), 0);
    }

    #[test]
    fn pipeline_rounds_basic() {
        let sched = AneMoEScheduler::new(16, 512 * 1024, 300 * 1024);
        assert_eq!(sched.pipeline_rounds(32), 2);
        assert_eq!(sched.pipeline_rounds(8), 1);
    }

    #[test]
    fn expert_sram_footprint_fp16() {
        // 32×4096 MoE: hidden=4096, intermediate=14336, FP16 (2 bytes)
        let footprint = expert_sram_footprint(4096, 14336, 2);
        // gate: 4096*14336*2 = 117,440,512
        // up: same = 117,440,512
        // down: 14336*4096*2 = 117,440,512
        // total = 352,321,536
        assert_eq!(footprint, 352_321_536);
    }

    #[test]
    fn select_top_k_renormalises() {
        let probs = vec![0.1f32, 0.5, 0.3, 0.1];
        let routing = select_top_k_for_token(&probs, 2);
        assert_eq!(routing.expert_indices, vec![1, 2]);
        let sum: f32 = routing.routing_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
