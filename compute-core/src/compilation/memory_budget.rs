//! Memory budget planner for the M1 16 GB unified-memory system.
//!
//! Establishes a hard process budget of 10.0 GB by default, with a 10.75 GB
//! emergency ceiling. Each compile region has a declared MemoryPlan with
//! resident/transient/peak bytes and a spill policy. The scheduler computes
//! predicted peak live bytes before dispatch and must react before memory
//! pressure becomes swap.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Region kind ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionKind {
    DenseTeacher,
    TernaryCandidate,
    ActivationFrontier,
    AccelerateWorkspace,
    ReceiptBuffer,
    Contingency,
}

// ── Spill policy ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpillPolicy {
    NoSpill,
    SpillOldestSealed,
    ReduceMicrobatch,
    SerializeProvider,
}

// ── Memory plan ─────────────────────────────────────────────────────────────

/// Declared memory plan for one region.
///
/// The scheduler computes predicted peak live bytes before dispatch. If the
/// sum exceeds the process budget, it must either reduce microbatch size,
/// serialize providers, spill the oldest sealed frontier shard to disk, or
/// choose a lower-memory policy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub region_kind: RegionKind,
    pub resident_bytes: u64,
    pub transient_bytes: u64,
    pub peak_bytes: u64,
    pub spill_policy: SpillPolicy,
    pub fallback_microbatch_sizes: Vec<usize>,
}

// ── Budget check result ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetCheck {
    pub fits: bool,
    pub suggested_actions: Vec<SpillPolicy>,
    pub predicted_peak: u64,
    pub headroom_bytes: u64,
}

// ── Memory budget ───────────────────────────────────────────────────────────

/// Hard process budget with per-region ceilings.
///
/// Defaults for the M1 16 GB machine (the primary development target):
/// - Process budget:   10.0 GB
/// - Emergency ceiling: 10.75 GB
/// - Dense teacher:     3.25 GB
/// - Ternary candidate:  2.75 GB
/// - Activation frontier: 2.0 GB
/// - Accelerate workspaces: 512 MB
/// - Receipt + digests:  256 MB
/// - Contingency:        ~1.2 GB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub process_budget_bytes: u64,
    pub emergency_ceiling_bytes: u64,
    pub per_region_ceilings: HashMap<RegionKind, u64>,
}

impl MemoryBudget {
    /// Default budget for an M1 with 16 GB unified memory.
    ///
    /// The remaining ~5.25 GB of system RAM stays available for macOS, Core ML
    /// compilation internals, Metal driver allocations, filesystem cache, and
    /// normal desktop use.
    pub fn m1_16gb_default() -> Self {
        let mut ceilings = HashMap::new();
        ceilings.insert(RegionKind::DenseTeacher, 3_250_000_000);          // 3.25 GB
        ceilings.insert(RegionKind::TernaryCandidate, 2_750_000_000);     // 2.75 GB
        ceilings.insert(RegionKind::ActivationFrontier, 2_000_000_000);   // 2.0 GB
        ceilings.insert(RegionKind::AccelerateWorkspace, 512_000_000);    // 512 MB
        ceilings.insert(RegionKind::ReceiptBuffer, 256_000_000);          // 256 MB
        ceilings.insert(RegionKind::Contingency, 1_200_000_000);          // ~1.2 GB

        MemoryBudget {
            process_budget_bytes: 10_000_000_000,      // 10.0 GB
            emergency_ceiling_bytes: 10_750_000_000,    // 10.75 GB
            per_region_ceilings: ceilings,
        }
    }

    /// Check whether a set of memory plans fits within the budget.
    ///
    /// Returns a `BudgetCheck` with suggested actions if the budget is exceeded.
    pub fn check_plans(&self, plans: &[MemoryPlan], current_usage: u64) -> BudgetCheck {
        let mut predicted_peak = current_usage;
        let mut suggested_actions = Vec::new();

        for plan in plans {
            predicted_peak = predicted_peak.saturating_add(plan.peak_bytes);

            // Check per-region ceiling
            if let Some(&ceiling) = self.per_region_ceilings.get(&plan.region_kind) {
                if plan.peak_bytes > ceiling {
                    suggested_actions.push(SpillPolicy::ReduceMicrobatch);
                }
            }

            // Check spill policy
            match plan.spill_policy {
                SpillPolicy::SpillOldestSealed => {
                    suggested_actions.push(SpillPolicy::SpillOldestSealed);
                }
                SpillPolicy::SerializeProvider => {
                    suggested_actions.push(SpillPolicy::SerializeProvider);
                }
                _ => {}
            }
        }

        let fits = predicted_peak <= self.process_budget_bytes;
        if !fits {
            if !suggested_actions.contains(&SpillPolicy::ReduceMicrobatch) {
                suggested_actions.push(SpillPolicy::ReduceMicrobatch);
            }
        }

        let headroom_bytes = if predicted_peak > self.emergency_ceiling_bytes {
            0
        } else {
            self.emergency_ceiling_bytes - predicted_peak
        };

        BudgetCheck {
            fits,
            suggested_actions,
            predicted_peak,
            headroom_bytes,
        }
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::m1_16gb_default()
    }
}
