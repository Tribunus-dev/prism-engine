//! Compile-path planning — memory budget, region catalogue, profitability
//! scoring, and packaging receipts.
//!
//! This module owns the canonical authority for the four planning-time
//! decisions that happen between graph construction and kernel lowering:
//!
//! 1. **ANE eligibility** — for each `CompilePhaseDescriptor` tensor in
//!    the world, evaluate whether the shape / operator family meets the
//!    `RegionCatalogue` for ANE placement.
//! 2. **Memory budget check** — given a set of per-region `MemoryPlan`
//!    values, evaluate whether they fit the process budget.
//! 3. **Region catalogue & planner** — for each `Model` entity, build
//!    `RegionPlan` components by walking the catalogue entries and
//!    selecting a backend per family.
//! 4. **Packaging receipt** — for each `Executable` entity, produce a
//!    `PhaseExecutionRecord` and attach the profitability summary.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The IR (owned by `prism-ecs-ir` / `prism-ecs-compile::uop`).
//! - The kernel lowerer (owned by `prism-ecs-kernel`).
//! - The region's runtime placement (owned by `prism-ecs-runtime`).
//!
//! All state authority is staged through `WorldTxn` and durable
//! components. The module never reads or mutates the world outside
//! a `WorldTxn` boundary.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RegionKind {
    DenseTeacher,
    TernaryCandidate,
    ActivationFrontier,
    AccelerateWorkspace,
    ReceiptBuffer,
    CoreMLReserve,
    Contingency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpillPolicy {
    NoSpill,
    SpillOldestSealed,
    ReduceMicrobatch,
    SerializeProvider,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub region_kind: RegionKind,
    pub resident_bytes: u64,
    pub transient_bytes: u64,
    pub peak_bytes: u64,
    pub spill_policy: SpillPolicy,
    pub fallback_microbatch_sizes: Vec<usize>,
}

impl prism_ecs_core::Component for MemoryPlan {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetCheck {
    pub fits: bool,
    pub suggested_actions: Vec<SpillPolicy>,
    pub predicted_peak: u64,
    pub headroom_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBudget {
    pub process_budget_bytes: u64,
    pub emergency_ceiling_bytes: u64,
    pub per_region_ceilings: BTreeMap<RegionKind, u64>,
}

impl MemoryBudget {
    pub fn m1_16gb_default() -> Self {
        let mut ceilings: BTreeMap<RegionKind, u64> = BTreeMap::new();
        ceilings.insert(RegionKind::DenseTeacher, 3_250_000_000);
        ceilings.insert(RegionKind::TernaryCandidate, 2_750_000_000);
        ceilings.insert(RegionKind::ActivationFrontier, 2_000_000_000);
        ceilings.insert(RegionKind::AccelerateWorkspace, 512_000_000);
        ceilings.insert(RegionKind::ReceiptBuffer, 256_000_000);
        ceilings.insert(RegionKind::CoreMLReserve, 750_000_000);
        ceilings.insert(RegionKind::Contingency, 500_000_000);

        Self {
            process_budget_bytes: 10_000_000_000,
            emergency_ceiling_bytes: 10_750_000_000,
            per_region_ceilings: ceilings,
        }
    }

    pub fn check_plans(&self, plans: &[MemoryPlan], current_usage: u64) -> BudgetCheck {
        let mut predicted_peak = current_usage;
        let mut suggested_actions = Vec::new();

        for plan in plans {
            predicted_peak = predicted_peak.saturating_add(plan.peak_bytes);

            if let Some(&ceiling) = self.per_region_ceilings.get(&plan.region_kind) {
                if plan.peak_bytes > ceiling
                    && !suggested_actions.contains(&SpillPolicy::ReduceMicrobatch)
                {
                    suggested_actions.push(SpillPolicy::ReduceMicrobatch);
                }
            }

            match plan.spill_policy {
                SpillPolicy::SpillOldestSealed => {
                    if !suggested_actions.contains(&SpillPolicy::SpillOldestSealed) {
                        suggested_actions.push(SpillPolicy::SpillOldestSealed);
                    }
                }
                SpillPolicy::SerializeProvider => {
                    if !suggested_actions.contains(&SpillPolicy::SerializeProvider) {
                        suggested_actions.push(SpillPolicy::SerializeProvider);
                    }
                }
                _ => {}
            }
        }

        let fits = predicted_peak <= self.process_budget_bytes;
        if !fits && !suggested_actions.contains(&SpillPolicy::ReduceMicrobatch) {
            suggested_actions.push(SpillPolicy::ReduceMicrobatch);
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProfitabilityScore {
    pub score: f64,
    pub confidence: f64,
    pub reason: String,
}

impl prism_ecs_core::Component for ProfitabilityScore {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionGate {
    pub name: String,
    pub passed: bool,
    pub evidence: Option<String>,
}

impl prism_ecs_core::Component for AdmissionGate {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionPlan {
    pub region_id: String,
    pub backend: String,
    pub schedule: Vec<String>,
}

impl prism_ecs_core::Component for RegionPlan {}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PlanningError {
    #[error("region `{0}` is missing from the per-region ceiling map")]
    UnknownRegion(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionAdmission {
    CoreAiProduction,
    MetalProduction,
    CpuProduction,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionCatalogueEntry {
    pub operator_family: String,
    pub primary_admission: RegionAdmission,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionCatalogue {
    pub entries: Vec<RegionCatalogueEntry>,
}

impl RegionCatalogue {
    pub fn fp16_alpha() -> Self {
        use RegionAdmission::*;
        Self {
            entries: vec![
                RegionCatalogueEntry {
                    operator_family: "matmul".into(),
                    primary_admission: MetalProduction,
                },
                RegionCatalogueEntry {
                    operator_family: "mlp_gate_up".into(),
                    primary_admission: MetalProduction,
                },
                RegionCatalogueEntry {
                    operator_family: "rms_norm".into(),
                    primary_admission: CoreAiProduction,
                },
                RegionCatalogueEntry {
                    operator_family: "softmax".into(),
                    primary_admission: CoreAiProduction,
                },
                RegionCatalogueEntry {
                    operator_family: "embedding".into(),
                    primary_admission: CpuProduction,
                },
                RegionCatalogueEntry {
                    operator_family: "rope".into(),
                    primary_admission: CoreAiProduction,
                },
            ],
        }
    }

    pub fn coreai_production_ops(&self) -> Vec<&RegionCatalogueEntry> {
        self.entries
            .iter()
            .filter(|e| matches!(e.primary_admission, RegionAdmission::CoreAiProduction))
            .collect()
    }

    pub fn backend_for(&self, operator_family: &str) -> &'static str {
        self.entries
            .iter()
            .find(|e| e.operator_family == operator_family)
            .map(|e| match e.primary_admission {
                RegionAdmission::CoreAiProduction => "ane",
                RegionAdmission::MetalProduction => "metal",
                RegionAdmission::CpuProduction => "cpu",
                RegionAdmission::Fallback => "fallback",
            })
            .unwrap_or("fallback")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseExecutionRecord {
    pub phase_id: u64,
    pub phase_type: String,
    pub provider: String,
    pub started_at_ns: u64,
    pub completed_at_ns: u64,
    pub input_slots: Vec<u32>,
    pub output_slots: Vec<u32>,
    pub peak_bytes: u64,
    pub transition_count: u32,
}

impl prism_ecs_core::Component for PhaseExecutionRecord {}

pub fn default_executable_memory_plan() -> MemoryPlan {
    MemoryPlan {
        region_kind: RegionKind::DenseTeacher,
        resident_bytes: 1_500_000_000,
        transient_bytes: 1_000_000_000,
        peak_bytes: 2_500_000_000,
        spill_policy: SpillPolicy::SpillOldestSealed,
        fallback_microbatch_sizes: vec![1],
    }
}

pub fn profitability_from_budget(check: &BudgetCheck) -> ProfitabilityScore {
    ProfitabilityScore {
        score: if check.fits { 1.0 } else { 0.0 },
        confidence: 0.9,
        reason: format!(
            "mem_budget: peak={}, headroom={}, actions={:?}",
            check.predicted_peak, check.headroom_bytes, check.suggested_actions,
        ),
    }
}

pub fn profitability_from_catalogue(
    region_count: usize,
    ane_count: usize,
) -> ProfitabilityScore {
    let score = if region_count == 0 {
        0.0
    } else {
        ane_count as f64 / region_count as f64
    };
    ProfitabilityScore {
        score,
        confidence: 0.95,
        reason: format!(
            "catalogue: {} regions, {} ANE-eligible ops",
            region_count, ane_count,
        ),
    }
}

pub fn profitability_from_receipt(record: &PhaseExecutionRecord) -> ProfitabilityScore {
    ProfitabilityScore {
        score: 1.0,
        confidence: 1.0,
        reason: format!(
            "receipt: phase={}, peak={}B, transitions={}",
            record.phase_type, record.peak_bytes, record.transition_count,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m1_16gb_default_has_expected_ceilings() {
        let b = MemoryBudget::m1_16gb_default();
        assert_eq!(b.per_region_ceilings[&RegionKind::DenseTeacher], 3_250_000_000);
        assert_eq!(
            b.per_region_ceilings[&RegionKind::TernaryCandidate],
            2_750_000_000
        );
        assert!(b.per_region_ceilings.contains_key(&RegionKind::Contingency));
    }

    #[test]
    fn check_plans_fits_when_under_budget() {
        let b = MemoryBudget::m1_16gb_default();
        let plans = vec![MemoryPlan {
            region_kind: RegionKind::DenseTeacher,
            resident_bytes: 1_000_000_000,
            transient_bytes: 500_000_000,
            peak_bytes: 1_500_000_000,
            spill_policy: SpillPolicy::NoSpill,
            fallback_microbatch_sizes: vec![1],
        }];
        let check = b.check_plans(&plans, 0);
        assert!(check.fits);
        assert!(check.suggested_actions.is_empty());
        assert_eq!(check.predicted_peak, 1_500_000_000);
    }

    #[test]
    fn check_plans_fails_when_over_budget() {
        let b = MemoryBudget::m1_16gb_default();
        let plans = vec![MemoryPlan {
            region_kind: RegionKind::DenseTeacher,
            resident_bytes: 8_000_000_000,
            transient_bytes: 4_000_000_000,
            peak_bytes: 12_000_000_000,
            spill_policy: SpillPolicy::NoSpill,
            fallback_microbatch_sizes: vec![1],
        }];
        let check = b.check_plans(&plans, 0);
        assert!(!check.fits);
        assert!(check.suggested_actions.contains(&SpillPolicy::ReduceMicrobatch));
    }

    #[test]
    fn check_plans_surfaces_per_region_violation() {
        let b = MemoryBudget::m1_16gb_default();
        let plans = vec![MemoryPlan {
            region_kind: RegionKind::DenseTeacher,
            resident_bytes: 4_000_000_000,
            transient_bytes: 0,
            peak_bytes: 4_000_000_000,
            spill_policy: SpillPolicy::NoSpill,
            fallback_microbatch_sizes: vec![1],
        }];
        let check = b.check_plans(&plans, 0);
        assert!(check.suggested_actions.contains(&SpillPolicy::ReduceMicrobatch));
    }

    #[test]
    fn check_plans_propagates_spill_oldest_sealed() {
        let b = MemoryBudget::m1_16gb_default();
        let plans = vec![MemoryPlan {
            region_kind: RegionKind::ActivationFrontier,
            resident_bytes: 100_000_000,
            transient_bytes: 0,
            peak_bytes: 100_000_000,
            spill_policy: SpillPolicy::SpillOldestSealed,
            fallback_microbatch_sizes: vec![1],
        }];
        let check = b.check_plans(&plans, 0);
        assert!(check.suggested_actions.contains(&SpillPolicy::SpillOldestSealed));
    }

    #[test]
    fn check_plans_headroom_caps_at_emergency_ceiling() {
        let b = MemoryBudget::m1_16gb_default();
        let plans = vec![MemoryPlan {
            region_kind: RegionKind::DenseTeacher,
            resident_bytes: 11_000_000_000,
            transient_bytes: 0,
            peak_bytes: 11_000_000_000,
            spill_policy: SpillPolicy::NoSpill,
            fallback_microbatch_sizes: vec![1],
        }];
        let check = b.check_plans(&plans, 0);
        assert!(!check.fits);
        assert_eq!(check.headroom_bytes, 0);
    }

    #[test]
    fn catalogue_alpha_classifies_ane_ops() {
        let c = RegionCatalogue::fp16_alpha();
        let ane = c.coreai_production_ops();
        let families: Vec<&str> = ane.iter().map(|e| e.operator_family.as_str()).collect();
        assert!(families.contains(&"rms_norm"));
        assert!(families.contains(&"softmax"));
        assert!(families.contains(&"rope"));
    }

    #[test]
    fn catalogue_backend_for_returns_known_family() {
        let c = RegionCatalogue::fp16_alpha();
        assert_eq!(c.backend_for("matmul"), "metal");
        assert_eq!(c.backend_for("rms_norm"), "ane");
        assert_eq!(c.backend_for("unknown_op"), "fallback");
    }

    #[test]
    fn default_executable_memory_plan_is_conservative() {
        let p = default_executable_memory_plan();
        assert_eq!(p.region_kind, RegionKind::DenseTeacher);
        assert_eq!(p.peak_bytes, 2_500_000_000);
        assert_eq!(p.fallback_microbatch_sizes, vec![1]);
        assert!(matches!(p.spill_policy, SpillPolicy::SpillOldestSealed));
    }

    #[test]
    fn profitability_from_budget_marks_unfit() {
        let check = BudgetCheck {
            fits: false,
            suggested_actions: vec![SpillPolicy::ReduceMicrobatch],
            predicted_peak: 12_000_000_000,
            headroom_bytes: 0,
        };
        let s = profitability_from_budget(&check);
        assert_eq!(s.score, 0.0);
        assert!(s.reason.contains("mem_budget"));
        assert!(s.reason.contains("peak=12000000000"));
    }

    #[test]
    fn profitability_from_catalogue_is_ratio() {
        let s = profitability_from_catalogue(8, 2);
        assert!((s.score - 0.25).abs() < 1e-9);
        let s2 = profitability_from_catalogue(0, 0);
        assert_eq!(s2.score, 0.0);
    }

    #[test]
    fn profitability_from_receipt_captures_peak() {
        let r = PhaseExecutionRecord {
            phase_id: 1,
            phase_type: "compilation".into(),
            provider: "coreml".into(),
            started_at_ns: 0,
            completed_at_ns: 1_000_000,
            input_slots: vec![0, 1],
            output_slots: vec![2],
            peak_bytes: 256_000_000,
            transition_count: 3,
        };
        let s = profitability_from_receipt(&r);
        assert_eq!(s.score, 1.0);
        assert!(s.reason.contains("peak=256000000"));
        assert!(s.reason.contains("transitions=3"));
    }

    #[test]
    fn ceiling_iteration_is_sorted_for_determinism() {
        let b = MemoryBudget::m1_16gb_default();
        let keys: Vec<RegionKind> = b.per_region_ceilings.keys().copied().collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
    }
}
