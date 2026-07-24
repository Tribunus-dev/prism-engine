//! Evolution budget tracking components.
//!
//! Defines per-search budget constraints for candidate evolution:
//! wall time, memory, energy, and candidate count limits.

use prism_ecs_core::Component;

/// Budget constraints for one evolutionary search session.
///
/// Attach this component to the search entity to enforce resource
/// limits during candidate evolution.
#[derive(Debug, Clone)]
pub struct EvolutionBudget {
    /// Maximum number of candidates to evaluate.
    pub max_candidates: usize,
    /// Maximum wall time in milliseconds.
    pub max_wall_time_ms: u64,
    /// Maximum memory in bytes for the search state.
    pub max_memory_bytes: u64,
    /// Maximum energy budget in microjoules (where measurable).
    pub max_energy_uj: u64,
}

impl Component for EvolutionBudget {}

impl EvolutionBudget {
    /// A generous budget for full exploration.
    pub fn generous() -> Self {
        Self {
            max_candidates: 10_000,
            max_wall_time_ms: 600_000,           // 10 minutes
            max_memory_bytes: 512 * 1024 * 1024, // 512 MiB
            max_energy_uj: 100_000_000,          // 100 J
        }
    }

    /// A tight budget for quick validation runs.
    pub fn quick() -> Self {
        Self {
            max_candidates: 100,
            max_wall_time_ms: 30_000,           // 30 seconds
            max_memory_bytes: 64 * 1024 * 1024, // 64 MiB
            max_energy_uj: 10_000_000,          // 10 J
        }
    }

    /// Whether we've exceeded any budget limit.
    pub fn is_exhausted(
        &self,
        candidates_used: usize,
        wall_time_ms: u64,
        memory_bytes: u64,
        energy_uj: u64,
    ) -> bool {
        candidates_used >= self.max_candidates
            || wall_time_ms >= self.max_wall_time_ms
            || memory_bytes >= self.max_memory_bytes
            || energy_uj >= self.max_energy_uj
    }
}

impl Default for EvolutionBudget {
    fn default() -> Self {
        Self::generous()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generous_budget_allows_large_search() {
        let budget = EvolutionBudget::generous();
        assert!(!budget.is_exhausted(5000, 300_000, 256 * 1024 * 1024, 50_000_000));
    }

    #[test]
    fn budget_exhausted_by_candidates() {
        let budget = EvolutionBudget::generous();
        assert!(budget.is_exhausted(10_000, 0, 0, 0));
        assert!(budget.is_exhausted(10_001, 0, 0, 0));
    }

    #[test]
    fn budget_exhausted_by_wall_time() {
        let budget = EvolutionBudget::generous();
        assert!(!budget.is_exhausted(0, 599_999, 0, 0));
        assert!(budget.is_exhausted(0, 600_000, 0, 0));
    }

    #[test]
    fn quick_budget_tight() {
        let budget = EvolutionBudget::quick();
        assert!(budget.is_exhausted(101, 0, 0, 0));
        assert!(budget.is_exhausted(0, 31_000, 0, 0));
    }
}
