#[allow(unused_imports)]
use crate::ecs::Entity;

/// Search budget — time, memory, energy, candidate-count, and kill-bit limits.
/// Plan Section 2: "Bounded execution: Search and training missions carry
/// time, memory, energy, candidate-count, and kill-bit limits."
#[derive(Debug, Clone)]
pub struct SearchBudget {
    pub max_candidates: usize,
    pub max_generations: usize,
    pub max_wall_time_ms: u64,
    pub max_memory_bytes: u64,
    pub kill_bit: bool,
}

impl SearchBudget {
    pub fn unlimited() -> Self {
        Self {
            max_candidates: usize::MAX,
            max_generations: usize::MAX,
            max_wall_time_ms: u64::MAX,
            max_memory_bytes: u64::MAX,
            kill_bit: false,
        }
    }

    pub fn is_exhausted(&self, candidates_used: usize, wall_ms: u64) -> bool {
        if self.kill_bit {
            return true;
        }
        candidates_used >= self.max_candidates || wall_ms >= self.max_wall_time_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlimited_never_exhausted() {
        let budget = SearchBudget::unlimited();
        assert!(!budget.is_exhausted(0, 0));
        assert!(!budget.is_exhausted(usize::MAX - 1, u64::MAX - 1));
    }

    #[test]
    fn test_exhausted_by_candidates() {
        let budget = SearchBudget {
            max_candidates: 10,
            max_generations: 100,
            max_wall_time_ms: 60_000,
            max_memory_bytes: 1_000_000_000,
            kill_bit: false,
        };
        assert!(budget.is_exhausted(10, 0));
        assert!(budget.is_exhausted(100, 0));
    }

    #[test]
    fn test_exhausted_by_wall_time() {
        let budget = SearchBudget {
            max_candidates: 1000,
            max_generations: 1000,
            max_wall_time_ms: 100,
            max_memory_bytes: 1_000_000_000,
            kill_bit: false,
        };
        assert!(budget.is_exhausted(0, 100));
        assert!(budget.is_exhausted(0, 200));
    }

    #[test]
    fn test_not_exhausted_within_limits() {
        let budget = SearchBudget {
            max_candidates: 100,
            max_generations: 100,
            max_wall_time_ms: 60_000,
            max_memory_bytes: 1_000_000_000,
            kill_bit: false,
        };
        assert!(!budget.is_exhausted(50, 30_000));
    }

    #[test]
    fn test_kill_bit_immediately_exhausts() {
        let budget = SearchBudget {
            max_candidates: 1000,
            max_generations: 1000,
            max_wall_time_ms: 60_000,
            max_memory_bytes: 1_000_000_000,
            kill_bit: true,
        };
        assert!(budget.is_exhausted(0, 0));
    }
}
