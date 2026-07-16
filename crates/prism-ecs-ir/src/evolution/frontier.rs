//! Frontier exploration system for Pareto-optimal candidate discovery.
//!
//! Maintains the Pareto frontier across generations of the evolutionary
//! search. The system tracks which candidates dominate others and preserves
//! the non-dominated set for selection and reporting.

use crate::evolution::foundation::FitnessScore;
use prism_ecs_core::{Component, Entity};

/// Configuration for frontier exploration.
///
/// Attached to the frontier entity to control exploration behaviour.
#[derive(Debug, Clone)]
pub struct FrontierConfig {
    /// Maximum number of candidates on the frontier.
    pub max_frontier_size: usize,
    /// Whether to allow candidates with equal fitness to coexist.
    pub allow_equal_coexistence: bool,
}

impl Component for FrontierConfig {}

impl Default for FrontierConfig {
    fn default() -> Self {
        Self {
            max_frontier_size: 100,
            allow_equal_coexistence: false,
        }
    }
}

/// A single frontier entry: candidate entity with its fitness vector.
#[derive(Debug, Clone)]
pub struct FrontierEntry {
    /// The candidate entity.
    pub entity: Entity,
    /// Multi-dimensional fitness scores.
    pub fitness: Vec<FitnessScore>,
    /// Generation in which this entry was added.
    pub generation: u64,
}

/// The Pareto frontier resource.
///
/// Stores the set of non-dominated candidates discovered during the search.
/// A candidate A dominates candidate B if A is at least as good in every
/// dimension and strictly better in at least one.
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    /// Current frontier entries (always non-dominated).
    pub entries: Vec<FrontierEntry>,
    /// Number of dimensions in the fitness vector.
    pub num_dimensions: usize,
}

impl ParetoFrontier {
    /// Create a new empty frontier with the given number of fitness dimensions.
    pub fn new(num_dimensions: usize) -> Self {
        Self {
            entries: Vec::new(),
            num_dimensions,
        }
    }

    /// Check whether candidate `a` dominates candidate `b`.
    ///
    /// Returns `true` when `a` is at least as good as `b` in every dimension
    /// and strictly better in at least one dimension.
    pub fn dominates(a: &[FitnessScore], b: &[FitnessScore]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut better_in_any = false;
        for (score_a, score_b) in a.iter().zip(b.iter()) {
            if score_a.value() < score_b.value() {
                return false; // a is worse in this dimension
            }
            if score_a.value() > score_b.value() {
                better_in_any = true;
            }
        }
        better_in_any
    }

    /// Try to insert a candidate into the frontier.
    ///
    /// Returns `true` if the candidate was inserted (i.e., it is
    /// non-dominated by existing entries). Existing entries dominated
    /// by the new candidate are removed.
    pub fn insert(
        &mut self,
        entity: Entity,
        fitness: Vec<FitnessScore>,
        generation: u64,
        config: &FrontierConfig,
    ) -> bool {
        // Reject if any existing entry dominates the new candidate.
        for entry in &self.entries {
            if Self::dominates(&entry.fitness, &fitness) {
                return false;
            }
        }

        // Remove entries dominated by the new candidate.
        self.entries
            .retain(|entry| !Self::dominates(&fitness, &entry.fitness));

        // Prune equal-fitness duplicates when disallowed.
        if !config.allow_equal_coexistence {
            self.entries.retain(|entry| {
                !entry
                    .fitness
                    .iter()
                    .zip(fitness.iter())
                    .all(|(a, b)| (a.value() - b.value()).abs() < 1e-9)
            });
        }

        // Add the new entry.
        self.entries.push(FrontierEntry {
            entity,
            fitness,
            generation,
        });

        // Enforce max frontier size by removing oldest entries.
        while self.entries.len() > config.max_frontier_size {
            self.entries.remove(0);
        }

        true
    }

    /// Number of entries on the frontier.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the frontier is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the best entry by single-dimensional fitness.
    pub fn best_by_dimension(&self, dim: usize) -> Option<&FrontierEntry> {
        if dim >= self.num_dimensions {
            return None;
        }
        self.entries.iter().max_by(|a, b| {
            a.fitness[dim]
                .value()
                .partial_cmp(&b.fitness[dim].value())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// Frontier exploration system.
///
/// Maintains the Pareto frontier across generations. Attach to an entity
/// with a `FrontierConfig` component.
#[derive(Debug, Clone)]
pub struct FrontierExplorationSystem {
    pub frontier: ParetoFrontier,
}

impl FrontierExplorationSystem {
    /// Create a new frontier system with the given number of fitness dimensions.
    pub fn new(num_dimensions: usize) -> Self {
        Self {
            frontier: ParetoFrontier::new(num_dimensions),
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fitness(values: &[f64]) -> Vec<FitnessScore> {
        values.iter().map(|&v| FitnessScore::new(v)).collect()
    }

    #[test]
    fn dominance_basic() {
        let a = fitness(&[0.9, 0.8]);
        let b = fitness(&[0.7, 0.6]);
        assert!(ParetoFrontier::dominates(&a, &b));
        assert!(!ParetoFrontier::dominates(&b, &a));
    }

    #[test]
    fn no_dominance_when_worse_in_one_dimension() {
        let a = fitness(&[0.9, 0.3]);
        let b = fitness(&[0.7, 0.8]);
        assert!(!ParetoFrontier::dominates(&a, &b));
        assert!(!ParetoFrontier::dominates(&b, &a));
    }

    #[test]
    fn frontier_insertion() {
        let config = FrontierConfig::default();
        let mut frontier = ParetoFrontier::new(2);

        let e1 = Entity(1, 0);
        let e2 = Entity(2, 0);

        assert!(frontier.insert(e1, fitness(&[0.8, 0.7]), 0, &config));
        assert!(frontier.insert(e2, fitness(&[0.9, 0.8]), 0, &config));

        // e2 dominates e1, so e1 should be removed
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier.entries[0].entity, e2);
    }

    #[test]
    fn frontier_rejects_dominated() {
        let config = FrontierConfig::default();
        let mut frontier = ParetoFrontier::new(2);

        let e1 = Entity(1, 0);
        let e2 = Entity(2, 0);

        assert!(frontier.insert(e1, fitness(&[0.9, 0.8]), 0, &config));
        // e2 is dominated by e1, so it is rejected
        assert!(!frontier.insert(e2, fitness(&[0.7, 0.6]), 0, &config));
    }
}
