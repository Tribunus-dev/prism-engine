//! Multi-objective fitness and quality-diversity primitives.
//!
//! Objectives are kept separate until selection/reporting.  This avoids the
//! information loss caused by reducing hardware, fidelity, memory, and
//! scheduling evidence to one weighted scalar.

use crate::evolution::foundation::CandidateGenome;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjectiveDirection {
    Maximize,
    Minimize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectiveValue {
    pub name: String,
    pub value: f64,
    pub direction: ObjectiveDirection,
}

impl ObjectiveValue {
    pub fn maximize(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            direction: ObjectiveDirection::Maximize,
        }
    }

    pub fn minimize(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
            direction: ObjectiveDirection::Minimize,
        }
    }

    pub fn normalized(&self) -> f64 {
        match self.direction {
            ObjectiveDirection::Maximize => self.value,
            ObjectiveDirection::Minimize => -self.value,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObjectiveVector {
    pub values: Vec<ObjectiveValue>,
}

impl ObjectiveVector {
    pub fn new(values: Vec<ObjectiveValue>) -> Self {
        Self { values }
    }

    pub fn dominates(&self, other: &Self) -> bool {
        if self.values.len() != other.values.len() {
            return false;
        }
        let mut strictly_better = false;
        for (a, b) in self.values.iter().zip(&other.values) {
            if a.name != b.name
                || a.direction != b.direction
                || !a.value.is_finite()
                || !b.value.is_finite()
            {
                return false;
            }
            if a.normalized() < b.normalized() {
                return false;
            }
            strictly_better |= a.normalized() > b.normalized();
        }
        strictly_better
    }

    pub fn scalar_compatibility_score(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.values.iter().map(|v| v.normalized()).sum();
        (sum / self.values.len() as f64).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BehaviorDescriptor {
    pub representation_family: u8,
    pub backend_split: u8,
    pub fusion_complexity: u8,
    pub memory_residency: u8,
    pub scheduler_behavior: u8,
}

impl BehaviorDescriptor {
    pub fn from_genome(genome: &CandidateGenome) -> Self {
        use crate::evolution::foundation::{FusionAxis, RepresentationAxis};
        let representation_family = match genome.representation {
            RepresentationAxis::Binary1 => 0,
            RepresentationAxis::Ternary158 => 1,
            RepresentationAxis::Int4 | RepresentationAxis::Nf4 => 2,
            RepresentationAxis::Int8 | RepresentationAxis::Nf8 => 3,
            _ => 4,
        };
        let fusion_complexity = match genome.fusion {
            FusionAxis::None => 0,
            FusionAxis::ElementWise => 1,
            FusionAxis::KernelFusion => 2,
        };
        let threads = genome
            .metal_geometry
            .threadgroup_width
            .saturating_mul(genome.metal_geometry.threadgroup_height);
        Self {
            representation_family,
            backend_split: 0,
            fusion_complexity,
            memory_residency: (genome.memory.shared_memory_bytes / 65536).min(3) as u8,
            scheduler_behavior: if threads >= 256 {
                2
            } else if threads >= 64 {
                1
            } else {
                0
            },
        }
    }

    pub fn from_execution(
        genome: &CandidateGenome,
        ane_score: f64,
        metal_score: f64,
        measured: bool,
        latency_ms: Option<f64>,
    ) -> Self {
        let mut descriptor = Self::from_genome(genome);
        let total = (ane_score.max(0.0) + metal_score.max(0.0)).max(f64::EPSILON);
        let ane_share = ane_score.max(0.0) / total;
        descriptor.backend_split = if !measured {
            0
        } else if ane_share >= 0.75 {
            3
        } else if ane_share >= 0.55 {
            2
        } else if ane_share >= 0.35 {
            1
        } else {
            0
        };
        descriptor.scheduler_behavior = match latency_ms {
            Some(latency) if latency.is_finite() && latency <= 1.0 => 2,
            Some(latency) if latency.is_finite() && latency <= 5.0 => 1,
            _ => descriptor.scheduler_behavior,
        };
        descriptor
    }

    /// Manhattan distance across compiler-behavior dimensions. A distance of
    /// zero means that two candidates occupy the same behavioral niche.
    pub fn distance(&self, other: &Self) -> u32 {
        [
            self.representation_family,
            self.backend_split,
            self.fusion_complexity,
            self.memory_residency,
            self.scheduler_behavior,
        ]
        .into_iter()
        .zip([
            other.representation_family,
            other.backend_split,
            other.fusion_complexity,
            other.memory_residency,
            other.scheduler_behavior,
        ])
        .map(|(left, right)| left.abs_diff(right) as u32)
        .sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveEntry {
    pub genome: CandidateGenome,
    pub objectives: ObjectiveVector,
    pub descriptor: BehaviorDescriptor,
    pub generation: u64,
    pub novelty: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityDiversityArchive {
    pub cells: std::collections::HashMap<BehaviorDescriptor, ArchiveEntry>,
}

impl QualityDiversityArchive {
    pub fn insert(&mut self, mut entry: ArchiveEntry) -> bool {
        entry.novelty = self
            .cells
            .keys()
            .map(|descriptor| descriptor.distance(&entry.descriptor))
            .min()
            .unwrap_or(5) as f64
            / 5.0;
        let replace = self
            .cells
            .get(&entry.descriptor)
            .map(|old| entry.objectives.dominates(&old.objectives))
            .unwrap_or(true);
        if replace {
            self.cells.insert(entry.descriptor, entry);
        }
        replace
    }

    pub fn elites(&self) -> impl Iterator<Item = &ArchiveEntry> {
        self.cells.values()
    }

    /// Return archive entries ordered by nondomination rank and crowding.
    /// This is the selection authority for the research search path; scalar
    /// compatibility scores are used only as deterministic tie-breakers.
    pub fn ranked_elites(&self) -> Vec<&ArchiveEntry> {
        let entries: Vec<&ArchiveEntry> = self.cells.values().collect();
        let mut ranks = vec![usize::MAX; entries.len()];
        let mut remaining: Vec<usize> = (0..entries.len()).collect();
        let mut rank = 0;
        while !remaining.is_empty() {
            let front: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&candidate| {
                    !remaining.iter().copied().any(|other| {
                        other != candidate
                            && entries[other]
                                .objectives
                                .dominates(&entries[candidate].objectives)
                    })
                })
                .collect();
            if front.is_empty() {
                break;
            }
            for &index in &front {
                ranks[index] = rank;
            }
            remaining.retain(|index| !front.contains(index));
            rank += 1;
        }
        let mut ranked: Vec<usize> = (0..entries.len()).collect();
        ranked.sort_by(|&a, &b| {
            ranks[a]
                .cmp(&ranks[b])
                .then_with(|| entries[b].novelty.total_cmp(&entries[a].novelty))
                .then_with(|| {
                    entries[b]
                        .objectives
                        .scalar_compatibility_score()
                        .total_cmp(&entries[a].objectives.scalar_compatibility_score())
                })
        });
        ranked.into_iter().map(|index| entries[index]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::foundation::CandidateGenome;

    #[test]
    fn mixed_direction_objectives_preserve_dominance() {
        let better = ObjectiveVector::new(vec![
            ObjectiveValue::maximize("fidelity", 0.9),
            ObjectiveValue::minimize("latency", 10.0),
        ]);
        let worse = ObjectiveVector::new(vec![
            ObjectiveValue::maximize("fidelity", 0.8),
            ObjectiveValue::minimize("latency", 12.0),
        ]);
        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
    }

    #[test]
    fn incompatible_objective_schemas_never_dominate() {
        let left = ObjectiveVector::new(vec![ObjectiveValue::maximize("fidelity", 1.0)]);
        let right = ObjectiveVector::new(vec![ObjectiveValue::maximize("throughput", 0.1)]);
        assert!(!left.dominates(&right));
        assert!(!right.dominates(&left));
    }

    #[test]
    fn archive_replaces_only_with_a_dominating_elite() {
        let genome = CandidateGenome::new();
        let descriptor = BehaviorDescriptor::from_genome(&genome);
        let mut archive = QualityDiversityArchive::default();
        let entry = |fidelity| ArchiveEntry {
            genome: genome.clone(),
            objectives: ObjectiveVector::new(vec![ObjectiveValue::maximize("fidelity", fidelity)]),
            descriptor,
            generation: 0,
            novelty: 0.0,
        };
        assert!(archive.insert(entry(0.8)));
        assert!(!archive.insert(entry(0.7)));
        assert!(archive.insert(entry(0.9)));
        assert_eq!(archive.cells.len(), 1);
        assert_eq!(
            archive.elites().next().unwrap().objectives.values[0].value,
            0.9
        );
    }

    #[test]
    fn ranked_elites_prioritize_nondominated_tradeoffs() {
        let genome = CandidateGenome::new();
        let mut archive = QualityDiversityArchive::default();
        for (descriptor, fidelity, latency) in [
            (
                BehaviorDescriptor {
                    representation_family: 0,
                    backend_split: 0,
                    fusion_complexity: 0,
                    memory_residency: 0,
                    scheduler_behavior: 0,
                },
                0.9,
                0.9,
            ),
            (
                BehaviorDescriptor {
                    representation_family: 1,
                    backend_split: 1,
                    fusion_complexity: 1,
                    memory_residency: 1,
                    scheduler_behavior: 1,
                },
                0.8,
                0.8,
            ),
        ] {
            archive.insert(ArchiveEntry {
                genome: genome.clone(),
                objectives: ObjectiveVector::new(vec![
                    ObjectiveValue::maximize("fidelity", fidelity),
                    ObjectiveValue::maximize("latency", latency),
                ]),
                descriptor,
                generation: 0,
                novelty: 1.0,
            });
        }
        assert_eq!(archive.ranked_elites().len(), 2);
    }

    #[test]
    fn archive_records_behavioral_novelty() {
        let first = CandidateGenome::new();
        let mut second = first.clone();
        second.memory.shared_memory_bytes = 262_144;
        let mut archive = QualityDiversityArchive::default();
        let objectives = |genome: &CandidateGenome| ArchiveEntry {
            genome: genome.clone(),
            objectives: ObjectiveVector::new(vec![ObjectiveValue::maximize("fitness", 0.8)]),
            descriptor: BehaviorDescriptor::from_genome(genome),
            generation: 0,
            novelty: 0.0,
        };
        archive.insert(objectives(&first));
        archive.insert(objectives(&second));
        assert!(archive.cells.values().any(|entry| entry.novelty > 0.0));
    }

    #[test]
    fn execution_descriptor_uses_measured_backend_behavior() {
        let mut genome = CandidateGenome::new();
        genome.metal_geometry.threadgroup_width = 8;
        genome.metal_geometry.threadgroup_height = 1;
        let ane_dominant = BehaviorDescriptor::from_execution(&genome, 0.95, 0.2, true, Some(0.8));
        let metal_dominant =
            BehaviorDescriptor::from_execution(&genome, 0.2, 0.95, true, Some(8.0));
        assert!(ane_dominant.backend_split > metal_dominant.backend_split);
        assert!(ane_dominant.scheduler_behavior > metal_dominant.scheduler_behavior);
    }

    #[test]
    fn finite_unmeasured_latency_is_json_serializable() {
        let vector = ObjectiveVector::new(vec![ObjectiveValue::minimize("latency_ms", f64::MAX)]);
        assert!(serde_json::to_string(&vector).is_ok());
    }
}
