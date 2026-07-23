//! Evidence-backed evolutionary memory and optional proposal interfaces.

use crate::evolution::chromosome::{Chromosome, GenomeChromosomes};
use crate::evolution::foundation::CandidateGenome;
use crate::evolution::objectives::{BehaviorDescriptor, ObjectiveVector};
use crate::evolution::variation::VariationOperator;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EvolutionContextKey {
    pub hardware: String,
    pub model_family: String,
    pub tensor_family: String,
}

impl EvolutionContextKey {
    pub fn matches(&self, query: &Self) -> bool {
        fn matches_field(value: &str, query: &str) -> bool {
            value == "*" || query == "*" || value == query
        }
        matches_field(&self.hardware, &query.hardware)
            && matches_field(&self.model_family, &query.model_family)
            && matches_field(&self.tensor_family, &query.tensor_family)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionReceipt {
    pub parent_digest: String,
    pub child_digest: String,
    pub operator: VariationOperator,
    pub context: EvolutionContextKey,
    pub descriptor: BehaviorDescriptor,
    pub objectives: ObjectiveVector,
    pub improvement: f64,
    pub measurement_receipt_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionaryMemory {
    pub receipts: Vec<EvolutionReceipt>,
    #[serde(default = "default_receipt_capacity")]
    pub max_receipts: usize,
}

fn default_receipt_capacity() -> usize {
    100_000
}

impl Default for EvolutionaryMemory {
    fn default() -> Self {
        Self {
            receipts: Vec::new(),
            max_receipts: default_receipt_capacity(),
        }
    }
}

impl EvolutionaryMemory {
    pub fn record(&mut self, receipt: EvolutionReceipt) {
        self.receipts.push(receipt);
        let capacity = self.max_receipts.max(1);
        if self.receipts.len() > capacity {
            let excess = self.receipts.len() - capacity;
            self.receipts.drain(0..excess);
        }
    }

    pub fn successful_mutations<'a>(
        &'a self,
        context: &EvolutionContextKey,
        minimum_improvement: f64,
    ) -> impl Iterator<Item = &'a EvolutionReceipt> {
        let context = context.clone();
        self.receipts.iter().filter(move |receipt| {
            receipt.context.matches(&context) && receipt.improvement >= minimum_improvement
        })
    }

    pub fn replay_candidates(
        &self,
        genome: &CandidateGenome,
        context: &EvolutionContextKey,
        minimum_improvement: f64,
    ) -> Vec<CandidateGenome> {
        let digest = genome_digest(genome);
        self.successful_mutations(context, minimum_improvement)
            .filter(|receipt| {
                receipt.parent_digest == digest
                    || receipt
                        .parent_digest
                        .rsplit_once('-')
                        .map(|(_, suffix)| suffix == digest)
                        .unwrap_or(false)
            })
            .filter_map(|receipt| serde_json::from_str(&receipt.child_digest).ok())
            .collect()
    }
}

/// Surrogates may reject or rank candidates, but cannot mark them measured.
pub trait SurrogateModel: Send + Sync {
    fn predict(&self, genome: &CandidateGenome, context: &[u8]) -> Option<ObjectiveVector>;
    fn predict_with_uncertainty(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Option<SurrogatePrediction> {
        self.predict(genome, context)
            .map(|objectives| SurrogatePrediction {
                objectives,
                uncertainty: 0.0,
                neighbors: 0,
            })
    }
    fn name(&self) -> &str;
    /// Feed measured evidence back into an online surrogate when supported.
    /// Implementations that are remote or immutable may keep the default.
    fn observe(&self, _genome: &CandidateGenome, _objectives: ObjectiveVector) {}
    fn observations(&self) -> u64 {
        0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogatePrediction {
    pub objectives: ObjectiveVector,
    pub uncertainty: f64,
    pub neighbors: usize,
}

/// Lightweight online surrogate trained directly from measured receipts.
/// It deliberately uses nearest-neighbor retrieval rather than pretending to
/// infer hardware behavior from unvalidated synthetic assumptions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReceiptSurrogate {
    pub observations: Vec<(Vec<f64>, ObjectiveVector)>,
    pub max_observations: usize,
}

impl ReceiptSurrogate {
    pub fn from_receipts(receipts: &[EvolutionReceipt], max_observations: usize) -> Self {
        let mut surrogate = Self {
            observations: Vec::new(),
            max_observations,
        };
        for receipt in receipts {
            if let Ok(genome) = serde_json::from_str::<CandidateGenome>(&receipt.child_digest) {
                ReceiptSurrogate::observe(&mut surrogate, &genome, receipt.objectives.clone());
            }
        }
        surrogate
    }

    pub fn observe(&mut self, genome: &CandidateGenome, objectives: ObjectiveVector) {
        let features = Self::features(genome);
        self.observations.push((features, objectives));
        let limit = self.max_observations.max(1);
        if self.observations.len() > limit {
            let excess = self.observations.len() - limit;
            self.observations.drain(0..excess);
        }
    }

    pub fn predict_with_uncertainty(
        &self,
        genome: &CandidateGenome,
    ) -> Option<SurrogatePrediction> {
        let features = Self::features(genome);
        let mut nearest: Vec<(f64, &ObjectiveVector)> = self
            .observations
            .iter()
            .map(|(candidate, objectives)| {
                (
                    candidate
                        .iter()
                        .zip(&features)
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                    objectives,
                )
            })
            .collect();
        nearest.sort_by(|left, right| left.0.total_cmp(&right.0));
        let (distance, objectives) = nearest.first().copied()?;
        Some(SurrogatePrediction {
            objectives: objectives.clone(),
            uncertainty: distance,
            neighbors: nearest.len().min(8),
        })
    }

    fn features(genome: &CandidateGenome) -> Vec<f64> {
        let chromosomes = GenomeChromosomes::from(genome);
        vec![
            chromosomes.representation.descriptor() as f64,
            chromosomes.packing.descriptor() as f64,
            chromosomes.schedule.descriptor() as f64,
            genome.metal_geometry.grid_tile_m as f64 / 256.0,
            genome.metal_geometry.grid_tile_n as f64 / 256.0,
            genome.metal_geometry.grid_tile_k as f64 / 128.0,
            genome.memory.shared_memory_bytes as f64 / 262144.0,
        ]
    }
}

impl SurrogateModel for ReceiptSurrogate {
    fn predict(&self, genome: &CandidateGenome, _context: &[u8]) -> Option<ObjectiveVector> {
        self.predict_with_uncertainty(genome)
            .map(|prediction| prediction.objectives)
    }
    fn predict_with_uncertainty(
        &self,
        genome: &CandidateGenome,
        _context: &[u8],
    ) -> Option<SurrogatePrediction> {
        ReceiptSurrogate::predict_with_uncertainty(self, genome)
    }
    fn name(&self) -> &str {
        "receipt-nearest-neighbor"
    }
}

impl SurrogateModel for std::sync::Mutex<ReceiptSurrogate> {
    fn predict(&self, genome: &CandidateGenome, context: &[u8]) -> Option<ObjectiveVector> {
        self.lock().ok()?.predict(genome, context)
    }

    fn name(&self) -> &str {
        // The name is static, so it is safe to return without holding the
        // mutex guard.
        "receipt-nearest-neighbor"
    }

    fn predict_with_uncertainty(
        &self,
        genome: &CandidateGenome,
        _context: &[u8],
    ) -> Option<SurrogatePrediction> {
        self.lock().ok()?.predict_with_uncertainty(genome)
    }

    fn observe(&self, genome: &CandidateGenome, objectives: ObjectiveVector) {
        if let Ok(mut surrogate) = self.lock() {
            ReceiptSurrogate::observe(&mut surrogate, genome, objectives);
        }
    }

    fn observations(&self) -> u64 {
        self.lock()
            .map(|surrogate| surrogate.observations.len() as u64)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationProposal {
    pub genome: CandidateGenome,
    pub rationale: String,
    pub proposer: String,
}

/// An optional semantic mutation source, suitable for an LLM-backed adapter.
/// The caller must still submit proposals to the ordinary evaluator.
pub trait MutationProposer: Send + Sync {
    fn propose(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
        relevant_receipts: &[EvolutionReceipt],
    ) -> Vec<MutationProposal>;
    fn name(&self) -> &str;
}

pub struct ClosureMutationProposer<F> {
    name: String,
    callback: F,
}

impl<F> ClosureMutationProposer<F> {
    pub fn new(name: impl Into<String>, callback: F) -> Self {
        Self {
            name: name.into(),
            callback,
        }
    }
}

impl<F> MutationProposer for ClosureMutationProposer<F>
where
    F: Fn(&CandidateGenome, &[u8], &[EvolutionReceipt]) -> Vec<MutationProposal> + Send + Sync,
{
    fn propose(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
        relevant_receipts: &[EvolutionReceipt],
    ) -> Vec<MutationProposal> {
        (self.callback)(genome, context, relevant_receipts)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub fn genome_digest(genome: &CandidateGenome) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{:?}", genome).hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::objectives::{ObjectiveValue, ObjectiveVector};

    fn context() -> EvolutionContextKey {
        EvolutionContextKey {
            hardware: "apple-m1".into(),
            model_family: "moe".into(),
            tensor_family: "attention".into(),
        }
    }

    #[test]
    fn memory_retrieves_only_matching_successful_mutations() {
        let parent = CandidateGenome::new();
        let child = CandidateGenome::default();
        let receipt = EvolutionReceipt {
            parent_digest: format!("gen7-{}", genome_digest(&parent)),
            child_digest: serde_json::to_string(&child).unwrap(),
            operator: VariationOperator::Geometry,
            context: context(),
            descriptor: BehaviorDescriptor::from_genome(&child),
            objectives: ObjectiveVector::new(vec![ObjectiveValue::maximize("fidelity", 0.9)]),
            improvement: 0.2,
            measurement_receipt_digest: "receipt-1".into(),
        };
        let mut memory = EvolutionaryMemory::default();
        memory.record(receipt);
        assert_eq!(memory.successful_mutations(&context(), 0.1).count(), 1);
        assert_eq!(memory.replay_candidates(&parent, &context(), 0.1).len(), 1);
        let wildcard = EvolutionContextKey {
            hardware: "*".into(),
            model_family: "moe".into(),
            tensor_family: "*".into(),
        };
        assert_eq!(memory.replay_candidates(&parent, &wildcard, 0.1).len(), 1);
        let specific = EvolutionContextKey {
            hardware: "apple-m1".into(),
            model_family: "moe".into(),
            tensor_family: "attention".into(),
        };
        assert_eq!(memory.successful_mutations(&specific, 0.1).count(), 1);
    }

    #[test]
    fn receipt_surrogate_learns_measured_objectives() {
        let mut surrogate = ReceiptSurrogate {
            max_observations: 8,
            ..Default::default()
        };
        let genome = CandidateGenome::new();
        ReceiptSurrogate::observe(
            &mut surrogate,
            &genome,
            ObjectiveVector::new(vec![ObjectiveValue::maximize("fitness", 0.91)]),
        );
        let prediction = surrogate.predict_with_uncertainty(&genome).unwrap();
        assert_eq!(prediction.neighbors, 1);
        assert!((prediction.objectives.values[0].value - 0.91).abs() < f64::EPSILON);
    }

    #[test]
    fn receipt_surrogate_bootstraps_from_serialized_receipts() {
        let genome = CandidateGenome::new();
        let receipt = EvolutionReceipt {
            parent_digest: "parent".into(),
            child_digest: serde_json::to_string(&genome).unwrap(),
            operator: VariationOperator::Geometry,
            context: context(),
            descriptor: BehaviorDescriptor::from_genome(&genome),
            objectives: ObjectiveVector::new(vec![ObjectiveValue::maximize("fitness", 0.88)]),
            improvement: 0.12,
            measurement_receipt_digest: "measurement".into(),
        };
        let surrogate = ReceiptSurrogate::from_receipts(&[receipt], 8);
        assert_eq!(surrogate.observations.len(), 1);
        assert_eq!(
            surrogate
                .predict_with_uncertainty(&genome)
                .unwrap()
                .neighbors,
            1
        );
    }

    #[test]
    fn shared_receipt_surrogate_accepts_online_observations() {
        let surrogate = std::sync::Mutex::new(ReceiptSurrogate {
            max_observations: 8,
            ..Default::default()
        });
        let genome = CandidateGenome::new();
        SurrogateModel::observe(
            &surrogate,
            &genome,
            ObjectiveVector::new(vec![ObjectiveValue::maximize("fitness", 0.9)]),
        );
        assert_eq!(SurrogateModel::observations(&surrogate), 1);
    }

    #[test]
    fn closure_mutation_proposer_is_a_valid_semantic_operator() {
        let proposer = ClosureMutationProposer::new(
            "test-llm",
            |genome: &CandidateGenome, _context: &[u8], _receipts: &[EvolutionReceipt]| {
                vec![MutationProposal {
                    genome: genome.clone(),
                    rationale: "preserve validated seed".into(),
                    proposer: "test-llm".into(),
                }]
            },
        );
        let proposals = proposer.propose(&CandidateGenome::new(), b"context", &[]);
        assert_eq!(proposer.name(), "test-llm");
        assert_eq!(proposals.len(), 1);
    }

    #[test]
    fn receipt_memory_evicts_oldest_evidence_at_capacity() {
        let mut memory = EvolutionaryMemory {
            receipts: Vec::new(),
            max_receipts: 1,
        };
        let make_receipt = |child: &str| EvolutionReceipt {
            parent_digest: "parent".into(),
            child_digest: child.into(),
            operator: VariationOperator::Geometry,
            context: context(),
            descriptor: BehaviorDescriptor::from_genome(&CandidateGenome::new()),
            objectives: ObjectiveVector::new(vec![ObjectiveValue::maximize("fitness", 0.8)]),
            improvement: 0.1,
            measurement_receipt_digest: child.into(),
        };
        memory.record(make_receipt("first"));
        memory.record(make_receipt("second"));
        assert_eq!(memory.receipts.len(), 1);
        assert_eq!(memory.receipts[0].child_digest, "second");
    }
}
