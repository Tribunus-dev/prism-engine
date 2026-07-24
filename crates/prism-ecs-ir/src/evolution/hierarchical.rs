use super::CandidateGenome;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FrozenHierarchy;
impl FrozenHierarchy {
    pub fn apply(&self, _candidate: &mut CandidateGenome) {}
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalStagePlan {
    pub tensor_keys: Vec<String>,
    pub tensor_scope: Vec<String>,
}
impl HierarchicalStagePlan {
    pub fn from_tensor_keys(k: &[String]) -> Self {
        Self {
            tensor_keys: k.to_vec(),
            tensor_scope: k.to_vec(),
        }
    }
    pub fn run<F, G>(
        &self,
        seed: CandidateGenome,
        mut f: F,
        mut score: G,
    ) -> (CandidateGenome, FrozenHierarchy)
    where
        F: FnMut(CandidateGenome, usize, &FrozenHierarchy) -> Vec<CandidateGenome>,
        G: FnMut(&CandidateGenome, &Self, &FrozenHierarchy) -> f64,
    {
        let frozen = FrozenHierarchy;
        let c = f(seed, 0, &frozen);
        let best = c
            .into_iter()
            .max_by(|a, b| {
                score(a, self, &frozen)
                    .partial_cmp(&score(b, self, &frozen))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(CandidateGenome::new);
        (best, frozen)
    }
}
