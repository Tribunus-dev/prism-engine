use crate::ecs::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateResult {
    pub nrmse: f64,
    pub perplexity_delta: f64,
    pub passed: bool,
}
impl Component for QualityGateResult {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionReceipt {
    pub recipe: String,
    pub evidence: Vec<u8>,
    pub signature: Vec<u8>,
}
impl Component for AdmissionReceipt {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AOTProfileMatch {
    pub profile_id: String,
    pub match_confidence: f64,
}
impl Component for AOTProfileMatch {}
