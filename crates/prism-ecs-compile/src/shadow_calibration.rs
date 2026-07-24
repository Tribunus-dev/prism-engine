//! Shadow calibration and evaluation for candidate Living CImage generations.

use crate::agentic_workload::AgenticCalibrationCorpus;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowCalibrationPolicy {
    pub min_episodes: usize,
    pub min_task_success: f64,
    pub min_tool_call_correctness: f64,
    pub max_retry_regression: f64,
    pub max_planning_step_regression: f64,
    pub max_logit_divergence: Option<f64>,
    pub max_rollout_divergence: Option<f64>,
    pub require_zero_user_visible_candidate_outputs: bool,
}

impl Default for ShadowCalibrationPolicy {
    fn default() -> Self {
        Self {
            min_episodes: 8,
            min_task_success: 0.98,
            min_tool_call_correctness: 0.99,
            max_retry_regression: 0.0,
            max_planning_step_regression: 0.05,
            max_logit_divergence: None,
            max_rollout_divergence: None,
            require_zero_user_visible_candidate_outputs: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowEpisodeComparison {
    pub episode_digest: String,
    pub baseline_task_success: f64,
    pub candidate_task_success: f64,
    pub baseline_tool_call_correctness: f64,
    pub candidate_tool_call_correctness: f64,
    pub baseline_retries: u32,
    pub candidate_retries: u32,
    pub baseline_planning_steps: u32,
    pub candidate_planning_steps: u32,
    pub logit_divergence: Option<f64>,
    pub rollout_divergence: Option<f64>,
    pub candidate_output_user_visible: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShadowCalibrationReceipt {
    pub baseline_generation_digest: String,
    pub candidate_generation_digest: String,
    pub corpus_digest: String,
    pub comparisons: Vec<ShadowEpisodeComparison>,
    pub admitted: bool,
    pub reasons: Vec<String>,
    pub receipt_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ShadowCalibrationError {
    #[error("generation digest is empty")]
    MissingGeneration,
    #[error("corpus does not contain enough episodes")]
    InsufficientEpisodes,
    #[error("comparison count does not match corpus")]
    ComparisonMismatch,
    #[error("metric is not finite")]
    InvalidMetric,
}

pub fn evaluate_shadow_candidate(
    baseline_generation_digest: impl Into<String>,
    candidate_generation_digest: impl Into<String>,
    corpus: &AgenticCalibrationCorpus,
    comparisons: Vec<ShadowEpisodeComparison>,
    policy: &ShadowCalibrationPolicy,
) -> Result<ShadowCalibrationReceipt, ShadowCalibrationError> {
    let baseline_generation_digest = baseline_generation_digest.into();
    let candidate_generation_digest = candidate_generation_digest.into();
    if baseline_generation_digest.is_empty() || candidate_generation_digest.is_empty() {
        return Err(ShadowCalibrationError::MissingGeneration);
    }
    if corpus.episodes.len() < policy.min_episodes {
        return Err(ShadowCalibrationError::InsufficientEpisodes);
    }
    if comparisons.len() != corpus.episodes.len() {
        return Err(ShadowCalibrationError::ComparisonMismatch);
    }

    let mut reasons = Vec::new();
    let mut task_sum = 0.0;
    let mut tool_sum = 0.0;
    let mut retry_regression_sum = 0.0;
    let mut planning_regression_sum = 0.0;
    for comparison in &comparisons {
        for value in [
            comparison.baseline_task_success,
            comparison.candidate_task_success,
            comparison.baseline_tool_call_correctness,
            comparison.candidate_tool_call_correctness,
        ] {
            if !value.is_finite() {
                return Err(ShadowCalibrationError::InvalidMetric);
            }
        }
        task_sum += comparison.candidate_task_success;
        tool_sum += comparison.candidate_tool_call_correctness;
        retry_regression_sum += comparison.candidate_retries.saturating_sub(comparison.baseline_retries) as f64;
        let planning_delta = comparison.candidate_planning_steps as f64 - comparison.baseline_planning_steps as f64;
        planning_regression_sum += planning_delta.max(0.0);
        if policy.require_zero_user_visible_candidate_outputs && comparison.candidate_output_user_visible {
            reasons.push(format!("candidate output was visible for {}", comparison.episode_digest));
        }
        if let (Some(limit), Some(value)) = (policy.max_logit_divergence, comparison.logit_divergence) {
            if value > limit { reasons.push(format!("logit divergence exceeded for {}", comparison.episode_digest)); }
        }
        if let (Some(limit), Some(value)) = (policy.max_rollout_divergence, comparison.rollout_divergence) {
            if value > limit { reasons.push(format!("rollout divergence exceeded for {}", comparison.episode_digest)); }
        }
    }
    let count = comparisons.len() as f64;
    let task_mean = task_sum / count;
    let tool_mean = tool_sum / count;
    let retry_regression = retry_regression_sum / count;
    let planning_regression = planning_regression_sum / count;
    if task_mean < policy.min_task_success { reasons.push("candidate task success below gate".into()); }
    if tool_mean < policy.min_tool_call_correctness { reasons.push("candidate tool-call correctness below gate".into()); }
    if retry_regression > policy.max_retry_regression { reasons.push("candidate retry regression exceeds gate".into()); }
    if planning_regression > policy.max_planning_step_regression { reasons.push("candidate planning-step regression exceeds gate".into()); }

    let mut receipt = ShadowCalibrationReceipt {
        baseline_generation_digest,
        candidate_generation_digest,
        corpus_digest: corpus.corpus_digest.clone(),
        comparisons,
        admitted: reasons.is_empty(),
        reasons,
        receipt_digest: String::new(),
    };
    receipt.receipt_digest = digest_json(&receipt);
    Ok(receipt)
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("shadow calibration canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}
