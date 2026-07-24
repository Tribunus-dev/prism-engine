//! Privacy-bounded agentic workload episodes for Living CImage refinement.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticWorkloadClass {
    Coding,
    RepositoryAnalysis,
    BrowserResearch,
    ToolOrchestration,
    LongContextSynthesis,
    MultimodalInspection,
    ComputerControl,
    PlanningAndExecution,
    RepeatedPersonalWorkflow,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyClass {
    Ephemeral,
    LocalOnly,
    Redacted,
    FeatureOnly,
    Exportable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub privacy_class: PrivacyClass,
    pub retain_raw_prompt: bool,
    pub retain_raw_output: bool,
    pub retain_tool_arguments: bool,
    pub expires_after_days: Option<u32>,
    pub explicit_opt_in: bool,
}

impl RetentionPolicy {
    pub fn privacy_safe(&self) -> bool {
        match self.privacy_class {
            PrivacyClass::Ephemeral | PrivacyClass::FeatureOnly => {
                !self.retain_raw_prompt && !self.retain_raw_output && !self.retain_tool_arguments
            }
            PrivacyClass::LocalOnly | PrivacyClass::Redacted => !self.retain_tool_arguments,
            PrivacyClass::Exportable => self.explicit_opt_in,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngramAccessEvent {
    pub engram_id: String,
    pub layer_or_region: String,
    pub retrieved: bool,
    pub utility_delta: Option<f64>,
    pub interference_delta: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticOutcome {
    pub task_success: f64,
    pub tool_call_correctness: f64,
    pub retry_count: u32,
    pub planning_steps: u32,
    pub final_answer_accepted: bool,
    pub failure_class: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticWorkloadEpisode {
    pub episode_id: String,
    pub workload_class: AgenticWorkloadClass,
    pub model_generation: u64,
    pub prompt_digest: String,
    pub output_digest: String,
    pub tool_schema_digest: Option<String>,
    pub step_count: u32,
    pub activation_probe_refs: Vec<String>,
    pub router_probe_refs: Vec<String>,
    pub engram_access_trace: Vec<EngramAccessEvent>,
    pub outcome: AgenticOutcome,
    pub retention: RetentionPolicy,
    #[serde(default)]
    pub feature_summary: BTreeMap<String, f64>,
    #[serde(default)]
    pub episode_digest: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgenticCalibrationCorpus {
    pub corpus_id: String,
    pub generation_digest: String,
    pub episodes: Vec<AgenticWorkloadEpisode>,
    pub workload_counts: BTreeMap<AgenticWorkloadClass, u64>,
    pub corpus_digest: String,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AgenticWorkloadError {
    #[error("episode identifier is empty")]
    MissingEpisodeId,
    #[error("prompt/output digest is empty")]
    MissingDigest,
    #[error("retention policy is not privacy safe")]
    UnsafeRetention,
    #[error("metric is outside [0, 1]")]
    InvalidMetric,
    #[error("episode digest mismatch")]
    DigestMismatch,
    #[error("corpus generation digest is empty")]
    MissingGeneration,
}

impl AgenticWorkloadEpisode {
    pub fn canonical_digest(&self) -> String {
        let mut canonical = self.clone();
        canonical.episode_digest.clear();
        digest_json(&canonical)
    }

    pub fn seal(mut self) -> Result<Self, AgenticWorkloadError> {
        self.episode_digest = self.canonical_digest();
        self.verify()?;
        Ok(self)
    }

    pub fn verify(&self) -> Result<(), AgenticWorkloadError> {
        if self.episode_id.is_empty() {
            return Err(AgenticWorkloadError::MissingEpisodeId);
        }
        if self.prompt_digest.is_empty() || self.output_digest.is_empty() {
            return Err(AgenticWorkloadError::MissingDigest);
        }
        if !self.retention.privacy_safe() {
            return Err(AgenticWorkloadError::UnsafeRetention);
        }
        for metric in [
            self.outcome.task_success,
            self.outcome.tool_call_correctness,
        ] {
            if !metric.is_finite() || !(0.0..=1.0).contains(&metric) {
                return Err(AgenticWorkloadError::InvalidMetric);
            }
        }
        if !self.episode_digest.is_empty() && self.episode_digest != self.canonical_digest() {
            return Err(AgenticWorkloadError::DigestMismatch);
        }
        Ok(())
    }
}

impl AgenticCalibrationCorpus {
    pub fn build(
        corpus_id: impl Into<String>,
        generation_digest: impl Into<String>,
        episodes: Vec<AgenticWorkloadEpisode>,
    ) -> Result<Self, AgenticWorkloadError> {
        let generation_digest = generation_digest.into();
        if generation_digest.is_empty() {
            return Err(AgenticWorkloadError::MissingGeneration);
        }
        let mut workload_counts = BTreeMap::new();
        for episode in &episodes {
            episode.verify()?;
            *workload_counts
                .entry(episode.workload_class.clone())
                .or_insert(0) += 1;
        }
        let mut corpus = Self {
            corpus_id: corpus_id.into(),
            generation_digest,
            episodes,
            workload_counts,
            corpus_digest: String::new(),
        };
        corpus.corpus_digest = digest_json(&corpus);
        Ok(corpus)
    }
}

fn digest_json<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("agentic workload canonical serialization");
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_only_episode_rejects_raw_retention() {
        let episode = AgenticWorkloadEpisode {
            episode_id: "ep-1".into(),
            workload_class: AgenticWorkloadClass::Coding,
            model_generation: 1,
            prompt_digest: "p".into(),
            output_digest: "o".into(),
            tool_schema_digest: None,
            step_count: 1,
            activation_probe_refs: vec![],
            router_probe_refs: vec![],
            engram_access_trace: vec![],
            outcome: AgenticOutcome {
                task_success: 1.0,
                tool_call_correctness: 1.0,
                retry_count: 0,
                planning_steps: 1,
                final_answer_accepted: true,
                failure_class: None,
            },
            retention: RetentionPolicy {
                privacy_class: PrivacyClass::FeatureOnly,
                retain_raw_prompt: true,
                retain_raw_output: false,
                retain_tool_arguments: false,
                expires_after_days: None,
                explicit_opt_in: false,
            },
            feature_summary: BTreeMap::new(),
            episode_digest: String::new(),
        };
        assert_eq!(episode.verify(), Err(AgenticWorkloadError::UnsafeRetention));
    }
}
