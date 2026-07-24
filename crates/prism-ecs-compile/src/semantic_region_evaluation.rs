//! Comparative evaluation contracts for semantic-region plans.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SemanticRegionBaseline {
    UniformPerTensor,
    FixedGroupWise,
    SensitivityOnly,
    GraphSemantic,
    NumericalBlockSelection,
    ChannelAssignmentRegularized,
    PrismSemanticOnly,
    PrismSemanticSensitivity,
    PrismSemanticSensitivityPlacement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionEvaluationMetrics {
    pub quality_score: Option<f64>,
    pub model_size_bytes: Option<u64>,
    pub prefill_latency_ms: Option<f64>,
    pub decode_latency_ms: Option<f64>,
    pub tokens_per_second: Option<f64>,
    pub energy_per_token_joules: Option<f64>,
    pub compile_time_ms: Option<u64>,
    pub search_time_ms: Option<u64>,
    pub materialization_bytes: u64,
    pub conversion_bytes: u64,
    pub kernel_count: u32,
    pub layout_fragmentation: f64,
    pub region_count: u32,
    pub fallback_frequency: f64,
    pub receipt_completeness: f64,
    pub measured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionEvaluationRecord {
    pub baseline: SemanticRegionBaseline,
    pub model_digest: String,
    pub hardware_fingerprint: String,
    pub region_plan_digest: Option<String>,
    pub metrics: SemanticRegionEvaluationMetrics,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRegionAblation {
    pub name: String,
    pub removed_feature: String,
    pub record: SemanticRegionEvaluationRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SemanticRegionEvaluationStudy {
    pub records: Vec<SemanticRegionEvaluationRecord>,
    pub ablations: Vec<SemanticRegionAblation>,
}

impl SemanticRegionEvaluationRecord {
    pub fn authoritative(&self) -> bool {
        self.metrics.measured
            && !self.model_digest.is_empty()
            && !self.hardware_fingerprint.is_empty()
            && !self.evidence_refs.is_empty()
            && self.metrics.receipt_completeness >= 1.0
            && self.metrics.fallback_frequency.is_finite()
            && self.metrics.layout_fragmentation.is_finite()
    }
}

impl SemanticRegionEvaluationStudy {
    pub fn by_baseline(&self) -> BTreeMap<SemanticRegionBaseline, Vec<&SemanticRegionEvaluationRecord>> {
        let mut grouped = BTreeMap::new();
        for record in &self.records {
            grouped.entry(record.baseline).or_insert_with(Vec::new).push(record);
        }
        grouped
    }

    pub fn measured_frontier(&self) -> Vec<&SemanticRegionEvaluationRecord> {
        let mut candidates = self.records.iter().filter(|record| record.authoritative()).collect::<Vec<_>>();
        candidates.retain(|candidate| {
            !self.records.iter().filter(|other| other.authoritative()).any(|other| {
                let cq = candidate.metrics.quality_score.unwrap_or(f64::NEG_INFINITY);
                let oq = other.metrics.quality_score.unwrap_or(f64::NEG_INFINITY);
                let ct = candidate.metrics.tokens_per_second.unwrap_or(0.0);
                let ot = other.metrics.tokens_per_second.unwrap_or(0.0);
                let cm = candidate.metrics.model_size_bytes.unwrap_or(u64::MAX);
                let om = other.metrics.model_size_bytes.unwrap_or(u64::MAX);
                (oq >= cq && ot >= ct && om <= cm) && (oq > cq || ot > ct || om < cm)
            })
        });
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(baseline: SemanticRegionBaseline, quality: f64, throughput: f64, bytes: u64) -> SemanticRegionEvaluationRecord {
        SemanticRegionEvaluationRecord {
            baseline,
            model_digest: "model".into(),
            hardware_fingerprint: "hardware".into(),
            region_plan_digest: Some("plan".into()),
            metrics: SemanticRegionEvaluationMetrics {
                quality_score: Some(quality),
                model_size_bytes: Some(bytes),
                prefill_latency_ms: None,
                decode_latency_ms: None,
                tokens_per_second: Some(throughput),
                energy_per_token_joules: None,
                compile_time_ms: Some(1),
                search_time_ms: Some(1),
                materialization_bytes: 0,
                conversion_bytes: 0,
                kernel_count: 1,
                layout_fragmentation: 0.0,
                region_count: 3,
                fallback_frequency: 0.0,
                receipt_completeness: 1.0,
                measured: true,
            },
            evidence_refs: vec!["receipt".into()],
        }
    }

    #[test]
    fn unmeasured_records_are_not_authoritative() {
        let mut r = record(SemanticRegionBaseline::PrismSemanticOnly, 1.0, 1.0, 1);
        r.metrics.measured = false;
        assert!(!r.authoritative());
    }

    #[test]
    fn frontier_removes_dominated_record() {
        let study = SemanticRegionEvaluationStudy {
            records: vec![record(SemanticRegionBaseline::UniformPerTensor, 0.9, 10.0, 100), record(SemanticRegionBaseline::PrismSemanticOnly, 0.95, 12.0, 90)],
            ablations: vec![],
        };
        let frontier = study.measured_frontier();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].baseline, SemanticRegionBaseline::PrismSemanticOnly);
    }
}
