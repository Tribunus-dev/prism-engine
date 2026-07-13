//! Evidence ledger — durable JSONL records for each sweep candidate.
//! Enables cross-run mining of substitution patterns.

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// One observation from a sweep trial on a single tensor.
/// Appended to the experiment ledger as a JSONL row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentReceipt {
    /// Run identifier — consistent across a single sweep invocation.
    pub run_id: String,
    /// Model family (e.g. "Gemma4", "Qwen3TTS").
    pub model_family: String,
    /// Model digest or version string.
    pub model_digest: String,
    /// Full tensor key (e.g. "model.layers.5.self_attn.q_proj.weight").
    pub tensor_key: String,
    /// Tensor class (e.g. "DecoderAttentionProjection").
    pub tensor_class: String,
    /// Tensor shape [out_features, in_features].
    pub tensor_shape: Vec<usize>,
    /// Logical matmul layout (e.g. "output_major").
    pub logical_layout: String,
    /// Candidate codec family.
    pub codec_family: String,
    /// Candidate codec parameters as JSON.
    pub codec_params: serde_json::Value,
    /// Packed payload bytes.
    pub packed_bytes: u64,
    /// Raw F32 weight bytes.
    pub raw_f32_bytes: u64,
    /// Weight-space NRMSE (if evaluated).
    pub weight_nrmse: Option<f64>,
    /// Weight-space zero-collapse ratio (if evaluated).
    pub weight_zero_collapse: Option<f64>,
    /// Operator NRMSE (if evaluated).
    pub operator_nrmse: Option<f64>,
    /// Operator cosine similarity (if evaluated).
    pub operator_cosine: Option<f64>,
    /// Operator max-absolute error (if evaluated).
    pub operator_max_abs: Option<f64>,
    /// Was activation weighting used?
    pub activation_weighted: bool,
    /// Evidence level reached.
    pub evidence_level: String,
    /// Final decision: "substituted", "rejected", "unavailable".
    pub decision: String,
    /// Failure reason if rejected.
    pub rejection_reason: String,
    /// Human-readable summary.
    pub summary: String,
    /// Timestamp for ordering.
    pub timestamp: String,
}

/// Appends one experiment receipt to the JSONL ledger.
pub fn append_experiment_receipt(
    receipt: &ExperimentReceipt,
    ledger_path: &Path,
) -> std::io::Result<()> {
    let mut line = serde_json::to_vec(receipt)?;
    line.push(b'\n');
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_path)?;
    file.write_all(&line)?;
    Ok(())
}

/// Read all experiment receipts from a JSONL ledger.
pub fn read_experiment_ledger(ledger_path: &Path) -> std::io::Result<Vec<ExperimentReceipt>> {
    let content = std::fs::read_to_string(ledger_path)?;
    Ok(content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receipt_roundtrip() {
        let r = ExperimentReceipt {
            run_id: "test-run".into(),
            model_family: "Gemma4".into(),
            model_digest: "abc".into(),
            tensor_key: "model.layers.5.self_attn.q_proj.weight".into(),
            tensor_class: "DecoderAttentionProjection".into(),
            tensor_shape: vec![4096, 3840],
            logical_layout: "output_major".into(),
            codec_family: "NF4".into(),
            codec_params: serde_json::json!({}),
            packed_bytes: 1000000,
            raw_f32_bytes: 62914560,
            weight_nrmse: Some(0.095),
            weight_zero_collapse: Some(0.0005),
            operator_nrmse: Some(0.0012),
            operator_cosine: Some(0.9995),
            operator_max_abs: Some(0.3),
            activation_weighted: false,
            evidence_level: "WeightSpaceValidated".into(),
            decision: "substituted".into(),
            rejection_reason: String::new(),
            summary: "NF4 g32 passes weight gate".into(),
            timestamp: "2026-07-08T03:13:00Z".into(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: ExperimentReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tensor_key, "model.layers.5.self_attn.q_proj.weight");
        assert_eq!(back.codec_family, "NF4");
        assert_eq!(back.decision, "substituted");
    }

    #[test]
    fn test_append_and_read_ledger() {
        let dir = std::env::temp_dir().join("prism_test_evidence_ledger");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("experiments.jsonl");
        let r1 = ExperimentReceipt {
            run_id: "r1".into(),
            model_family: "Gemma4".into(),
            model_digest: "d1".into(),
            tensor_key: "t1".into(),
            tensor_class: "Test".into(),
            tensor_shape: vec![64, 64],
            logical_layout: "output_major".into(),
            codec_family: "NF4".into(),
            codec_params: serde_json::json!({}),
            packed_bytes: 100,
            raw_f32_bytes: 16384,
            weight_nrmse: None,
            weight_zero_collapse: None,
            operator_nrmse: None,
            operator_cosine: None,
            operator_max_abs: None,
            activation_weighted: false,
            evidence_level: "Hypothesis".into(),
            decision: "rejected".into(),
            rejection_reason: "DisallowedByPolicy".into(),
            summary: "NF4 disallowed for speech".into(),
            timestamp: "2026-07-08T03:13:00Z".into(),
        };
        append_experiment_receipt(&r1, &path).unwrap();
        let records = read_experiment_ledger(&path).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].tensor_key, "t1");

        // Append second record
        let r2 = ExperimentReceipt {
            run_id: "r2".into(),
            model_family: "Qwen3TTS".into(),
            ..r1.clone()
        };
        append_experiment_receipt(&r2, &path).unwrap();
        let records = read_experiment_ledger(&path).unwrap();
        assert_eq!(records.len(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
