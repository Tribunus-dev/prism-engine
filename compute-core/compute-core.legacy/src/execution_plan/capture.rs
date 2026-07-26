//! Model reference vector capture — saves RawF32 reference outputs
//! as deterministic test vectors for quantized cimage comparison.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::execution_plan::vectors::{LogprobEntry, ModelReferenceVector};

/// Configuration for a reference vector capture run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub output_dir: PathBuf,
    pub model_digest: String,
    pub tokenizer_digest: String,
    pub prompts: Vec<CapturePrompt>,
    pub prefill_chunk_size: usize,
    pub max_generated_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturePrompt {
    pub id: String,
    pub tokens: Vec<u32>,
}

/// Captures a reference vector from a running inference session.
pub fn capture_reference_vector(
    config: &CaptureConfig,
    session: &mut dyn InferenceSession,
) -> Result<Vec<ModelReferenceVector>, String> {
    let mut vectors = Vec::new();

    for prompt in &config.prompts {
        let mut greedy_tokens = Vec::new();
        let mut logits_topk = Vec::new();
        let checkpoints = Vec::new();

        session
            .load_prompt(&prompt.tokens)
            .map_err(|e| format!("load: {:?}", e))?;

        for step in 0..config.max_generated_tokens {
            let output = session
                .step()
                .map_err(|e| format!("step {}: {:?}", step, e))?;
            greedy_tokens.push(output.argmax_token);
            logits_topk.push(
                output
                    .topk_logprobs
                    .iter()
                    .map(|(t, lp)| LogprobEntry {
                        token: *t,
                        logprob: *lp,
                    })
                    .collect(),
            );
        }

        vectors.push(ModelReferenceVector {
            vector_id: format!("{}_{}", config.model_digest, prompt.id),
            model_digest: config.model_digest.clone(),
            tokenizer_digest: config.tokenizer_digest.clone(),
            prompt_tokens: prompt.tokens.clone(),
            prefill_chunk_size: config.prefill_chunk_size,
            expected_greedy_tokens: greedy_tokens,
            logits_topk,
            hidden_checkpoints: checkpoints,
        });
    }

    Ok(vectors)
}

/// Trait abstracting inference session for capture.
pub trait InferenceSession {
    fn load_prompt(&mut self, tokens: &[u32]) -> Result<(), String>;
    fn step(&mut self) -> Result<InferenceOutput, String>;
}

#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub argmax_token: u32,
    pub topk_logprobs: Vec<(u32, f64)>,
}

/// Save reference vectors as JSON files.
pub fn save_vectors(
    vectors: &[ModelReferenceVector],
    config: &CaptureConfig,
) -> Result<(), String> {
    std::fs::create_dir_all(&config.output_dir).map_err(|e| format!("mkdir: {:?}", e))?;
    for vec in vectors {
        let path = config.output_dir.join(format!("{}.json", vec.vector_id));
        let json = serde_json::to_string_pretty(vec).map_err(|e| format!("ser: {:?}", e))?;
        std::fs::write(&path, &json).map_err(|e| format!("write: {:?}", e))?;
    }
    Ok(())
}

/// Load a reference vector from a JSON file.
pub fn load_vector(path: &Path) -> Result<ModelReferenceVector, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read: {:?}", e))?;
    serde_json::from_str(&content).map_err(|e| format!("parse: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_capture_config_serialization() {
        let config = CaptureConfig {
            output_dir: PathBuf::from("/tmp/capture_out"),
            model_digest: "sha256:abc123".into(),
            tokenizer_digest: "sha256:def456".into(),
            prompts: vec![CapturePrompt {
                id: "prompt_hello".into(),
                tokens: vec![1, 2, 3, 4],
            }],
            prefill_chunk_size: 512,
            max_generated_tokens: 10,
        };

        let json = serde_json::to_string_pretty(&config).expect("serialize");
        let recovered: CaptureConfig = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(config.output_dir, recovered.output_dir);
        assert_eq!(config.model_digest, recovered.model_digest);
        assert_eq!(config.tokenizer_digest, recovered.tokenizer_digest);
        assert_eq!(config.prompts.len(), recovered.prompts.len());
        assert_eq!(config.prompts[0].id, recovered.prompts[0].id);
        assert_eq!(config.prompts[0].tokens, recovered.prompts[0].tokens);
        assert_eq!(config.prefill_chunk_size, recovered.prefill_chunk_size);
        assert_eq!(config.max_generated_tokens, recovered.max_generated_tokens);
    }

    #[test]
    fn test_save_load_vector() {
        let tmp = TempDir::new().expect("tempdir");

        let vector = ModelReferenceVector {
            vector_id: "test_vec".into(),
            model_digest: "sha256:abc".into(),
            tokenizer_digest: "sha256:def".into(),
            prompt_tokens: vec![1, 2, 3],
            prefill_chunk_size: 512,
            expected_greedy_tokens: vec![42, 99, 7],
            logits_topk: vec![
                vec![
                    LogprobEntry {
                        token: 42,
                        logprob: -0.5,
                    },
                    LogprobEntry {
                        token: 99,
                        logprob: -1.2,
                    },
                ],
                vec![
                    LogprobEntry {
                        token: 99,
                        logprob: -0.3,
                    },
                    LogprobEntry {
                        token: 7,
                        logprob: -0.9,
                    },
                ],
            ],
            hidden_checkpoints: vec![],
        };

        let config = CaptureConfig {
            output_dir: tmp.path().to_path_buf(),
            model_digest: "sha256:abc".into(),
            tokenizer_digest: "sha256:def".into(),
            prompts: vec![],
            prefill_chunk_size: 512,
            max_generated_tokens: 10,
        };

        save_vectors(&[vector.clone()], &config).expect("save");

        let path = tmp.path().join("test_vec.json");
        assert!(path.exists(), "saved file exists");

        let loaded = load_vector(&path).expect("load");
        assert_eq!(loaded.vector_id, vector.vector_id);
        assert_eq!(loaded.model_digest, vector.model_digest);
        assert_eq!(loaded.tokenizer_digest, vector.tokenizer_digest);
        assert_eq!(loaded.prompt_tokens, vector.prompt_tokens);
        assert_eq!(loaded.prefill_chunk_size, vector.prefill_chunk_size);
        assert_eq!(loaded.expected_greedy_tokens, vector.expected_greedy_tokens);
        assert_eq!(loaded.logits_topk.len(), vector.logits_topk.len());
        assert_eq!(
            loaded.logits_topk[0][0].token,
            vector.logits_topk[0][0].token
        );
        assert!(
            (loaded.logits_topk[0][0].logprob - vector.logits_topk[0][0].logprob).abs() < 1e-12
        );
        assert!(loaded.hidden_checkpoints.is_empty());
    }
}
