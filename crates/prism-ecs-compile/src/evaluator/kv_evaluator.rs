//! KV-cache candidate evaluation.
//!
//! This module owns the canonical MI300X-backed evaluator for KV-cache
//! compression candidates ([`Mi300xKvEvaluator`]) and the free function
//! that drives the reference-cache evaluation and persists the evidence
//! sidecar consumed by CImage emission. The evaluator is a semantic gate
//! over a backend scorer (the HIP reduction) and Prism's native
//! TurboQuant quantizer — the quantization and reconstruction paths stay
//! in their native implementations; this module only composes them and
//! produces [`KvCompressionEvidence`].
//!
//! The module is canonical. The HIP scorer is a borrowed handle
//! (`Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>`) and the
//! quantizer is a value type with no hardware coupling in this layer.
//! Hardware execution is delegated to the runtime crates via the
//! borrowed handle.

#![forbid(unsafe_code)]

use std::sync::Arc;

use prism_ecs_quantization::kv_search::{
    KvCompressionCandidate, KvCompressionEvaluator, KvCompressionEvidence,
};
use prism_ecs_quantization::turboquant_kv::{KvQuantMode, TurboQuantKvCache};
use prism_rocm_runtime::ternary::Mi300xTernaryScorer;

/// MI300X-backed evaluator for KV-cache candidates. Quantization and
/// reconstruction stay in Prism's native TurboQuant implementation; the
/// expensive reference-vs-reconstruction reductions run through the
/// existing HIP scorer.
pub struct Mi300xKvEvaluator {
    keys: Vec<f32>,
    values: Vec<f32>,
    scorer: Arc<Mi300xTernaryScorer>,
}

impl Mi300xKvEvaluator {
    /// Build an evaluator with the given reference keys/values and a
    /// pre-resolved HIP scorer handle. The keys and values must be
    /// non-empty and of equal length.
    pub fn new(
        keys: Vec<f32>,
        values: Vec<f32>,
        scorer: Arc<Mi300xTernaryScorer>,
    ) -> Result<Self, String> {
        if keys.is_empty() || keys.len() != values.len() {
            return Err("KV evaluator keys and values must be nonempty and equal length".into());
        }
        Ok(Self {
            keys,
            values,
            scorer,
        })
    }
}

impl KvCompressionEvaluator for Mi300xKvEvaluator {
    fn evaluate(
        &mut self,
        candidate: KvCompressionCandidate,
    ) -> Result<KvCompressionEvidence, String> {
        let mut cache = TurboQuantKvCache::new(
            KvQuantMode::Polar(candidate.key_bits as u32),
            candidate.group_size as usize,
            1,
        );
        cache
            .quantize_asymmetric(0, &self.keys, &self.values, &candidate.mode())
            .map_err(|error| format!("quantize KV candidate: {error}"))?;
        let (reconstructed_keys, reconstructed_values) = cache
            .dequantize(0)
            .map_err(|error| format!("dequantize KV candidate: {error}"))?;
        let key_mse = self
            .scorer
            .mean_squared_error(&self.keys, &reconstructed_keys)?;
        let value_mse = self
            .scorer
            .mean_squared_error(&self.values, &reconstructed_values)?;
        let key_scale = self
            .keys
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
        let value_scale = self
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
        let key_error = (key_mse.sqrt() as f32) / key_scale;
        let value_error = (value_mse.sqrt() as f32) / value_scale;
        Ok(KvCompressionEvidence {
            candidate,
            key_error,
            value_error,
            attention_loss: (key_error + value_error) * 0.5,
            bytes_per_token: ((self.keys.len() as u64 * candidate.key_bits as u64).div_ceil(8))
                + ((self.values.len() as u64 * candidate.value_bits as u64).div_ceil(8)),
        })
    }
}

/// Evaluate a reference KV sample and persist the complete candidate
/// evidence sidecar consumed by CImage emission. MI300X is preferred when
/// enabled; otherwise the identical native CPU evaluator is used.
pub fn evaluate_kv_reference_cache(
    keys: &[f32],
    values: &[f32],
    evidence_path: &std::path::Path,
) -> Result<Vec<KvCompressionEvidence>, String> {
    let search = prism_ecs_quantization::kv_search::KvCompressionSearch::default();
    let evidence =
        if let Ok(Some(scorer)) = Mi300xTernaryScorer::from_env() {
            let mut evaluator =
                Mi300xKvEvaluator::new(keys.to_vec(), values.to_vec(), Arc::new(scorer))?;
            search
                .candidates
                .iter()
                .copied()
                .map(|candidate| evaluator.evaluate(candidate))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            search.evaluate_reference_cache(keys, values)?
        };
    if let Some(parent) = evidence_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create KV evidence directory: {error}"))?;
    }
    let bytes = serde_json::to_vec_pretty(&evidence)
        .map_err(|error| format!("encode KV compression evidence: {error}"))?;
    std::fs::write(evidence_path, bytes)
        .map_err(|error| format!("write KV compression evidence: {error}"))?;
    Ok(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// We cannot construct a real `Mi300xTernaryScorer` without a HIP
    /// device, so the construction-path tests focus on the data-side
    /// invariants of [`Mi300xKvEvaluator`]. The hardware path is
    /// covered by the integration suite that requires the rocm
    /// feature and a connected accelerator.
    #[test]
    fn mi300x_kv_evaluator_rejects_empty_inputs() {
        // The Arc::new on a non-constructable type would fail to compile
        // here — instead we use a pointer-sized dummy that we never
        // dereference, by leaning on the constructor's own guard.
        let dummy_scorer: Arc<Mi300xTernaryScorer> = match Mi300xTernaryScorer::from_env() {
            Ok(Some(s)) => Arc::new(s),
            _ => return, // no HIP device in this environment; skip
        };
        assert!(Mi300xKvEvaluator::new(vec![], vec![], dummy_scorer).is_err());
    }

    #[test]
    fn mi300x_kv_evaluator_rejects_mismatched_lengths() {
        let dummy_scorer: Arc<Mi300xTernaryScorer> = match Mi300xTernaryScorer::from_env() {
            Ok(Some(s)) => Arc::new(s),
            _ => return,
        };
        let res = Mi300xKvEvaluator::new(vec![1.0, 2.0, 3.0], vec![1.0, 2.0], dummy_scorer);
        assert!(res.is_err());
    }
}
