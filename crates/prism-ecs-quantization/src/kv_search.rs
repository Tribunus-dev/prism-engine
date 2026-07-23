//! Evolutionary KV-cache compression selection.
//!
//! The search chooses the most aggressive key/value policy that remains
//! effectively lossless against a reference cache. The winning policy is
//! serializable and can be embedded in a CImage manifest.

use crate::turboquant_kv::AsymmetricQuantMode;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCompressionCandidate {
    pub mode: AsymmetricQuantModeId,
    pub key_bits: u8,
    pub value_bits: u8,
    pub group_size: u16,
    pub qjl_bits: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsymmetricQuantModeId {
    KeyLightValueHeavy,
    KeyPolarValueProd,
    KeySplitValuePolar,
}

impl KvCompressionCandidate {
    pub fn mode(self) -> AsymmetricQuantMode {
        match self.mode {
            AsymmetricQuantModeId::KeyLightValueHeavy => AsymmetricQuantMode::KeyLightValueHeavy {
                k_bits: self.key_bits as u32,
                v_bits: self.value_bits as u32,
            },
            AsymmetricQuantModeId::KeyPolarValueProd => AsymmetricQuantMode::KeyPolarValueProd {
                k_bits: self.key_bits as u32,
                v_bits: self.value_bits as u32,
            },
            AsymmetricQuantModeId::KeySplitValuePolar => AsymmetricQuantMode::KeySplitValuePolar {
                k_bits: self.key_bits as u32,
                v_bits: self.value_bits as u32,
            },
        }
    }

    pub fn compression_ratio(self) -> f64 {
        self.mode().compression_ratio()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct KvCompressionEvidence {
    pub candidate: KvCompressionCandidate,
    pub key_error: f32,
    pub value_error: f32,
    pub attention_loss: f32,
    pub bytes_per_token: u64,
}

impl KvCompressionEvidence {
    pub fn lossless(self, max_error: f32) -> bool {
        self.key_error.is_finite()
            && self.value_error.is_finite()
            && self.attention_loss.is_finite()
            && self.key_error <= max_error
            && self.value_error <= max_error
            && self.attention_loss <= max_error
    }
}

pub struct KvCompressionSearch {
    pub max_error: f32,
    pub candidates: Vec<KvCompressionCandidate>,
}

/// Backend-neutral hook for evaluating KV candidates. GPU backends can batch
/// reconstruction and attention probes without making the quantization crate
/// depend on a particular runtime or accelerator API.
pub trait KvCompressionEvaluator {
    fn evaluate(
        &mut self,
        candidate: KvCompressionCandidate,
    ) -> Result<KvCompressionEvidence, String>;
}

impl Default for KvCompressionSearch {
    fn default() -> Self {
        let mut candidates = Vec::new();
        for &(mode, k, v) in &[
            (AsymmetricQuantModeId::KeyLightValueHeavy, 2, 4),
            (AsymmetricQuantModeId::KeyLightValueHeavy, 3, 4),
            (AsymmetricQuantModeId::KeyPolarValueProd, 2, 4),
            (AsymmetricQuantModeId::KeyPolarValueProd, 3, 4),
            (AsymmetricQuantModeId::KeySplitValuePolar, 2, 4),
        ] {
            for &group_size in &[32, 64, 128] {
                candidates.push(KvCompressionCandidate {
                    mode,
                    key_bits: k,
                    value_bits: v,
                    group_size,
                    qjl_bits: 0,
                });
            }
        }
        Self {
            max_error: 0.01,
            candidates,
        }
    }
}

impl KvCompressionSearch {
    /// Measure every candidate against one reference KV slice. This is the
    /// native evaluator used by compiler-side evolutionary search; it does
    /// not infer losslessness from nominal bit widths.
    pub fn evaluate_reference_cache(
        &self,
        keys: &[f32],
        values: &[f32],
    ) -> Result<Vec<KvCompressionEvidence>, String> {
        if keys.is_empty() || values.is_empty() || keys.len() != values.len() {
            return Err("KV reference keys and values must be nonempty and equal length".into());
        }
        let mut evidence = Vec::with_capacity(self.candidates.len());
        for candidate in self.candidates.iter().copied() {
            let mut cache = crate::turboquant_kv::TurboQuantKvCache::new(
                crate::turboquant_kv::KvQuantMode::Polar(candidate.key_bits as u32),
                candidate.group_size as usize,
                1,
            );
            cache
                .quantize_asymmetric(0, keys, values, &candidate.mode())
                .map_err(|error| format!("evaluate KV candidate {:?}: {error}", candidate))?;
            let (reconstructed_keys, reconstructed_values) = cache
                .dequantize(0)
                .map_err(|error| format!("dequantize KV candidate {:?}: {error}", candidate))?;
            let key_error = normalized_rmse(keys, &reconstructed_keys);
            let value_error = normalized_rmse(values, &reconstructed_values);
            let attention_loss = (key_error + value_error) * 0.5;
            let bytes_per_token = ((keys.len() as u64 * candidate.key_bits as u64).div_ceil(8))
                .saturating_add((values.len() as u64 * candidate.value_bits as u64).div_ceil(8));
            evidence.push(KvCompressionEvidence {
                candidate,
                key_error,
                value_error,
                attention_loss,
                bytes_per_token,
            });
        }
        Ok(evidence)
    }

    /// Evaluate and select the most aggressive effectively-lossless policy in
    /// one compiler-facing operation.
    pub fn search_reference_cache(
        &self,
        keys: &[f32],
        values: &[f32],
    ) -> Result<Option<KvCompressionEvidence>, String> {
        let evidence = self.evaluate_reference_cache(keys, values)?;
        Ok(self.select_from_evidence(evidence))
    }

    /// Run the same admission policy through an external evaluator, such as
    /// the MI300X HIP path. Selection remains here so CPU and GPU searches
    /// produce identical, auditable decisions.
    pub fn search_with_evaluator<E: KvCompressionEvaluator>(
        &self,
        evaluator: &mut E,
    ) -> Result<Option<KvCompressionEvidence>, String> {
        let mut evidence = Vec::with_capacity(self.candidates.len());
        for candidate in self.candidates.iter().copied() {
            evidence.push(evaluator.evaluate(candidate)?);
        }
        Ok(self.select_from_evidence(evidence))
    }

    /// Select from measurements produced by the compiler's reference-cache
    /// evaluator. Keeping this separate from `select` makes it impossible for
    /// artifact emission to silently treat an unmeasured candidate as valid.
    pub fn select_from_evidence(
        &self,
        evidence: impl IntoIterator<Item = KvCompressionEvidence>,
    ) -> Option<KvCompressionEvidence> {
        let mut measured = evidence.into_iter().collect::<Vec<_>>();
        measured.retain(|entry| self.candidates.contains(&entry.candidate));
        measured
            .into_iter()
            .filter(|entry| entry.lossless(self.max_error))
            .min_by(|a, b| {
                a.bytes_per_token
                    .cmp(&b.bytes_per_token)
                    .then_with(|| a.attention_loss.total_cmp(&b.attention_loss))
            })
    }

    /// Read a deterministic evidence sidecar emitted by the KV evaluator.
    pub fn load_evidence(path: &std::path::Path) -> Result<Vec<KvCompressionEvidence>, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read KV compression evidence {}: {error}", path.display()))?;
        serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse KV compression evidence {}: {error}", path.display()))
    }

    pub fn select<F>(&self, mut evaluate: F) -> Option<KvCompressionEvidence>
    where
        F: FnMut(KvCompressionCandidate) -> KvCompressionEvidence,
    {
        self.candidates
            .iter()
            .copied()
            .map(&mut evaluate)
            .filter(|e| e.lossless(self.max_error))
            .min_by(|a, b| {
                a.bytes_per_token
                    .cmp(&b.bytes_per_token)
                    .then_with(|| a.attention_loss.total_cmp(&b.attention_loss))
            })
    }
}

fn normalized_rmse(reference: &[f32], reconstructed: &[f32]) -> f32 {
    let n = reference.len().min(reconstructed.len());
    if n == 0 {
        return f32::INFINITY;
    }
    let mse = reference
        .iter()
        .zip(reconstructed.iter())
        .take(n)
        .map(|(a, b)| {
            let error = a - b;
            error * error
        })
        .sum::<f32>()
        / n as f32;
    let scale = reference
        .iter()
        .take(n)
        .map(|value| value.abs())
        .fold(0.0f32, f32::max)
        .max(1.0);
    (mse.sqrt() / scale).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_most_compressed_lossless_candidate() {
        let search = KvCompressionSearch {
            max_error: 0.01,
            candidates: vec![
                KvCompressionCandidate {
                    mode: AsymmetricQuantModeId::KeyLightValueHeavy,
                    key_bits: 2,
                    value_bits: 4,
                    group_size: 32,
                    qjl_bits: 0,
                },
                KvCompressionCandidate {
                    mode: AsymmetricQuantModeId::KeyLightValueHeavy,
                    key_bits: 3,
                    value_bits: 4,
                    group_size: 64,
                    qjl_bits: 0,
                },
            ],
        };
        let winner = search
            .select(|candidate| KvCompressionEvidence {
                candidate,
                key_error: if candidate.key_bits == 2 { 0.02 } else { 0.001 },
                value_error: 0.001,
                attention_loss: if candidate.key_bits == 2 { 0.02 } else { 0.001 },
                bytes_per_token: if candidate.key_bits == 2 { 8 } else { 10 },
            })
            .unwrap();
        assert_eq!(winner.candidate.key_bits, 3);
    }

    #[test]
    fn evaluator_hook_preserves_admission_selection() {
        struct Evaluator;
        impl KvCompressionEvaluator for Evaluator {
            fn evaluate(
                &mut self,
                candidate: KvCompressionCandidate,
            ) -> Result<KvCompressionEvidence, String> {
                Ok(KvCompressionEvidence {
                    candidate,
                    key_error: if candidate.key_bits == 2 { 0.02 } else { 0.001 },
                    value_error: 0.001,
                    attention_loss: 0.001,
                    bytes_per_token: candidate.key_bits as u64,
                })
            }
        }
        let search = KvCompressionSearch {
            max_error: 0.01,
            candidates: vec![
                KvCompressionCandidate {
                    mode: AsymmetricQuantModeId::KeyLightValueHeavy,
                    key_bits: 2,
                    value_bits: 4,
                    group_size: 32,
                    qjl_bits: 0,
                },
                KvCompressionCandidate {
                    mode: AsymmetricQuantModeId::KeyLightValueHeavy,
                    key_bits: 3,
                    value_bits: 4,
                    group_size: 64,
                    qjl_bits: 0,
                },
            ],
        };
        let winner = search
            .search_with_evaluator(&mut Evaluator)
            .unwrap()
            .unwrap();
        assert_eq!(winner.candidate.key_bits, 3);
    }

    #[test]
    fn sidecar_selection_ignores_unsearched_candidates() {
        let search = KvCompressionSearch {
            max_error: 0.01,
            candidates: vec![KvCompressionCandidate {
                mode: AsymmetricQuantModeId::KeyLightValueHeavy,
                key_bits: 3,
                value_bits: 4,
                group_size: 64,
                qjl_bits: 0,
            }],
        };
        let unsearched = KvCompressionEvidence {
            candidate: KvCompressionCandidate {
                mode: AsymmetricQuantModeId::KeyLightValueHeavy,
                key_bits: 2,
                value_bits: 4,
                group_size: 32,
                qjl_bits: 0,
            },
            key_error: 0.0,
            value_error: 0.0,
            attention_loss: 0.0,
            bytes_per_token: 1,
        };
        assert!(search.select_from_evidence([unsearched]).is_none());
    }

    #[test]
    fn evaluates_reference_cache_before_selecting_policy() {
        let search = KvCompressionSearch::default();
        let keys = vec![-1.0, -0.5, 0.25, 0.75, 1.0, 0.1, -0.2, 0.4];
        let values = vec![0.8, -0.3, 0.2, 1.2, -0.9, 0.4, 0.1, -0.6];
        let evidence = search.evaluate_reference_cache(&keys, &values).unwrap();
        assert_eq!(evidence.len(), search.candidates.len());
        assert!(evidence.iter().all(|entry| {
            entry.key_error.is_finite()
                && entry.value_error.is_finite()
                && entry.attention_loss.is_finite()
                && entry.bytes_per_token > 0
        }));
        let measured_search = KvCompressionSearch {
            max_error: 1.0,
            candidates: search.candidates.clone(),
        };
        assert!(measured_search.select_from_evidence(evidence).is_some());
    }
}
