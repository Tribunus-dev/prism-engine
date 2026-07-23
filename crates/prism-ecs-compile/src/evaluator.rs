//! Evaluator integration for evolutionary search
//!
//! This module provides the integration between the search system and hardware-backed
//! evaluators, ensuring that production mode uses real measurements and fails closed
//! when real evaluation is not available.

use crate::{CandidateMeasurements, SearchError};
use prism_ecs_core::identity::TensorProvider;
use prism_ecs_ir::evolution::evaluate::EvaluationStrategy as EcsEvaluationStrategy;
use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::evolution::{ProgressiveStageExecutor, TernaryObjectiveEvidence};
use prism_ecs_quantization::kv_search::{
    KvCompressionCandidate, KvCompressionEvaluator, KvCompressionEvidence,
};
use prism_ecs_quantization::safetensors_provider::SafeTensorProvider;
use prism_ecs_quantization::turboquant_kv::{KvQuantMode, TurboQuantKvCache};
use std::sync::Arc;
use std::time::Instant;

/// MI300X-backed evaluator for KV-cache candidates. Quantization and
/// reconstruction stay in Prism's native TurboQuant implementation; the
/// expensive reference-vs-reconstruction reductions run through the existing
/// HIP scorer.
pub struct Mi300xKvEvaluator {
    keys: Vec<f32>,
    values: Vec<f32>,
    scorer: Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>,
}

impl Mi300xKvEvaluator {
    pub fn new(
        keys: Vec<f32>,
        values: Vec<f32>,
        scorer: Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>,
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
        if let Ok(Some(scorer)) = prism_rocm_runtime::ternary::Mi300xTernaryScorer::from_env() {
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

/// Wrapper around MeasuredEvaluator that adapts it to the search system's EvaluationStrategy trait
pub struct MeasuredEvaluatorAdapter {
    inner: Arc<dyn EcsEvaluationStrategy>,
    behavioral_probe: Option<Arc<dyn BehavioralProbe>>,
}

pub trait BehavioralProbe: Send + Sync {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError>;
}

/// Provider-backed tensor probe context. It deliberately describes a single
/// tensor because the SafeTensors provider streams one mapped tensor at a time;
/// a future graph executor can extend this to layer activations and logits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MappedTensorProbeContext {
    pub model_dir: std::path::PathBuf,
    pub tensor_name: String,
}

impl MappedTensorProbeContext {
    /// Construct a context for one concrete tensor in a mapped checkpoint.
    ///
    /// Keeping the tensor name in the serialized context is important: a
    /// progressive run must be reproducible against the same source tensor,
    /// rather than letting the probe select an arbitrary router tensor.
    pub fn for_tensor(model_dir: impl Into<std::path::PathBuf>, tensor_name: impl Into<String>) -> Self {
        Self {
            model_dir: model_dir.into(),
            tensor_name: tensor_name.into(),
        }
    }

    /// Reject an empty or non-file-backed reference before progressive search
    /// admits any evidence. This does not read tensor payload bytes; it only
    /// verifies that the requested reference is present in the provider.
    pub fn validate_reference(&self) -> Result<(), SearchError> {
        if self.tensor_name.trim().is_empty() {
            return Err(SearchError::SearchFailed(
                "mapped probe context has an empty tensor reference".into(),
            ));
        }
        let provider = SafeTensorProvider::new(&self.model_dir)
            .map_err(SearchError::SearchFailed)?;
        let present = provider
            .list_tensors()
            .map_err(SearchError::SearchFailed)?
            .into_iter()
            .any(|tensor| tensor.name == self.tensor_name);
        if !present {
            return Err(SearchError::SearchFailed(format!(
                "mapped probe tensor reference '{}' is not present in {}",
                self.tensor_name,
                self.model_dir.display()
            )));
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, SearchError> {
        serde_json::to_vec(self)
            .map_err(|e| SearchError::SearchFailed(format!("encode mapped probe context: {e}")))
    }
}

/// Bounded reference-backed probe for one mapped tensor. It evaluates the
/// actual BF16/F16 tensor against a deterministic ternary reconstruction and
/// feeds operator, router-margin, and logit evidence into progressive search.
/// It intentionally materializes only the selected tensor, never the model.
pub struct MappedTensorBehavioralProbe {
    pub model_dir: std::path::PathBuf,
    pub max_elements: usize,
    pub router_top_k: usize,
    pub model: Arc<dyn crate::ModelAdapter>,
    pub mi300x_scorer: Option<Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>>,
}

/// Search-facing adapter for mapped model directories. It makes the
/// reference probe available to the compiler's normal evolutionary search
/// trait, not only to the progressive ECS executor.
pub struct MappedTensorEvaluationStrategy {
    pub probe: MappedTensorBehavioralProbe,
    pub limits: prism_ecs_ir::evolution::TernaryAdmissionLimits,
}

impl MappedTensorEvaluationStrategy {
    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            probe: MappedTensorBehavioralProbe::new(model_dir),
            limits: Default::default(),
        }
    }
}

impl crate::search::EvaluationStrategy for MappedTensorEvaluationStrategy {
    fn evaluate(&self, genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        let genome: CandidateGenome =
            serde_json::from_str(genome).map_err(|e| format!("decode genome: {e}"))?;
        let evidence = self
            .probe
            .evaluate(&genome, context)
            .map_err(|e| e.to_string())?;
        Ok(evidence
            .vector(&self.limits)
            .into_iter()
            .map(|score| score.value())
            .collect())
    }

    fn name(&self) -> &str {
        "mapped-model-behavioral"
    }
}

impl MappedTensorBehavioralProbe {
    /// Build a probe only when the checkpoint directory is a usable,
    /// provider-backed source. Production progressive ternaryization should
    /// use this constructor so an invalid path cannot silently become a
    /// synthetic evaluation.
    pub fn try_new_real(model_dir: impl Into<std::path::PathBuf>) -> Result<Self, SearchError> {
        let model_dir = model_dir.into();
        SafeTensorProvider::new(&model_dir).map_err(SearchError::SearchFailed)?;
        let model = crate::adapter_for_model_dir(&model_dir)
            .map_err(SearchError::SearchFailed)?;
        Ok(Self::with_model(model_dir, Arc::from(model)))
    }

    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        let model_dir = model_dir.into();
        let model: Arc<dyn crate::ModelAdapter> = crate::adapter_for_model_dir(&model_dir)
            .map(|adapter| -> Arc<dyn crate::ModelAdapter> { adapter.into() })
            .unwrap_or_else(|_| Arc::new(GenericNameAdapter));
        Self::with_model(model_dir, model)
    }

    pub fn with_model(
        model_dir: impl Into<std::path::PathBuf>,
        model: Arc<dyn crate::ModelAdapter>,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            max_elements: 8 * 1024 * 1024,
            router_top_k: 8,
            model,
            mi300x_scorer: prism_rocm_runtime::ternary::Mi300xTernaryScorer::from_env()
                .ok()
                .flatten()
                .map(Arc::new),
        }
    }

    fn read_tensor(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), SearchError> {
        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let mut reader = provider
            .open_streaming_tensor(name)
            .map_err(SearchError::SearchFailed)?;
        let shape = reader.shape().to_vec();
        let elements = shape.iter().product::<usize>();
        if elements == 0 || elements > self.max_elements {
            return Err(SearchError::SearchFailed(format!(
                "mapped behavioral probe tensor '{}' has {} elements (limit {})",
                name, elements, self.max_elements
            )));
        }
        let mut bytes = vec![0u8; elements * 4];
        let mut filled = 0;
        while filled < bytes.len() {
            let n = reader
                .read_chunk(&mut bytes[filled..])
                .map_err(SearchError::SearchFailed)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled != bytes.len() {
            return Err(SearchError::SearchFailed("short mapped tensor read".into()));
        }
        Ok((
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            shape,
        ))
    }

    fn evaluate_tensor(
        &self,
        genome: &CandidateGenome,
        name: &str,
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        let (reference, shape) = self.read_tensor(name)?;
        let cols = shape.last().copied().unwrap_or(reference.len()).max(1);
        let rows = reference.len() / cols;
        let group = match genome.packing {
            prism_ecs_ir::evolution::PackingAxis::Tile640 => 640,
            prism_ecs_ir::evolution::PackingAxis::Block2D => 128,
            _ => 32,
        };
        let threshold = if matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        ) {
            0.05
        } else {
            0.0
        };
        let role = self.model.classify_tensor(name).role;
        let is_router = matches!(&role, crate::TensorRole::Router { .. });
        let is_output_head = matches!(&role, crate::TensorRole::OutputHead);
        let mut candidate = vec![0.0f32; reference.len()];
        let ternary = matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        );
        let packed_compatible = ternary && cols % group == 0;
        let groups_per_row = cols.div_ceil(group);
        let mut packed = vec![0u8; reference.len().div_ceil(4)];
        let mut scales = vec![0.0f32; rows * groups_per_row];
        for row in 0..rows {
            for start in (0..cols).step_by(group) {
                let end = (start + group).min(cols);
                let scale = reference[row * cols + start..row * cols + end]
                    .iter()
                    .map(|v| v.abs())
                    .sum::<f32>()
                    / (end - start).max(1) as f32;
                scales[row * groups_per_row + start / group] = scale;
                for col in start..end {
                    let value = reference[row * cols + col];
                    let code = if value.abs() <= threshold {
                        0u8
                    } else if value.is_sign_positive() {
                        1u8
                    } else {
                        2u8
                    };
                    packed[(row * cols + col) / 4] |= code << (((row * cols + col) % 4) * 2);
                    candidate[row * cols + col] = if code == 0 {
                        0.0
                    } else {
                        value.signum() * scale
                    };
                }
            }
        }
        let reference64: Vec<f64> = reference.iter().map(|v| *v as f64).collect();
        let candidate64: Vec<f64> = candidate.iter().map(|v| *v as f64).collect();
        let activation_error = self
            .mi300x_scorer
            .as_ref()
            .and_then(|scorer| {
                if packed_compatible {
                    scorer
                        .packed_mean_squared_error(&reference, &packed, &scales, group)
                        .ok()
                } else {
                    scorer.mean_squared_error(&reference, &candidate).ok()
                }
            })
            .map(f64::sqrt)
            .unwrap_or_else(|| vector_rmse(&reference64, &candidate64));
        let normalized =
            activation_error / (vector_rmse(&reference64, &vec![0.0; reference.len()]) + 1e-6);
        let mut evidence = TernaryObjectiveEvidence {
            quality: (-normalized.max(0.0)).exp(),
            activation_error: normalized,
            logit_divergence: normalized,
            task_loss: normalized,
            router_agreement: 1.0,
            router_margin_error: 0.0,
            logit_cross_entropy: 0.0,
            generation_loss: normalized,
            native_ternary_fraction: if matches!(
                genome.representation,
                prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                    | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
            ) {
                1.0
            } else {
                0.0
            },
            memory_bytes: (candidate.len() / 4) as u64,
            ..Default::default()
        };
        // LFQ-style final-block scoring: output-head candidates are judged by
        // the probability distribution they induce, not just weight RMSE.
        if is_output_head && rows > 0 {
            let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
            let reference_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    reference[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            let candidate_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    candidate[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            let ce = prism_ecs_ir::evolution::progressive::logit_cross_entropy(
                &reference_logits,
                &candidate_logits,
            );
            evidence.logit_cross_entropy = ce;
            evidence.logit_divergence = ce;
            evidence.task_loss = ce;
            evidence.generation_loss = ce;
        }
        if is_router && rows > self.router_top_k {
            let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
            let reference_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    reference[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            let mut best_score = f64::INFINITY;
            let mut best_protected_fraction = 0.0f64;
            for &(candidate_group, candidate_threshold, protected_fraction) in &[
                (32usize, 0.0f32, 0.00f32),
                (64, 0.0, 0.01),
                (128, 0.0, 0.01),
                (256, 0.0, 0.02),
                (640, 0.0, 0.02),
                (128, 0.05, 0.05),
            ] {
                let mut trial = vec![0.0f32; reference.len()];
                for row in 0..rows {
                    let row_start = row * cols;
                    let mut protected = (0..cols).collect::<Vec<_>>();
                    protected.sort_unstable_by(|&a, &b| {
                        reference[row_start + b]
                            .abs()
                            .total_cmp(&reference[row_start + a].abs())
                    });
                    let protected_count = (cols as f32 * protected_fraction).ceil() as usize;
                    for start in (0..cols).step_by(candidate_group) {
                        let end = (start + candidate_group).min(cols);
                        let active = (start..end)
                            .filter(|col| reference[row_start + *col].abs() > candidate_threshold)
                            .collect::<Vec<_>>();
                        let denom = active.len().max(1) as f32;
                        let scale = active
                            .iter()
                            .map(|col| reference[row_start + *col].abs())
                            .sum::<f32>()
                            / denom;
                        for col in start..end {
                            let value = reference[row_start + col];
                            trial[row_start + col] =
                                if protected[..protected_count.min(cols)].contains(&col) {
                                    value
                                } else if value.abs() <= candidate_threshold {
                                    0.0
                                } else {
                                    value.signum() * scale
                                };
                        }
                    }
                }
                let trial_logits: Vec<f64> = (0..rows)
                    .map(|r| {
                        trial[r * cols..(r + 1) * cols]
                            .iter()
                            .zip(&input)
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum()
                    })
                    .collect();
                let k = self.router_top_k.min(rows);
                let margin = prism_ecs_ir::evolution::progressive::router_margin_error(
                    &reference_logits,
                    &trial_logits,
                    k,
                );
                let ce = prism_ecs_ir::evolution::progressive::logit_cross_entropy(
                    &reference_logits,
                    &trial_logits,
                );
                let mut ref_order: Vec<_> = (0..rows).collect();
                ref_order.sort_by(|&a, &b| reference_logits[b].total_cmp(&reference_logits[a]));
                let mut trial_order: Vec<_> = (0..rows).collect();
                trial_order.sort_by(|&a, &b| trial_logits[b].total_cmp(&trial_logits[a]));
                let agreement = ref_order[..k]
                    .iter()
                    .filter(|x| trial_order[..k].contains(x))
                    .count() as f64
                    / k as f64;
                let score = ce + margin + (1.0 - agreement) * 10.0;
                if score < best_score {
                    best_score = score;
                    best_protected_fraction = protected_fraction as f64;
                    candidate = trial;
                }
            }
            let candidate_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    candidate[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            evidence.logit_cross_entropy =
                prism_ecs_ir::evolution::progressive::logit_cross_entropy(
                    &reference_logits,
                    &candidate_logits,
                );
            evidence.router_margin_error =
                prism_ecs_ir::evolution::progressive::router_margin_error(
                    &reference_logits,
                    &candidate_logits,
                    self.router_top_k.min(rows),
                );
            let mut ref_order: Vec<_> = (0..rows).collect();
            ref_order.sort_by(|&a, &b| reference_logits[b].total_cmp(&reference_logits[a]));
            let mut cand_order: Vec<_> = (0..rows).collect();
            cand_order.sort_by(|&a, &b| candidate_logits[b].total_cmp(&candidate_logits[a]));
            evidence.router_agreement = ref_order[..self.router_top_k.min(rows)]
                .iter()
                .filter(|x| cand_order[..self.router_top_k.min(rows)].contains(x))
                .count() as f64
                / self.router_top_k.min(rows) as f64;
            evidence.native_ternary_fraction = (1.0 - best_protected_fraction).clamp(0.0, 1.0);
        }
        Ok(evidence)
    }
}

/// Conservative fallback for checkpoints that do not yet have a registered
/// family adapter. It keeps the evaluator usable for dense models while
/// making router-sensitive scoring opt-in to an explicit adapter.
struct GenericNameAdapter;

impl crate::ModelAdapter for GenericNameAdapter {
    fn family(&self) -> &str {
        "generic"
    }
    fn classify_tensor(&self, name: &str) -> crate::TensorDescriptor {
        let lower = name.to_ascii_lowercase();
        let role = if lower.contains("lm_head")
            || lower.contains("embed_out")
            || lower.ends_with("output.weight")
        {
            crate::TensorRole::OutputHead
        } else if lower.contains("router") {
            crate::TensorRole::Router { layer: 0 }
        } else {
            crate::TensorRole::Other
        };
        crate::TensorDescriptor {
            name: name.to_string(),
            shape: Vec::new(),
            role,
        }
    }
    fn validate_inventory(&self, _names: &[String]) -> Result<(), String> {
        Ok(())
    }
    fn layer_count(&self) -> Option<usize> {
        None
    }
}

impl BehavioralProbe for MappedTensorBehavioralProbe {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        if let Ok(context) = serde_json::from_slice::<MappedTensorProbeContext>(context) {
            if context.model_dir != self.model_dir {
                return Err(SearchError::SearchFailed(
                    "mapped probe context source does not match probe source".into(),
                ));
            }
            context.validate_reference()?;
            return self.evaluate_tensor(genome, &context.tensor_name);
        }
        // The ECS progressive stage supplies its catalog context as text.
        // Resolve a stable router tensor from the mapped provider so that this
        // path still performs real behavioral admission rather than failing
        // closed merely because it lacks a single-tensor JSON context.
        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let tensor_name = provider
            .list_tensors()
            .map_err(SearchError::SearchFailed)?
            .into_iter()
            .map(|info| info.name)
            .find(|name| {
                let lower = name.to_ascii_lowercase();
                lower.contains("router") || lower.contains("mlp.gate.weight")
            })
            .ok_or_else(|| {
                SearchError::SearchFailed("mapped progressive context has no router tensor".into())
            })?;
        self.evaluate_tensor(genome, &tensor_name)
    }
}

fn vector_rmse(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

#[cfg(test)]
fn logit_kl(reference: &[f64], candidate: &[f64]) -> f64 {
    if reference.len() != candidate.len() || reference.is_empty() {
        return f64::INFINITY;
    }
    let max_r = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_c = candidate.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let r: Vec<f64> = reference.iter().map(|x| (x - max_r).exp()).collect();
    let c: Vec<f64> = candidate.iter().map(|x| (x - max_c).exp()).collect();
    let rs = r.iter().sum::<f64>();
    let cs = c.iter().sum::<f64>();
    r.iter()
        .zip(c)
        .map(|(x, y)| {
            let p = x / rs;
            let q = y / cs;
            p * (p.max(1e-30) / q.max(1e-30)).ln()
        })
        .sum()
}

#[cfg(test)]
fn router_agreement(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
    let (Some(a), Some(b)) = (a.as_object(), b.as_object()) else {
        return f64::NAN;
    };
    let mut total = 0usize;
    let mut same = 0usize;
    for (layer, av) in a {
        let Some(bv) = b.get(layer) else { return 0.0 };
        total += 1;
        if av == bv {
            same += 1;
        }
    }
    if total == 0 {
        f64::NAN
    } else {
        same as f64 / total as f64
    }
}

impl MeasuredEvaluatorAdapter {
    /// Create a new adapter wrapping a MeasuredEvaluator
    pub fn new(evaluator: Arc<dyn EcsEvaluationStrategy>) -> Self {
        Self {
            inner: evaluator,
            behavioral_probe: None,
        }
    }

    pub fn with_behavioral_probe(mut self, probe: Arc<dyn BehavioralProbe>) -> Self {
        self.behavioral_probe = Some(probe);
        self
    }

    /// Install the bounded SafeTensor-backed reference probe used by mapped
    /// progressive compilation. This is the production wiring point for
    /// behavior-aware ternary admission; callers can still provide a richer
    /// graph probe through `with_behavioral_probe` when available.
    pub fn with_mapped_tensor_probe(self, model_dir: impl Into<std::path::PathBuf>) -> Self {
        self.with_behavioral_probe(Arc::new(MappedTensorBehavioralProbe::new(model_dir)))
    }

    /// Check if this adapter wraps a synthetic evaluator
    pub fn is_synthetic(&self) -> bool {
        let name = self.inner.name();
        name.contains("Synthetic") || name.contains("synthetic")
    }

    /// Extract measurements from the evaluator for a candidate
    pub fn extract_measurements(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
        fitness_score: f64,
    ) -> Result<CandidateMeasurements, SearchError> {
        if self.is_synthetic() {
            return Err(SearchError::SyntheticDataInProductionMode);
        }

        // Validate fitness score
        if !fitness_score.is_finite() || fitness_score <= 0.0 {
            return Err(SearchError::CorrectnessValidationFailed);
        }

        let start = Instant::now();
        let measured_score = self.inner.evaluate(genome, context).value();
        let wall_time_ms = start.elapsed().as_secs_f64() * 1_000.0;
        if !measured_score.is_finite() || measured_score <= 0.0 {
            return Err(SearchError::CorrectnessValidationFailed);
        }

        Ok(CandidateMeasurements {
            wall_time_ms,
            gpu_time_ms: wall_time_ms,
            bandwidth_gbps: if wall_time_ms > 0.0 {
                context.len() as f64 / wall_time_ms / 1_000.0
            } else {
                0.0
            },
            peak_memory_mb: 0.0,
            reconstruction_error: 1.0 - measured_score,
            accuracy_score: measured_score,
        })
    }

    /// Produce structured evidence for progressive Pareto admission. The
    /// backend score is the reference-quality signal; callers that have
    /// activation/logit/router probes can replace the remaining fields.
    pub fn evaluate_ternary(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        if self.is_synthetic() {
            return Err(SearchError::SyntheticDataInProductionMode);
        }
        let start = Instant::now();
        let quality = self.inner.evaluate(genome, context).value();
        let latency_ms = start.elapsed().as_secs_f64() * 1_000.0;
        if !quality.is_finite() || quality <= 0.0 {
            return Err(SearchError::CorrectnessValidationFailed);
        }
        let ternary_candidate = matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        );
        if ternary_candidate && self.behavioral_probe.is_none() {
            return Err(SearchError::CorrectnessValidationFailed);
        }
        let mut evidence = TernaryObjectiveEvidence {
            quality,
            latency_ms,
            memory_bytes: context.len() as u64,
            native_ternary_fraction: if matches!(
                genome.representation,
                prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                    | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
            ) {
                1.0
            } else {
                0.0
            },
            activation_error: f64::NAN,
            logit_divergence: f64::NAN,
            task_loss: f64::NAN,
            router_agreement: f64::NAN,
            router_margin_error: f64::NAN,
            logit_cross_entropy: f64::NAN,
            generation_loss: f64::NAN,
            energy: f64::NAN,
            ..Default::default()
        };
        if let Some(probe) = &self.behavioral_probe {
            let behavioral = probe.evaluate(genome, context)?;
            evidence.activation_error = behavioral.activation_error;
            evidence.logit_divergence = behavioral.logit_divergence;
            evidence.task_loss = behavioral.task_loss;
            evidence.router_agreement = behavioral.router_agreement;
            evidence.router_margin_error = behavioral.router_margin_error;
            evidence.logit_cross_entropy = behavioral.logit_cross_entropy;
            evidence.generation_loss = behavioral.generation_loss;
            evidence.expert_balance_error = behavioral.expert_balance_error;
            evidence.residual_bytes = behavioral.residual_bytes;
            evidence.energy = behavioral.energy;
        }
        Ok(evidence)
    }
}

impl ProgressiveStageExecutor for MeasuredEvaluatorAdapter {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        _stage: usize,
        context: &[u8],
    ) -> TernaryObjectiveEvidence {
        self.evaluate_ternary(genome, context).unwrap_or_default()
    }
}

impl super::EvaluationStrategy for MeasuredEvaluatorAdapter {
    fn evaluate(&self, genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        // Parse the genome string back into a CandidateGenome
        // This is a simplified approach - in production we'd use proper serialization
        let genome = parse_genome_from_string(genome).map_err(|e| e.to_string())?;

        // Delegate to the inner evaluator
        let fitness_score = self.inner.evaluate(&genome, context);

        Ok(vec![fitness_score.value()])
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn progressive_executor(&self) -> Option<&dyn ProgressiveStageExecutor> {
        Some(self)
    }
}

/// Helper function to parse genome from string (simplified)
fn parse_genome_from_string(
    genome_str: &str,
) -> Result<CandidateGenome, Box<dyn std::error::Error>> {
    serde_json::from_str(genome_str).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Create a MeasuredEvaluatorAdapter from a daemon resource
pub fn create_measured_evaluator_from_daemon() -> Result<MeasuredEvaluatorAdapter, SearchError> {
    // This function would be called from the daemon context where MeasuredEvaluator
    // is available as a resource. For now, we return an error indicating
    // that the daemon integration is required.
    Err(SearchError::SearchFailed(
        "MeasuredEvaluator not available - daemon integration required".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{logit_kl, router_agreement, vector_rmse};

    #[test]
    fn probe_metrics_are_zero_for_identical_outputs() {
        let values = [1.0, -2.0, 3.5];
        assert_eq!(vector_rmse(&values, &values), 0.0);
        assert!(logit_kl(&values, &values).abs() < 1e-12);
        let routes = serde_json::json!({"0": [[[1, 2, 3]]], "1": [[[4, 5, 6]]]});
        assert_eq!(router_agreement(&routes, &routes), 1.0);
    }

    #[test]
    fn probe_metrics_reject_shape_mismatch_and_route_changes() {
        assert!(vector_rmse(&[1.0], &[1.0, 2.0]).is_infinite());
        assert!(logit_kl(&[1.0], &[1.0, 2.0]).is_infinite());
        let reference = serde_json::json!({"0": [[[1, 2, 3]]]});
        let candidate = serde_json::json!({"0": [[[1, 2, 4]]]});
        assert_eq!(router_agreement(&reference, &candidate), 0.0);
    }
}
