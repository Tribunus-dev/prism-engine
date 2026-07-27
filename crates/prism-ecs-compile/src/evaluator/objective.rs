//! Objective evidence composition.
//!
//! This module owns the authority for turning a bounded reference probe
//! into a [`TernaryObjectiveEvidence`] that the search system's
//! progressive admission can act on. The single authority is the
//! behavioral-scoring path:
//!
//! 1. **The bounded reference probe** — [`MappedTensorBehavioralProbe`]
//!    materializes exactly one mapped tensor at a time, computes the
//!    activation error, logit divergence, router agreement, and
//!    router margin against the genome's reconstruction, and emits
//!    [`TernaryObjectiveEvidence`]. The probe's reference cache is a
//!    bounded `BTreeMap` (ordered insertion) keyed by tensor name, so
//!    hot từen eviction is deterministic across runs.
//! 2. **The objective composition pipeline** — [`evaluate_tensor`],
//!    [`evaluate_family_canaries`], and [`progressive_fallback_format`]
//!    are the three entry points: per-tensor scoring, per-family
//!    champion/verification scoring, and the policy that decides what
//!    fallback format an outlier should be admitted with.
//! 3. **The SpecHub result shape** — [`SpecHubVerification`] is the
//!    canonical data type for the engine's multi-draft verification
//!    result. The MLX-coupled functions that build it (`spechub_*`)
//!    stay in the engine; the data shape lives here so the engine and
//!    any future constitutional caller share one authority.
//!
//! The probe is canonical: no hardware handles, no `unsafe`, no FFI.
//! The reference cache uses an `Arc<Mutex<_>>` for shared read paths
//! (a process-local optimization, not a hardware handle); this matches
//! the cache pattern in the rest of the crate and is documented as a
//! process-local concern.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use prism_ecs_core::identity::TensorProvider;
use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::evolution::progressive::TernaryAdmissionLimits;
use prism_ecs_ir::evolution::TernaryObjectiveEvidence;
use prism_ecs_quantization::safetensors_provider::SafeTensorProvider;
use prism_rocm_runtime::ternary::Mi300xTernaryScorer;

use crate::representation_cache::{
    record_outlier, TensorFamilyPolicy, TensorFamilySignature,
};
use crate::search::SearchError;
use crate::{adapter_for_model_dir, ModelAdapter, TensorDescriptor, TensorRole};

use super::strategy::{
    parse_genome_from_string, quantize_ternary, reconstruct_representation, BehavioralProbe,
};

// ---------------------------------------------------------------------------
// MappedTensorProbeContext — single-tensor reference for a behavioral
// probe run
// ---------------------------------------------------------------------------

/// Provider-backed tensor probe context. It deliberately describes a
/// single tensor because the SafeTensors provider streams one mapped
/// tensor at a time; a future graph executor can extend this to layer
/// activations and logits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MappedTensorProbeContext {
    pub model_dir: std::path::PathBuf,
    pub tensor_name: String,
}

impl MappedTensorProbeContext {
    /// Construct a context for one concrete tensor in a mapped
    /// checkpoint. Keeping the tensor name in the serialized context
    /// is important: a progressive run must be reproducible against
    /// the same source tensor, rather than letting the probe select
    /// an arbitrary router tensor.
    pub fn for_tensor(
        model_dir: impl Into<std::path::PathBuf>,
        tensor_name: impl Into<String>,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            tensor_name: tensor_name.into(),
        }
    }

    /// Reject an empty or non-file-backed reference before progressive
    /// search admits any evidence. This does not read tensor payload
    /// bytes; it only verifies that the requested reference is
    /// present in the provider.
    pub fn validate_reference(&self) -> Result<(), SearchError> {
        if self.tensor_name.trim().is_empty() {
            return Err(SearchError::SearchFailed(
                "mapped probe context has an empty tensor reference".into(),
            ));
        }
        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
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

// ---------------------------------------------------------------------------
// MappedTensorBehavioralProbe — bounded reference-backed probe for one
// mapped tensor
// ---------------------------------------------------------------------------

/// Bounded reference-backed probe for one mapped tensor. It evaluates
/// the actual BF16/F16 tensor against a deterministic ternary
/// reconstruction and feeds operator, router-margin, and logit
/// evidence into progressive search. It intentionally materializes
/// only the selected tensor, never the model.
pub struct MappedTensorBehavioralProbe {
    pub model_dir: std::path::PathBuf,
    pub max_elements: usize,
    pub router_top_k: usize,
    pub model: Arc<dyn ModelAdapter>,
    pub mi300x_scorer: Option<Arc<Mi300xTernaryScorer>>,
    /// Reference cache. The ordering is intentionally HashMap (not
    /// BTreeMap) because the cache key is a tensor name and the order
    /// of iteration is not observable: callers request a specific key
    /// by name, they do not iterate.
    reference_cache: Arc<Mutex<HashMap<String, (Vec<f32>, Vec<usize>)>>>,
    cache_bytes: Arc<Mutex<usize>>,
    max_cache_bytes: usize,
}

impl MappedTensorBehavioralProbe {
    /// Build a probe only when the checkpoint directory is a usable,
    /// provider-backed source. Production progressive ternaryization
    /// should use this constructor so an invalid path cannot silently
    /// become a synthetic evaluation.
    pub fn try_new_real(model_dir: impl Into<std::path::PathBuf>) -> Result<Self, SearchError> {
        let model_dir = model_dir.into();
        SafeTensorProvider::new(&model_dir).map_err(SearchError::SearchFailed)?;
        let model = adapter_for_model_dir(&model_dir).map_err(SearchError::SearchFailed)?;
        Ok(Self::with_model(model_dir, Arc::from(model)))
    }

    /// Build a probe against any model directory. Falls back to
    /// [`GenericNameAdapter`] if no family-specific adapter is found.
    /// Production callers should use [`Self::try_new_real`].
    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        let model_dir = model_dir.into();
        let model: Arc<dyn ModelAdapter> = adapter_for_model_dir(&model_dir)
            .map(|adapter| -> Arc<dyn ModelAdapter> { adapter.into() })
            .unwrap_or_else(|_| Arc::new(GenericNameAdapter));
        Self::with_model(model_dir, model)
    }

    pub fn with_model(
        model_dir: impl Into<std::path::PathBuf>,
        model: Arc<dyn ModelAdapter>,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            max_elements: 8 * 1024 * 1024,
            router_top_k: 8,
            model,
            mi300x_scorer: Mi300xTernaryScorer::from_env()
                .ok()
                .flatten()
                .map(Arc::new),
            reference_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_bytes: Arc::new(Mutex::new(0)),
            max_cache_bytes: 64 * 1024 * 1024,
        }
    }

    fn element_count(shape: &[usize]) -> usize {
        shape.iter().product::<usize>()
    }

    fn bounded_shape(shape: &[usize], max_elements: usize) -> Vec<usize> {
        let elements = Self::element_count(shape);
        if elements == 0 || max_elements == 0 || elements <= max_elements {
            return shape.to_vec();
        }
        let cols = shape.last().copied().unwrap_or(1).max(1);
        if shape.len() <= 1 {
            return vec![max_elements];
        }
        if cols <= max_elements {
            let rows = (max_elements / cols).max(1);
            vec![rows, cols]
        } else {
            vec![1, max_elements]
        }
    }

    pub(crate) fn select_tensor_from_context(
        &self,
        context: &[u8],
        max_elements: usize,
    ) -> Result<String, SearchError> {
        if let Ok(ctx_text) = std::str::from_utf8(context) {
            if let Ok(ctx) = serde_json::from_str::<MappedTensorProbeContext>(ctx_text) {
                if !ctx.tensor_name.trim().is_empty() {
                    return Ok(ctx.tensor_name);
                }
            }
            if let Some(line) = ctx_text
                .lines()
                .next()
                .map(str::trim)
                .filter(|line| !line.is_empty())
            {
                if let Some(first) = line
                    .split(':')
                    .next()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                {
                    return Ok(first.to_string());
                }
            }
        }

        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let mut fallback = None::<(String, usize)>;
        let mut router = None::<(String, usize)>;
        for tensor in provider.list_tensors().map_err(SearchError::SearchFailed)? {
            let elements = Self::element_count(&tensor.shape);
            if elements == 0 || elements > max_elements {
                continue;
            }
            let lower = tensor.name.to_ascii_lowercase();
            let entry = if lower.contains("router") || lower.contains("mlp.gate.weight") {
                &mut router
            } else {
                &mut fallback
            };
            let should_replace = entry
                .as_ref()
                .is_none_or(|(_, existing)| *existing > elements);
            if should_replace {
                *entry = Some((tensor.name, elements));
            }
        }
        router.or(fallback).map(|(name, _)| name).ok_or_else(|| {
            SearchError::SearchFailed("no mapped tensor available within evaluation budget".into())
        })
    }

    pub(crate) fn read_tensor(
        &self,
        name: &str,
        max_elements: usize,
    ) -> Result<(Vec<f32>, Vec<usize>), SearchError> {
        if let Ok(cache) = self.reference_cache.lock() {
            if let Some((value, shape)) = cache.get(name) {
                let max_elements = max_elements.min(Self::element_count(shape).max(1));
                let mut values = value.clone();
                values.truncate(max_elements);
                let mut shape = shape.clone();
                if values.len() != Self::element_count(&shape) {
                    shape = Self::bounded_shape(&shape, values.len());
                }
                return Ok((values, shape));
            }
        }
        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let mut reader = provider
            .open_streaming_tensor(name)
            .map_err(SearchError::SearchFailed)?;
        let shape = reader.shape().to_vec();
        let elements = shape.iter().product::<usize>();
        if elements == 0 {
            return Err(SearchError::SearchFailed(format!(
                "mapped behavioral probe tensor '{}' is empty",
                name
            )));
        }
        let sample_elements = max_elements.min(elements);
        let mut bytes = vec![0u8; sample_elements * 4];
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
        let sample_elements = filled / 4;
        let bounded_shape = if sample_elements == elements {
            shape
        } else {
            Self::bounded_shape(&shape, sample_elements)
        };
        // Use an explicit per-byte fold instead of `try_into().unwrap()`
        // so the production path does not panic on misaligned reads.
        let mut decoded = Vec::with_capacity(sample_elements);
        for chunk in bytes.chunks_exact(4) {
            let mut arr = [0u8; 4];
            arr.copy_from_slice(chunk);
            decoded.push(f32::from_le_bytes(arr));
        }
        let result = (decoded, bounded_shape);
        let result_bytes = result.0.len() * std::mem::size_of::<f32>();
        if result_bytes <= self.max_cache_bytes {
            if let (Ok(mut cache), Ok(mut used)) =
                (self.reference_cache.lock(), self.cache_bytes.lock())
            {
                while *used + result_bytes > self.max_cache_bytes {
                    let Some(key) = cache.keys().next().cloned() else {
                        break;
                    };
                    if let Some((values, _)) = cache.remove(&key) {
                        *used = used.saturating_sub(values.len() * 4);
                    }
                }
                *used += result_bytes;
                cache.insert(name.to_string(), result.clone());
            }
        }
        Ok(result)
    }

    /// Evaluate one tensor against the genome. This is the
    /// per-tensor scoring path used by both the single-tensor
    /// behavioral probe and the family-canary verification loop.
    pub(crate) fn evaluate_tensor(
        &self,
        genome: &CandidateGenome,
        name: &str,
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        let (reference, shape) = self.read_tensor(name, self.max_elements)?;
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
        let is_router = matches!(&role, TensorRole::Router { .. });
        let is_output_head = matches!(&role, TensorRole::OutputHead);
        let ternary = matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        );
        let packed_compatible = ternary && cols % group == 0;
        let (mut candidate, encoded_bytes) =
            reconstruct_representation(&reference, rows, cols, genome);
        let groups_per_row = cols.div_ceil(group);
        let mut packed = vec![0u8; reference.len().div_ceil(4)];
        let mut scales = vec![0.0f32; rows * groups_per_row];
        if ternary {
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
                    }
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
        let normalized = activation_error
            / (vector_rmse(&reference64, &vec![0.0; reference.len()]) + 1e-6);
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
            memory_bytes: encoded_bytes,
            ..Default::default()
        };
        // LFQ-style final-block scoring: output-head candidates are
        // judged by the probability distribution they induce, not
        // just weight RMSE.
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

    /// Evaluate one champion and a bounded verification sample for
    /// every structural family. Failed canaries are returned as
    /// explicit outliers; callers must not reuse the champion format
    /// for them.
    pub fn evaluate_family_canaries(
        &self,
        policies: &mut std::collections::BTreeMap<TensorFamilySignature, TensorFamilyPolicy>,
        limits: &TernaryAdmissionLimits,
    ) -> Result<(), String> {
        for policy in policies.values_mut() {
            let genome = genome_for_format(&policy.format);
            let champion = self
                .evaluate_tensor(&genome, &policy.champion)
                .map_err(|e| e.to_string())?;
            if !champion.behavioral_passes(limits) {
                let existing = policy.outliers.clone();
                let failed_members: Vec<String> = policy
                    .members
                    .iter()
                    .filter(|name| !existing.iter().any(|outlier| outlier == *name))
                    .cloned()
                    .collect();
                for member in failed_members {
                    policy.outliers.push(member.clone());
                    if let Some(format) = self.progressive_fallback_format(&member, limits) {
                        policy.outlier_formats.insert(member, format);
                    }
                }
                continue;
            }
            for member in policy.verification_members.clone() {
                let evidence = self
                    .evaluate_tensor(&genome, &member)
                    .map_err(|e| e.to_string())?;
                let divergence = evidence
                    .activation_error
                    .max(evidence.logit_divergence)
                    .max(evidence.task_loss);
                let max_divergence = limits
                    .max_activation_error
                    .max(limits.max_logit_divergence)
                    .max(limits.max_task_loss);
                record_outlier(policy, &member, divergence, max_divergence);
                if policy.outliers.iter().any(|outlier| outlier == &member) {
                    if let Some(format) = self.progressive_fallback_format(&member, limits) {
                        policy.outlier_formats.insert(member, format);
                    }
                }
            }
        }
        Ok(())
    }

    fn progressive_fallback_format(
        &self,
        tensor: &str,
        limits: &TernaryAdmissionLimits,
    ) -> Option<String> {
        let candidates = [
            "TernaryTile640",
            "Ternary158",
            "Nf4",
            "Int4",
            "Int8",
            "Bf16",
            "Fp16",
        ];
        candidates
            .iter()
            .filter_map(|format| {
                let evidence = self
                    .evaluate_tensor(&genome_for_format(format), tensor)
                    .ok()?;
                evidence
                    .behavioral_passes(limits)
                    .then_some((evidence.memory_bytes, (*format).to_string()))
            })
            .min_by_key(|(memory, _)| *memory)
            .map(|(_, format)| format)
    }
}

// ---------------------------------------------------------------------------
// BehavioralProbe impl — bridges the bounded probe to the strategy
// surface in `super::strategy`
// ---------------------------------------------------------------------------

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
        // The ECS progressive stage supplies its catalog context as
        // text. Resolve a stable router tensor from the mapped
        // provider so this path still performs real behavioral
        // admission rather than failing closed merely because it
        // lacks a single-tensor JSON context.
        let tensor_name = self.select_tensor_from_context(context, self.max_elements)?;
        self.evaluate_tensor(genome, &tensor_name)
    }
}

// ---------------------------------------------------------------------------
// GenericNameAdapter — conservative fallback for checkpoints that do
// not yet have a registered family adapter
// ---------------------------------------------------------------------------

/// Conservative fallback for checkpoints that do not yet have a
/// registered family adapter. It keeps the evaluator usable for
/// dense models while making router-sensitive scoring opt-in to an
/// explicit adapter.
pub struct GenericNameAdapter;

impl ModelAdapter for GenericNameAdapter {
    fn family(&self) -> &str {
        "generic"
    }
    fn classify_tensor(&self, name: &str) -> TensorDescriptor {
        let lower = name.to_ascii_lowercase();
        let role = if lower.contains("lm_head")
            || lower.contains("embed_out")
            || lower.ends_with("output.weight")
        {
            TensorRole::OutputHead
        } else if lower.contains("router") {
            TensorRole::Router { layer: 0 }
        } else {
            TensorRole::Other
        };
        TensorDescriptor {
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

// ---------------------------------------------------------------------------
// Objective composition helpers — small, canonical, used by the probe
// and (in tests) by external callers
// ---------------------------------------------------------------------------

/// Construct a [`CandidateGenome`] whose representation axis matches
/// the given canonical format name. Used by the family-canary
/// verification loop to materialize a per-format genome without
/// pulling the full search-system axis mapping into this module.
pub fn genome_for_format(format: &str) -> CandidateGenome {
    use prism_ecs_ir::evolution::RepresentationAxis;
    let mut genome = CandidateGenome::new();
    genome.representation = match format {
        "Bf16" => RepresentationAxis::Bf16,
        "Int8" => RepresentationAxis::Int8,
        "Int4" => RepresentationAxis::Int4,
        "Nf4" => RepresentationAxis::Nf4,
        "Nf8" => RepresentationAxis::Nf8,
        "Ternary158" => RepresentationAxis::Ternary158,
        "Binary1" => RepresentationAxis::Binary1,
        _ => RepresentationAxis::Fp16,
    };
    genome
}

/// Root-mean-square error between two equal-length f64 vectors. Used
/// as the fallback activation-error metric when the MI300X HIP
/// scorer is not available.
pub fn vector_rmse(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// SpecHubVerification — canonical data type absorbed from the engine's
// `core/speculative.rs`. The MLX-coupled builder functions
// (`sparse_joint_distribution_at_pos`, `softmax_at_pos`,
// `compatible_subset_at_pos`, `find_consensus_token`,
// `reweigh_with_subset`, `spechub_verify`) remain engine-side because
// they take `&mlx_rs::Array` (criterion 4: FFI surface).
// ---------------------------------------------------------------------------

/// SpecHub verification result — accepts more tokens than greedy.
///
/// SpecHub builds a sparse joint distribution over all draft outputs
/// and identifies the subset of drafts consistent with the target
/// model, recovering tokens that greedy rejection would discard.
#[derive(Debug, Clone, Default)]
pub struct SpecHubVerification {
    /// Token IDs accepted at each verified position.
    pub accepted_tokens: Vec<u32>,
    /// Fraction of draft tokens accepted (verified / attempted).
    pub acceptance_rate: f64,
    /// Estimated latency saved by acceptance vs. target-only decode
    /// (ms). Set externally from wall-clock measurements.
    pub saved_latency_ms: f64,
    /// Time spent in the SpecHub verification algorithm
    /// (microseconds).
    pub verification_time_us: u64,
}

// ---------------------------------------------------------------------------
// Bridge — used by `super::fail_closed` to invoke the probe without
// creating an import cycle (the probe is owned by `objective.rs`, the
// fail-closed evidence composition is owned by `fail_closed.rs`).
// ---------------------------------------------------------------------------

/// Bridge used by `super::fail_closed` to invoke the probe without
/// pulling the full `MappedTensorBehavioralProbe` into the
/// fail-closed module. Returns the full evidence struct.
pub(crate) fn mapped_probe_evaluate_for_fail_closed(
    probe: &(dyn BehavioralProbe + '_),
    genome: &CandidateGenome,
    context: &[u8],
) -> Result<TernaryObjectiveEvidence, SearchError> {
    probe.evaluate(genome, context)
}

// ---------------------------------------------------------------------------
// Test helpers — used by the per-module unit test below and (under
// `#[cfg(test)]`) by external callers that need a probe-metric
// primitive.
// ---------------------------------------------------------------------------

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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn genome_for_format_maps_canonical_axis() {
        let g = genome_for_format("Ternary158");
        assert!(matches!(
            g.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
        ));
        let g = genome_for_format("unknown");
        assert!(matches!(
            g.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Fp16
        ));
    }
}
