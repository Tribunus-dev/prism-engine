//! Stage 1 weight-only sweep runner.
//!
//! Loads a model's safetensors, matches tensor selectors against the tensor
//! inventory, generates all candidate format variants for each selected tensor,
//! runs weight-only validation, scores each candidate, and emits receipts.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use memmap2::Mmap;
use safetensors::SafeTensors;
use serde_json::json;
use uuid::Uuid;
use rayon::prelude::*;

use crate::quantization::contract::{
    QuantizationValidationProfile, TensorClass,
};
use crate::quantization::sweep::SweepCandidateStatus;
use crate::quantization::sweep::candidate::{
    FamilyPolicyEntry, PerClassPolicy, QuantSweepReceipt, QuantFamilyId, ByteAccounting, MatrixShape, PackedTileLayout,
    quant_family_id_name,
};
use crate::quantization::contract::SourceMatrixLayout;
use crate::quantization::sweep::families::{
    generate_all_candidates, FamilyCandidate,
};
use crate::quantization::sweep::spec::{
    QuantSweepSpec, SweepResourceLimits, SweepScoringConfig,
    SweepValidationConfig, TensorSelector, PolicyMode,
};
use crate::quantization::validation::validate_weight_space;

/// Fully-resolved tensor metadata from scanning safetensors.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub key: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub tensor_class: TensorClass,
    pub layer_index: Option<u32>,
}

/// Result of running a single candidate against a single tensor.
#[derive(Debug, Clone)]
pub struct CandidateRun {
    pub tensor_key: String,
    pub tensor_class: TensorClass,
    pub receipt: QuantSweepReceipt,
}

/// Top-level result of a sweep run.
#[derive(Debug, Clone)]
pub struct SweepRunResult {
    pub run_id: String,
    pub spec_version: u16,
    pub num_tensors: usize,
    pub num_candidates: usize,
    pub wall_ms: u64,
    pub per_class_policies: Vec<PerClassPolicy>,
    pub candidates: Vec<QuantSweepReceipt>,
}

/// Scan safetensors directory for all tensor keys and shapes.
pub fn scan_tensors(source_dir: &Path) -> Result<Vec<TensorEntry>, String> {
    let mut entries = Vec::new();
    if !source_dir.is_dir() {
        return Err(format!("source dir not found: {:?}", source_dir));
    }
    let mut dir =
        fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))?;
    while let Some(entry) = dir.next().transpose().map_err(|e| format!("entry: {e}"))? {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "safetensors") {
            continue;
        }
        let file = fs::File::open(&path).map_err(|e| format!("open {:?}: {e}", path))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let tensors =
            SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;
        for (key, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            let tensor_class = classify_tensor(&key);
            let layer_index = key
                .split('.')
                .filter_map(|s| s.parse::<u32>().ok())
                .next();
            entries.push(TensorEntry {
                key: key.to_string(),
                dtype,
                shape,
                tensor_class,
                layer_index,
            });
        }
    }
    Ok(entries)
}

/// Classify a tensor key into a TensorClass.
fn classify_tensor(key: &str) -> TensorClass {
    if key.contains("self_attn") {
        TensorClass::DecoderAttentionProjection
    } else if key.contains("mlp") {
        TensorClass::DecoderMlpProjection
    } else if key.contains("embed_tokens") || key.contains("embedding") {
        TensorClass::TokenEmbedding
    } else if key.contains("embed_vision") {
        TensorClass::VisionPatchProjection
    } else if key.contains("embed_audio") || key.contains("cross_modal") {
        TensorClass::CrossModalBridge
    } else if key.contains("lm_head") || key.contains("output") {
        TensorClass::OutputHead
    } else {
        TensorClass::Unknown
    }
}

/// Match tensor selectors against scanned entries.
pub fn select_tensors(
    entries: &[TensorEntry],
    selectors: &[TensorSelector],
) -> Vec<TensorEntry> {
    let mut selected = Vec::new();
    for sel in selectors {
        match sel {
            TensorSelector::ExactKey(k) => {
                if let Some(e) = entries.iter().find(|e| e.key == *k) {
                    selected.push(e.clone());
                }
            }
            TensorSelector::Regex(pat) => {
                if let Ok(re) = regex::Regex::new(pat) {
                    for e in entries.iter() {
                        if re.is_match(&e.key) {
                            selected.push(e.clone());
                        }
                    }
                }
            }
            TensorSelector::TensorClass {
                class,
                max_tensors,
            } => {
                let mut count = 0;
                for e in entries.iter() {
                    if e.tensor_class == *class && count < *max_tensors {
                        selected.push(e.clone());
                        count += 1;
                    }
                }
            }
        }
    }
    selected
}

/// Load a single tensor's f32 data from safetensors into a Vec<f32>.
pub fn load_tensor_f32(source_dir: &Path, target_key: &str) -> Result<Vec<f32>, String> {
    let mut dir =
        fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))?;
    while let Some(entry) = dir.next().transpose().map_err(|e| format!("entry: {e}"))? {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "safetensors") {
            continue;
        }
        let file = fs::File::open(&path).map_err(|e| format!("open {:?}: {e}", path))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let tensors =
            SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;
        for (key, view) in tensors.tensors() {
            if key != target_key {
                continue;
            }
            let dtype = view.dtype();
            let data = view.data().to_vec();
            return Ok(match dtype {
                safetensors::Dtype::F32 => data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                safetensors::Dtype::BF16 => data
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = ((c[0] as u32) << 16) | ((c[1] as u32) << 24);
                        f32::from_bits(bits)
                    })
                    .collect(),
                _ => {
                    return Err(format!(
                        "unsupported dtype {:?} for tensor {}",
                        dtype, target_key
                    ))
                }
            });
        }
    }
    Err(format!("tensor not found: {target_key}"))
}

/// Default scoring config for the sweep.
pub fn default_scoring_config() -> SweepScoringConfig {
    let mut max_weight_nrmse_by_family = HashMap::new();
    max_weight_nrmse_by_family
        .insert("Nf4".to_string(), 0.15);
    max_weight_nrmse_by_family
        .insert("SymInt4".to_string(), 0.15);
    max_weight_nrmse_by_family.insert("Int8".to_string(), 0.02);
    max_weight_nrmse_by_family
        .insert("Ternary".to_string(), 0.90);
    max_weight_nrmse_by_family
        .insert("MixedTile".to_string(), 0.10);
    SweepScoringConfig {
        max_weight_nrmse_by_family,
        max_zero_collapse: 0.01,
        byte_weight: 0.3,
    }
}

/// Default validation config.
/// Default validation config.
#[allow(deprecated)]
pub fn default_validation_config() -> SweepValidationConfig {
    SweepValidationConfig {
        run_weight_validation: true,
        max_candidates: None,
        max_candidates_per_tensor: 200,
       max_total_candidates: None,
        policy_mode: PolicyMode::ProductionCandidateOnly,
    }
}

/// Default resource limits.
pub fn default_resource_limits() -> SweepResourceLimits {
    SweepResourceLimits { max_workers: 4 }
}

/// Map a FamilyCandidate label prefix to a QuantFamilyId.
fn family_id_from_label(label: &str) -> QuantFamilyId {
    if label.starts_with("Nf4") {
        QuantFamilyId::Nf4
    } else if label.starts_with("SymInt4") || label.starts_with("SymInt") {
        QuantFamilyId::SymInt4
    } else if label.starts_with("Int8") {
        QuantFamilyId::Int8
    } else if label.starts_with("Ternary") {
        QuantFamilyId::Ternary
    } else if label.starts_with("MixedTile") || label.starts_with("Mixed") {
        QuantFamilyId::MixedTile
    } else {
        // Default fallback — avoid panic
        QuantFamilyId::Nf4
    }
}

/// Score a candidate receipt according to the scoring config.
/// Higher is better.
fn score_receipt(receipt: &QuantSweepReceipt, config: &SweepScoringConfig) -> f64 {
    let max_nrmse = config
        .max_weight_nrmse_by_family
        .get(&quant_family_id_name(&receipt.family).to_string())
        .copied()
        .unwrap_or(1.0);
    let quality_score = 1.0 - (receipt.weight.nrmse / max_nrmse).min(1.0);
    // Normalize total bytes: roughly how many bytes per element
    let total_elements_f64 = receipt.source_shape.iter().product::<usize>() as f64;
    let bytes_per_elem = if total_elements_f64 > 0.0 {
        receipt.bytes.total_bytes as f64 / total_elements_f64
    } else {
        4.0
    };
    let size_penalty = config.byte_weight * (bytes_per_elem / 4.0).min(1.0);
    quality_score - size_penalty
}

/// Run the full stage-1 weight-only sweep.
///
/// 1. Scan source directory for safetensors and matching tensor entries.
/// 2. For each tensor, generate all candidates from all active families.
/// 3. For each candidate: pack → validate → score → emit receipt.
/// 4. Compute per-class policies from best candidates.
pub fn run_weight_sweep(
    spec: &QuantSweepSpec,
    source_dir: &Path,
) -> Result<SweepRunResult, String> {
    let t0 = Instant::now();
    let run_id = Uuid::new_v4().to_string();

    // 1. Scan and select tensors
    let entries = scan_tensors(source_dir)?;
    let selected = select_tensors(&entries, &spec.tensor_selectors);
    if selected.is_empty() {
        return Err(
            "No tensors matched the provided selectors; check --tensor-regex".into(),
        );
    }

    let scoring_config = if spec.scoring.byte_weight > 0.0
        || !spec.scoring.max_weight_nrmse_by_family.is_empty()
    {
        &spec.scoring
    } else {
        // Use defaults
        &default_scoring_config()
    };

    let max_candidates = spec.validation.max_candidates_per_tensor;
    let max_total = spec.validation.max_total_candidates.unwrap_or(usize::MAX);
    let mut all_receipts: Vec<QuantSweepReceipt> = Vec::new();
    let tensor_count = Arc::new(AtomicU64::new(0));
    let total_tensors = selected.len();

    // For each tensor, generate candidates and run them.
    for tensor_entry in &selected {
        let mut tensor_candidate_count = 0usize;
        let tensor_count =
            tensor_count.fetch_add(1, Ordering::SeqCst) + 1;
        eprintln!(
            "  [{}/{}] {} ({})",
            tensor_count, total_tensors, tensor_entry.key, tensor_entry.tensor_class as u8
        );

        if tensor_entry.shape.len() != 2 {
            eprintln!("    skipping non-2D tensor: {:?}", tensor_entry.shape);
            continue;
        }
        // Check global candidate limit before generating for this tensor
        if all_receipts.len() >= max_total {
            eprintln!("    reached max total candidates ({max_total}), stopping");
            break;
        }

        // Load the tensor
        let weights = match load_tensor_f32(source_dir, &tensor_entry.key) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("    load error: {e}");
                continue;
            }
        };
        let in_features = tensor_entry.shape[0];
        let out_features = tensor_entry.shape[1];

        // Generate all candidates from all families
        let families = &spec.families;
        let family_candidates = generate_all_candidates(families);
        let per_tensor_remaining = max_candidates;
        let total_remaining = max_total.saturating_sub(all_receipts.len());
        let remaining = per_tensor_remaining.min(total_remaining);
        let candidates_slice: Vec<&FamilyCandidate> = family_candidates
            .iter()
            .take(remaining)
            .collect();

        let new_receipts: std::sync::Mutex<Vec<QuantSweepReceipt>> = std::sync::Mutex::new(Vec::new());
        candidates_slice.par_iter().for_each(|fc| {
            let t1 = Instant::now();
            let family_id = family_id_from_label(&fc.label);

            // Pack returns Vec<u8> extra directly — no unsafe cast
            let (codes, scales, biases, extra_bytes) =
            (fc.packer)(&weights, in_features, out_features);

            // Unpack uses extra_bytes directly (no reinterpret cast)
            let recon =
                (fc.unpacker)(&codes, &scales, &biases, &extra_bytes, in_features, out_features);

            // Key thresholds by family_id, not fc.label
            let family_key = format!("{:?}", family_id);
            let profile = QuantizationValidationProfile {
                tensor_class: tensor_entry.tensor_class,
                phase: crate::quantization::contract::ProfilePhase::Promotion,
                max_weight_nrmse: scoring_config
                    .max_weight_nrmse_by_family
                    .get(&family_key)
                    .copied()
                    .unwrap_or(f64::MAX),
                investigation_nrmse_ceiling: f64::MAX,
                max_zero_collapse_ratio: scoring_config.max_zero_collapse,
                max_operator_nrmse: f32::MAX,
                min_mean_cosine: 0.0,
                min_worst_cosine: 0.0,
                max_norm_ratio_drift: f32::MAX,
            };
            let wr = validate_weight_space(&weights, &recon, &profile);

            // Determine status
            let status = if wr.passes(&profile) {
                SweepCandidateStatus::Passed
            } else if wr.nrmse <= profile.investigation_nrmse_ceiling
                && wr.zero_collapse_ratio <= profile.max_zero_collapse_ratio
            {
                SweepCandidateStatus::InvestigationBand {
                    warning: format!(
                        "wNRMSE={:.4} exceeds target {:.4}, within ceiling",
                        wr.nrmse, profile.max_weight_nrmse
                    ),
                }
            } else {
                SweepCandidateStatus::Rejected {
                    reason: if wr.zero_collapse_ratio > profile.max_zero_collapse_ratio {
                        format!(
                            "zeroCollapse={:.4} > max={:.4}",
                            wr.zero_collapse_ratio, profile.max_zero_collapse_ratio
                        )
                    } else {
                        format!(
                            "wNRMSE={:.4} > ceiling={:.4}",
                            wr.nrmse, profile.investigation_nrmse_ceiling
                        )
                    },
                }
            };

            // ByteAccounting from actual payload lengths (codes=Vec<u8>, scales+biases=Vec<f32>, extra=Vec<u8>)
            let elem_count = in_features * out_features;
            // Convert scales+biases to LE bytes for accurate metadata accounting
            let mut meta_bytes = Vec::with_capacity((scales.len() + biases.len()) * 4);
            for &s in &scales { meta_bytes.extend_from_slice(&s.to_le_bytes()); }
            for &b in &biases { meta_bytes.extend_from_slice(&b.to_le_bytes()); }
            let bytes = ByteAccounting::from_payloads(&codes, &meta_bytes, &extra_bytes, &[], elem_count);

            let mut tmp_receipt = QuantSweepReceipt {
                receipt_version: 1,
                run_id: run_id.clone(),
                tensor_key: tensor_entry.key.clone(),
                tensor_class: tensor_entry.tensor_class,
                source_shape: tensor_entry.shape.clone(),
                family: family_id,
                parameters: fc.parameters.clone(),
                bytes,
                source_layout: SourceMatrixLayout::CheckpointOutByIn,
                logical_shape: MatrixShape { in_features, out_features },
                packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
                weight: wr.clone(),
                status,
                score: 0.0,
                wall_ms: t1.elapsed().as_millis() as u64,
            };
            let score = score_receipt(&tmp_receipt, scoring_config);
            tmp_receipt.score = score;

            new_receipts.lock().unwrap().push(tmp_receipt);
        }); // end par_iter for_each
        all_receipts.extend(new_receipts.into_inner().unwrap());
    } // end for tensor_entry

    // Deduplicate per (tensor_key, family) — keep best score (lowest)
    // Group by (tensor_key, family)
    let mut best_per_tensor_family: HashMap<(String, QuantFamilyId), usize> = HashMap::new();
    for (i, r) in all_receipts.iter().enumerate() {
        let key = (r.tensor_key.clone(), r.family);
        let best = best_per_tensor_family.entry(key).or_insert(i);
        if r.score > all_receipts[*best].score {
            *best = i;
        }
    }

    // Build per-class policies
    let mut per_class: HashMap<TensorClass, Vec<(f64, QuantSweepReceipt)>> = HashMap::new();
    for &idx in best_per_tensor_family.values() {
        let r = &all_receipts[idx];
        per_class
            .entry(r.tensor_class)
            .or_default()
            .push((r.score, r.clone()));
    }

    let per_class_policies: Vec<PerClassPolicy> = per_class
        .into_iter()
        .map(|(tc, mut receipts)| {
            // Sort descending: higher score is better
            receipts.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            // In production mode, only include passed candidates
            let policy_mode = spec.validation.policy_mode;
            let filtered: Vec<(f64, QuantSweepReceipt)> = match policy_mode {
                PolicyMode::ProductionCandidateOnly => {
                    receipts.into_iter().filter(|(_, r)| matches!(r.status, SweepCandidateStatus::Passed)).collect()
                }
                PolicyMode::Exploratory => receipts,
            };
            let preferred: Vec<FamilyPolicyEntry> = filtered
                .into_iter()
                .take(3)
                .map(|(_score, r)| FamilyPolicyEntry {
                    family: format!("{:?}", r.family),
                    parameters: r.parameters.clone(),
                    weight_nrmse: r.weight.nrmse,
                    score: r.score,
                    total_bytes: r.bytes.total_bytes,
                })
                .collect();
            PerClassPolicy {
                tensor_class: tc,
                preferred,
                fallback: "RawF32".to_string(),
            }
        })
        .collect();

    let result = SweepRunResult {
        run_id,
        spec_version: spec.spec_version,
        num_tensors: selected.len(),
        num_candidates: all_receipts.len(),
        wall_ms: t0.elapsed().as_millis() as u64,
        per_class_policies,
        candidates: all_receipts,
    };

    Ok(result)
}

/// Write sweep output to the spec's output directory.
pub fn write_sweep_output(out_dir: &Path, result: &SweepRunResult) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|e| format!("create output dir: {e}"))?;
    fs::create_dir_all(out_dir.join("summaries"))
        .map_err(|e| format!("create summaries dir: {e}"))?;
    fs::create_dir_all(out_dir.join("best"))
        .map_err(|e| format!("create best dir: {e}"))?;

    // Run manifest
    let manifest = json!({
        "run_id": result.run_id,
        "spec_version": result.spec_version,
        "num_tensors": result.num_tensors,
        "num_candidates": result.num_candidates,
        "wall_ms": result.wall_ms,
        "generated_at": "unknown",
    });
    let manifest_path = out_dir.join("run_manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).map_err(|e| format!("serialize manifest: {e}"))?,
    )
    .map_err(|e| format!("write manifest: {e}"))?;

    // Candidates JSONL
    let candidates_path = out_dir.join("candidates.jsonl");
    let mut lines = Vec::new();
    for r in &result.candidates {
        let v = json!({
            "receipt_version": r.receipt_version,
            "run_id": r.run_id,
            "tensor_key": r.tensor_key,
            "tensor_class": r.tensor_class as u8,
            "source_shape": r.source_shape,
            "family": format!("{:?}", r.family),
            "parameters": r.parameters,
            "code_bytes": r.bytes.code_bytes,
            "metadata_bytes": r.bytes.metadata_bytes,
            "total_bytes": r.bytes.total_bytes,
            "compression_ratio_vs_f32": r.bytes.compression_ratio_vs_f32,
            "weight_nrmse": (r.weight.nrmse * 10000.0).round() / 10000.0,
            "weight_zero_collapse": (r.weight.zero_collapse_ratio * 10000.0).round() / 10000.0,
            "weight_rmse": (r.weight.rmse * 10000.0).round() / 10000.0,
            "weight_max_abs_error": (r.weight.max_abs_error * 10000.0).round() / 10000.0,
            "status": format!("{:?}", r.status),
            "score": (r.score * 10000.0).round() / 10000.0,
            "wall_ms": r.wall_ms,
        });
        lines.push(serde_json::to_string(&v).map_err(|e| format!("serialize candidate: {e}"))?);
    }
    fs::write(&candidates_path, lines.join("\n"))
        .map_err(|e| format!("write candidates: {e}"))?;

    // Best per-class policy
    let policy_path = out_dir.join("best").join("per_class_policy.json");
    let policy_json: Vec<serde_json::Value> = result
        .per_class_policies
        .iter()
        .map(|p| {
            let preferred: Vec<serde_json::Value> = p
                .preferred
                .iter()
                .map(|fpe| {
                    json!({
                        "family": fpe.family,
                        "parameters": fpe.parameters,
                        "weight_nrmse": (fpe.weight_nrmse * 10000.0).round() / 10000.0,
                        "score": (fpe.score * 10000.0).round() / 10000.0,
                        "total_bytes": fpe.total_bytes,
                    })
                })
                .collect();
            json!({
                "tensor_class": format!("{:?}", p.tensor_class),
                "tensor_class_id": p.tensor_class as u8,
                "preferred": preferred,
                "fallback": p.fallback,
            })
        })
        .collect();
    fs::write(
        &policy_path,
        serde_json::to_string_pretty(&policy_json)
            .map_err(|e| format!("serialize policy: {e}"))?,
    )
    .map_err(|e| format!("write policy: {e}"))?;

    Ok(())
}
