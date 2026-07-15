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
use rayon::prelude::*;
use safetensors::SafeTensors;
use serde_json::json;
use uuid::Uuid;

use crate::contract::SourceMatrixLayout;
use crate::contract::{
    QuantizationValidationProfile, TensorClass, WeightValidationReport,
};
use crate::sweep::candidate::{
    quant_family_id_name, ByteAccounting, FamilyPolicyEntry, MatrixShape, PackedTileLayout,
    PerClassPolicy, QuantFamilyId, QuantSweepReceipt,
};
use crate::sweep::families::{generate_all_candidates, FamilyCandidate};
use crate::sweep::spec::{
    PolicyMode, QuantSweepSpec, SweepResourceLimits, SweepScoringConfig, SweepValidationConfig,
    TensorSelector,
};
use crate::sweep::{SweepCandidateStatus, SweepFailureReason};
use crate::validation::validate_weight_space;

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
    let mut dir = fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))?;
    while let Some(entry) = dir.next().transpose().map_err(|e| format!("entry: {e}"))? {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "safetensors") {
            continue;
        }
        let file = fs::File::open(&path).map_err(|e| format!("open {:?}: {e}", path))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;
        for (key, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();
            let dtype = format!("{:?}", view.dtype());
            let tensor_class = classify_tensor(&key);
            let layer_index = key.split('.').filter_map(|s| s.parse::<u32>().ok()).next();
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
pub fn select_tensors(entries: &[TensorEntry], selectors: &[TensorSelector]) -> Vec<TensorEntry> {
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
            TensorSelector::TensorClass { class, max_tensors } => {
                let mut count = 0;
                for e in entries.iter() {
                    if e.tensor_class == *class && count < *max_tensors {
                        selected.push(e.clone());
                        count += 1;
                    }
                }
            }
            TensorSelector::DepthAware(sel) => {
                let ranges = match parse_depth_ranges(&sel.depth_ranges) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let class_filter = if sel.tensor_class.is_empty() {
                    None
                } else {
                    Some(sel.tensor_class.as_str())
                };
                let mut count = 0;
                for e in entries.iter() {
                    if count >= sel.max_tensors {
                        break;
                    }
                    // Filter by tensor class if specified
                    if let Some(class_name) = class_filter {
                        let entry_class = format!("{:?}", e.tensor_class);
                        if entry_class != class_name {
                            continue;
                        }
                    }
                    // Filter by depth (layer index fast-path, then string fallback)
                    if matches_depth_by_entry(e, &ranges) {
                        selected.push(e.clone());
                        count += 1;
                    }
                }
            }
        }
    }
    selected
}

/// Parse "start-end" depth range strings into (usize, usize) pairs.
fn parse_depth_ranges(ranges: &[String]) -> Result<Vec<(usize, usize)>, String> {
    ranges
        .iter()
        .map(|r| {
            let parts: Vec<&str> = r.split('-').collect();
            if parts.len() != 2 {
                return Err(format!("Invalid range: {}", r));
            }
            let start: usize = parts[0]
                .parse()
                .map_err(|_| format!("Invalid start in range: {}", r))?;
            let end: usize = parts[1]
                .parse()
                .map_err(|_| format!("Invalid end in range: {}", r))?;
            Ok((start, end))
        })
        .collect()
}

/// Check whether a tensor entry falls within any of the given depth ranges.
/// Uses `layer_index` as a fast-path when available, falling back to string
/// parsing of the "layers.N." pattern in the tensor key.
fn matches_depth_by_entry(entry: &TensorEntry, depth_ranges: &[(usize, usize)]) -> bool {
    let layer = match entry.layer_index {
        Some(idx) => idx as usize,
        None => {
            // Fallback: try to extract layer number from "layers.N." pattern
            let Some(dot_idx) = entry.key.find("layers.") else {
                return false;
            };
            let remainder = &entry.key[dot_idx + 7..]; // skip "layers."
            let num_str: String = remainder
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            let Ok(n) = num_str.parse::<usize>() else {
                return false;
            };
            n
        }
    };
    depth_ranges
        .iter()
        .any(|(start, end)| layer >= *start && layer <= *end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::spec::DepthAwareSelector;

    #[test]
    fn test_parse_depth_ranges() {
        let ranges = vec!["0-3".to_string(), "20-25".to_string(), "42-46".to_string()];
        let parsed = parse_depth_ranges(&ranges).unwrap();
        assert_eq!(parsed, vec![(0, 3), (20, 25), (42, 46)]);
    }

    #[test]
    fn test_parse_depth_ranges_invalid() {
        assert!(parse_depth_ranges(&["abc".to_string()]).is_err());
        assert!(parse_depth_ranges(&["5-3-2".to_string()]).is_err());
        assert!(parse_depth_ranges(&["-3".to_string()]).is_err());
    }

    #[test]
    fn test_matches_depth_by_entry_layer_index() {
        let e = TensorEntry {
            key: "model.layers.5.self_attn.q_proj.weight".into(),
            dtype: "F32".into(),
            shape: vec![4096, 4096],
            tensor_class: TensorClass::DecoderAttentionProjection,
            layer_index: Some(5),
        };
        let ranges = [(0, 3), (20, 25), (42, 46)];
        assert!(!matches_depth_by_entry(&e, &ranges));
        let ranges2 = [(4, 6), (20, 25)];
        assert!(matches_depth_by_entry(&e, &ranges2));
    }

    #[test]
    fn test_matches_depth_by_entry_fallback() {
        // Key has "layers.N." pattern but layer_index is None
        let e = TensorEntry {
            key: "model.layers.12.self_attn.v_proj.weight".into(),
            dtype: "F32".into(),
            shape: vec![4096, 4096],
            tensor_class: TensorClass::DecoderAttentionProjection,
            layer_index: None, // fallback path
        };
        let ranges = [(10, 15)];
        assert!(matches_depth_by_entry(&e, &ranges));
        let ranges2 = [(0, 3)];
        assert!(!matches_depth_by_entry(&e, &ranges2));
    }

    #[test]
    fn test_matches_depth_by_entry_no_layer() {
        // Key doesn't look like a layer at all
        let e = TensorEntry {
            key: "model.lm_head.weight".into(),
            dtype: "F32".into(),
            shape: vec![32000, 4096],
            tensor_class: TensorClass::OutputHead,
            layer_index: None,
        };
        let ranges = [(0, 100)];
        assert!(!matches_depth_by_entry(&e, &ranges));
    }

    #[test]
    fn test_depth_aware_selector_empty_class() {
        // Empty tensor_class means select all classes
        let entries = vec![
            TensorEntry {
                key: "model.layers.0.self_attn.q_proj.weight".into(),
                dtype: "F32".into(),
                shape: vec![4096, 4096],
                tensor_class: TensorClass::DecoderAttentionProjection,
                layer_index: Some(0),
            },
            TensorEntry {
                key: "model.layers.25.self_attn.q_proj.weight".into(),
                dtype: "F32".into(),
                shape: vec![4096, 4096],
                tensor_class: TensorClass::DecoderAttentionProjection,
                layer_index: Some(25),
            },
        ];
        let selector = TensorSelector::DepthAware(DepthAwareSelector {
            tensor_class: String::new(),
            depth_ranges: vec!["0-5".to_string()],
            max_tensors: 10,
        });
        let selected = select_tensors(&entries, &[selector]);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].key, "model.layers.0.self_attn.q_proj.weight");
    }

    #[test]
    fn test_depth_aware_selector_class_filter() {
        let entries = vec![
            TensorEntry {
                key: "model.layers.0.self_attn.q_proj.weight".into(),
                dtype: "F32".into(),
                shape: vec![4096, 4096],
                tensor_class: TensorClass::DecoderAttentionProjection,
                layer_index: Some(0),
            },
            TensorEntry {
                key: "model.layers.0.mlp.gate_proj.weight".into(),
                dtype: "F32".into(),
                shape: vec![4096, 11008],
                tensor_class: TensorClass::DecoderMlpProjection,
                layer_index: Some(0),
            },
        ];
        // Filter only MLP projections
        let selector = TensorSelector::DepthAware(DepthAwareSelector {
            tensor_class: "DecoderMlpProjection".to_string(),
            depth_ranges: vec!["0-5".to_string()],
            max_tensors: 10,
        });
        let selected = select_tensors(&entries, &[selector]);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].tensor_class, TensorClass::DecoderMlpProjection);
    }

    #[test]
    fn test_depth_aware_selector_max_tensors() {
        let entries: Vec<TensorEntry> = (0..20)
            .map(|i| TensorEntry {
                key: format!("model.layers.{}.self_attn.q_proj.weight", i),
                dtype: "F32".into(),
                shape: vec![4096, 4096],
                tensor_class: TensorClass::DecoderAttentionProjection,
                layer_index: Some(i),
            })
            .collect();
        let selector = TensorSelector::DepthAware(DepthAwareSelector {
            tensor_class: String::new(),
            depth_ranges: vec!["0-30".to_string()],
            max_tensors: 5,
        });
        let selected = select_tensors(&entries, &[selector]);
        assert_eq!(selected.len(), 5);
    }

    #[test]
    fn test_depth_aware_serde_roundtrip() {
        let sel = DepthAwareSelector {
            tensor_class: "DecoderAttentionProjection".to_string(),
            depth_ranges: vec!["0-3".to_string(), "20-25".to_string()],
            max_tensors: 50,
        };
        let json = serde_json::to_string(&sel).unwrap();
        let back: DepthAwareSelector = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tensor_class, "DecoderAttentionProjection");
        assert_eq!(back.depth_ranges, vec!["0-3", "20-25"]);
        assert_eq!(back.max_tensors, 50);
    }

    #[test]
    fn test_depth_aware_enum_serde_roundtrip() {
        let sel = TensorSelector::DepthAware(DepthAwareSelector {
            tensor_class: String::new(),
            depth_ranges: vec!["0-3".to_string()],
            max_tensors: 10,
        });
        let json = serde_json::to_string(&sel).unwrap();
        let back: TensorSelector = serde_json::from_str(&json).unwrap();
        match &back {
            TensorSelector::DepthAware(d) => {
                assert!(d.tensor_class.is_empty());
                assert_eq!(d.depth_ranges, vec!["0-3"]);
                assert_eq!(d.max_tensors, 10);
            }
            _ => panic!("expected DepthAware"),
        }
    }
}
/// Load a single tensor's f32 data from safetensors into a Vec<f32>.
pub fn load_tensor_f32(source_dir: &Path, target_key: &str) -> Result<Vec<f32>, String> {
    let mut dir = fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))?;
    while let Some(entry) = dir.next().transpose().map_err(|e| format!("entry: {e}"))? {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "safetensors") {
            continue;
        }
        let file = fs::File::open(&path).map_err(|e| format!("open {:?}: {e}", path))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;
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
    max_weight_nrmse_by_family.insert("Nf4".to_string(), 0.15);
    max_weight_nrmse_by_family.insert("SymInt4".to_string(), 0.15);
    max_weight_nrmse_by_family.insert("Int8".to_string(), 0.02);
    max_weight_nrmse_by_family.insert("Ternary".to_string(), 0.90);
    max_weight_nrmse_by_family.insert("MixedTile".to_string(), 0.10);
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
        return Err("No tensors matched the provided selectors; check --tensor-regex".into());
    }

    let scoring_config =
        if spec.scoring.byte_weight > 0.0 || !spec.scoring.max_weight_nrmse_by_family.is_empty() {
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
        let _tensor_candidate_count = 0usize;
        let tensor_count = tensor_count.fetch_add(1, Ordering::SeqCst) + 1;
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
        let candidates_slice: Vec<&FamilyCandidate> =
            family_candidates.iter().take(remaining).collect();

        let new_receipts: std::sync::Mutex<Vec<QuantSweepReceipt>> =
            std::sync::Mutex::new(Vec::new());

        // ── Metal batch path for NF4 candidates ──────────────────────────
        // Build (rmse, nrmse, max_abs) lookup from GPU eval, keyed by
        // candidate index in candidates_slice. Empty = use CPU fallback.
        let metal_metrics: std::collections::HashMap<usize, (f64, f64, f64)> = {
            #[cfg(feature = "metal-dispatch")]
            {
                let nf4_entries: Vec<(usize, [u32; 4])> = candidates_slice
                    .iter()
                    .enumerate()
                    .filter_map(|(i, fc)| {
                        if family_id_from_label(&fc.label) != QuantFamilyId::Nf4 {
                            return None;
                        }
                        // Parse label to extract codebook_id, group_size, affine_mode
                        let p = &fc.parameters;
                        // codebook is a JSON string like "PrismCurrent", "BitsAndBytesNf4", "SymmetricNormalFloat"
                        let cb = p
                            .get("codebook")
                            .and_then(|v| v.as_str())
                            .map(|s| match s {
                                "BitsAndBytesNf4" => 1u32,
                                "SymmetricNormalFloat" => 2u32,
                                _ => 0u32, // PrismCurrent or unknown
                            })
                            .unwrap_or(0);
                        let gs = p.get("group_size").and_then(|v| v.as_u64()).unwrap_or(128) as u32;
                        let am = p
                            .get("affine_mode")
                            .and_then(|v| v.as_str())
                            .map(|s| if s == "ScaleBias" { 1 } else { 0 })
                            .unwrap_or(0);
                        Some((i, [cb, gs, am, 0]))
                    })
                    .collect();
                if nf4_entries.is_empty() {
                    std::collections::HashMap::new()
                } else {
                    let params: Vec<[u32; 4]> = nf4_entries.iter().map(|(_, p)| *p).collect();
                    let indices: Vec<usize> = nf4_entries.iter().map(|(i, _)| *i).collect();
                    match crate::sweep::metal::evaluate_nf4_batch(
                        &weights,
                        &params,
                        params.len(),
                        in_features,
                        out_features,
                    ) {
                        Ok(metrics) => indices
                            .into_iter()
                            .zip(metrics.into_iter())
                            .map(|(i, m)| (i, (m.rmse, m.nrmse, m.max_abs_error)))
                            .collect(),
                        Err(e) => {
                            eprintln!("    Metal eval failed (falling back to CPU): {e}");
                            std::collections::HashMap::new()
                        }
                    }
                }
            }
            #[cfg(not(feature = "metal-dispatch"))]
            {
                std::collections::HashMap::new()
            }
        };

        candidates_slice
            .par_iter()
            .enumerate()
            .for_each(|(idx, fc)| {
                let t1 = Instant::now();
                let family_id = family_id_from_label(&fc.label);

                // Conditional pack — skip for Metal-evaluated NF4 candidates
                let (codes, scales, biases, extra_bytes, _recon): (
                    Vec<u8>,
                    Vec<f32>,
                    Vec<f32>,
                    Vec<u8>,
                    Vec<f32>,
                ) = if metal_metrics.contains_key(&idx) {
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
                } else {
                    // TODO(imatrix): load per-column quant_weights and pass to packer
                    // for activation-weighted candidates when imatrix data is available.
                    let (c, s, b, e) = (fc.packer)(&weights, in_features, out_features);
                    let r = (fc.unpacker)(&c, &s, &b, &e, in_features, out_features);
                    (c, s, b, e, r)
                };

                // Key thresholds by family_id
                let family_key = format!("{:?}", family_id);
                let profile = QuantizationValidationProfile {
                    tensor_class: tensor_entry.tensor_class,
                    phase: crate::ProfilePhase::Promotion,
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

                // Compute weight report — either from Metal metrics or CPU validation
                let wr = if let Some(&(rmse, nrmse, max_abs)) = metal_metrics.get(&idx) {
                    // Metal path: metrics already computed on GPU, zero-collapse not available
                    WeightValidationReport {
                        rmse,
                        nrmse,
                        max_abs_error: max_abs,
                        zero_collapse_ratio: 0.0,
                    }
                } else {
                    let recon = (fc.unpacker)(
                        &codes,
                        &scales,
                        &biases,
                        &extra_bytes,
                        in_features,
                        out_features,
                    );
                    validate_weight_space(&weights, &recon, &profile)
                };

                // Determine status
                let (status, failure_reason) = if wr.passes(&profile) {
                    (SweepCandidateStatus::Passed, SweepFailureReason::None)
                } else if wr.nrmse <= profile.investigation_nrmse_ceiling
                    && wr.zero_collapse_ratio <= profile.max_zero_collapse_ratio
                {
                    (
                        SweepCandidateStatus::InvestigationBand {
                            warning: format!(
                                "wNRMSE={:.4} exceeds target {:.4}, within ceiling",
                                wr.nrmse, profile.max_weight_nrmse
                            ),
                        },
                        SweepFailureReason::None,
                    )
                } else {
                    let reason = if wr.zero_collapse_ratio > profile.max_zero_collapse_ratio {
                        format!(
                            "zeroCollapse={:.4} > max={:.4}",
                            wr.zero_collapse_ratio, profile.max_zero_collapse_ratio
                        )
                    } else {
                        format!(
                            "wNRMSE={:.4} > ceiling={:.4}",
                            wr.nrmse, profile.investigation_nrmse_ceiling
                        )
                    };
                    let fr = if wr.zero_collapse_ratio > profile.max_zero_collapse_ratio {
                        SweepFailureReason::ZeroCollapse
                    } else {
                        SweepFailureReason::WeightNrmse
                    };
                    (SweepCandidateStatus::Rejected { reason }, fr)
                };

                let elem_count = in_features * out_features;
                let bytes = if metal_metrics.contains_key(&idx) {
                    let cb = (fc.code_bytes_fn)(in_features, out_features);
                    let mb = (fc.metadata_bytes_fn)(in_features, out_features);
                    ByteAccounting::from_payloads(
                        &vec![0u8; cb as usize],
                        &vec![0u8; mb as usize],
                        &[],
                        &[],
                        elem_count,
                    )
                } else {
                    let mut meta = Vec::with_capacity((scales.len() + biases.len()) * 4);
                    for &s in &scales {
                        meta.extend_from_slice(&s.to_le_bytes());
                    }
                    for &b in &biases {
                        meta.extend_from_slice(&b.to_le_bytes());
                    }
                    ByteAccounting::from_payloads(&codes, &meta, &extra_bytes, &[], elem_count)
                };

                let score = score_receipt(
                    &QuantSweepReceipt {
                        receipt_version: 1,
                        run_id: run_id.clone(),
                        tensor_key: tensor_entry.key.clone(),
                        tensor_class: tensor_entry.tensor_class,
                        source_shape: tensor_entry.shape.clone(),
                        family: family_id,
                        parameters: fc.parameters.clone(),
                        bytes,
                        source_layout: SourceMatrixLayout::CheckpointOutByIn,
                        logical_shape: MatrixShape {
                            in_features,
                            out_features,
                        },
                        packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
                        weight: wr.clone(),
                        status: status.clone(),
                        failure_reason: failure_reason.clone(),
                        score: 0.0,
                        wall_ms: t1.elapsed().as_millis() as u64,
                    },
                    scoring_config,
                );

                new_receipts.lock().unwrap().push(QuantSweepReceipt {
                    receipt_version: 1,
                    run_id: run_id.clone(),
                    tensor_key: tensor_entry.key.clone(),
                    tensor_class: tensor_entry.tensor_class,
                    source_shape: tensor_entry.shape.clone(),
                    family: family_id,
                    parameters: fc.parameters.clone(),
                    bytes,
                    source_layout: SourceMatrixLayout::CheckpointOutByIn,
                    logical_shape: MatrixShape {
                        in_features,
                        out_features,
                    },
                    packed_layout: PackedTileLayout::OutputChannelContiguousReductionTiles,
                    weight: wr,
                    status,
                    failure_reason,
                    score,
                    wall_ms: t1.elapsed().as_millis() as u64,
                });
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
                PolicyMode::ProductionCandidateOnly => receipts
                    .into_iter()
                    .filter(|(_, r)| matches!(r.status, SweepCandidateStatus::Passed))
                    .collect(),
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
    fs::create_dir_all(out_dir.join("best")).map_err(|e| format!("create best dir: {e}"))?;

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
            "failure_reason": r.failure_reason,
            "score": (r.score * 10000.0).round() / 10000.0,
            "wall_ms": r.wall_ms,
        });
        lines.push(serde_json::to_string(&v).map_err(|e| format!("serialize candidate: {e}"))?);
    }
    fs::write(&candidates_path, lines.join("\n")).map_err(|e| format!("write candidates: {e}"))?;

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
        serde_json::to_string_pretty(&policy_json).map_err(|e| format!("serialize policy: {e}"))?,
    )
    .map_err(|e| format!("write policy: {e}"))?;

    Ok(())
}
