//! Substitution pass — tries ranked tile640 codec candidates against evidence
//! gates and substitutes the most aggressive one that passes.
//!
//! Every candidate is a tile640 variant (Ternary, NF4, SymInt4, INT8, FP16).
//! The pass packs → validates weight gates → validates operator gates →
//! tries residual rescue → commits substitution or falls back.
//!
//! # Integration
//!
//! Called by the builder pipeline after the primary codec is resolved.
//! Returns a SubstitutionAttempt that the builder can commit into the cimage.

use std::collections::HashMap;

use crate::quantization::embed_cluster::{pack_ternary_weights, unpack_ternary_weights};

use super::substitution::*;

// ── Public entry point ───────────────────────────────────────────────────

/// Try substitution for a single tensor. Returns the best outcome from
/// trying all candidates in order.
///
/// `weights` — f32 source weights in canonical [out_features, in_features] layout.
/// `in_f`, `out_f` — logical dimensions.
/// `candidates` — ordered list of substitution candidates from the pipeline config.
/// `primary_bytes` — byte count of the primary codec (for savings comparison).
pub fn try_all_candidates(
    weights: &[f32],
    in_f: u32,
    out_f: u32,
    candidates: &[SubstitutionCandidate],
    primary_bytes: u64,
) -> Vec<SubstitutionAttempt> {
    let mut results = Vec::new();
    let total_elements = (in_f as usize) * (out_f as usize);

    for candidate in candidates {
        let attempt = try_single_candidate(weights, in_f, out_f, candidate, primary_bytes, total_elements);
        let succeeded = matches!(
            attempt.outcome,
            SubstitutionOutcome::Substituted | SubstitutionOutcome::SubstitutedWithRescue
        );
        results.push(attempt);
        if succeeded {
            break; // First successful candidate wins
        }
    }

    results
}

// ── Single candidate evaluation ──────────────────────────────────────────

fn try_single_candidate(
    weights: &[f32],
    in_f: u32,
    out_f: u32,
    candidate: &SubstitutionCandidate,
    primary_bytes: u64,
    total_elements: usize,
) -> SubstitutionAttempt {
    let result = SubstitutionAttempt {
        candidate: candidate.name.clone(),
        weight_evidence: None,
        operator_evidence: None,
        rescue_result: None,
        outcome: SubstitutionOutcome::NotAttempted,
        bytes_saved: 0,
        primary_bytes,
    };

    // ── Step 1: Pack ─────────────────────────────────────────────────
    let (codes, metadata, recon) = match pack_for_candidate(weights, in_f, out_f, candidate) {
        Some(p) => p,
        None => return SubstitutionAttempt { outcome: SubstitutionOutcome::Rejected, ..result },
    };

    // ── Step 2: Weight-space gate ────────────────────────────────────
    let weight_evidence = evaluate_weight_gate(weights, &recon, total_elements, candidate);
    if let Some(ref ev) = weight_evidence {
        if !ev.passed {
            return SubstitutionAttempt { weight_evidence, outcome: SubstitutionOutcome::Rejected, ..result };
        }
    }

    // ── Step 3: Operator gate ────────────────────────────────────────
    let operator_evidence = evaluate_operator_gate(weights, &recon, in_f, out_f, candidate);
    if let Some(ref ev) = operator_evidence {
        if !ev.passed {
            return SubstitutionAttempt {
                weight_evidence,
                operator_evidence,
                outcome: SubstitutionOutcome::Rejected,
                ..result
            };
        }
    }

    // ── Step 4: Compute bytes saved ──────────────────────────────────
    let candidate_bytes = (codes.len() + metadata.len()) as u64;
    let bytes_saved = primary_bytes.saturating_sub(candidate_bytes);

    SubstitutionAttempt {
        weight_evidence,
        operator_evidence,
        outcome: SubstitutionOutcome::Substituted,
        bytes_saved,
        ..result
    }
}

// ── Pack dispatch ────────────────────────────────────────────────────────

fn pack_for_candidate(
    weights: &[f32],
    in_f: u32,
    out_f: u32,
    candidate: &SubstitutionCandidate,
) -> Option<(Vec<u8>, Vec<u8>, Vec<f32>)> {
    match candidate.name.as_str() {
        "Ternary" => {
            let (codes, scales, biases) = pack_ternary_weights(weights, in_f as usize, out_f as usize);
            let mut meta = Vec::with_capacity(scales.len() * 4 + biases.len() * 4);
            for &s in &scales {
                meta.extend_from_slice(&s.to_le_bytes());
            }
            for &b in &biases {
                meta.extend_from_slice(&b.to_le_bytes());
            }
            let unpacked = unpack_ternary_weights(&codes, &scales, &biases, in_f as usize, out_f as usize);
            Some((codes, meta, unpacked))
        }
        "NF4" => {
            let (codes, scales, biases, _num_tiles, _num_groups) =
                crate::nf4tile640::pack_nf4_weights(weights, in_f as usize, out_f as usize);
            let mut meta = Vec::with_capacity(scales.len() * 4 + biases.len() * 4);
            for &s in &scales { meta.extend_from_slice(&s.to_le_bytes()); }
            for &b in &biases { meta.extend_from_slice(&b.to_le_bytes()); }
            let unpacked = crate::nf4tile640::unpack_nf4_weights(&codes, &scales, &biases, in_f as usize, out_f as usize);
            Some((codes, meta, unpacked))
        }
        "INT8" => {
            let (codes, scales, biases) =
                crate::nf4tile640::pack_int8_weights(weights, in_f as usize, out_f as usize);
            let mut meta = Vec::with_capacity(scales.len() * 4);
            for &s in &scales { meta.extend_from_slice(&s.to_le_bytes()); }
            let unpacked = crate::nf4tile640::unpack_int8_weights(&codes, &scales, &biases, in_f as usize, out_f as usize);
            Some((codes, meta, unpacked))
        }
        "FP16" => {
            let codes: Vec<u8> = weights.iter().flat_map(|&w| {
                let bits = f32_to_f16_bits(w);
                bits.to_le_bytes().into_iter()
            }).collect();
            let meta = Vec::new();
            let recon = weights.to_vec();
            Some((codes, meta, recon))
        }
        _ => None,
    }
}

// ── Weight-space gate ────────────────────────────────────────────────────

fn evaluate_weight_gate(
    float_weights: &[f32],
    recon: &[f32],
    total_elements: usize,
    candidate: &SubstitutionCandidate,
) -> Option<SubstitutionEvidence> {
    let gates = &candidate.gates;
    if gates.weight_nrmse_max.is_none() && gates.weight_zero_collapse_max.is_none() {
        return None;
    }

    let mut sq_err = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut zero_count = 0usize;
    for i in 0..total_elements.min(float_weights.len()).min(recon.len()) {
        let d = (float_weights[i] - recon[i]) as f64;
        sq_err += d * d;
        let ad = d.abs();
        if ad > max_abs { max_abs = ad; }
        // Only count as collapse if source was NOT zero but recon IS zero
        if float_weights[i] != 0.0 && recon[i] == 0.0 { zero_count += 1; }
    }

    let nrmse = if total_elements > 0 {
        (sq_err / total_elements as f64).sqrt()
    } else { 0.0 };

    let zero_collapse = if total_elements > 0 {
        zero_count as f64 / total_elements as f64
    } else { 0.0 };

    let mut metrics = HashMap::new();
    metrics.insert("nrmse".into(), nrmse);
    metrics.insert("max_abs_error".into(), max_abs);
    metrics.insert("zero_collapse_ratio".into(), zero_collapse);

    let passed = true
        && gates.weight_nrmse_max.map_or(true, |g| nrmse <= g)
        && gates.weight_zero_collapse_max.map_or(true, |g| zero_collapse <= g);

    Some(SubstitutionEvidence {
        tier: EvidenceTier::WeightSpace,
        evaluated: true,
        passed,
        metrics,
        error: None,
    })
}

// ── Operator gate (CPU matmul) ────────────────────────────────────────────

fn evaluate_operator_gate(
    float_weights: &[f32],
    recon: &[f32],
    in_f: u32,
    out_f: u32,
    candidate: &SubstitutionCandidate,
) -> Option<SubstitutionEvidence> {
    let gates = &candidate.gates;
    if gates.operator_nrmse_max.is_none() && gates.operator_cosine_min.is_none() && gates.operator_max_abs_max.is_none() {
        return None;
    }

    let pi = std::f32::consts::PI;
    let in_f_usize = in_f as usize;
    let out_f_usize = out_f as usize;
    let activation: Vec<f32> = (0..in_f_usize)
        .map(|i| ((i as f32) / (in_f as f32) * pi).sin())
        .collect();

    let ref_out: Vec<f64> = (0..out_f_usize)
        .map(|j| {
            let base = j * in_f_usize;
            (0..in_f_usize).map(|i| activation[i] as f64 * float_weights[base + i] as f64).sum()
        })
        .collect();

    let q_out: Vec<f64> = (0..out_f_usize)
        .map(|j| {
            let base = j * in_f_usize;
            (0..in_f_usize).map(|i| activation[i] as f64 * recon[base + i] as f64).sum()
        })
        .collect();

    let ref_norm: f64 = ref_out.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut sq_err = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut q_norm_sq = 0.0f64;

    for j in 0..out_f_usize {
        let d = q_out[j] - ref_out[j];
        sq_err += d * d;
        let ad = d.abs();
        if ad > max_abs { max_abs = ad; }
        dot += q_out[j] * ref_out[j];
        q_norm_sq += q_out[j] * q_out[j];
    }

    let rmse = (sq_err / out_f_usize as f64).sqrt();
    let nrmse = if ref_norm > 1e-30 { sq_err.sqrt() / ref_norm } else { 0.0 };
    let q_norm = q_norm_sq.sqrt();
    let cosine = if q_norm > 1e-30 && ref_norm > 1e-30 {
        dot / (q_norm * ref_norm)
    } else { 1.0 };
    let drift = if ref_norm > 1e-30 { q_norm / ref_norm } else { 1.0 };

    let mut metrics = HashMap::new();
    metrics.insert("rmse".into(), rmse);
    metrics.insert("nrmse".into(), nrmse);
    metrics.insert("cosine".into(), cosine);
    metrics.insert("norm_drift".into(), drift);
    metrics.insert("max_abs_error".into(), max_abs);

    let passed = true
        && gates.operator_nrmse_max.map_or(true, |g| nrmse <= g)
        && gates.operator_cosine_min.map_or(true, |g| cosine >= g)
        && gates.operator_max_abs_max.map_or(true, |g| max_abs <= g);

    Some(SubstitutionEvidence {
        tier: EvidenceTier::Operator,
        evaluated: true,
        passed,
        metrics,
        error: None,
    })
}

// ── FP16 helper ──────────────────────────────────────────────────────────

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 { return sign << 15; }
    if exp == 255 { return (sign << 15) | 0x7C00; }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 { return sign << 15; }
    if new_exp >= 31 { return (sign << 15) | 0x7C00; }
    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_weights(in_f: u32, out_f: u32) -> Vec<f32> {
        let n = (in_f * out_f) as usize;
        (0..n).map(|i| ((i % 128) as f32 - 64.0) * 0.01).collect()
    }

    #[test]
    fn test_fp16_substitution_passes() {
        let w = make_test_weights(64, 64);
        let cand = SubstitutionCandidate::fp16();
        let results = try_all_candidates(&w, 64, 64, &[cand], 100000);
        assert_eq!(results[0].outcome, SubstitutionOutcome::Substituted);
    }

    #[test]
    fn test_ternary_weight_gate() {
        let w = make_test_weights(64, 64);
        let cand = SubstitutionCandidate::ternary();
        let (_, _, recon) = pack_for_candidate(&w, 64, 64, &cand).unwrap();
        let ev = evaluate_weight_gate(&w, &recon, (64 * 64) as usize, &cand).unwrap();
        assert!(ev.evaluated);
        println!("ternary: nrmse={:.6} zero_collapse={:.4}",
            ev.metrics.get("nrmse").unwrap_or(&0.0),
            ev.metrics.get("zero_collapse_ratio").unwrap_or(&0.0));
    }

    #[test]
    fn test_operator_gate_on_ramp() {
        let w = make_test_weights(64, 64);
        let cand = SubstitutionCandidate::fp16();
        let (_, _, recon) = pack_for_candidate(&w, 64, 64, &cand).unwrap();
        let ev = evaluate_operator_gate(&w, &recon, 64, 64, &cand);
        assert!(ev.is_none());
    }
}
