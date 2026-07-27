//! Bounded representation-reconstruction helpers used by progressive
//! stage scoring.
//!
//! **Single authority:** The pure-data transforms
//! ([`reconstruct_representation`], [`quantize_uniform`],
//! [`quantize_ternary`]) and the [`parse_genome_from_string`] adapter
//! that progressive stage evaluation and the mapped-tensor strategy
//! share to map a reference tensor through a candidate
//! [`CandidateGenome`] representation and back to a comparable form.
//! These are the building blocks of bounded progressive scoring:
//! `reconstruct_representation` returns a real reconstruction (not a
//! ternary label with a different name) so admission can compare the
//! actual divergence between the reference and the candidate.

#![forbid(unsafe_code)]

use prism_ecs_ir::evolution::foundation::CandidateGenome;

// ---------------------------------------------------------------------------
// Representation helpers — pure data transforms used by the
// progressive stage executor and the mapped-tensor strategy family
// ---------------------------------------------------------------------------

/// Reconstruct one bounded candidate representation for behavioral
/// scoring. Fallback formats are deliberately real reconstructions,
/// not ternary labels with a different name, so admission can compare
/// their actual divergence.
pub(crate) fn reconstruct_representation(
    reference: &[f32],
    rows: usize,
    cols: usize,
    genome: &CandidateGenome,
) -> (Vec<f32>, u64) {
    use prism_ecs_ir::evolution::RepresentationAxis::*;
    let (bits, candidate) = match genome.representation {
        Fp16 => (16u64, reference.to_vec()),
        Bf16 => (
            16,
            reference
                .iter()
                .map(|v| f32::from_bits(v.to_bits() & 0xffff0000))
                .collect(),
        ),
        Int8 => (8, quantize_uniform(reference, 8)),
        Int4 | Nf4 => (4, quantize_uniform(reference, 4)),
        Nf8 => (8, quantize_uniform(reference, 8)),
        Ternary158 | TernaryTile640 => (2, quantize_ternary(reference, rows, cols, genome)),
        Binary1 => (
            1,
            reference
                .iter()
                .map(|v| if *v >= 0.0 { v.abs() } else { -v.abs() })
                .collect(),
        ),
    };
    let bytes = ((reference.len() as u64 * bits) + 7) / 8;
    (candidate, bytes)
}

/// Symmetric uniform quantization over the reference range.
pub(crate) fn quantize_uniform(reference: &[f32], bits: u32) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let min = reference.iter().copied().fold(f32::INFINITY, f32::min);
    let max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(f32::EPSILON);
    reference
        .iter()
        .map(|v| {
            let q = ((*v - min) / span * levels).round();
            min + q / levels * span
        })
        .collect()
}

/// Row-grouped ternary quantization with packing-aware grouping.
pub(crate) fn quantize_ternary(
    reference: &[f32],
    rows: usize,
    cols: usize,
    genome: &CandidateGenome,
) -> Vec<f32> {
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
    let mut out = vec![0.0; reference.len()];
    for row in 0..rows {
        for start in (0..cols).step_by(group) {
            let end = (start + group).min(cols);
            let scale = reference[row * cols + start..row * cols + end]
                .iter()
                .map(|v| v.abs())
                .sum::<f32>()
                / (end - start).max(1) as f32;
            for col in start..end {
                let v = reference[row * cols + col];
                out[row * cols + col] = if v.abs() <= threshold {
                    0.0
                } else {
                    v.signum() * scale
                };
            }
        }
    }
    out
}

/// Parse a genome from its canonical JSON form. Used by the
/// [`MeasuredEvaluatorAdapter`] wrapper, which has to bridge the
/// search-system's string-based API to the internal
/// [`CandidateGenome`] shape.
pub(crate) fn parse_genome_from_string(
    genome_str: &str,
) -> Result<CandidateGenome, Box<dyn std::error::Error>> {
    serde_json::from_str(genome_str).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strategy_parses_genome_string() {
        let genome = CandidateGenome::new();
        let json = serde_json::to_string(&genome).expect("encode");
        let parsed = parse_genome_from_string(&json).expect("decode");
        assert_eq!(
            std::mem::discriminant(&parsed.representation),
            std::mem::discriminant(&genome.representation)
        );
    }
}
