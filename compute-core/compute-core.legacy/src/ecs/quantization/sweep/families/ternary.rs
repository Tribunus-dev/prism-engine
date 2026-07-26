//! Ternary family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, sparsity targets, scale policies, and residual
//! policies. Ternary uses 2-bit-per-element codes (64 bytes per 256-element block)
//! with one f32 scale per block.

use serde_json::json;

use crate::ecs::quantization::embed_cluster::{pack_ternary_weights, unpack_ternary_weights};
use crate::ecs::quantization::sweep::families::FamilyCandidate;
use crate::ecs::quantization::sweep::spec::TernarySweepGrid;

const BLOCK_SIZE: usize = 256;
const CODES_PER_BLOCK: usize = 64; // 256 × 2 bits / 8

// ── Byte-count estimators ────────────────────────────────────────────────

/// Ternary code bytes: each 256-element block stores 64 bytes of 2-bit codes.
fn ternary_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let blocks_per_row = out_features.div_ceil(BLOCK_SIZE);
    let total_blocks = in_features * blocks_per_row;
    (total_blocks * CODES_PER_BLOCK) as u64
}

/// Ternary metadata: one f32 scale per block (4 bytes).
fn ternary_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let blocks_per_row = out_features.div_ceil(BLOCK_SIZE);
    let total_blocks = in_features * blocks_per_row;
    (total_blocks * 4) as u64
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all Ternary family candidates from the sweep grid.

/// Default Ternary sweep grid for initial exploration.
pub fn create_ternary_grid() -> TernarySweepGrid {
    TernarySweepGrid {
        group_sizes: vec![32, 64, 128],
        sparsity_targets: vec![0.50, 0.70, 0.85],
        scale_policies: vec!["mean_abs_nonzero".to_string()],
        residual_policies: vec!["none".to_string()],
    }
}
pub fn generate_ternary_candidates(grid: &TernarySweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    for &_group_size in &grid.group_sizes {
        for &sparsity in &grid.sparsity_targets {
            for scale_policy in &grid.scale_policies {
                for residual_policy in &grid.residual_policies {
                    let params = json!({
                        "family": "Ternary",
                        "group_size": _group_size,
                        "sparsity_target": sparsity,
                        "scale_policy": scale_policy,
                        "residual_policy": residual_policy,
                    });

                    let packer = Box::new(|w: &[f32], r: usize, c: usize| {
                        let (codes, scales, biases) = pack_ternary_weights(w, r, c);
                        (codes, scales, biases, Vec::<u8>::new())
                    });

                    let unpacker = Box::new(
                        |codes: &[u8],
                         scales: &[f32],
                         biases: &[f32],
                         _extra: &[u8],
                         rows: usize,
                         cols: usize| {
                            unpack_ternary_weights(codes, scales, biases, rows, cols)
                        },
                    );

                    candidates.push(FamilyCandidate {
                        label: format!(
                            "Ternary_g{}_sp{:.2}_sp_{}_rp_{}",
                            _group_size, sparsity, scale_policy, residual_policy,
                        ),
                        parameters: params,
                        packer,
                        unpacker,
                        code_bytes_fn: ternary_code_bytes,
                        metadata_bytes_fn: Box::new(ternary_metadata_bytes),
                    });
                }
            }
        }
    }

    candidates
}
