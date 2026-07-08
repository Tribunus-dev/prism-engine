//! INT8 family candidate generation for QuantSweep.
//!
//! Sweeps over group sizes, clipping policies, and scale policies.
//! INT8 uses direct byte-per-element quantization (640 bytes per tile)
//! with one f32 scale per tile.

use serde_json::json;

use crate::nf4tile640::{pack_int8_weights, unpack_int8_weights, TILE_ELEMENTS};
use crate::quantization::sweep::spec::Int8SweepGrid;
use crate::quantization::sweep::families::FamilyCandidate;

// ── Byte-count estimators ────────────────────────────────────────────────

/// INT8 code bytes: each element is one byte, padded to tile boundary (640).
fn int8_code_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let tile_cols = tiles_per_row * TILE_ELEMENTS;
    (in_features * tile_cols) as u64
}

/// INT8 metadata: one f32 scale per tile (4 bytes).
fn int8_metadata_bytes(in_features: usize, out_features: usize) -> u64 {
    let tiles_per_row = out_features.div_ceil(TILE_ELEMENTS);
    let total_tiles = in_features * tiles_per_row;
    (total_tiles as u64) * 4 // one f32 per tile
}

// ── Candidate generation ──────────────────────────────────────────────────

/// Generate all INT8 family candidates from the sweep grid.

/// Default INT8 sweep grid for initial exploration.
pub fn create_int8_grid() -> Int8SweepGrid {
    use crate::quantization::sweep::spec::{ClippingPolicy, ScalePolicy};
    Int8SweepGrid {
        group_sizes: vec![128, 640],
        clipping_policies: vec![ClippingPolicy::None],
        scale_policies: vec![ScalePolicy::MaxAbs],
        per_channel: false,
    }
}
pub fn generate_int8_candidates(grid: &Int8SweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();

    for &_group_size in &grid.group_sizes {
        for clip in &grid.clipping_policies {
            for scale_policy in &grid.scale_policies {
                let params = json!({
                    "family": "Int8",
                    "group_size": _group_size,
                    "clipping_policy": format!("{:?}", clip),
                    "scale_policy": format!("{:?}", scale_policy),
                    "per_channel": grid.per_channel,
                });

                let packer = Box::new(|w: &[f32], r: usize, c: usize| {
                    let (codes, scales, biases) = pack_int8_weights(w, r, c);
                    (codes, scales, biases, Vec::new())
                });

                let unpacker = Box::new(|codes: &[u8], scales: &[f32], biases: &[f32], _extra: &[u8], rows: usize, cols: usize| {
                    unpack_int8_weights(codes, scales, biases, rows, cols)
                });

                candidates.push(FamilyCandidate {
                    label: format!(
                        "Int8_g{}_clip{:?}_sp{:?}",
                        _group_size, clip, scale_policy,
                    ),
                    parameters: params,
                    packer,
                    unpacker,
                    code_bytes_fn: int8_code_bytes,
                    metadata_bytes_fn: int8_metadata_bytes,
                });
            }
        }
    }

    candidates
}
