//! QuantFamily trait and candidate generation dispatch.
//!
//! Each codec family implements a candidate generator that sweeps its parameter
//! grid.  `generate_all_candidates` dispatches across all active families.

pub mod int8;
pub mod mixed_tile;
pub mod nf4;
pub mod sym_int4;
pub mod ternary;

use serde_json::Value;

use crate::quantization::sweep::spec::QuantFamilySweep;

// ── FamilyCandidate ─────────────────────────────────────────────────────────────

/// One fully-resolved parameter combination from a codec family sweep grid.
///
/// The packer/unpacker/codec-size closures are populated by the family's
/// candidate generation loop; the runner invokes them without needing to know
/// which family produced the candidate.
pub struct FamilyCandidate {
    /// Human-readable label (e.g. "Nf4Tile640").
    pub label: String,
    /// Resolved parameter set for this candidate, as JSON Value.
    pub parameters: Value,
    /// Pack weights into (codes, scales, biases, extra).
    pub packer: Box<dyn Fn(&[f32], usize, usize) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) + Send + Sync>,
    /// Reconstruct weights from packed representation.
    pub unpacker:
        Box<dyn Fn(&[u8], &[f32], &[f32], &[u8], usize, usize) -> Vec<f32> + Send + Sync>,
    /// Compute code byte count for given dimensions.
    pub code_bytes_fn: fn(usize, usize) -> u64,
    /// Compute metadata byte count for given dimensions.
    pub metadata_bytes_fn: fn(usize, usize) -> u64,
}

// ── Dispatch ────────────────────────────────────────────────────────────────────

/// Generate all candidates from all active families in the sweep spec.
pub fn generate_all_candidates(families: &[QuantFamilySweep]) -> Vec<FamilyCandidate> {
    let mut all = Vec::new();
    for family in families {
        match family {
            QuantFamilySweep::Nf4(grid) => {
                all.extend(super::families::nf4::generate_nf4_candidates(grid));
            }
            QuantFamilySweep::SymInt4(grid) => {
                all.extend(super::families::sym_int4::generate_sym_int4_candidates(grid));
            }
            QuantFamilySweep::Int8(grid) => {
                all.extend(super::families::int8::generate_int8_candidates(grid));
            }
            QuantFamilySweep::Ternary(grid) => {
                all.extend(super::families::ternary::generate_ternary_candidates(grid));
            }
            QuantFamilySweep::MixedTile(grid) => {
                all.extend(super::families::mixed_tile::generate_mixed_tile_candidates(
                    grid,
                ));
            }
        }
    }
    all
}
