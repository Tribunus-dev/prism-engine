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
use std::fmt;

use crate::ecs::quantization::sweep::candidate::{MatrixShape, PackedCandidate, QuantFamilyId};
use crate::ecs::quantization::sweep::spec::{LayoutSweepGrid, QuantFamilySweep};

// ── SweepScratch ────────────────────────────────────────────────────────────────

pub struct SweepScratch {
    pub tile_f32: Vec<f32>,
    pub recon_band: Vec<f32>,
    pub error_band: Vec<f32>,
    pub metrics_tmp: Vec<f32>,
}

impl SweepScratch {
    pub fn new() -> Self {
        Self {
            tile_f32: Vec::with_capacity(640),
            recon_band: Vec::new(),
            error_band: Vec::new(),
            metrics_tmp: Vec::new(),
        }
    }
}

// ── ParamError ──────────────────────────────────────────────────────────────────

pub enum ParamError {
    InvalidGroupSize { group_size: usize },
    InvalidCodebook { codebook: String },
    InvalidAffineMode { mode: String },
    InvalidClippingPolicy { policy: String },
}

// ── QuantError ──────────────────────────────────────────────────────────────────

pub enum QuantError {
    InvalidDimensions { expected: usize, got: usize },
    InvalidPayload { reason: String },
    ParamError(ParamError),
}

// ── QuantFamily trait ───────────────────────────────────────────────────────────

pub trait QuantFamily: Send + Sync {
    type Params: Clone + serde::Serialize + for<'de> serde::Deserialize<'de>;
    fn family_id(&self) -> QuantFamilyId;
    fn enumerate(&self, grid: &crate::quantization::sweep::spec::Nf4SweepGrid) -> Vec<Self::Params>
    where
        Self::Params: 'static;
    fn validate_params(&self, params: &Self::Params) -> Result<(), ParamError>;
    fn pack(
        &self,
        source: &[f32],
        logical_shape: &MatrixShape,
        params: &Self::Params,
        scratch: &mut SweepScratch,
    ) -> Result<PackedCandidate, QuantError>;
    fn unpack(
        &self,
        packed: &PackedCandidate,
        logical_shape: &MatrixShape,
        params: &Self::Params,
        scratch: &mut SweepScratch,
    ) -> Result<Vec<f32>, QuantError>;
}

// ── FamilyCandidate ─────────────────────────────────────────────────────────────

/// One fully-resolved parameter combination from a codec family sweep grid.
///
/// Deprecated: This transitional type will be replaced by the QuantFamily trait.
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
    /// extra is already LE bytes (Vec<u8>), not Vec<f32>.
    pub packer:
        Box<dyn Fn(&[f32], usize, usize) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>) + Send + Sync>,
    /// Reconstruct weights from packed representation.
    pub unpacker: Box<dyn Fn(&[u8], &[f32], &[f32], &[u8], usize, usize) -> Vec<f32> + Send + Sync>,
    /// Compute code byte count for given dimensions.
    pub code_bytes_fn: fn(usize, usize) -> u64,
    /// Compute metadata byte count for given dimensions (closure may capture group_size).
    pub metadata_bytes_fn: Box<dyn Fn(usize, usize) -> u64 + Send + Sync>,
}
impl fmt::Debug for FamilyCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FamilyCandidate")
            .field("label", &self.label)
            .field("parameters", &self.parameters)
            .finish_non_exhaustive()
    }
}

/// One fully-resolved parameter combination from a layout parameter sweep.
///
/// Wraps an existing codec-family candidate and augments it with layout
/// parameters — tile shape, group axis, metadata placement, and execution
/// lane — so the runner can compare hardware configuration trade-offs on
/// the same quantized representation.
pub struct LayoutFamilyCandidate {
    /// Base codec-family candidate (label, parameters, packer/unpacker/etc.).
    pub base: FamilyCandidate,
    /// Tile shape identifier (e.g. "640", "256", "1024").
    pub tile_shape: String,
    /// Group axis policy (e.g. "PackedContiguous", "InputAxis").
    pub group_axis: String,
    /// Metadata placement (e.g. "AdjacentTile", "SeparatedManifest").
    pub metadata_layout: String,
    /// Execution lane (e.g. "MetalFusedGpu", "MetalTensorApi").
    pub execution_lane: String,
}

impl std::fmt::Debug for LayoutFamilyCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayoutFamilyCandidate")
            .field("label", &self.base.label)
            .field("parameters", &self.base.parameters)
            .field("tile_shape", &self.tile_shape)
            .field("group_axis", &self.group_axis)
            .field("metadata_layout", &self.metadata_layout)
            .field("execution_lane", &self.execution_lane)
            .finish()
    }
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
                all.extend(super::families::sym_int4::generate_sym_int4_candidates(
                    grid,
                ));
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
            QuantFamilySweep::Layout(grid) => {
                all.extend(generate_layout_candidates(grid));
            }
        }
    }
    all
}

/// Generate candidates from a layout sweep grid.
///
/// Each candidate records one combination of tile shape, group axis,
/// metadata placement, and execution lane. The closures are identity/no-op
/// since layout candidates direct *how* an existing codec candidate runs
/// rather than performing quantization themselves.
fn generate_layout_candidates(grid: &LayoutSweepGrid) -> Vec<FamilyCandidate> {
    let mut candidates = Vec::new();
    for ts in &grid.tile_shapes {
        for ga in &grid.group_axes {
            for ml in &grid.metadata_layouts {
                for el in &grid.execution_lanes {
                    let label = format!("Layout::{}::{}::{}::{}", ts, ga, ml, el);
                    let parameters = serde_json::json!({
                        "tile_shape": ts,
                        "group_axis": ga,
                        "metadata_layout": ml,
                        "execution_lane": el,
                    });
                    candidates.push(FamilyCandidate {
                        label,
                        parameters,
                        packer: Box::new(|_, _, _| {
                            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                        }),
                        unpacker: Box::new(|_, _, _, _, _, _| Vec::new()),
                        code_bytes_fn: |_, _| 0,
                        metadata_bytes_fn: Box::new(|_, _| 0),
                    });
                }
            }
        }
    }
    candidates
}
