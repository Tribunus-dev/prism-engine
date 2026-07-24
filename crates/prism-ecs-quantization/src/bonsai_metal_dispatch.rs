//! Metal-backed Bonsai dispatch adapters.
//!
//! This module provides stubs for environments where a Metal runtime backend is
//! not built into the tree. The Bonsai cimage runtime gates all metal paths behind
//! the same `target_os = "macos"` and `metal-dispatch` conditions, so these
//! helpers preserve contract-compatibility without introducing placeholder
//! behavior in non-metal compilation profiles.

use crate::bonsai_ternary;

/// Return true only when a valid Metal kernel contract can be claimed.
pub fn verify_kernel_contract() -> bool {
    false
}

/// Run Tile640 ternary GEMV through a Metal-backed path.
///
/// This fallback implementation intentionally executes the CPU reference so that
/// behavior is still well-defined when the optimized dispatch implementation is
/// unavailable.
pub fn run_ternary_gemv(
    packed_bytes: &[u8],
    input: &[f32],
    page_scale_bytes: &[u8],
    lane_scale_bytes: &[u8],
    outlier_rows_bytes: Option<&[u8]>,
    outlier_cols_bytes: Option<&[u8]>,
    outlier_vals_bytes: Option<&[u8]>,
    dim_n: u32,
    dim_m: u32,
) -> Result<Vec<f32>, String> {
    let lane_scale_i8: Vec<i8> = lane_scale_bytes.iter().map(|&byte| byte as i8).collect();
    let packed_u32: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let page_scale_u16: Vec<u16> = page_scale_bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    let outlier_rows = outlier_rows_bytes.map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    });
    let outlier_cols = outlier_cols_bytes.map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    });
    let outlier_vals = outlier_vals_bytes.map(|bytes| {
        bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    });

    Ok(bonsai_ternary::apply_outlier_correction(
        &bonsai_ternary::ternary_gemv_ref(
            &packed_u32,
            input,
            &page_scale_u16,
            &lane_scale_i8,
            dim_n,
            dim_m,
        ),
        outlier_rows,
        outlier_cols,
        outlier_vals,
        page_scale_u16.as_slice(),
        &lane_scale_i8,
    ))
}
