use crate::ecs::ternary::codec::TernaryCodecError;
use crate::ecs::ternary::pack::unpack_ternary_codes;

/// Reference GEMV for ternary-packed weights.
///
/// Unpacks each group of `group_size` weights, multiplies by the per-group
/// f32 scale, and computes the dot product with the corresponding segment
/// of `activations`.
///
/// Returns `rows` output values (one per output row).
pub fn ternary_gemv_reference(
    activations: &[f32],
    codes: &[u8],
    scales: &[f32],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<Vec<f32>, TernaryCodecError> {
    if activations.len() < cols {
        return Err(TernaryCodecError::PackingError(format!(
            "activations length {} < cols {}",
            activations.len(),
            cols
        )));
    }

    let groups_per_row = cols.div_ceil(group_size);
    let expected_scales = rows * groups_per_row;
    if scales.len() < expected_scales {
        return Err(TernaryCodecError::PackingError(format!(
            "scales length {} < rows {} * groups_per_row {}",
            scales.len(),
            rows,
            groups_per_row
        )));
    }

    let _total_values = rows * cols;
    let bytes_per_row = groups_per_row * ((group_size + 3) / 4);
    let expected_bytes = rows * bytes_per_row;
    if codes.len() < expected_bytes {
        return Err(TernaryCodecError::LengthMismatch {
            expected: expected_bytes,
            actual: codes.len(),
        });
    }

    let mut output = vec![0.0f32; rows];

    for r in 0..rows {
        let row_code_offset = r * bytes_per_row;
        let row_scale_offset = r * groups_per_row;
        let mut acc = 0.0f32;

        for g in 0..groups_per_row {
            let col_start = g * group_size;
            let col_end = (col_start + group_size).min(cols);
            let n_weights = col_end - col_start;
            let n_padded = group_size; // full footprint even for last partial group

            let scale = scales[row_scale_offset + g];
            let group_byte_offset = row_code_offset + g * ((group_size + 3) / 4);
            let group_bytes = &codes[group_byte_offset..group_byte_offset + (n_padded + 3) / 4];

            let weights = unpack_ternary_codes(group_bytes, n_weights)?;

            for (k, &w) in weights.iter().enumerate() {
                let act = activations[col_start + k];
                acc += act * (w as f32) * scale;
            }
        }

        output[r] = acc;
    }

    Ok(output)
}
