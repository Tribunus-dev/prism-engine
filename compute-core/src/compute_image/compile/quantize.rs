//! Compile-time quantization transforms — NF4, 8-bit affine, INT4, ternary.

use super::source::{ensure_tensor_loaded, LoadedSource, SourceTensor};
use crate::config::CompileQuantMode;

// ═══════════════════════════════════════════════════════════════════════════
// NF4 codebook
// ═══════════════════════════════════════════════════════════════════════════

/// These are the 16 quantiles of a standard normal distribution,
/// symmetric around zero, with equal area under the curve per interval.
pub(crate) const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.8480, -0.5698, -0.3940, -0.2419, -0.1057, 0.0, 0.1057, 0.2419, 0.3940, 0.5698, 0.8480,
    1.0, 1.2588, 1.5862, 2.0,
];

/// Find the nearest NF4 codebook index for a given normalized value.
/// Returns index in [0, 15].
pub(crate) fn quantize_nf4_value(value: f32) -> u8 {
    let mut best_idx: u8 = 0;
    let mut best_dist: f32 = (value - NF4_CODEBOOK[0]).abs();
    for (i, &level) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let dist = (value - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Apply NF4 block quantization to a single group of F32 values.
/// Returns (packed_u32_words, scale_absmax, bias_zero_point).
/// For NF4: bias is always 0.0 (symmetric quantization).
pub(crate) fn quantize_nf4_group(values: &[f32]) -> (Vec<u32>, f32, f32) {
    if values.is_empty() {
        return (vec![0u32; 1], 0.0, 0.0);
    }
    // Find absolute maximum for the group (the scale factor).
    let absmax = values.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));

    let scale = if absmax > 1e-12 { absmax } else { 1.0 };
    let inv_scale = 1.0 / scale;

    // Quantize each value to a 4-bit NF4 index, pack 8 per U32 word.
    let n_words = (values.len() + 7) / 8;
    let mut packed = vec![0u32; n_words];
    for (i, &val) in values.iter().enumerate() {
        let normalized = val * inv_scale;
        // Clamp to [-1, 1] range (NF4 codebook bounds).
        let clamped = normalized.clamp(-1.0, 1.0);
        let idx = quantize_nf4_value(clamped);
        let word_idx = i / 8;
        let bit_shift = ((i % 8) * 4) as u32;
        packed[word_idx] |= (idx as u32) << bit_shift;
    }

    (packed, scale, 0.0) // NF4 is symmetric — bias = 0
}

/// Apply 8-bit affine block quantization to a single group of F32 values.
/// Returns (packed_u8_bytes, scale, bias).
pub(crate) fn quantize_af8_group(values: &[f32]) -> (Vec<u8>, f32, f32) {
    if values.is_empty() {
        return (vec![0u8; 1], 0.0, 0.0);
    }
    let min_val = values.iter().cloned().fold(f32::MAX, f32::min);
    let max_val = values.iter().cloned().fold(f32::MIN, f32::max);

    let range = max_val - min_val;
    let scale = if range > 1e-12 {
        range / 255.0
    } else {
        1.0 / 255.0
    };
    let bias = min_val;

    let mut q = Vec::with_capacity(values.len());
    for &v in values {
        let qv = ((v - min_val) / scale).round().clamp(0.0, 255.0) as u8;
        q.push(qv);
    }

    (q, scale, bias)
}

/// Apply 4-bit affine (INT4) block quantization to a single group of F32 values.
/// Uses standard signed 4-bit format: values in [-8, 7], stored as unsigned [0, 15].
/// This matches MLX's affine dequantization format.
pub fn quantize_int4_group(values: &[f32]) -> (Vec<u32>, f32, f32) {
    if values.is_empty() {
        return (vec![0u32; 1], 0.0, 0.0);
    }

    let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    if range == 0.0 {
        return (vec![0u32; 1], 1.0, min_val);
    }
    // MLX: scale = max_abs / 8 (unsigned 4-bit centers max_abs within [0-15]).
    // bias = max_val when min has wider negative range, else bias = min_val.
    // When bias = max, scale is negative (decrements from max).
    let max_abs = max_val.abs().max(min_val.abs());
    let scale_mag = max_abs / 8.0;
    let (scale, bias) = if max_val.abs() < min_val.abs() {
        // Negative side has larger magnitude: bias = max, scale negative
        (-scale_mag, max_val)
    } else {
        // Positive dominates or symmetric: bias = min, scale positive
        (scale_mag, min_val)
    };
    let n = values.len();
    let packed_len = (n + 7) / 8;
    let mut packed = vec![0u32; packed_len];

    for (i, &val) in values.iter().enumerate() {
        // MLX affine 4-bit: deq = scale * u + bias, u is unsigned [0, 15]
        // u = round((val - bias) / scale), clamped to [0, 15]
        let u = ((val - bias) / scale).round().clamp(0.0, 15.0) as u8;
        let word_idx = i / 8;
        let bit_shift = ((i % 8) * 4) as u32;
        packed[word_idx] |= (u as u32) << bit_shift;
    }
    (packed, scale, bias)
}

// ═══════════════════════════════════════════════════════════════════════════
// Top-level quantization dispatch
// ═══════════════════════════════════════════════════════════════════════════

/// Apply compile-time quantization to all FP16/BF16 weight tensors in the
/// loaded source. This modifies the source tensors in-place, converting
/// weight tensor bytes to packed quantized form and adding companion
/// scale/bias tensors. The TensorBinding packed_shape fields are also set
/// so the existing `emit_quantized_binding` pipeline writes the triplets.
pub(crate) fn apply_quantize_to_loaded(
    loaded: &mut LoadedSource,
    qmode: CompileQuantMode,
) -> crate::Result<()> {
    // Collect all weight bindings (global + per-layer) that are not already packed.
    #[allow(dead_code)]
    struct WeightBinding {
        name: String,
        role: String,
        logical_shape: Vec<u32>,
        is_global: bool,
        layer_index: Option<u32>,
    }

    let mut weight_bindings: Vec<WeightBinding> = Vec::new();

    // Collect global weight tensors.
    for binding in &loaded.spec.global_tensors {
        if binding.name.ends_with(".weight") {
            weight_bindings.push(WeightBinding {
                name: binding.name.clone(),
                role: format!("{:?}", binding.role),
                logical_shape: binding.logical_shape.clone(),
                is_global: true,
                layer_index: None,
            });
        }
    }

    // Collect per-layer weight tensors.
    for layer in &loaded.spec.layers {
        for binding in &layer.tensors {
            if binding.name.ends_with(".weight") {
                weight_bindings.push(WeightBinding {
                    name: binding.name.clone(),
                    role: format!("{:?}", binding.role),
                    logical_shape: binding.logical_shape.clone(),
                    is_global: false,
                    layer_index: Some(layer.index),
                });
            }
        }
    }

    // Lazy-load each weight tensor from mmap before quantizing
    for wb in &weight_bindings {
        if let Some(tensor) = loaded.source_tensors.get_mut(&wb.name) {
            for mmap in &loaded.mmap_bytes {
                ensure_tensor_loaded(tensor, mmap);
                if !tensor.data.is_empty() {
                    break;
                }
            }
        }
    }

    eprintln!(
        "[quantize] applying {} quantization to {} weight tensors",
        match qmode {
            CompileQuantMode::Nf4 { group_size } => {
                format!("NF4 (group_size={})", group_size)
            }
            CompileQuantMode::Af8 { group_size } => {
                format!("8-bit affine (group_size={})", group_size)
            }
            CompileQuantMode::Ternary { group_size } => {
                format!("ternary 1.58-bit (group_size={})", group_size)
            }
            CompileQuantMode::TernaryTile640 { group_size } => {
                format!("ternary tile640 (group_size={})", group_size)
            }
        },
        weight_bindings.len(),
    );

    for wb in &weight_bindings {
        let source_tensor = loaded.source_tensors.get(&wb.name).ok_or_else(|| {
            crate::Error::from_reason(format!("quantize: missing source tensor '{}'", wb.name))
        })?;

        // Only quantize FP16/BF16 dtypes.
        let dtype = source_tensor.dtype.clone();
        if dtype != "F16" && dtype != "BF16" {
            eprintln!(
                "[quantize] skipping {} (dtype={}, only FP16/BF16 supported)",
                wb.name, dtype
            );
            continue;
        }

        let shape = &source_tensor.shape;
        // Skip 1D tensors (RMS norm weights, etc.) — only quantize 2D weight matrices.
        if shape.len() != 2 {
            eprintln!(
                "[quantize] skipping {} (shape={:?}, only 2D weight matrices supported)",
                wb.name, shape
            );
            continue;
        }
        let out_dim = shape[0]; // rows
        let in_dim = shape[1]; // cols

        // Clone raw data to release the immutable borrow on loaded
        // before calling try_ternary_tile640_pack_gpu which needs &mut loaded.
        let raw = source_tensor.data.clone();

        // GPU-accelerated TernaryTile640: try Metal before falling back to CPU.
        // The GPU kernel reads BF16 directly from a shared-memory buffer,
        // skipping the CPU F32 conversion entirely.
        #[cfg(feature = "metal-dispatch")]
        if matches!(qmode, CompileQuantMode::TernaryTile640 { .. }) {
            if try_ternary_tile640_pack_gpu(loaded, &wb.name, &raw, out_dim, in_dim, None)? {
                eprintln!("[quantize] {} ternary tile640 → GPU", wb.name);
                continue;
            }
            eprintln!("[quantize] {} ternary tile640 → CPU fallback (Metal unavailable)", wb.name);
        }

        // Convert FP16/BF16 raw bytes to F32.
        let n_elements = raw.len() / 2;
        let mut f32_vals = Vec::with_capacity(n_elements);
        if dtype == "BF16" {
            // BF16: same exponent/mantissa layout as F32 top-16 bits.
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32_vals.push(f32::from_bits((bits as u32) << 16));
            }
        } else {
            // FP16: standard IEEE 754 half-precision.
            for chunk in raw.chunks_exact(2) {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32_vals.push(half_to_f32(bits));
            }
        }

        let group_size = match qmode {
            CompileQuantMode::Nf4 { group_size } => group_size,
            CompileQuantMode::Af8 { group_size } => group_size,
            CompileQuantMode::Ternary { group_size } => group_size,
            CompileQuantMode::TernaryTile640 { group_size } => group_size,
        };
        let groups_per_row = (in_dim + group_size - 1) / group_size;
        let total_groups = out_dim * groups_per_row;

        // Apply block quantization per group.
        match qmode {
            CompileQuantMode::Nf4 { .. } => {
                apply_nf4_quantize(
                    loaded,
                    &wb.name,
                    &f32_vals,
                    out_dim,
                    in_dim,
                    group_size,
                    groups_per_row,
                    total_groups,
                )?;
            }
            CompileQuantMode::Af8 { .. } => {
                apply_af8_quantize(
                    loaded,
                    &wb.name,
                    &f32_vals,
                    out_dim,
                    in_dim,
                    group_size,
                    groups_per_row,
                    total_groups,
                )?;
            }
            CompileQuantMode::Ternary { .. } => {
                apply_ternary_quantize(
                    loaded,
                    &wb.name,
                    &f32_vals,
                    out_dim,
                    in_dim,
                    group_size,
                    groups_per_row,
                    total_groups,
                )?;
            }
            CompileQuantMode::TernaryTile640 { .. } => {
                apply_ternary_tile640_quantize(
                    loaded,
                    &wb.name,
                    &f32_vals,
                    out_dim,
                    in_dim,
                    group_size,
                    groups_per_row,
                    total_groups,
                )?;
            }
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// 8-bit affine quantization
// ═══════════════════════════════════════════════════════════════════════════

/// Apply 8-bit affine quantization to a weight tensor and update the loaded source.
pub(crate) fn apply_af8_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // 8-bit quantized weights stored as U8.
    let packed_weight_len = (out_dim as usize) * in_dim_u;
    let mut packed_weight = vec![0u8; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);
    let mut biases = Vec::with_capacity(total_g);

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            let (q_bytes, scale, bias) = quantize_af8_group(group_vals);
            scales.push(scale);
            biases.push(bias);

            for (wi, &byte) in q_bytes.iter().enumerate() {
                packed_weight[group_start + wi] = byte;
            }
        }
    }

    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = biases
        .iter()
        .flat_map(|&b| b.to_le_bytes().to_vec())
        .collect();

    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let pack = 32 / 8; // 4 U8 per U32
    let packed_in = in_dim / pack;

    // Repack U8 weight bytes into U32 words (4 U8 per U32, little-endian).
    // MLX quantized matmul requires uint32 weight arrays.
    let u32_weight_len = (packed_weight.len() + 3) / 4;
    let mut packed_u32 = vec![0u32; u32_weight_len];
    for (i, chunk) in packed_weight.chunks(4).enumerate() {
        let mut word: u32 = 0;
        for (j, &byte) in chunk.iter().enumerate() {
            word |= (byte as u32) << (j * 8);
        }
        packed_u32[i] = word;
    }
    let u32_weight_bytes: Vec<u8> = packed_u32
        .iter()
        .flat_map(|&w| w.to_le_bytes().to_vec())
        .collect();

    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 8,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace weight source tensor.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = u32_weight_bytes;
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, packed_in];
    }

    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] 8-bit affine quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// NF4 quantization
// ═══════════════════════════════════════════════════════════════════════════

/// Apply NF4 quantization to a weight tensor and update the loaded source.
pub(crate) fn apply_nf4_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // Packed NF4 weights: each U32 stores 8 * 4-bit values.
    let pack_factor = 8; // 32 / 4
    let packed_in = (in_dim_u + pack_factor - 1) / pack_factor;
    let packed_weight_len = (out_dim as usize) * packed_in;
    let mut packed_weight = vec![0u32; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);
    let _biases = vec![0.0f32; total_g]; // NF4 is symmetric — biases are 0

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            let (_packed_group, _scale, _bias) = quantize_nf4_group(group_vals);
            let (packed_group, scale, _bias) = quantize_int4_group(group_vals);
            scales.push(scale);

            // Place packed U32 words into the correct position in packed_weight.
            let weight_row_offset = row * packed_in;
            let group_word_offset = g * ((gs + pack_factor - 1) / pack_factor);
            for (wi, &word) in packed_group.iter().enumerate() {
                let idx = weight_row_offset + group_word_offset + wi;
                if idx >= packed_weight.len() {
                    return Err(crate::Error::from_reason(format!(
                        "OOB: row={} g={} wi={} idx={} len={} out={} in={} packed_in={} gpr={}",
                        row,
                        g,
                        wi,
                        idx,
                        packed_weight.len(),
                        out_dim,
                        in_dim_u,
                        packed_in,
                        gpr
                    )));
                }
                packed_weight[idx] = word;
            }
        }
    }

    // Serialize packed weights as U32 bytes (little-endian).
    let packed_bytes: Vec<u8> = packed_weight
        .iter()
        .flat_map(|&w| w.to_le_bytes().to_vec())
        .collect();
    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = vec![0u8; total_g * 4]; // F32 zeros

    // Derive scale/bias tensor names.
    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    // Build the packed shape descriptor.
    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in as u32],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 4,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace the weight source tensor with packed data.
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = packed_bytes;
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, packed_in as u32];
    }

    // Add scale source tensor.
    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    // Add bias source tensor.
    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    // Update the TensorBinding in the spec to enable packed emission.
    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] NF4 quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Ternary 1.58-bit quantization (2-bit nibble, 4 per byte)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply ternary quantization (1.58-bit) to a weight tensor.
///
/// Each weight becomes one of { -1, 0, +1 }, stored as 2-bit nibbles
/// (4 per byte): 00=0, 01=+1, 10=-1. Block scaling (absmax per group)
/// is computed and stored as F32 scales alongside packed U8 weight data.
pub(crate) fn apply_ternary_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let gs = group_size as usize;
    let gpr = groups_per_row as usize;
    let total_g = total_groups as usize;

    // Packed ternary weights: 4 weights per byte (2 bits each)
    let packed_in = (in_dim_u + 3) / 4;
    let packed_weight_len = (out_dim as usize) * packed_in;
    let mut packed_weight = vec![0u8; packed_weight_len];
    let mut scales = Vec::with_capacity(total_g);

    for row in 0..out_dim as usize {
        let row_offset = row * in_dim_u;
        for g in 0..gpr {
            let group_start = row_offset + g * gs;
            let group_end = (group_start + gs).min(row_offset + in_dim_u);
            let group_vals = &f32_vals[group_start..group_end];

            // Compute absmax scale for this group
            let absmax = group_vals
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, |a, b| a.max(b));
            let scale = if absmax > 1e-12 { absmax } else { 1.0 };
            scales.push(scale);

            let inv_scale = 1.0 / scale;

            // Quantize to {0, +1, -1} → 2-bit nibble: 00=0, 01=+1, 10=-1
            for (wi, &val) in group_vals.iter().enumerate() {
                let normalized = val * inv_scale;
                let nibble: u8 = if normalized > 0.5 {
                    0b01 // +1
                } else if normalized < -0.5 {
                    0b10 // -1
                } else {
                    0b00 // 0
                };
                let abs_pos = group_start + wi;
                let byte_idx = row * packed_in + abs_pos / 4;
                let bit_shift = (abs_pos % 4) * 2;
                packed_weight[byte_idx] |= nibble << bit_shift;
            }
        }
    }

    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = vec![0u8; total_g * 4]; // F32 zeros (ternary is symmetric)

    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in as u32],
        scales: vec![out_dim, groups_per_row],
        biases: vec![out_dim, groups_per_row],
        bits: 2,
        group_size,
        groups: groups_per_row * out_dim,
    };

    // Replace weight source tensor with packed ternary U8 data
    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = packed_weight;
        st.dtype = "U8".to_string();
        st.shape = vec![out_dim, packed_in as u32];
    }

    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, groups_per_row],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] Ternary quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, groups_per_row
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Ternary tile640 quantization
// ═══════════════════════════════════════════════════════════════════════════

const TILE_SIZE: usize = 640;

/// Apply ternary tile640 quantization to a weight tensor.
///
/// Base-3 encoding (20 per u32) with 640-weight tiles, 32 lanes, 20 weights/lane.
/// Each row is pre-padded to TILE_SIZE multiple before processing.
/// Block scaling per tile.
pub(crate) fn apply_ternary_tile640_quantize(
    loaded: &mut LoadedSource,
    weight_name: &str,
    f32_vals: &[f32],
    out_dim: u32,
    in_dim: u32,
    group_size: u32,
    groups_per_row: u32,
    total_groups: u32,
) -> crate::Result<()> {
    let in_dim_u = in_dim as usize;
    let out_dim_u = out_dim as usize;
    let _gs = group_size as usize;
    let _gpr = groups_per_row as usize;
    let _total_g = total_groups as usize;

    // Pad each row to TILE_SIZE multiple
    let padded_cols = ((in_dim_u + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
    let tile_count = padded_cols / TILE_SIZE;

    // Base-3 encoding: 20 ternary values per u32 (3^20 fits in u32)
    let vals_per_u32 = 20;
    let u32_per_tile = TILE_SIZE / vals_per_u32; // 640 / 20 = 32 u32s per tile
    let total_u32s = out_dim_u * tile_count * u32_per_tile;
    let mut packed = vec![0u32; total_u32s];
    let mut scales = Vec::with_capacity(out_dim_u * tile_count);

    for row in 0..out_dim_u {
        let row_offset = row * in_dim_u;

        // Build padded row
        let row_padded: Vec<f32> = if padded_cols > in_dim_u {
            let mut r = Vec::with_capacity(padded_cols);
            r.extend_from_slice(&f32_vals[row_offset..row_offset + in_dim_u]);
            r.resize(padded_cols, 0.0);
            r
        } else {
            f32_vals[row_offset..row_offset + in_dim_u].to_vec()
        };

        for t in 0..tile_count {
            let tile_start = t * TILE_SIZE;
            let tile_end = (tile_start + TILE_SIZE).min(padded_cols);
            // SAFETY: row_padded is sized to padded_cols, tile_start < padded_cols
            let tile_slice = &row_padded[tile_start..tile_end];

            // Block scale for this tile
            let absmax = tile_slice
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, |a, b| a.max(b));
            let scale = if absmax > 1e-12 { absmax } else { 1.0 };
            scales.push(scale);
            let inv_scale = 1.0 / scale;

            // 32 lanes, 20 weights per lane → base-3 encoded
            for lane in 0..32 {
                let mut base3_value: u32 = 0;
                for w in 0..vals_per_u32 {
                    let idx = lane * vals_per_u32 + w;
                    let val = if idx < tile_slice.len() {
                        tile_slice[idx] * inv_scale
                    } else {
                        0.0
                    };
                    // Ternary: 0=0, 1=+1, 2=-1
                    let ternary: u32 = if val > 0.5 {
                        1
                    } else if val < -0.5 {
                        2
                    } else {
                        0
                    };
                    base3_value = base3_value * 3 + ternary;
                }
                let packed_idx = row * tile_count * u32_per_tile + t * u32_per_tile + lane;
                // SAFETY: packed_idx is bounded by total_u32s computed above
                if packed_idx < total_u32s {
                    packed[packed_idx] = base3_value;
                }
            }
        }
    }

    let scales_bytes: Vec<u8> = scales
        .iter()
        .flat_map(|&s| s.to_le_bytes().to_vec())
        .collect();
    let biases_bytes: Vec<u8> = vec![0u8; out_dim_u * tile_count * 4]; // F32 zeros (symmetric)

    let stem = weight_name.strip_suffix(".weight").unwrap_or(weight_name);
    let scales_name = format!("{}.scales", stem);
    let biases_name = format!("{}.biases", stem);

    // Packed shape: [out_dim, tile_count * u32_per_tile] in u32 units
    let packed_in = (tile_count * u32_per_tile) as u32;
    let packed_bytes: Vec<u8> = packed
        .iter()
        .flat_map(|&w| w.to_le_bytes().to_vec())
        .collect();

    let packed_shape = crate::config::PackedLinearShapes {
        weight: vec![out_dim, packed_in],
        scales: vec![out_dim, tile_count as u32],
        biases: vec![out_dim, tile_count as u32],
        bits: 2,
        group_size,
        groups: (out_dim_u * tile_count) as u32,
    };

    if let Some(st) = loaded.source_tensors.get_mut(weight_name) {
        st.data = packed_bytes;
        st.dtype = "U32".to_string();
        st.shape = vec![out_dim, packed_in];
    }

    loaded.source_tensors.insert(
        scales_name.clone(),
        SourceTensor {
            name: scales_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, tile_count as u32],
            data: scales_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    loaded.source_tensors.insert(
        biases_name.clone(),
        SourceTensor {
            name: biases_name.clone(),
            dtype: "F32".to_string(),
            shape: vec![out_dim, tile_count as u32],
            data: biases_bytes,
            source_filename: String::new(),
            source_sha256: String::new(),
            source_offset: 0,
        },
    );

    for binding in &mut loaded.spec.global_tensors {
        if binding.name == weight_name && binding.packed_shape.is_none() {
            binding.packed_shape = Some(packed_shape.clone());
        }
    }
    for layer in &mut loaded.spec.layers {
        for binding in &mut layer.tensors {
            if binding.name == weight_name && binding.packed_shape.is_none() {
                binding.packed_shape = Some(packed_shape.clone());
            }
        }
    }

    eprintln!(
        "[quantize] Ternary tile640 quantized {}: [{},{}] -> packed [{},{}] + scales [{},{}]",
        weight_name, out_dim, in_dim, out_dim, packed_in, out_dim, tile_count
    );

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// FP16 → F32 helper
// ═══════════════════════════════════════════════════════════════════════════

/// Fast half-precision (FP16) to F32 conversion.
pub(crate) fn half_to_f32(bits: u16) -> f32 {
    // FP16 format: 1 sign + 5 exponent + 10 mantissa
    let sign = ((bits >> 15) & 0x1) as f32;
    let exp = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x3ff;

    if exp == 0 {
        // Subnormal or zero
        if mantissa == 0 {
            0.0_f32.copysign(1.0 - 2.0 * sign)
        } else {
            f32::from_bits(
                ((sign as u32) << 31) | ((102u32 - 14 + 127) << 23) | ((mantissa as u32) << 13),
            ) * (1.0 / 16777216.0) // 2^-24
        }
    } else if exp == 31 {
        // Infinity or NaN
        let exp_f32: u32 = 255;
        let mantissa_f32: u32 = if mantissa == 0 {
            0
        } else {
            (mantissa as u32) << 13
        };
        f32::from_bits(((sign as u32) << 31) | (exp_f32 << 23) | mantissa_f32)
    } else {
        // Normal: FP16 exponent bias = 15, F32 exponent bias = 127
        let exp_f32: u32 = ((exp as u32) + 127 - 15) << 23;
        f32::from_bits(((sign as u32) << 31) | exp_f32 | ((mantissa as u32) << 13))
    }
}

// GPU-accelerated TernaryTile640 pack (behind metal-dispatch feature gate).
// Compiled as part of this module — the include! merges the source inline
// so the `use` items and `pub(crate)` visibility work naturally.
#[cfg(feature = "metal-dispatch")]
include!("gpu_pack.rs");
