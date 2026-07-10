use crate::profiled_model::LoadedProfiledModel;
use crate::ecs::vision::preprocess::load_resized_rgb_image;
use mlx_rs::Array;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

pub fn project_image_with_loaded_model(
    model: &LoadedProfiledModel,
    source: &str,
) -> Result<Array, String> {
    let bindings = model
        .multimodal_bindings()
        .ok_or_else(|| "missing sealed multimodal bindings".to_string())?;
    if !bindings.ready_for_direct_image_projection() {
        return Err("packed cimage does not expose a direct image projector bundle".to_string());
    }

    let vision_config = model
        .reader
        .manifest
        .vision_config
        .as_ref()
        .ok_or_else(|| "image input requires vision_config in model manifest".to_string())?;
    let channels = usize::from(bindings.descriptor.image_channels.max(1));
    let patch_record = bindings
        .image_patch_embedding()
        .ok_or_else(|| "missing image patch embedding record".to_string())?;
    let final_record = bindings
        .image_projection()
        .ok_or_else(|| "missing image projection record".to_string())?;

    let patch_input_width = patch_record.record.input_width as usize;
    if patch_input_width == 0 || patch_input_width % channels != 0 {
        return Err(format!(
            "invalid patch input width {} for {} channels",
            patch_input_width, channels
        ));
    }
    let patch_area = patch_input_width / channels;
    let patch_edge = (patch_area as f64).sqrt() as usize;
    if patch_edge * patch_edge * channels != patch_input_width {
        return Err(format!(
            "patch input width {} does not map to a square patch",
            patch_input_width
        ));
    }

    let image_size = vision_config.image_size as usize;
    let rgb = load_resized_rgb_image(source, image_size, image_size, channels)?;
    let patches_w = image_size / patch_edge;
    let patches_h = image_size / patch_edge;
    let num_patches = patches_w * patches_h;
    let patch_pixels = patchify_rgb(&rgb, image_size, patch_edge, channels);

    let patch_dense_name = model
        .find_tensor_name_with_suffixes(&[
            ".vision_embedder.patch_dense.weight",
            "vision_embedder.patch_dense.weight",
        ])
        .ok_or_else(|| "missing vision patch_dense.weight tensor".to_string())?;
    let patch_dense_bias_name = model.find_tensor_name_with_suffixes(&[
        ".vision_embedder.patch_dense.bias",
        "vision_embedder.patch_dense.bias",
    ]);
    let patch_ln2_weight_name = model
        .find_tensor_name_with_suffixes(&[
            ".vision_embedder.patch_ln2.weight",
            "vision_embedder.patch_ln2.weight",
        ])
        .ok_or_else(|| "missing vision patch_ln2.weight tensor".to_string())?;
    let patch_ln2_bias_name = model
        .find_tensor_name_with_suffixes(&[
            ".vision_embedder.patch_ln2.bias",
            "vision_embedder.patch_ln2.bias",
        ])
        .ok_or_else(|| "missing vision patch_ln2.bias tensor".to_string())?;
    let pos_embedding_name = model
        .find_tensor_name_with_suffixes(&[
            ".vision_embedder.pos_embedding",
            "vision_embedder.pos_embedding",
        ])
        .ok_or_else(|| "missing vision pos_embedding tensor".to_string())?;
    let pos_norm_weight_name = model
        .find_tensor_name_with_suffixes(&[
            ".vision_embedder.pos_norm.weight",
            "vision_embedder.pos_norm.weight",
        ])
        .ok_or_else(|| "missing vision pos_norm.weight tensor".to_string())?;
    let pos_norm_bias_name = model
        .find_tensor_name_with_suffixes(&[
            ".vision_embedder.pos_norm.bias",
            "vision_embedder.pos_norm.bias",
        ])
        .ok_or_else(|| "missing vision pos_norm.bias tensor".to_string())?;
    let final_proj_name = model
        .find_tensor_name_with_suffixes(&[
            ".embed_vision.embedding_projection.weight",
            "embed_vision.embedding_projection.weight",
        ])
        .ok_or_else(|| "missing vision embedding_projection.weight tensor".to_string())?;

    let (patch_dense_weight, patch_dense_shape) = model
        .load_tensor_f32_by_name(&patch_dense_name)
        .map_err(|e| format!("load {}: {}", patch_dense_name, e))?;
    let patch_dense_bias = patch_dense_bias_name
        .as_deref()
        .map(|name| {
            model
                .load_tensor_f32_by_name(name)
                .map(|(values, _)| values)
        })
        .transpose()
        .map_err(|e| format!("load patch_dense.bias: {}", e))?
        .unwrap_or_default();
    let (patch_ln2_weight, _) = model
        .load_tensor_f32_by_name(&patch_ln2_weight_name)
        .map_err(|e| format!("load {}: {}", patch_ln2_weight_name, e))?;
    let (patch_ln2_bias, _) = model
        .load_tensor_f32_by_name(&patch_ln2_bias_name)
        .map_err(|e| format!("load {}: {}", patch_ln2_bias_name, e))?;
    let (pos_embedding, pos_shape) = model
        .load_tensor_f32_by_name(&pos_embedding_name)
        .map_err(|e| format!("load {}: {}", pos_embedding_name, e))?;
    let (pos_norm_weight, _) = model
        .load_tensor_f32_by_name(&pos_norm_weight_name)
        .map_err(|e| format!("load {}: {}", pos_norm_weight_name, e))?;
    let (pos_norm_bias, _) = model
        .load_tensor_f32_by_name(&pos_norm_bias_name)
        .map_err(|e| format!("load {}: {}", pos_norm_bias_name, e))?;
    let (final_proj_weight, final_shape) = model
        .load_tensor_f32_by_name(&final_proj_name)
        .map_err(|e| format!("load {}: {}", final_proj_name, e))?;

    let patch_out = patch_record.record.output_width as usize;
    let final_out = final_record.record.output_width as usize;
    if patch_dense_shape.len() < 2
        || patch_dense_shape[0] != patch_out
        || patch_dense_shape[1] != patch_input_width
    {
        return Err(format!(
            "unexpected patch_dense shape {:?}, expected [{}, {}]",
            patch_dense_shape, patch_out, patch_input_width
        ));
    }
    if final_shape.len() < 2 || final_shape[0] != final_out || final_shape[1] != patch_out {
        return Err(format!(
            "unexpected embedding_projection shape {:?}, expected [{}, {}]",
            final_shape, final_out, patch_out
        ));
    }

    let mut projected = sgemm_row_major_a_bt(
        &patch_pixels,
        num_patches,
        patch_input_width,
        &patch_dense_weight,
        patch_out,
    )?;
    if !patch_dense_bias.is_empty() {
        add_row_bias(&mut projected, num_patches, patch_out, &patch_dense_bias)?;
    }
    apply_rms_norm_affine(
        &mut projected,
        num_patches,
        patch_out,
        &patch_ln2_weight,
        &patch_ln2_bias,
    )?;
    add_position_embeddings(
        &mut projected,
        patches_w,
        patches_h,
        patch_out,
        &pos_embedding,
        &pos_shape,
    )?;
    apply_affine(
        &mut projected,
        num_patches,
        patch_out,
        &pos_norm_weight,
        &pos_norm_bias,
    )?;

    let pooling_kernel = usize::from(bindings.descriptor.image_pooling_kernel.max(1));
    let pooled = pool_soft_tokens(&projected, patches_w, patches_h, patch_out, pooling_kernel);
    let soft_tokens = pooled.len() / patch_out;
    let decoder_embeds = sgemm_row_major_a_bt(
        &pooled,
        soft_tokens,
        patch_out,
        &final_proj_weight,
        final_out,
    )?;

    Ok(Array::from_slice(
        &decoder_embeds,
        &[soft_tokens as i32, final_out as i32],
    ))
}

fn patchify_rgb(rgb: &[u8], image_size: usize, patch_edge: usize, channels: usize) -> Vec<f32> {
    let patches_w = image_size / patch_edge;
    let patches_h = image_size / patch_edge;
    let mut out = Vec::with_capacity(patches_w * patches_h * patch_edge * patch_edge * channels);
    for py in 0..patches_h {
        for px in 0..patches_w {
            let y0 = py * patch_edge;
            let x0 = px * patch_edge;
            for dy in 0..patch_edge {
                for dx in 0..patch_edge {
                    let idx = ((y0 + dy) * image_size + (x0 + dx)) * channels;
                    for c in 0..channels {
                        out.push(rgb[idx + c] as f32 / 255.0);
                    }
                }
            }
        }
    }
    out
}

fn add_row_bias(matrix: &mut [f32], rows: usize, cols: usize, bias: &[f32]) -> Result<(), String> {
    if bias.len() != cols {
        return Err(format!("bias length {} != {}", bias.len(), cols));
    }
    for row in 0..rows {
        let start = row * cols;
        for col in 0..cols {
            matrix[start + col] += bias[col];
        }
    }
    Ok(())
}

fn apply_rms_norm_affine(
    matrix: &mut [f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<(), String> {
    if weight.len() != cols || bias.len() != cols {
        return Err("rms norm parameter length mismatch".to_string());
    }
    for row in 0..rows {
        let start = row * cols;
        let slice = &mut matrix[start..start + cols];
        let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_sq + 1e-6).sqrt();
        for col in 0..cols {
            slice[col] = slice[col] * inv_rms * weight[col] + bias[col];
        }
    }
    Ok(())
}

fn apply_affine(
    matrix: &mut [f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<(), String> {
    if weight.len() != cols || bias.len() != cols {
        return Err("affine parameter length mismatch".to_string());
    }
    for row in 0..rows {
        let start = row * cols;
        for col in 0..cols {
            matrix[start + col] = matrix[start + col] * weight[col] + bias[col];
        }
    }
    Ok(())
}

fn add_position_embeddings(
    matrix: &mut [f32],
    patches_w: usize,
    patches_h: usize,
    hidden: usize,
    pos_embedding: &[f32],
    pos_shape: &[usize],
) -> Result<(), String> {
    let (max_h, max_w, components, embed_hidden) = match pos_shape {
        [h, w, c, d] => (*h, *w, *c, *d),
        [hw, c, d] => {
            let side = (*hw as f64).sqrt() as usize;
            (side, side, *c, *d)
        }
        other => {
            return Err(format!("unsupported pos_embedding shape {:?}", other));
        }
    };
    if components < 2 || embed_hidden != hidden {
        return Err(format!(
            "unsupported pos_embedding contract [{}, {}, {}, {}] for hidden {}",
            max_h, max_w, components, embed_hidden, hidden
        ));
    }
    for py in 0..patches_h {
        for px in 0..patches_w {
            let fx = if patches_w > 1 {
                px as f32 / (patches_w - 1) as f32
            } else {
                0.0
            };
            let fy = if patches_h > 1 {
                py as f32 / (patches_h - 1) as f32
            } else {
                0.0
            };
            let pos_x = (fx * (max_w.saturating_sub(1)) as f32).round() as usize;
            let pos_y = (fy * (max_h.saturating_sub(1)) as f32).round() as usize;
            let base = ((pos_y * max_w + pos_x) * components) * hidden;
            let row = (py * patches_w + px) * hidden;
            for col in 0..hidden {
                matrix[row + col] += pos_embedding[base + col] + pos_embedding[base + hidden + col];
            }
        }
    }
    Ok(())
}

fn pool_soft_tokens(
    matrix: &[f32],
    patches_w: usize,
    patches_h: usize,
    hidden: usize,
    kernel: usize,
) -> Vec<f32> {
    let soft_w = patches_w / kernel;
    let soft_h = patches_h / kernel;
    let mut out = vec![0.0f32; soft_w * soft_h * hidden];
    let denom = (kernel * kernel) as f32;
    for sy in 0..soft_h {
        for sx in 0..soft_w {
            let out_row = (sy * soft_w + sx) * hidden;
            for ky in 0..kernel {
                for kx in 0..kernel {
                    let py = sy * kernel + ky;
                    let px = sx * kernel + kx;
                    let in_row = (py * patches_w + px) * hidden;
                    for col in 0..hidden {
                        out[out_row + col] += matrix[in_row + col];
                    }
                }
            }
            for col in 0..hidden {
                out[out_row + col] /= denom;
            }
        }
    }
    out
}

fn sgemm_row_major_a_bt(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
) -> Result<Vec<f32>, String> {
    if a.len() != m * k {
        return Err(format!("left matrix length {} != {}x{}", a.len(), m, k));
    }
    if b.len() != n * k {
        return Err(format!("right matrix length {} != {}x{}", b.len(), n, k));
    }
    let mut out = vec![0.0f32; m * n];
    #[cfg(target_os = "macos")]
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            k as i32,
            1.0,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            k as i32,
            0.0,
            out.as_mut_ptr(),
            n as i32,
        );
    }
    #[cfg(not(target_os = "macos"))]
    {
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0.0f32;
                let a_row = row * k;
                let b_row = col * k;
                for idx in 0..k {
                    acc += a[a_row + idx] * b[b_row + idx];
                }
                out[row * n + col] = acc;
            }
        }
    }
    Ok(out)
}
