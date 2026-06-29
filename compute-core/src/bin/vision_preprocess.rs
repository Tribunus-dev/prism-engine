//! vision_preprocess — Image embedding preprocessor for Gemma 4 12B Unified.
//!
//! Loads an image, patchifies it, projects patches through the vision embedder's
//! `patch_dense.weight`, and writes FP16 hidden vectors for the Metal pipeline.
//!
//! Usage:
//!   cargo run --bin vision-preprocess --features prism-backend -- \
//!     --image /path/to/photo.jpg \
//!     --model /path/to/model.safetensors \
//!     --output /path/to/embeddings.bin
//!
//! Algorithm:
//!   1. Load image, resize to 896×896.
//!   2. Slice into 48×48 non-overlapping patches (48×48×3 = 6912 values each).
//!   3. Load `model.vision_embedder.patch_dense.weight` [3840, 6912] from
//!      the safetensors file (BF16 precision).
//!   4. Project each flattened patch: output = W · patch → [3840].
//!   5. Write all patch vectors as contiguous FP16 bytes.

use std::path::PathBuf;

use clap::Parser;
use half::f16;

// ── Architecture constants ──────────────────────────────────────────

/// Expected input image resolution (square).
const IMAGE_SIZE: usize = 896;

/// Patch dimensions for the vision embedder.
///
/// `patch_dense.weight` has shape [3840, 6912] where 6912 = 48 × 48 × 3,
/// so each patch is 48×48 RGB pixels.
const PATCH_SIZE: usize = 48;

/// Stride between patches (non-overlapping).
const STRIDE: usize = 48;

/// Hidden dimension of Gemma 4.
const HIDDEN_DIM: usize = 3840;

/// Flattened size of one patch: 48 × 48 × 3 (RGB).
const PATCH_FLAT: usize = PATCH_SIZE * PATCH_SIZE * 3;

// ── CLI ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "vision-preprocess", about = "Image embedding preprocessor")]
struct Args {
    /// Path to input image (JPEG, PNG, etc.)
    #[arg(long)]
    image: PathBuf,

    /// Path to model.safetensors (Gemma 4 checkpoint)
    #[arg(long)]
    model: PathBuf,

    /// Path for output .bin file of FP16 embeddings
    #[arg(long)]
    output: PathBuf,
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Convert tensor bytes to Vec<f32>, handling F32 and BF16 dtypes.
fn tensor_to_f32(data: &[u8], dtype: safetensors::Dtype) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| {
                let u = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((u as u32) << 16)
            })
            .collect(),
        other => panic!("unsupported tensor dtype: {other:?} (expected F32 or BF16)"),
    }
}

// ── Main ────────────────────────────────────────────────────────────

fn main() -> Result<(), String> {
    let args = Args::parse();

    // ── 1. Load and resize image ─────────────────────────────────────
    let img = image::open(&args.image).map_err(|e| format!("open image {:?}: {e}", args.image))?;
    let img = img.resize_exact(
        IMAGE_SIZE as u32,
        IMAGE_SIZE as u32,
        image::imageops::FilterType::Lanczos3,
    );
    let rgb = img.to_rgb8();
    let pixels = rgb.as_raw();
    assert_eq!(pixels.len(), IMAGE_SIZE * IMAGE_SIZE * 3);
    println!("[1/5] Loaded image, resized to {IMAGE_SIZE}×{IMAGE_SIZE}");

    // ── 2. Load safetensors ──────────────────────────────────────────
    let raw =
        std::fs::read(&args.model).map_err(|e| format!("read model {:?}: {e}", args.model))?;
    let tensors = safetensors::SafeTensors::deserialize(&raw)
        .map_err(|e| format!("parse safetensors {:?}: {e}", args.model))?;
    println!(
        "[2/5] Loaded safetensors ({} tensors)",
        tensors.names().len()
    );

    // ── 3. Inspect vision-embedder tensors ───────────────────────────
    let vision_tensors = [
        "model.vision_embedder.patch_dense.weight",
        "model.vision_embedder.pos_embedding",
        "model.vision_embedder.patch_ln1.weight",
        "model.vision_embedder.patch_ln1.bias",
        "model.embed_vision.embedding_projection.weight",
    ];

    for name in &vision_tensors {
        if let Ok(view) = tensors.tensor(name) {
            println!(
                "  {name}: shape={:?}, dtype={:?}",
                view.shape(),
                view.dtype()
            );
        } else {
            println!("  {name}: (not found)");
        }
    }

    // ── 4. Patchify image ────────────────────────────────────────────
    // Extract non-overlapping 48×48 patches from 896×896 image.
    //   patches_per_dim = (896 - 48) / 48 + 1 = 18
    //   total_patches   = 18 × 18 = 324
    let patches_per_dim = (IMAGE_SIZE - PATCH_SIZE) / STRIDE + 1;
    let num_patches = patches_per_dim * patches_per_dim;

    // Precompute patch start indices for the inner loop.
    let mut patch_starts: Vec<(usize, usize)> = Vec::with_capacity(num_patches);
    for py in 0..patches_per_dim {
        for px in 0..patches_per_dim {
            patch_starts.push((py * STRIDE, px * STRIDE));
        }
    }

    println!(
        "[3/5] Extracting {num_patches} patches ({patches_per_dim}×{patches_per_dim}) of {PATCH_SIZE}×{PATCH_SIZE}"
    );

    // Allocate a single reusable buffer for the current flattened patch.
    let mut patch_buf = vec![0.0f32; PATCH_FLAT];

    // ── 5. Load projection weight and project ────────────────────────
    // Each projected patch yields a [3840] vector.
    let mut hidden = Vec::with_capacity(num_patches * HIDDEN_DIM);

    if let Ok(view) = tensors.tensor("model.vision_embedder.patch_dense.weight") {
        let shape = view.shape().to_vec();
        assert_eq!(
            shape,
            &[HIDDEN_DIM, PATCH_FLAT],
            "patch_dense.weight expected [{HIDDEN_DIM}, {PATCH_FLAT}], got {shape:?}"
        );

        let w_f32 = tensor_to_f32(view.data(), view.dtype());
        println!("[4/5] Loaded patch_dense.weight [{HIDDEN_DIM}, {PATCH_FLAT}]");

        // For each patch, flatten to [PATCH_FLAT], then project via W^T.
        for &(y0, x0) in &patch_starts {
            // Flatten patch into patch_buf.
            let mut idx = 0;
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    let pixel_idx = ((y0 + dy) * IMAGE_SIZE + (x0 + dx)) * 3;
                    // Normalize to [0, 1].
                    patch_buf[idx] = pixels[pixel_idx] as f32 / 255.0;
                    patch_buf[idx + 1] = pixels[pixel_idx + 1] as f32 / 255.0;
                    patch_buf[idx + 2] = pixels[pixel_idx + 2] as f32 / 255.0;
                    idx += 3;
                }
            }

            // Project: hidden[j] = sum_k W[j, k] * patch[k].
            for row in 0..HIDDEN_DIM {
                let start = row * PATCH_FLAT;
                let w_row = &w_f32[start..start + PATCH_FLAT];
                let dot = w_row
                    .iter()
                    .zip(patch_buf.iter())
                    .map(|(w, p)| w * p)
                    .sum::<f32>();
                hidden.push(dot);
            }
        }
    } else {
        // ── Fallback: synthetic embeddings ───────────────────────────
        eprintln!("WARNING: patch_dense.weight not found; generating zero embeddings");
        eprintln!("  Found tensors (sample):");
        for name in tensors.names().iter().take(20) {
            eprintln!("    {name}");
        }
        hidden.resize(num_patches * HIDDEN_DIM, 0.0f32);
    }

    println!("[5/5] Projected {num_patches} patches → {HIDDEN_DIM}-d vectors");

    // ── 6. Write output as FP16 bytes ────────────────────────────────
    let out_bytes: Vec<u8> = hidden
        .iter()
        .flat_map(|&v| f16::from_f32(v).to_le_bytes())
        .collect();

    std::fs::write(&args.output, &out_bytes)
        .map_err(|e| format!("write output {:?}: {e}", args.output))?;

    let total_vectors = hidden.len() / HIDDEN_DIM;
    println!(
        "  Wrote {total_vectors} × {HIDDEN_DIM} = {} FP16 vectors ({:.1} MB) to {}",
        hidden.len(),
        out_bytes.len() as f64 / 1_048_576.0,
        args.output.display(),
    );

    Ok(())
}
