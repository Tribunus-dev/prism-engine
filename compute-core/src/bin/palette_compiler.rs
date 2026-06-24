//! palette_compiler — offline vDSP palette compilation for the .cimage format.
//!
//! Iterates all weight tensors in a model's .safetensors, runs k-means
//! to fit per-row 16-entry codebooks, packs indices to 4-bit nibbles,
//! and writes a 16 KB page-aligned .cimage file ready for zero-copy mmap.
//!
//! Usage:
//!   RUST_LOG=info cargo run --release --bin palette_compiler -- \
//!       --model-path ~/.tribunus/models/qwen2.5-0.5b \
//!       --output-dir ~/.tribunus/compiled/

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use half::f16;
use tribunus_compute_core::quantization::cimage::CImageWriter;
use tribunus_compute_core::quantization::palette::palettize_matrix;

#[derive(Parser)]
#[command(name = "palette_compiler", about = "Offline k-means palette compiler for .cimage")]
struct Args {
    /// Path to the HuggingFace model directory containing .safetensors.
    #[arg(long)]
    model_path: PathBuf,

    /// Output directory for the compiled .cimage file.
    #[arg(long)]
    output_dir: PathBuf,

    /// Target bits per parameter (currently ignored; always uses 16-entry LUT).
    #[arg(long, default_value = "4")]
    _target_bpp: u8,
}

fn main() -> Result<(), String> {
    let args = Args::parse();

    // Discover all .safetensors files
    let mut shards: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&args.model_path).map_err(|e| format!("read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("entry: {e}"))?;
        let path = entry.path();
        if path.extension().map_or(false, |ext| ext == "safetensors") {
            shards.push(path);
        }
    }
    if shards.is_empty() {
        return Err("No .safetensors files found".to_string());
    }
    shards.sort();

    let cimage_path = args.output_dir.join("model.cimage");
    std::fs::create_dir_all(&args.output_dir).map_err(|e| format!("create output dir: {e}"))?;
    let mut cimage = CImageWriter::new(&cimage_path)?;

    for shard_path in &shards {
        eprintln!("[compiler] Processing shard: {}", shard_path.display());
        let data = std::fs::read(shard_path).map_err(|e| format!("read shard: {e}"))?;
        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| format!("parse safetensors: {e}"))?;

        let names: Vec<&String> = tensors.names();

        for name in names {
            let view = tensors.tensor(name).map_err(|e| format!("tensor {name}: {e}"))?;
            let shape: Vec<usize> = view.shape().to_vec();

            // Skip rank-0 and rank-1 tensors (biases, norms, etc.)
            if shape.len() < 2 {
                eprintln!("  [skip] {name} (rank < 2)");
                continue;
            }

            let out_dim = shape[0];
            let in_dim = shape[1];
            let total = out_dim * in_dim;

            // Skip small tensors (< 4 KB)
            if total < 1024 {
                eprintln!("  [skip] {name} (too small)");
                continue;
            }

            let t0 = Instant::now();

            // Convert safetensors data to f32
            let f32_vals = match view.dtype() {
                safetensors::Dtype::F32 => {
                    view.data().chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect::<Vec<_>>()
                }
                safetensors::Dtype::BF16 => {
                    view.data().chunks_exact(2)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect::<Vec<_>>()
                }
                _ => {
                    eprintln!("  [skip] {name} (unsupported dtype {:?})", view.dtype());
                    continue;
                }
            };

            // Palette-compress (per-row 16-entry LUT)
            let pal = palettize_matrix(&f32_vals, out_dim, in_dim, 16, 50);
            let bpp = pal.effective_bpp();

            // Build split-block payload: [all codebooks (f16) | all indices (u8 packed)]
            // Each row: codebook = [f32; 16] → f16 via half crate
            let cb_bytes = pal.rows.len() * 16 * 2;
            let idx_bytes: usize = pal.rows.iter().map(|r| r.indices.len()).sum();
            let mut payload = Vec::with_capacity(cb_bytes + idx_bytes);
            for row in &pal.rows {
                for &cb_f32 in &row.codebook {
                    let cb_f16 = f16::from_f32(cb_f32);
                    payload.extend_from_slice(&cb_f16.to_bits().to_le_bytes());
                }
            }
            for row in &pal.rows {
                payload.extend_from_slice(&row.indices);
            }

            cimage.append_palettized(name, &payload, out_dim as u32, in_dim as u32)?;

            let elapsed = t0.elapsed();
            eprintln!(
                "  [ok] {name} ({out_dim}×{in_dim}) bpp={bpp:.3} {:.2}s",
                elapsed.as_secs_f64(),
            );
        }
    }

    cimage.finalize()?;
    eprintln!("\n[compiler] Done — written to {}", cimage_path.display());
    Ok(())
}
