//! Qwen2.5-Omni cimage ingest — packs Thinker LM, ViT vision encoder,
//! audio encoder, and projector weights into a Prism cimage.
//!
//! Usage:
//!   cargo run --bin qwen25-omni-ingest --features prism-backend -- \
//!     --model-dir ./qwen25-omni-7b --output ./qwen25-omni-7b.cimage

use std::path::Path;
use std::time::Instant;

use tribunus_compute_core::compute_image::legacy_compute_image_compile::ternary::{
    CimageHeader, SegmentEntry, SegmentKind, CIMAGE_SEGMENT_CAPACITY,
};

// ── Qwen2.5-Omni 7B architecture constants ────────────────────────
const NUM_LAYERS: usize = 28;
const HIDDEN_DIM: usize = 3584;
const NUM_HEADS: usize = 28;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;
const FFN_INTERMEDIATE: usize = 18944;
const VOCAB_SIZE: usize = 151936;

// Vision encoder
const VISION_HIDDEN: usize = 1152;
#[allow(dead_code)]
const VISION_LAYERS: usize = 27;

// Decoder weight matrix list (Thinker LM)
const DECODER_MATRICES: &[(&str, usize, usize)] = &[
    (
        "model.language_model.layers.{}.self_attn.q_proj.weight",
        HIDDEN_DIM,
        NUM_HEADS * HEAD_DIM,
    ),
    (
        "model.language_model.layers.{}.self_attn.k_proj.weight",
        HIDDEN_DIM,
        NUM_KV_HEADS * HEAD_DIM,
    ),
    (
        "model.language_model.layers.{}.self_attn.v_proj.weight",
        HIDDEN_DIM,
        NUM_KV_HEADS * HEAD_DIM,
    ),
    (
        "model.language_model.layers.{}.self_attn.o_proj.weight",
        NUM_HEADS * HEAD_DIM,
        HIDDEN_DIM,
    ),
    (
        "model.language_model.layers.{}.mlp.gate_proj.weight",
        HIDDEN_DIM,
        FFN_INTERMEDIATE,
    ),
    (
        "model.language_model.layers.{}.mlp.up_proj.weight",
        HIDDEN_DIM,
        FFN_INTERMEDIATE,
    ),
    (
        "model.language_model.layers.{}.mlp.down_proj.weight",
        FFN_INTERMEDIATE,
        HIDDEN_DIM,
    ),
];

// Vision encoder weight matrices (ViT)
const VISION_MATRICES: &[(&str, usize, usize)] = &[
    (
        "model.visual.vision_model.embeddings.patch_embedding.weight",
        VISION_HIDDEN,
        588,
    ),
    (
        "model.visual.vision_model.embeddings.position_embedding.weight",
        1025,
        VISION_HIDDEN,
    ),
];

// Vision projector
const VISION_PROJECTOR: (&str, usize, usize) = (
    "model.visual.merger.linear.weight",
    HIDDEN_DIM,
    VISION_HIDDEN * 4,
);

// Text embeddings
#[allow(dead_code)]
const EMBED_TOKENS: (&str, usize, usize) = (
    "model.language_model.embed_tokens.weight",
    VOCAB_SIZE,
    HIDDEN_DIM,
);

#[allow(dead_code)]
const FINAL_NORM: (&str, usize, usize) = ("model.language_model.norm.weight", HIDDEN_DIM, 1);

fn get_opt(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = get_opt(&args, "--model-dir").unwrap_or_else(|| {
        eprintln!("Usage: qwen25-omni-ingest --model-dir <PATH> [--output <PATH>]");
        std::process::exit(1);
    });
    let output = get_opt(&args, "--output")
        .unwrap_or_else(|| format!("{}/qwen25-omni-7b.cimage", model_dir));

    let dir = Path::new(&model_dir);
    if !dir.is_dir() {
        eprintln!("model-dir not found: {}", model_dir);
        std::process::exit(1);
    }

    println!("Qwen2.5-Omni cimage ingest");
    println!("  Model dir:  {}", model_dir);
    println!("  Output:     {}", output);
    println!("  Architecture: Thinker LM (28 layers, 3584 hidden) + ViT encoder (27 layers, 1152 hidden)");

    let total_start = Instant::now();

    // ── Step 1: Read safetensors ─────────────────────────────────
    let st_path = dir.join("model.safetensors");
    if !st_path.exists() {
        eprintln!(
            "ERROR: model.safetensors not found at {}",
            st_path.display()
        );
        eprintln!("Please download the model first: huggingface-cli download Qwen/Qwen2.5-Omni-7B --local-dir {}", model_dir);
        std::process::exit(1);
    }

    let st_bytes = std::fs::read(&st_path).unwrap_or_else(|e| {
        eprintln!("ERROR: cannot read safetensors: {e}");
        std::process::exit(1);
    });

    // Parse safetensors header
    let header_len = u64::from_le_bytes(st_bytes[0..8].try_into().unwrap()) as usize;
    let header_json: serde_json::Value =
        serde_json::from_slice(&st_bytes[8..8 + header_len]).unwrap_or_default();

    println!(
        "  Safetensors: {} tensors",
        header_json.as_object().map(|o| o.len()).unwrap_or(0)
    );

    // ── Step 2: Quantize decoder weights ──────────────────────────
    println!("\n  Step 2: Ternary quantization of Thinker LM weights...");

    use tribunus_compute_core::compute_image::legacy_compute_image_compile::ternary::ternary_quantize_block;

    let mut all_weights = Vec::new();
    let mut all_scales = Vec::new();
    let _total_elements: u64 = 0;

    for layer in 0..NUM_LAYERS {
        for (template, out_dim, in_dim) in DECODER_MATRICES.iter() {
            let name = template.replace("{}", &layer.to_string());
            let tensor_info = header_json.get(&name);

            if let Some(info) = tensor_info {
                let dtype = info.get("dtype").and_then(|v| v.as_str()).unwrap_or("F32");
                let _shape = info
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64()).collect::<Vec<_>>())
                    .unwrap_or_default();
                let start: u64 = info
                    .get("data_offsets")
                    .and_then(|v| v[0].as_u64())
                    .unwrap_or(0);
                let end: u64 = info
                    .get("data_offsets")
                    .and_then(|v| v[1].as_u64())
                    .unwrap_or(0);
                let _len = (end - start) as usize;

                let rows = *out_dim as usize;
                let cols = *in_dim as usize;

                // Extract and quantize
                let raw = &st_bytes[8 + header_len + start as usize..8 + header_len + end as usize];

                if dtype == "BF16" {
                    let nt = ((cols + 639) / 640) as usize;
                    let packed_len = rows * nt * 32 * 4;
                    let scales_len = rows * nt * 2;

                    let mut weights_buf = vec![0u8; packed_len];
                    let mut scales_buf = vec![0u8; scales_len];

                    for r in 0..rows {
                        for t in 0..nt {
                            let c_start = t * 640;
                            let c_end = (c_start + 640).min(cols);
                            let n = c_end - c_start;
                            // Quantize in 256-element sub-blocks within each 640-tile
                            for sb in 0..((n + 255) / 256) {
                                let sb_start = sb * 256;
                                let sb_end = (sb_start + 256).min(n);
                                let sb_n = sb_end - sb_start;
                                let mut blk = [0.0f32; 256];
                                for j in 0..sb_n {
                                    let bo = (r * cols + c_start + sb_start + j) * 2;
                                    let bf16 = u16::from_le_bytes([raw[bo], raw[bo + 1]]);
                                    blk[j] = tribunus_compute_core::compute_image::legacy_compute_image_compile::ternary::fp16_to_f32(bf16.to_le_bytes());
                                }
                                // Zero-fill trailing elements in the block
                                for j in sb_n..256 {
                                    blk[j] = 0.0;
                                }
                                let (sc, nib) = ternary_quantize_block(&blk);
                                let sc_offset = (r * nt + t) * 2 + sb * 2;
                                if sc_offset + 2 <= scales_buf.len() {
                                    scales_buf[sc_offset..sc_offset + 2].copy_from_slice(&sc);
                                }
                                for j in 0..sb_n {
                                    let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 {
                                        0b01 => 1u32,
                                        0b10 => 2u32,
                                        _ => 0u32,
                                    };
                                    let po = r * nt * 32 * 4
                                        + t * 32 * 4
                                        + (sb_start / 20 + (j / 20)) * 4;
                                    if po + 4 > weights_buf.len() {
                                        continue;
                                    }
                                    let mut pk = u32::from_le_bytes([
                                        weights_buf[po],
                                        weights_buf[po + 1],
                                        weights_buf[po + 2],
                                        weights_buf[po + 3],
                                    ]);
                                    let sub = j % 20;
                                    let mut mul = 1u32;
                                    for _ in 0..sub {
                                        mul *= 3;
                                    }
                                    pk = (pk / (mul * 3)) * (mul * 3) + d * mul + pk % mul;
                                    weights_buf[po..po + 4].copy_from_slice(&pk.to_le_bytes());
                                }
                            }
                        }
                    }
                    all_weights.extend_from_slice(&weights_buf);
                    all_scales.extend_from_slice(&scales_buf);
                }
            }
        }
    }

    println!("    Decoder weights: {} bytes packed", all_weights.len());
    println!("    Decoder scales:  {} bytes", all_scales.len());

    // ── Step 3: Pack vision encoder + projector (FP16 for now) ────
    println!("\n  Step 3: Packing vision encoder (ViT) + projector...");
    let mut vision_weights = Vec::new();
    let _vision_scales: Vec<u8> = Vec::new();

    for (name, _, _) in VISION_MATRICES.iter() {
        if let Some(info) = header_json.get(*name) {
            let start: u64 = info
                .get("data_offsets")
                .and_then(|v| v[0].as_u64())
                .unwrap_or(0);
            let end: u64 = info
                .get("data_offsets")
                .and_then(|v| v[1].as_u64())
                .unwrap_or(0);
            let _len = (end - start) as usize;
            let raw = &st_bytes[8 + header_len + start as usize..8 + header_len + end as usize];
            vision_weights.extend_from_slice(raw);
        }
    }

    // Pack vision projector
    let (proj_name, _, _) = VISION_PROJECTOR;
    if let Some(info) = header_json.get(proj_name) {
        let start: u64 = info
            .get("data_offsets")
            .and_then(|v| v[0].as_u64())
            .unwrap_or(0);
        let end: u64 = info
            .get("data_offsets")
            .and_then(|v| v[1].as_u64())
            .unwrap_or(0);
        let _len = (end - start) as usize;
        let raw = &st_bytes[8 + header_len + start as usize..8 + header_len + end as usize];
        vision_weights.extend_from_slice(raw);
    }

    println!("    Vision weights: {} bytes (FP16)", vision_weights.len());

    // ── Step 4: Write cimage ──────────────────────────────────────
    println!("\n  Step 4: Writing cimage...");

    use sha2::{Digest, Sha256};
    use std::io::{BufWriter, Seek, SeekFrom, Write};

    let file = std::fs::File::create(&output).unwrap();
    let mut writer = BufWriter::new(file);

    let header_size = std::mem::size_of::<CimageHeader>() as u64;
    writer.write_all(&vec![0u8; header_size as usize]).unwrap();

    let page_align = |w: &mut BufWriter<std::fs::File>| -> std::io::Result<u64> {
        let pos = w.stream_position()?;
        let aligned = ((pos + 16383) / 16384) * 16384;
        if aligned > pos {
            w.write_all(&vec![0u8; (aligned - pos) as usize])?;
        }
        Ok(aligned)
    };

    let weights_off = page_align(&mut writer).unwrap();
    writer.write_all(&all_weights).unwrap();

    let scales_off = page_align(&mut writer).unwrap();
    writer.write_all(&all_scales).unwrap();

    let _vision_off = if !vision_weights.is_empty() {
        let off = page_align(&mut writer).unwrap();
        writer.write_all(&vision_weights).unwrap();
        Some(off)
    } else {
        None
    };

    let mut segments = [SegmentEntry {
        kind: SegmentKind::MetalLib as u32,
        offset: 0,
        length: 0,
    }; CIMAGE_SEGMENT_CAPACITY];
    segments[0] = SegmentEntry {
        kind: SegmentKind::TernaryWeights as u32,
        offset: weights_off,
        length: all_weights.len() as u64,
    };
    segments[1] = SegmentEntry {
        kind: SegmentKind::BlockScales as u32,
        offset: scales_off,
        length: all_scales.len() as u64,
    };

    let mut hasher = Sha256::new();
    hasher.update(&all_weights);
    hasher.update(&all_scales);
    let payload_hash: [u8; 32] = hasher.finalize().into();

    let header = CimageHeader {
        magic: *b"PRISM\0\0\0",
        version: 5,
        segment_count: 2,
        payload_hash,
        num_layers: NUM_LAYERS as u32,
        num_heads: NUM_HEADS as u32,
        head_dim: HEAD_DIM as u32,
        hidden_dim: HIDDEN_DIM as u32,
        intermediate_dim: FFN_INTERMEDIATE as u32,
        vocab_size: VOCAB_SIZE as u32,
        quantization_schema: 0,
        draft_num_layers: 0,
        segments,
        _pad: [0u8; 8],
    };
    writer.seek(SeekFrom::Start(0)).unwrap();
    let hb = unsafe {
        std::slice::from_raw_parts(
            &header as *const CimageHeader as *const u8,
            header_size as usize,
        )
    };
    writer.write_all(hb).unwrap();
    writer.flush().unwrap();
    drop(writer);

    let cimage_bytes = std::fs::read(&output).unwrap();
    println!(
        "    Written: {} bytes ({:.1} MB)",
        cimage_bytes.len(),
        cimage_bytes.len() as f64 / (1024.0 * 1024.0)
    );

    let elapsed = total_start.elapsed();
    println!("\n  Done in {:.1}s", elapsed.as_secs_f64());
    println!("  Output: {}", output);
}
