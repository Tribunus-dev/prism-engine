//! GGUF v3 writer for test fixtures and synthetic model weights.
//!
//! Produces a valid GGUF v3 binary with metadata KV pairs and tensor data.
//! The layout follows the canonical GGUF specification:
//!
//! ```text
//! [magic: b"GGUF" (4 bytes)]
//! [version: u32 LE = 3]
//! [tensor_count: u64 LE]
//! [metadata_kv_count: u64 LE]
//! [metadata KV pairs ...]
//! [tensor info entries ...]
//! [padding to 32-byte alignment]
//! [tensor data at sequential offsets ...]
//! ```

use crate::{ggml_type, GgufError};
use std::fs::File;
use std::io::Write;
use std::path::Path;

// ── GGUF value-type constants (local) ────────────────────────
const GGUF_TYPE_UINT32: u32 = 4;
const GGUF_TYPE_FLOAT32: u32 = 6;
const GGUF_TYPE_STRING: u32 = 8;

/// Describes a single tensor to be written into the GGUF.
struct TensorEntry {
    /// GGUF-style tensor name (e.g. `"blk.0.attn_q.weight"`).
    name: String,
    /// GGML dtype code (e.g. `ggml_type::F16`).
    dtype: u32,
    /// Shape dimensions (row-major).
    dims: Vec<u64>,
    /// Raw serialized data bytes (already in the target dtype encoding).
    data: Vec<u8>,
}

/// Write a minimal Bonsai-format GGUF v3 file for pipeline integration testing.
///
/// The generated file contains the minimal set of tensors needed to exercise
/// the `compile_to_cimage` path with a custom tiny `ModelGraph`. All tensors
/// use F16 encoding and are filled with deterministic pseudo-random data.
///
/// The GGUF includes only the tensors needed for a tiny model graph built
/// from a `UnifiedConfig` with 1 layer, small dimensions, and separate QKV
/// attention projections (the default for layer 0 in the Bonsai path).
pub fn create_mini_bonsai_gguf(path: &Path) -> Result<(), GgufError> {
    let mut f = File::create(path)?;

    // ── Tensor descriptions ────────────────────────────────────
    // These match the model graph created by `ModelGraph::build` with:
    //   family: Bonsai, num_layers: 1, hidden_size: 32,
    //   intermediate_size: 64, vocab_size: 16,
    //   num_heads: 2, head_dim: 16, num_kv_heads: 2
    //   layer_types["0"] = "separate_qkv"
    //
    // Separate QKV dimensions (Bonsai expansion factor):
    //   q_proj_dim = num_heads * head_dim * 8 = 2 * 16 * 8 = 256
    //   k_proj_dim = num_kv_heads * head_dim * 4 = 2 * 16 * 4 = 128
    //   v_proj_dim = num_kv_heads * head_dim * 4 = 2 * 16 * 4 = 128
    //   o_proj_dim = num_heads * head_dim * 4 = 2 * 16 * 4 = 128

    let tensors: Vec<TensorEntry> = vec![
        // Token embedding
        make_f16_tensor("token_embd.weight", &[16, 32]),
        // Layer 0 — separate QKV (standard for layer 0)
        make_f16_tensor("blk.0.attn_q.weight", &[256, 32]),
        make_f16_tensor("blk.0.attn_k.weight", &[128, 32]),
        make_f16_tensor("blk.0.attn_v.weight", &[128, 32]),
        make_f16_tensor("blk.0.attn_output.weight", &[128, 32]),
        // Layer 0 — FFN
        make_f16_tensor("blk.0.ffn_gate.weight", &[64, 32]),
        make_f16_tensor("blk.0.ffn_up.weight", &[64, 32]),
        make_f16_tensor("blk.0.ffn_down.weight", &[32, 64]),
        // LM head (not tied)
        make_f16_tensor("output.weight", &[16, 32]),
    ];

    let tensor_count = tensors.len() as u64;

    // We'll compute offsets after writing header + tensor infos.
    // Placeholder: write header first, remember position for data.
    // Strategy: two-pass — write header+infos, then backfill offsets.

    // ── Stage 1: collect (name, dims, dtype, data) — we already have it.
    // Compute tensor data offsets from the end of the info section.
    // We'll write the info entries with placeholder offsets, then seek back.

    // ── Stage 2: compute the header footprint before writing anything.
    // ── Compute header footprint ────────────────────────────────
    let mut header_end: u64 = 0;
    header_end += 4; // magic
    header_end += 4; // version
    header_end += 8; // tensor_count
    header_end += 8; // metadata_kv_count

    let meta_entries: Vec<(Vec<u8>, u32, Vec<u8>)> = vec![
        make_string_meta("general.architecture", "qwen35"),
        make_uint32_meta("general.file_type", 1),
        make_uint32_meta("qwen35.block_count", 64),
        make_uint32_meta("qwen35.context_length", 32768),
        make_uint32_meta("qwen35.embedding_length", 5120),
        make_uint32_meta("qwen35.feed_forward_length", 17408),
        make_uint32_meta("qwen35.attention.head_count", 24),
        make_uint32_meta("qwen35.attention.head_count_kv", 2),
        make_uint32_meta("qwen35.attention.key_length", 64),
        make_uint32_meta("qwen35.attention.value_length", 64),
        make_float32_meta("qwen35.rope.freq_base", 10000000.0),
        make_uint32_meta("qwen35.rope.dimension_count", 128),
        make_float32_meta("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
        make_uint32_meta("qwen35.expert_count", 0),
        make_uint32_meta("qwen35.experts_used_count", 0),
    ];

    for (key_bytes, _value_type, value_bytes) in &meta_entries {
        header_end += key_bytes.len() as u64; // string key (already includes u64 len prefix)
        header_end += 4; // _value_type
        header_end += value_bytes.len() as u64; // value payload
    }

    // Tensor info entries
    for tensor in &tensors {
        header_end += 8 + tensor.name.len() as u64; // name string (u64 len + bytes)
        header_end += 4; // n_dims
        header_end += (tensor.dims.len() as u64) * 8; // dims (u64 each, v3)
        header_end += 4; // dtype
        header_end += 8; // offset (placeholder)
    }

    // Align to 32 bytes
    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;
    let mut data_offset = data_start;
    let mut tensor_info: Vec<(u64, &TensorEntry)> = Vec::new(); // (offset, tensor)
    for tensor in &tensors {
        tensor_info.push((data_offset, tensor));
        data_offset += tensor.data.len() as u64;
    }

    // ── Stage 3: write ─────────────────────────────────────────
    // Magic
    f.write_all(b"GGUF")?;
    // Version
    f.write_all(&3u32.to_le_bytes())?;
    // Tensor count
    f.write_all(&tensor_count.to_le_bytes())?;
    // Metadata KV count
    f.write_all(&(meta_entries.len() as u64).to_le_bytes())?;

    // Metadata KV pairs
    for (key_bytes, value_type, value_bytes) in &meta_entries {
        f.write_all(&key_bytes)?; // includes u64 length prefix
        f.write_all(&value_type.to_le_bytes())?;
        f.write_all(value_bytes)?;
    }

    // Tensor info entries
    for (offset, tensor) in &tensor_info {
        write_string_v3(&mut f, &tensor.name)?;
        f.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
        for dim in &tensor.dims {
            f.write_all(&dim.to_le_bytes())?;
        }
        f.write_all(&tensor.dtype.to_le_bytes())?;
        f.write_all(&offset.to_le_bytes())?;
    }

    // Padding to 32-byte alignment
    let pad_buf = vec![0u8; pad as usize];
    f.write_all(&pad_buf)?;

    // Tensor data
    for (_, tensor) in &tensor_info {
        f.write_all(&tensor.data)?;
    }

    f.flush()?;
    Ok(())
}

/// Write a GGUF file with metadata values matching Bonsai27B constants.
///
/// Uses the same minimal tensor data as `create_mini_bonsai_gguf` but writes
/// metadata matching the exact values expected by
/// `BonsaiCheckpointIngestion::ingest`. This allows callers to exercise
/// `compile_gguf` (which validates against Bonsai27B constants) without a
/// full-size Bonsai-27B checkpoint.
pub fn create_bonsai27b_gguf(path: &Path) -> Result<(), GgufError> {
    use std::io::Write;

    let mut f = File::create(path)?;

    // ── Tensor descriptions — same as create_mini_bonsai_gguf ──
    let tensors: Vec<TensorEntry> = vec![
        make_f16_tensor("token_embd.weight", &[16, 32]),
        make_f16_tensor("blk.0.attn_q.weight", &[256, 32]),
        make_f16_tensor("blk.0.attn_k.weight", &[128, 32]),
        make_f16_tensor("blk.0.attn_v.weight", &[128, 32]),
        make_f16_tensor("blk.0.attn_output.weight", &[128, 32]),
        make_f16_tensor("blk.0.ffn_gate.weight", &[64, 32]),
        make_f16_tensor("blk.0.ffn_up.weight", &[64, 32]),
        make_f16_tensor("blk.0.ffn_down.weight", &[32, 64]),
        make_f16_tensor("output.weight", &[16, 32]),
    ];

    let tensor_count = tensors.len() as u64;

    // ── Metadata with CORRECT Bonsai27B constants ────────────
    let meta_entries: Vec<(Vec<u8>, u32, Vec<u8>)> = vec![
        make_string_meta("general.architecture", "qwen35"),
        make_uint32_meta("general.file_type", 1),
        make_uint32_meta("qwen35.block_count", 64),
        make_uint32_meta("qwen35.context_length", 262_144),
        make_uint32_meta("qwen35.embedding_length", 5120),
        make_uint32_meta("qwen35.feed_forward_length", 17408),
        make_uint32_meta("qwen35.attention.head_count", 24),
        make_uint32_meta("qwen35.attention.head_count_kv", 4),
        make_uint32_meta("qwen35.attention.key_length", 256),
        make_uint32_meta("qwen35.attention.value_length", 256),
        make_float32_meta("qwen35.rope.freq_base", 10_000_000.0),
        make_uint32_meta("qwen35.rope.dimension_count", 64),
        make_float32_meta("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
        make_uint32_meta("qwen35.expert_count", 0),
        make_uint32_meta("qwen35.experts_used_count", 0),
    ];

    // ── Compute header footprint ─────────────────────────────
    let mut header_end: u64 = 0;
    header_end += 4; // magic
    header_end += 4; // version
    header_end += 8; // tensor_count
    header_end += 8; // metadata_kv_count

    for (key_bytes, _value_type, value_bytes) in &meta_entries {
        header_end += key_bytes.len() as u64;
        header_end += 4;
        header_end += value_bytes.len() as u64;
    }

    for tensor in &tensors {
        header_end += 8 + tensor.name.len() as u64;
        header_end += 4;
        header_end += (tensor.dims.len() as u64) * 8;
        header_end += 4;
        header_end += 8;
    }

    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;
    let mut data_offset = data_start;
    let mut tensor_info: Vec<(u64, &TensorEntry)> = Vec::new();
    for tensor in &tensors {
        tensor_info.push((data_offset, tensor));
        data_offset += tensor.data.len() as u64;
    }

    // ── Write ────────────────────────────────────────────────
    f.write_all(b"GGUF")?;
    f.write_all(&3u32.to_le_bytes())?;
    f.write_all(&tensor_count.to_le_bytes())?;
    f.write_all(&(meta_entries.len() as u64).to_le_bytes())?;

    for (key_bytes, value_type, value_bytes) in &meta_entries {
        f.write_all(key_bytes)?;
        f.write_all(&value_type.to_le_bytes())?;
        f.write_all(value_bytes)?;
    }

    for (offset, tensor) in &tensor_info {
        write_string_v3(&mut f, &tensor.name)?;
        f.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
        for dim in &tensor.dims {
            f.write_all(&dim.to_le_bytes())?;
        }
        f.write_all(&tensor.dtype.to_le_bytes())?;
        f.write_all(&offset.to_le_bytes())?;
    }

    let pad_buf = vec![0u8; pad as usize];
    f.write_all(&pad_buf)?;

    for (_, tensor) in &tensor_info {
        f.write_all(&tensor.data)?;
    }

    f.flush()?;
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────

/// Create a GGUF v3 string: u64 LE length + UTF-8 bytes.
fn write_string_v3(f: &mut File, s: &str) -> Result<(), GgufError> {
    let bytes = s.as_bytes();
    f.write_all(&(bytes.len() as u64).to_le_bytes())?;
    f.write_all(bytes)?;
    Ok(())
}

/// Build a string-typed metadata entry (pre-serialized bytes).
fn make_string_meta(key: &str, value: &str) -> (Vec<u8>, u32, Vec<u8>) {
    let mut key_bytes = Vec::new();
    write_key(&mut key_bytes, key);

    let mut val_bytes = Vec::new();
    // u64 length prefix + value bytes
    val_bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    val_bytes.extend_from_slice(value.as_bytes());

    (key_bytes, GGUF_TYPE_STRING, val_bytes)
}

/// Build a u32-typed metadata entry (pre-serialized bytes).
fn make_uint32_meta(key: &str, value: u32) -> (Vec<u8>, u32, Vec<u8>) {
    let mut key_bytes = Vec::new();
    write_key(&mut key_bytes, key);
    (key_bytes, GGUF_TYPE_UINT32, value.to_le_bytes().to_vec())
}

/// Build a f32-typed metadata entry (pre-serialized bytes).
fn make_float32_meta(key: &str, value: f32) -> (Vec<u8>, u32, Vec<u8>) {
    let mut key_bytes = Vec::new();
    write_key(&mut key_bytes, key);
    (key_bytes, GGUF_TYPE_FLOAT32, value.to_le_bytes().to_vec())
}

/// Write a GGUF v3 string key (u64 LE length + UTF-8 bytes) into a buffer.
fn write_key(buf: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    buf.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(bytes);
}

/// Create an F16 tensor entry filled with deterministic data.
///
/// The data is computed as `(row_idx * ncols + col_idx) as f16` so
/// values are deterministic and vary per position for debugging.
fn make_f16_tensor(name: &str, shape: &[u64]) -> TensorEntry {
    let num_elems: usize = shape.iter().map(|&d| d as usize).product();
    use half::f16;

    let mut data = Vec::with_capacity(num_elems * 2);
    if shape.len() == 2 {
        let rows = shape[0] as usize;
        let cols = shape[1] as usize;
        for r in 0..rows {
            for c in 0..cols {
                // Deterministic value: small variation per position
                let val = f16::from_f32(((r * cols + c) as f32) * 0.01);
                data.extend_from_slice(&val.to_bits().to_le_bytes());
            }
        }
    } else if shape.len() == 1 {
        for i in 0..num_elems {
            let val = f16::from_f32((i as f32) * 0.01);
            data.extend_from_slice(&val.to_bits().to_le_bytes());
        }
    }

    TensorEntry {
        name: name.to_string(),
        dtype: ggml_type::F16,
        dims: shape.to_vec(),
        data,
    }
}

/// Create an F16 tensor entry from explicit f32 values.
///
/// The values are converted to FP16 for exact representation, then
/// packed as little-endian bytes. The number of values must match
/// the product of the shape dimensions.
fn make_f16_tensor_from_values(name: &str, shape: &[u64], values: &[f32]) -> TensorEntry {
    let num_elems: usize = shape.iter().map(|&d| d as usize).product();
    assert_eq!(
        values.len(),
        num_elems,
        "make_f16_tensor_from_values: {} expected {} elements, got {}",
        name,
        num_elems,
        values.len()
    );
    use half::f16;
    let mut data = Vec::with_capacity(num_elems * 2);
    for &v in values {
        data.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    TensorEntry {
        name: name.to_string(),
        dtype: ggml_type::F16,
        dims: shape.to_vec(),
        data,
    }
}

/// Write a miniature 2-layer standard GGUF v3 fixture with
/// deterministic small-integer weights for numerical reference testing.
///
/// Dimensions: HIDDEN=8, INTERMEDIATE=8, VOCAB=8, N_LAYERS=2,
/// N_HEADS=2, HEAD_DIM=4, KV_HEADS=2.
///
/// All weights are small integers (-3..=+3) ensuring exact FP16
/// representation and reproducible numerical results.
pub fn create_numerical_fixture_gguf(path: &Path) -> Result<(), GgufError> {
    const VOCAB: u64 = 8;
    const HIDDEN: u64 = 8;
    const INTERMEDIATE: u64 = 8;
    const N_LAYERS: u64 = 2;
    const N_HEADS: u64 = 2;
    const KV_HEADS: u64 = 2;
    const HEAD_DIM: u64 = 4;

    let mut tensors: Vec<TensorEntry> = Vec::new();

    // Identity matrix [HIDDEN x HIDDEN]
    let identity_8x8: Vec<f32> = {
        let mut v = vec![0.0f32; (HIDDEN * HIDDEN) as usize];
        for i in 0..HIDDEN as usize {
            v[i * HIDDEN as usize + i] = 1.0;
        }
        v
    };

    // Ones vector [HIDDEN]
    let ones_8: Vec<f32> = vec![1.0; HIDDEN as usize];

    // Token embedding: identity
    tensors.push(make_f16_tensor_from_values(
        "token_embd.weight",
        &[VOCAB, HIDDEN],
        &identity_8x8,
    ));

    // Per-layer tensors
    for layer in 0..N_LAYERS {
        let prefix = format!("blk.{}", layer);

        // Input layernorm (all 1.0)
        tensors.push(make_f16_tensor_from_values(
            &format!("{}.input_layernorm.weight", prefix),
            &[HIDDEN],
            &ones_8,
        ));

        // Q/K/V/O attention projections (identity)
        for proj in &[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ] {
            tensors.push(make_f16_tensor_from_values(
                &format!("{}.{}.weight", prefix, proj),
                &[HIDDEN, HIDDEN],
                &identity_8x8,
            ));
        }

        // Post-attention layernorm (all 1.0)
        tensors.push(make_f16_tensor_from_values(
            &format!("{}.post_attention_layernorm.weight", prefix),
            &[HIDDEN],
            &ones_8,
        ));

        // FFN gate [INTERMEDIATE, HIDDEN]: row i has 1 at column (i % HIDDEN)
        {
            let mut vals = vec![0.0f32; (INTERMEDIATE * HIDDEN) as usize];
            for i in 0..INTERMEDIATE as usize {
                vals[i * HIDDEN as usize + (i % HIDDEN as usize)] = 1.0;
            }
            tensors.push(make_f16_tensor_from_values(
                &format!("{}.mlp.gate_proj.weight", prefix),
                &[INTERMEDIATE, HIDDEN],
                &vals,
            ));
        }

        // FFN up [INTERMEDIATE, HIDDEN]: same sparse pattern
        {
            let mut vals = vec![0.0f32; (INTERMEDIATE * HIDDEN) as usize];
            for i in 0..INTERMEDIATE as usize {
                vals[i * HIDDEN as usize + (i % HIDDEN as usize)] = 1.0;
            }
            tensors.push(make_f16_tensor_from_values(
                &format!("{}.mlp.up_proj.weight", prefix),
                &[INTERMEDIATE, HIDDEN],
                &vals,
            ));
        }

        // FFN down [HIDDEN, INTERMEDIATE]: row j has 1 at column j
        {
            let mut vals = vec![0.0f32; (HIDDEN * INTERMEDIATE) as usize];
            for j in 0..HIDDEN as usize {
                vals[j * INTERMEDIATE as usize + j] = 1.0;
            }
            tensors.push(make_f16_tensor_from_values(
                &format!("{}.mlp.down_proj.weight", prefix),
                &[HIDDEN, INTERMEDIATE],
                &vals,
            ));
        }
    }

    // Final norm (all 1.0)
    tensors.push(make_f16_tensor_from_values(
        "output_norm.weight",
        &[HIDDEN],
        &ones_8,
    ));

    // LM head / output projection (identity)
    tensors.push(make_f16_tensor_from_values(
        "output.weight",
        &[VOCAB, HIDDEN],
        &identity_8x8,
    ));

    // ── Write GGUF v3 file ─────────────────────────────────
    let mut f = File::create(path)?;
    let tensor_count = tensors.len() as u64;

    let meta_entries: Vec<(Vec<u8>, u32, Vec<u8>)> = vec![
        make_string_meta("general.architecture", "qwen2"),
        make_uint32_meta("general.file_type", 1),
        make_uint32_meta("qwen2.block_count", N_LAYERS as u32),
        make_uint32_meta("qwen2.context_length", 64),
        make_uint32_meta("qwen2.embedding_length", HIDDEN as u32),
        make_uint32_meta("qwen2.feed_forward_length", INTERMEDIATE as u32),
        make_uint32_meta("qwen2.attention.head_count", N_HEADS as u32),
        make_uint32_meta("qwen2.attention.head_count_kv", KV_HEADS as u32),
        make_float32_meta("qwen2.rope.freq_base", 10000.0),
        make_uint32_meta("qwen2.rope.dimension_count", (N_HEADS * HEAD_DIM) as u32),
        make_float32_meta("qwen2.attention.layer_norm_rms_epsilon", 1e-5),
        make_uint32_meta("qwen2.expert_count", 0),
        make_uint32_meta("qwen2.experts_used_count", 0),
    ];

    // Compute header footprint
    let mut header_end: u64 = 0;
    header_end += 4; // magic
    header_end += 4; // version
    header_end += 8; // tensor_count
    header_end += 8; // metadata_kv_count

    for (key_bytes, _value_type, value_bytes) in &meta_entries {
        header_end += key_bytes.len() as u64;
        header_end += 4;
        header_end += value_bytes.len() as u64;
    }

    for tensor in &tensors {
        header_end += 8 + tensor.name.len() as u64;
        header_end += 4;
        header_end += (tensor.dims.len() as u64) * 8;
        header_end += 4;
        header_end += 8;
    }

    let pad = (32 - (header_end % 32)) % 32;
    let data_start = header_end + pad;
    let mut data_offset = data_start;
    let mut tensor_info: Vec<(u64, &TensorEntry)> = Vec::new();
    for tensor in &tensors {
        tensor_info.push((data_offset, tensor));
        data_offset += tensor.data.len() as u64;
    }

    // Write header
    f.write_all(b"GGUF")?;
    f.write_all(&3u32.to_le_bytes())?;
    f.write_all(&tensor_count.to_le_bytes())?;
    f.write_all(&(meta_entries.len() as u64).to_le_bytes())?;

    for (key_bytes, value_type, value_bytes) in &meta_entries {
        f.write_all(key_bytes)?;
        f.write_all(&value_type.to_le_bytes())?;
        f.write_all(value_bytes)?;
    }

    for (offset, tensor) in &tensor_info {
        write_string_v3(&mut f, &tensor.name)?;
        f.write_all(&(tensor.dims.len() as u32).to_le_bytes())?;
        for dim in &tensor.dims {
            f.write_all(&dim.to_le_bytes())?;
        }
        f.write_all(&tensor.dtype.to_le_bytes())?;
        f.write_all(&offset.to_le_bytes())?;
    }

    let pad_buf = vec![0u8; pad as usize];
    f.write_all(&pad_buf)?;

    for (_, tensor) in &tensor_info {
        f.write_all(&tensor.data)?;
    }

    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_gguf_header;

    #[test]
    fn test_create_mini_bonsai_gguf_roundtrip() {
        let dir = std::env::temp_dir().join("gguf-writer-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("mini-test.gguf");

        create_mini_bonsai_gguf(&path).expect("create GGUF");

        // Parse it back and verify contents
        let (metadata, tensors) = parse_gguf_header(&path).expect("parse GGUF");

        // Check architecture metadata
        assert!(metadata
            .iter()
            .any(|(k, v)| { k == "general.architecture" && v == "qwen35" }));

        // Check we have all expected tensors
        let tensor_names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            tensor_names.contains(&"token_embd.weight"),
            "should have token_embd.weight, got {tensor_names:?}"
        );
        assert!(
            tensor_names.contains(&"blk.0.attn_q.weight"),
            "should have attn_q.weight"
        );

        // Check tensor sizes make sense (F16 = 2 bytes per element)
        let token_embd = tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .unwrap();
        assert_eq!(token_embd.shape, &[16, 32]);
        assert_eq!(token_embd.dtype, "f16");
        assert_eq!(token_embd.byte_size, 16 * 32 * 2);

        // Verify file size matches expected
        let file_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        assert!(file_size > 100, "GGUF file should be non-trivial");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_create_bonsai27b_gguf_roundtrip() {
        let dir = std::env::temp_dir().join("gguf-bonsai27b-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("bonsai27b.gguf");

        create_bonsai27b_gguf(&path).expect("create Bonsai27B GGUF");

        // Parse it back and verify metadata values match Bonsai27B constants
        let (metadata, tensors) = parse_gguf_header(&path).expect("parse GGUF");

        // Check key metadata values
        let context_len = metadata
            .iter()
            .find(|(k, _)| k == "qwen35.context_length")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        assert_eq!(context_len, "262144", "context_length should be 262144");

        let kv_heads = metadata
            .iter()
            .find(|(k, _)| k == "qwen35.attention.head_count_kv")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        assert_eq!(kv_heads, "4", "head_count_kv should be 4");

        let key_len = metadata
            .iter()
            .find(|(k, _)| k == "qwen35.attention.key_length")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        assert_eq!(key_len, "256", "key_length should be 256");

        let value_len = metadata
            .iter()
            .find(|(k, _)| k == "qwen35.attention.value_length")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        assert_eq!(value_len, "256", "value_length should be 256");

        let rope_dim = metadata
            .iter()
            .find(|(k, _)| k == "qwen35.rope.dimension_count")
            .map(|(_, v)| v.as_str())
            .unwrap_or("");
        assert_eq!(rope_dim, "64", "rope.dimension_count should be 64");

        // Same tensor structure as the mini variant
        assert_eq!(tensors.len(), 9, "should have 9 tensors");
        assert!(tensors.iter().any(|t| t.name == "token_embd.weight"));
        assert!(tensors.iter().any(|t| t.name == "blk.0.attn_q.weight"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
#[ignore = "requires bonsai-27b/Ternary-Bonsai-27B-Q2_0.gguf"]
fn test_real_bonsai_gguf_structure() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap() // workspace root
        .join("bonsai-27b")
        .join("Ternary-Bonsai-27B-Q2_0.gguf");
    if !path.exists() {
        eprintln!("Skipping: Bonsai GGUF not found at {}", path.display());
        return;
    }

    let (metadata, tensors) = crate::parse_gguf_header(&path).expect("parse real Bonsai GGUF");

    // Verify metadata
    let arch = metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert_eq!(arch, "qwen35", "architecture should be qwen35");

    let block_count = metadata
        .iter()
        .find(|(k, _)| k == "qwen35.block_count")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert_eq!(block_count, "64", "should have 64 layers");

    let emb_len = metadata
        .iter()
        .find(|(k, _)| k == "qwen35.embedding_length")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert_eq!(emb_len, "5120", "embedding length should be 5120");

    // Verify tensor count is reasonable
    assert!(
        tensors.len() > 500,
        "should have 500+ tensors, got {}",
        tensors.len()
    );

    // Verify tensor naming convention
    let layer_tensors: Vec<_> = tensors
        .iter()
        .filter(|t| t.name.starts_with("blk."))
        .collect();
    assert!(!layer_tensors.is_empty(), "should have layer tensors");

    // Verify specific tensor names exist
    let has_embed = tensors.iter().any(|t| t.name == "token_embd.weight");
    assert!(has_embed, "should have token_embd.weight");

    let has_output = tensors.iter().any(|t| t.name == "output.weight");
    assert!(has_output, "should have output.weight");

    // Verify at least one layer has attention weights
    let has_attn_q = tensors.iter().any(|t| t.name.contains("attn_q.weight"));
    let has_attn_k = tensors.iter().any(|t| t.name.contains("attn_k.weight"));
    let has_attn_v = tensors.iter().any(|t| t.name.contains("attn_v.weight"));
    assert!(has_attn_q, "should have attn_q.weight");
    assert!(has_attn_k, "should have attn_k.weight");
    assert!(has_attn_v, "should have attn_v.weight");

    // Verify dtype is Q2_0 (quantized)
    let q2_tensors: Vec<_> = tensors.iter().filter(|t| t.dtype == "q2_0").collect();
    assert!(!q2_tensors.is_empty(), "should have Q2_0 quantized tensors");

    // Count per-layer tensors: each layer should have the same number of tensors
    let mut per_layer: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for t in &layer_tensors {
        if let Some(dot) = t.name[4..].find('.') {
            let layer_key = &t.name[..4 + dot + 1];
            *per_layer.entry(layer_key.to_string()).or_insert(0) += 1;
        }
    }
    let layer_count = per_layer.len();
    assert_eq!(layer_count, 64, "should have 64 layers with tensors");
    let counts: std::collections::HashSet<&usize> = per_layer.values().collect();
    assert_eq!(
        counts.len(),
        1,
        "each layer should have the same number of tensors, got {counts:?}"
    );
    let tensors_per_layer = *per_layer.values().next().unwrap();
    assert!(tensors_per_layer > 0, "each layer should have tensors");

    eprintln!(
        "Bonsai GGUF validation passed: {} tensors ({} per layer), {} metadata keys",
        tensors.len(),
        tensors_per_layer,
        metadata.len()
    );
}

#[test]
fn test_mini_bonsai_structure() {
    let dir = std::env::temp_dir().join("prism-gguf-structure-test");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("mini-structure.gguf");

    crate::writer::create_mini_bonsai_gguf(&path).expect("create mini bonsai gguf");

    let (metadata, tensors) = crate::parse_gguf_header(&path).expect("parse mini gguf");

    // The mini GGUF has 15 metadata keys
    assert_eq!(metadata.len(), 15, "should have 15 metadata keys");

    // 2 shared (token_embd, output) + 7 layer tensors (1 layer)
    assert_eq!(tensors.len(), 9, "should have 2 shared + 7 layer tensors");

    // Verify metadata structure
    let arch = metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .map(|(_, v)| v.as_str())
        .unwrap_or("");
    assert_eq!(arch, "qwen35", "mini architecture should be qwen35");

    // Verify tensor naming
    assert!(tensors.iter().any(|t| t.name == "token_embd.weight"));
    assert!(tensors.iter().any(|t| t.name == "output.weight"));
    assert!(tensors.iter().any(|t| t.name == "blk.0.attn_q.weight"));
    assert!(tensors.iter().any(|t| t.name == "blk.0.ffn_down.weight"));

    // Verify all tensors are f16 in mini
    for t in &tensors {
        assert_eq!(t.dtype, "f16", "mini tensor {} should be f16", t.name);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
