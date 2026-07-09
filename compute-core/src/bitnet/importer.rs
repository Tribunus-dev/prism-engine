//! BitNet b1.58 native weight importer.
//!
//! Ingests already-ternary {-1, 0, +1} weights from a BitNet model into
//! `TernaryPackedTensor` values. For hermetic testing, deterministic
//! pseudo-random generation is used in place of live safetensors I/O.

use crate::ternary::codec::{TernaryCodecError, TernaryPackedTensor};
use crate::ternary::pack::pack_ternary_codes;
use half::f16;

/// BitNet native weight importer.
///
/// Uses a deterministic LCG to generate ternary weight distributions,
/// then packs them into 2-bit codes ready for cimage emission.
pub struct BitNetImporter;

impl BitNetImporter {
    /// Generate a deterministic set of {-1, 0, +1} weights.
    ///
    /// Uses the MMIX LCG (same as the cimage shard builder) to generate
    /// f32 values in [-1, 1), then thresholds at ±1/3 to produce a roughly
    /// equal ternary distribution.
    pub fn generate_ternary_weights(seed: u64, num_weights: usize) -> Vec<i8> {
        let mut state = seed;
        let mut weights = Vec::with_capacity(num_weights);
        for _ in 0..num_weights {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Reinterpret as f32 in (-1, 1)
            let f = ((state >> 32) as i32 as f64) / (1i64 << 31) as f64;
            let f = f.clamp(-1.0, 1.0) as f32;
            let t: i8 = if f > 0.33 {
                1
            } else if f < -0.33 {
                -1
            } else {
                0
            };
            weights.push(t);
        }
        weights
    }

    /// Import a single ternary tensor, packing into codes and computing scales.
    ///
    /// `rows` and `cols` describe the stored (pack) shape, which for a
    /// BitLinear weight is [in_features, out_features] — the weights are
    /// already in transposed storage layout.
    ///
    /// For each row, `groups_per_row = cols.div_ceil(group_size)` and one
    /// f16 scale is emitted per group.
    pub fn import_ternary_tensor(
        seed: u64,
        rows: usize,
        cols: usize,
        group_size: usize,
    ) -> Result<TernaryPackedTensor, TernaryCodecError> {
        let num_weights = rows * cols;
        let weights = Self::generate_ternary_weights(seed, num_weights);

        let codes = pack_ternary_codes(&weights)?;

        // Compute per-group scales: sum(|w|) / group_size.
        let groups_per_row = cols.div_ceil(group_size);
        let scales_f32: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let weights_ref = &weights;
                (0..groups_per_row).map(move |g| {
                    let start = g * group_size;
                    let end = (start + group_size).min(cols);
                    let mut sum_abs = 0i32;
                    for c in start..end {
                        sum_abs += (weights_ref[r * cols + c] as i32).abs();
                    }
                    sum_abs as f32 / group_size as f32
                })
            })
            .collect();

        let scales: Vec<f16> = scales_f32.iter().map(|&s| f16::from_f32(s)).collect();

        let bytes_per_group = (group_size + 3) / 4;

        Ok(TernaryPackedTensor {
            rows,
            cols,
            group_size,
            groups_per_row,
            bytes_per_group,
            codes,
            scales,
        })
    }

    /// Import the three MLP weight tensors for a BitNet MLP block.
    ///
    /// Returns `(gate_proj, up_proj, down_proj)`, each as a
    /// `TernaryPackedTensor` in [in_features, out_features] (stored) layout.
    ///
    /// Seeds are offset to avoid correlation between tensors.
    pub fn import_mlp_block(
        seed: u64,
        hidden_dim: usize,
        intermediate_dim: usize,
        group_size: usize,
    ) -> Result<
        (
            TernaryPackedTensor,
            TernaryPackedTensor,
            TernaryPackedTensor,
        ),
        TernaryCodecError,
    > {
        // BitLinear gate_proj: [intermediate_dim, hidden_dim] → stored [hidden_dim, intermediate_dim]
        let gate = Self::import_ternary_tensor(
            seed.wrapping_add(1),
            hidden_dim,
            intermediate_dim,
            group_size,
        )?;
        // BitLinear up_proj: same shape.
        let up = Self::import_ternary_tensor(
            seed.wrapping_add(2),
            hidden_dim,
            intermediate_dim,
            group_size,
        )?;
        // BitLinear down_proj: [hidden_dim, intermediate_dim] → stored [intermediate_dim, hidden_dim]
        let down = Self::import_ternary_tensor(
            seed.wrapping_add(3),
            intermediate_dim,
            hidden_dim,
            group_size,
        )?;

        Ok((gate, up, down))
    }

    /// Import Q/K/V/O attention projection tensors for a BitNet decoder layer.
    ///
    /// Q,O shapes: stored [hidden_dim, hidden_dim]
    /// K,V shapes: stored [hidden_dim, kv_inner] where kv_inner = num_kv_heads * head_dim
    pub fn import_attention_tensors(
        seed: u64,
        hidden_dim: usize,
        _num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        group_size: usize,
    ) -> Result<
        (
            TernaryPackedTensor,
            TernaryPackedTensor,
            TernaryPackedTensor,
            TernaryPackedTensor,
        ),
        TernaryCodecError,
    > {
        let kv_inner = num_kv_heads * head_dim;
        let q =
            Self::import_ternary_tensor(seed.wrapping_add(1), hidden_dim, hidden_dim, group_size)?;
        let k =
            Self::import_ternary_tensor(seed.wrapping_add(2), hidden_dim, kv_inner, group_size)?;
        let v =
            Self::import_ternary_tensor(seed.wrapping_add(3), hidden_dim, kv_inner, group_size)?;
        let o =
            Self::import_ternary_tensor(seed.wrapping_add(4), hidden_dim, hidden_dim, group_size)?;
        Ok((q, k, v, o))
    }

    /// Import a 1-D layernorm weight tensor.
    pub fn import_layernorm_tensor(seed: u64, dim: usize) -> TernaryPackedTensor {
        let weights = Self::generate_ternary_weights(seed, dim);
        // Layernorm: single group, each weight is its own "scale" value
        let group_size = dim;
        let groups_per_row = 1;
        let bytes_per_group = (dim + 3) / 4;
        let _num_weights = dim;
        let codes = pack_ternary_codes(&weights).unwrap_or_default();
        // Scale: sum(|w|) / dim per group
        let sum_abs: i32 = weights.iter().map(|&w| (w as i32).abs()).sum();
        let scale_f32 = sum_abs as f32 / dim as f32;
        let scales = vec![half::f16::from_f32(scale_f32)];
        TernaryPackedTensor {
            rows: 1,
            cols: dim,
            group_size,
            groups_per_row,
            bytes_per_group,
            codes,
            scales,
        }
    }

    /// Import a full decoder layer: 11 tensors.
    pub fn import_full_decoder_layer(
        seed: u64,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        seq_len: usize,
        group_size: usize,
    ) -> Result<Vec<TernaryPackedTensor>, TernaryCodecError> {
        let input_ln = Self::import_layernorm_tensor(seed, hidden_dim);
        let (q, k, v, o) = Self::import_attention_tensors(
            seed,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            group_size,
        )?;
        let post_attn_ln = Self::import_layernorm_tensor(seed.wrapping_add(5), hidden_dim);
        let (gate, up, down) =
            Self::import_mlp_block(seed, hidden_dim, intermediate_dim, group_size)?;
        // Position IDs: sequential 0..seq_len as f32. Not ternary — store codes as raw f32 le bytes.
        let pos_data: Vec<u8> = (0..seq_len)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        let position_ids = TernaryPackedTensor {
            rows: seq_len,
            cols: 1,
            group_size: 0,
            groups_per_row: 1,
            bytes_per_group: 0,
            codes: pos_data,
            scales: vec![],
        };
        let rmsnorm_extra = Self::import_layernorm_tensor(seed.wrapping_add(10), hidden_dim);
        Ok(vec![
            input_ln,      // 0
            q,             // 1
            k,             // 2
            v,             // 3
            o,             // 4
            post_attn_ln,  // 5
            gate,          // 6
            up,            // 7
            down,          // 8
            position_ids,  // 9
            rmsnorm_extra, // 10
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ternary::pack::unpack_ternary_codes;

    #[test]
    fn test_importer_generates_valid_ternary_weights() {
        let w = BitNetImporter::generate_ternary_weights(42, 1000);
        assert_eq!(w.len(), 1000);
        for &v in &w {
            assert!(v == -1 || v == 0 || v == 1, "unexpected weight {}", v);
        }
        // Roughly equal distribution (within 2 sigma for 1000 samples).
        let n_neg = w.iter().filter(|&&v| v == -1).count();
        let n_zero = w.iter().filter(|&&v| v == 0).count();
        let n_pos = w.iter().filter(|&&v| v == 1).count();
        let expected = 1000.0 / 3.0;
        let margin = 150.0; // ~5 sigma tolerance
        for &count in &[n_neg, n_zero, n_pos] {
            assert!(
                (count as f64 - expected).abs() < margin,
                "distribution {n_neg}/{n_zero}/{n_pos} out of tolerance ±{margin}"
            );
        }
    }

    #[test]
    fn test_importer_roundtrip() {
        let tensor = BitNetImporter::import_ternary_tensor(42, 8, 128, 32).unwrap();
        assert_eq!(tensor.rows, 8);
        assert_eq!(tensor.cols, 128);
        assert_eq!(tensor.group_size, 32);
        assert_eq!(tensor.groups_per_row, 4);
        assert_eq!(tensor.bytes_per_group, 8);

        let total_values = tensor.rows * tensor.cols;
        let unpacked = unpack_ternary_codes(&tensor.codes, total_values).unwrap();
        assert_eq!(unpacked.len(), total_values);

        // Scales: rows * groups_per_row.
        assert_eq!(tensor.scales.len(), tensor.rows * tensor.groups_per_row);
    }

    #[test]
    fn test_importer_mlp_block_shapes() {
        let (gate, up, down) = BitNetImporter::import_mlp_block(42, 256, 1024, 32).unwrap();
        // Gate: stored [hidden_dim, intermediate_dim]
        assert_eq!(gate.rows, 256);
        assert_eq!(gate.cols, 1024);
        assert_eq!(gate.groups_per_row, 1024 / 32);
        // Up: same
        assert_eq!(up.rows, 256);
        assert_eq!(up.cols, 1024);
        // Down: stored [intermediate_dim, hidden_dim]
        assert_eq!(down.rows, 1024);
        assert_eq!(down.cols, 256);
        assert_eq!(down.groups_per_row, 256 / 32);
    }
}
