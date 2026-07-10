//! BitNet KV cache — per-layer key/value state for decoder inference.
//!
//! Each layer stores K and V tensors as flat buffers of shape
//! `[max_seq_len * kv_inner]` where `kv_inner = num_kv_heads * head_dim`.
//! Data for position `pos` spans `[pos * kv_inner .. (pos+1) * kv_inner]`.

use crate::ecs::cimage::manifest::{CImagePayloadRef, CImageTensorEntry, PhysicalTileLayout};
use crate::execution_plan::{CodecFamily, DType};

/// In-memory KV cache for BitNet decoder layers.
///
/// Shape convention (illustrative):
///   k_cache[layer][head][pos * head_dim .. (pos+1) * head_dim]
///
/// where `head_dim = kv_inner / num_kv_heads`. The implementation stores all
/// heads concatenated per position in a single flat buffer per layer.
pub struct BitNetKvCache {
    pub k_cache: Vec<Vec<Vec<f32>>>,
    pub v_cache: Vec<Vec<Vec<f32>>>,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub kv_inner: usize,
}

impl BitNetKvCache {
    /// Create a new empty KV cache with capacity for `max_seq_len` tokens.
    pub fn new(num_layers: usize, kv_inner: usize, max_seq_len: usize) -> Self {
        let k_cache: Vec<Vec<Vec<f32>>> = (0..num_layers)
            .map(|_| vec![vec![0.0f32; max_seq_len * kv_inner]])
            .collect();
        let v_cache: Vec<Vec<Vec<f32>>> = (0..num_layers)
            .map(|_| vec![vec![0.0f32; max_seq_len * kv_inner]])
            .collect();
        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            max_seq_len,
            num_layers,
            kv_inner,
        }
    }

    /// Append K/V slices for a single token position.
    ///
    /// Panics if `layer >= num_layers` or `pos >= max_seq_len` or
    /// `k_slice.len() != kv_inner` or `v_slice.len() != kv_inner`.
    pub fn append(&mut self, layer: usize, pos: usize, k_slice: &[f32], v_slice: &[f32]) {
        assert!(layer < self.num_layers, "layer out of bounds");
        assert!(pos < self.max_seq_len, "position out of bounds");
        assert_eq!(k_slice.len(), self.kv_inner, "k_slice length mismatch");
        assert_eq!(v_slice.len(), self.kv_inner, "v_slice length mismatch");

        let start = pos * self.kv_inner;
        let end = start + self.kv_inner;
        self.k_cache[layer][0][start..end].copy_from_slice(k_slice);
        self.v_cache[layer][0][start..end].copy_from_slice(v_slice);

        if pos >= self.seq_len {
            self.seq_len = pos + 1;
        }
    }

    /// Read K/V slices for a single token position.
    ///
    /// Panics if `layer >= num_layers` or `pos >= seq_len`.
    pub fn read(&self, layer: usize, pos: usize) -> (&[f32], &[f32]) {
        assert!(layer < self.num_layers, "layer out of bounds");
        assert!(pos < self.seq_len, "position beyond current seq_len");

        let start = pos * self.kv_inner;
        let end = start + self.kv_inner;
        (
            &self.k_cache[layer][0][start..end],
            &self.v_cache[layer][0][start..end],
        )
    }

    /// Extend the cache with a prefill of `seq_len` tokens.
    ///
    /// `k_tensor` and `v_tensor` are flat slices of length `seq_len * kv_inner`
    /// (all heads concatenated, positions laid out sequentially).
    pub fn extend_prefill(
        &mut self,
        layer: usize,
        k_tensor: &[f32],
        v_tensor: &[f32],
        seq_len: usize,
    ) {
        assert!(layer < self.num_layers, "layer out of bounds");
        assert!(
            seq_len <= self.max_seq_len,
            "prefill seq_len exceeds capacity"
        );
        assert_eq!(
            k_tensor.len(),
            seq_len * self.kv_inner,
            "k_tensor length mismatch"
        );
        assert_eq!(
            v_tensor.len(),
            seq_len * self.kv_inner,
            "v_tensor length mismatch"
        );

        let byte_len = seq_len * self.kv_inner;
        self.k_cache[layer][0][..byte_len].copy_from_slice(k_tensor);
        self.v_cache[layer][0][..byte_len].copy_from_slice(v_tensor);

        self.seq_len = self.seq_len.max(seq_len);
    }
}

/// Build a `CImageTensorEntry` that references the state-store schema
/// for KV cache dimensions.
///
/// The returned entry describes a virtual "KV cache" tensor whose payload
/// is the state-store schema JSON blob. This is used in the cimage manifest
/// to declare KV cache residency without embedding actual tensor data.
pub fn build_bitnet_kv_manifest_entry(
    num_layers: usize,
    kv_inner: usize,
    seq_len: usize,
) -> CImageTensorEntry {
    let data_len = num_layers * kv_inner * seq_len;
    CImageTensorEntry {
        tensor_id: "kv_cache".into(),
        tensor_key: "state_store.kv_cache".into(),
        tensor_class: "KvCache".into(),
        logical_shape: vec![num_layers as u32, kv_inner as u32, seq_len as u32],
        source_dtype: DType::F32,
        codec: CodecFamily::RawF32,
        precision_plan: None,
        physical_layout: PhysicalTileLayout {
            tile_m: 1,
            tile_n: (kv_inner * seq_len) as u32,
            tiles_per_row: 1,
            total_tiles: num_layers as u32,
            padded_cols: (kv_inner * seq_len) as u32,
            group_size: 0,
            groups_per_tile: 0,
            packed_bytes_per_tile: (data_len / num_layers * 4) as u32,
            metadata_f32_per_tile: 0,
        },
        payload_ref: CImagePayloadRef::Single {
            payload_id: "state_store_schema".into(),
        },
        raw_f32_reference_ref: None,
        tensor_sha256: String::new(),
        validation_digest: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitnet_kv_append_read_roundtrip() {
        let num_layers = 2;
        let kv_inner = 256; // e.g., 4 kv_heads × 64 head_dim
        let max_seq_len = 8;

        let mut cache = BitNetKvCache::new(num_layers, kv_inner, max_seq_len);
        assert_eq!(cache.seq_len, 0);
        assert_eq!(cache.num_layers, num_layers);
        assert_eq!(cache.kv_inner, kv_inner);
        assert_eq!(cache.max_seq_len, max_seq_len);

        // Append one token to layer 0.
        let k0: Vec<f32> = (0..kv_inner).map(|i| (i as f32) * 0.1).collect();
        let v0: Vec<f32> = (0..kv_inner).map(|i| (i as f32) * 0.2).collect();
        cache.append(0, 0, &k0, &v0);
        assert_eq!(cache.seq_len, 1);

        // Read it back.
        let (rk, rv) = cache.read(0, 0);
        assert_eq!(rk, k0.as_slice());
        assert_eq!(rv, v0.as_slice());

        // Append a second token.
        let k1: Vec<f32> = (0..kv_inner).map(|i| (i as f32) * 0.3).collect();
        let v1: Vec<f32> = (0..kv_inner).map(|i| (i as f32) * 0.4).collect();
        cache.append(0, 1, &k1, &v1);
        assert_eq!(cache.seq_len, 2);

        // Both positions readable.
        let (rk0, rv0) = cache.read(0, 0);
        let (rk1, rv1) = cache.read(0, 1);
        assert_eq!(rk0, k0.as_slice());
        assert_eq!(rv0, v0.as_slice());
        assert_eq!(rk1, k1.as_slice());
        assert_eq!(rv1, v1.as_slice());

        // Layer 1 is still empty (zeros).
        let (rk_l1, rv_l1) = cache.read(1, 0);
        assert!(rk_l1.iter().all(|&v| v == 0.0));
        assert!(rv_l1.iter().all(|&v| v == 0.0));

        // --- extend_prefill ---
        let kv_inner2 = 256;
        let mut cache2 = BitNetKvCache::new(1, kv_inner2, 8);
        let prefill_len = 3;
        let k_pre: Vec<f32> = (0..prefill_len * kv_inner2)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let v_pre: Vec<f32> = (0..prefill_len * kv_inner2)
            .map(|i| (i as f32) * 0.02)
            .collect();
        cache2.extend_prefill(0, &k_pre, &v_pre, prefill_len);
        assert_eq!(cache2.seq_len, prefill_len);

        // Each position should match the original slices.
        for pos in 0..prefill_len {
            let (rk, rv) = cache2.read(0, pos);
            let start = pos * kv_inner2;
            let end = start + kv_inner2;
            assert_eq!(rk, &k_pre[start..end]);
            assert_eq!(rv, &v_pre[start..end]);
        }
    }

    #[test]
    fn test_bitnet_kv_manifest_entry() {
        let entry = build_bitnet_kv_manifest_entry(30, 640, 4096);
        assert_eq!(entry.tensor_id, "kv_cache");
        assert_eq!(entry.tensor_class, "KvCache");
        assert_eq!(entry.logical_shape, vec![30, 640, 4096]);
        assert_eq!(entry.codec, CodecFamily::RawF32);
        assert!(matches!(
            entry.payload_ref,
            CImagePayloadRef::Single { ref payload_id } if payload_id == "state_store_schema"
        ));
    }
}
