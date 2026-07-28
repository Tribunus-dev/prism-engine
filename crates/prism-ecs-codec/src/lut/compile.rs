//! `CompiledTensor` — the immutable AOT-compiled payload for a
//! single palettized LUT tensor.
//!
//! This module owns the canonical authority for the
//! `CompiledTensor` data type: the key used to look up the
//! tensor in a [`crate::lut::graph::ModelGraph`], the matmul
//! dimensions `(dim_m, dim_n)`, the on-disk payload bytes
//! (palettized codebook + packed 4-bit indices, see
//! [`crate::lut::table_builder`]), and the effective bits-per-
//! parameter metric reported by the AOT compiler.
//!
//! The hardware-specific AOT compile functions
//! (`compile_to_cimage`, `compile_gguf_to_cimage`) live in
//! the engine's compile path because they depend on
//! engine-specific I/O (CImageWriter, GGUF parser, palette
//! k-means); only the data type is constitutional.

/// The immutable AOT-compiled payload for a single palettized
/// LUT tensor. The `payload` bytes are content-addressed by the
/// tensor's position in a [`crate::lut::graph::ModelGraph`].
#[derive(Debug, Clone)]
pub struct CompiledTensor {
    /// The lookup key in the parent
    /// [`crate::lut::graph::ModelGraph`].
    pub key: String,
    /// Output dimension M (rows).
    pub dim_m: u32,
    /// Input dimension N (columns).
    pub dim_n: u32,
    /// On-disk palettized payload bytes
    /// (`[codebook × dim_m][packed 4-bit indices]`).
    pub payload: Vec<u8>,
    /// Effective bits-per-parameter reported by the AOT
    /// compiler; lower means more compression.
    pub effective_bpp: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_tensor_carries_key_and_dimensions() {
        let t = CompiledTensor {
            key: "model.layers.0.self_attn.q_proj.weight".to_string(),
            dim_m: 1024,
            dim_n: 1024,
            payload: vec![0u8; 64],
            effective_bpp: 4.5,
        };
        assert_eq!(t.key, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(t.dim_m, 1024);
        assert_eq!(t.dim_n, 1024);
        assert_eq!(t.payload.len(), 64);
        assert!((t.effective_bpp - 4.5).abs() < 1e-6);
    }

    #[test]
    fn compiled_tensor_clone_is_independent() {
        let t = CompiledTensor {
            key: "k".to_string(),
            dim_m: 8,
            dim_n: 8,
            payload: vec![1, 2, 3, 4],
            effective_bpp: 4.0,
        };
        let mut t2 = t.clone();
        t2.payload[0] = 99;
        assert_eq!(t.payload[0], 1, "original payload is untouched");
        assert_eq!(t2.payload[0], 99);
    }
}
