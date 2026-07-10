pub struct ProjectorConfig {
    pub input_dim: u32,
    pub output_dim: u32,
    /// Optional model path for loading a compute-core projection layer
    /// (only used when feature `prism-backend` is enabled).
    pub model_path: Option<String>,
}

impl ProjectorConfig {
    pub fn forward(&self, features: &[u16]) -> Vec<u16> {
        #[cfg(feature = "prism-backend")]
        {
            self.forward_via_compute_core(features)
        }
        #[cfg(not(feature = "prism-backend"))]
        {
            features.to_vec()
        }
    }

    /// Run the projection through compute-core's MLX-backed linear transform.
    ///
    /// Converts the u16 input to f32, applies a linear projection
    /// (input_dim → output_dim), and converts back to u16.
    /// When model weights are unavailable (no loaded model), falls back
    /// to a truncated identity projection that pads/truncates to output_dim.
    #[cfg(feature = "prism-backend")]
    fn forward_via_compute_core(&self, features: &[u16]) -> Vec<u16> {
        // MLX-backed projection has been retired; keep the fallback behavior
        // as the canonical path for now.
        let out_dim = self.output_dim as usize;
        let mut projected = vec![0u16; out_dim];
        let copy_len = features.len().min(out_dim);
        projected[..copy_len].copy_from_slice(&features[..copy_len]);
        projected
    }
}
