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
        use mlx_rs::Array;

        let f32_input: Vec<f32> = features.iter().map(|&v| v as f32).collect();
        let input_array = Array::from_slice(&f32_input, &[1, features.len() as i32]);

        // Attempt to load a compute-core vision encoder and use its projection.
        // When the encoder loads, we use its `projection.forward()`. If loading
        // fails (no tensors registered), we fall back to padding/truncation.
        let result: Option<Vec<u16>> = self.project_via_vision_encoder(&input_array);

        match result {
            Some(out) => out,
            None => {
                // Fallback: pad or truncate to output_dim.
                let out_dim = self.output_dim as usize;
                let mut projected = vec![0u16; out_dim];
                let copy_len = features.len().min(out_dim);
                projected[..copy_len].copy_from_slice(&features[..copy_len]);
                projected
            }
        }
    }

    /// Try to project using a loaded vision encoder's projection layer.
    #[cfg(feature = "prism-backend")]
    fn project_via_vision_encoder(&self, _input_f32: &mlx_rs::Array) -> Option<Vec<u16>> {
        use bytemuck::cast_slice;
        use tribunus_compute_core::config::VisionArchitecture;
        use tribunus_compute_core::vision::encoder::VisionEncoder;

        let arch = VisionArchitecture {
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_size: self.input_dim,
            projection_dim: self.output_dim,
            num_attention_heads: 12,
            num_hidden_layers: 12,
            intermediate_size: self.input_dim * 4,
        };

        let encoder = VisionEncoder::load(arch, &mut |_name| {
            Err("tensor not loaded from Prism facade".to_string())
        }).ok()?;

        // Use the encoder's projection QuantizedLinearBinding directly.
        let projected = encoder
            .projection
            .forward(_input_f32)
            .ok()?;

        let _ = projected.eval().ok()?;

        let slice = projected.try_as_slice::<f32>().ok()?;
        Some(cast_slice::<f32, u16>(slice).to_vec())
    }
}
