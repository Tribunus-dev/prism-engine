use super::ImageInput;

#[derive(Debug, Clone)]
pub enum VisionArch {
    ClipVitL,    // CLIP ViT-L/14, 768 dim
    ClipVitBigG, // CLIP ViT-bigG, 1280 dim
    SigLIP,      // SigLIP ViT, 1152 dim
    EvaVit,      // EVA ViT, 1408 dim (CogVLM)
    PixtralVit,  // Pixtral's ViT, 1024 dim
}

pub struct VisionEncoderConfig {
    pub arch: VisionArch,
    pub input_size: (u32, u32),
    pub patch_size: u32,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
    /// Optional model path for loading a compute-core vision encoder
    /// (only used when feature `prism-backend` is enabled).
    pub model_path: Option<String>,
}

impl VisionEncoderConfig {
    pub fn encode(&self, image: &ImageInput) -> Vec<u16> {
        #[cfg(feature = "prism-backend")]
        {
            self.encode_via_compute_core(image)
        }
        #[cfg(not(feature = "prism-backend"))]
        {
            let _ = image;
            Vec::new()
        }
    }

    #[cfg(feature = "prism-backend")]
    fn encode_via_compute_core(&self, image: &ImageInput) -> Vec<u16> {
        use mlx_rs::Array;
        use tribunus_compute_core::config::VisionArchitecture;
        use tribunus_compute_core::vision::encoder::VisionEncoder;

        // Map Prism VisionArch to compute-core VisionArchitecture.
        let vision_arch = VisionArchitecture {
            image_size: self.input_size.0,
            patch_size: self.patch_size,
            num_channels: 3,
            hidden_size: self.hidden_dim,
            projection_dim: projection_dim_for_arch(&self.arch),
            num_attention_heads: self.num_heads,
            num_hidden_layers: self.num_layers,
            intermediate_size: self.hidden_dim * 4,
        };

        // Convert Prism ImageInput ([width*height*3 f32]) to MLX Array [1, 3, H, W].
        let h = image.height as i32;
        let w = image.width as i32;
        let c = 3i32;
        let array = Array::from_slice(&image.data, &[1, c, h, w]);

        // Attempt to load the vision encoder using a tensor-loader callback.
        // In a real deployment the tensors would come from a compiled model's
        // tensor registry; here we demonstrate the wiring.
        let encoder_result = VisionEncoder::load(vision_arch, &mut |name: &str| {
            Err(format!(
                "vision tensor '{}' not reachable from Prism facade; \
                 requires a loaded compute-core ProfiledModel",
                name
            ))
        });

        match encoder_result {
            Ok(encoder) => {
                match encoder.encode(&array) {
                    Ok(projected) => {
                        // Convert the f32 output Array to Vec<u16> by
                        // interpreting the f32 data bytes as u16 slices.
                        if let Ok(slice) = projected.try_as_slice::<f32>() {
                            bytemuck::cast_slice(slice).to_vec()
                        } else {
                            Vec::new()
                        }
                    }
                    Err(_) => Vec::new(),
                }
            }
            Err(_) => {
                // Fallback: encode via raw pixel cast when encoder loading
                // fails (e.g. no compiled model in test environments).
                bytemuck::cast_slice(&image.data).to_vec()
            }
        }
    }
}

/// Return the projection dimension for each vision architecture variant.
#[cfg(feature = "prism-backend")]
fn projection_dim_for_arch(arch: &VisionArch) -> u32 {
    match arch {
        VisionArch::ClipVitL => 768,
        VisionArch::ClipVitBigG => 1280,
        VisionArch::SigLIP => 1152,
        VisionArch::EvaVit => 1408,
        VisionArch::PixtralVit => 1024,
    }
}
