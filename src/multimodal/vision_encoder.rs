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
        // MLX-backed vision encoding has been retired from the canonical
        // build. Keep the legacy facade as a deterministic reshape/cast.
        let _ = (
            self.input_size,
            self.patch_size,
            self.num_layers,
            self.hidden_dim,
            self.num_heads,
        );
        bytemuck::cast_slice(&image.data).to_vec()
    }
}

/// Return the projection dimension for each vision architecture variant.
#[allow(dead_code)]
fn projection_dim_for_arch(arch: &VisionArch) -> u32 {
    match arch {
        VisionArch::ClipVitL => 768,
        VisionArch::ClipVitBigG => 1280,
        VisionArch::SigLIP => 1152,
        VisionArch::EvaVit => 1408,
        VisionArch::PixtralVit => 1024,
    }
}
