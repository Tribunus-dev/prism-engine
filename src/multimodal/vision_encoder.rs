use super::ImageInput;

pub enum VisionArch {
    ClipVitL,        // CLIP ViT-L/14, 768 dim
    ClipVitBigG,     // CLIP ViT-bigG, 1280 dim
    SigLIP,          // SigLIP ViT, 1152 dim
    EvaVit,          // EVA ViT, 1408 dim (CogVLM)
    PixtralVit,      // Pixtral's ViT, 1024 dim
}

pub struct VisionEncoderConfig {
    pub arch: VisionArch,
    pub input_size: (u32, u32),
    pub patch_size: u32,
    pub num_layers: u32,
    pub hidden_dim: u32,
    pub num_heads: u32,
}

impl VisionEncoderConfig {
    pub fn encode(&self, _image: &ImageInput) -> Vec<u16> {
        // Dummy implementation of encoding an image into a sequence of tokens
        Vec::new()
    }
}
