use std::path::Path;
use anyhow::Result;

pub mod sd3;
pub mod flux;
pub mod sdxl;
pub mod scheduler_registry;

pub struct GpuInfo {
    pub id: usize,
    pub memory_gb: usize,
}

pub struct CImage {
    pub sharded: bool,
}

pub struct TextEncoderInfo {
    pub name: String,
    pub params_billion: f32,
}

pub struct VaeConfig {
    pub channels: u32,
}

pub enum Sd3Variant {
    Sd3_5,
    Sd3Medium,
}

pub enum FluxVariant {
    Dev,
    Schnell,
    Pro,
}

pub enum SdxlVariant {
    Base,
    Refiner,
}

pub enum DgVariant {
    Base,
}

pub enum ModelType {
    StableDiffusion3 { variant: Sd3Variant },
    Flux { variant: FluxVariant },
    Sdxl { variant: SdxlVariant },
    DiffusionGemma { variant: DgVariant },
    Custom { encoder: String, denoiser: String },
}

pub trait DiffusionModel: Send {
    fn model_type(&self) -> ModelType;
    fn steps(&self) -> (u32, u32, u32);
    fn latent_shape(&self) -> (u32, u32, u32);
    fn text_encoders(&self) -> Vec<TextEncoderInfo>;
    fn vae_config(&self) -> VaeConfig;
    fn guidance_range(&self) -> (f32, f32);
    fn has_cfg(&self) -> bool;
}

pub struct Metadata {}

pub mod diffusion_gemma {
    use super::*;
    pub struct DgCompiler;
    impl DgCompiler {
        pub fn compile(_gguf_path: &Path, _variant: DgVariant) -> Result<Metadata> {
            Ok(Metadata {})
        }
    }
}

pub fn compile_diffusion_model(
    gguf_path: &Path,
    model_type: ModelType,
    gpu_topology: &[GpuInfo],
) -> Result<CImage> {
    let metadata = match model_type {
        ModelType::Flux { variant } => {
            flux::FluxCompiler::compile(gguf_path, variant)?
        }
        ModelType::StableDiffusion3 { variant } => {
            sd3::Sd3Compiler::compile(gguf_path, variant)?
        }
        ModelType::Sdxl { variant } => {
            sdxl::SdxlCompiler::compile(gguf_path, variant)?
        }
        ModelType::DiffusionGemma { variant } => {
            diffusion_gemma::DgCompiler::compile(gguf_path, variant)?
        }
        ModelType::Custom { .. } => {
            anyhow::bail!("Custom model compilation not supported yet");
        }
    };
    
    build_sharded_cimage(metadata, gpu_topology)
}

fn build_sharded_cimage(_metadata: Metadata, _gpu_topology: &[GpuInfo]) -> Result<CImage> {
    Ok(CImage { sharded: true })
}
