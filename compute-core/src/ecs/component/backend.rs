use crate::ecs::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendTarget {
    Metal,
    ROCm,
    CUDA,
    Vulkan,
    CPU,
}
impl Component for BackendTarget {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUArch {
    pub arch: String,
    pub compute_units: u32,
    pub wave_size: u32,
    pub max_lds_bytes: u32,
    pub max_registers: u32,
    pub memory_bw_gbs: f64,
}
impl Component for GPUArch {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningSpec {
    pub tile_shape: [u32; 3],
    pub vector_width: u32,
    pub unroll_factor: u32,
    pub lds_usage_bytes: u32,
    pub wave_limit: Option<u32>,
}
impl Component for TuningSpec {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShaderLanguage {
    MSL,
    HIP,
    SPIRV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSource {
    pub language: ShaderLanguage,
    pub source: String,
    pub entry_point: String,
}
impl Component for KernelSource {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BinaryFormat {
    Metallib,
    HSACO,
    SPIRV,
    LLVMBitcode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledBinary {
    pub format: BinaryFormat,
    pub data: Vec<u8>,
    pub fingerprint: String,
}
impl Component for CompiledBinary {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileConfig {
    pub mode: CompileMode,
    pub features: Vec<String>,
}
impl Component for CompileConfig {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompileMode {
    Debug,
    Coverage,
    Profiling,
    Optimized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AOTVariantRef(pub String);
impl Component for AOTVariantRef {}

/// Identifies the binary format and variant label for an Executable entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutableFormat {
    pub binary_format: BinaryFormat,
    pub variant_label: String,
}
impl Component for ExecutableFormat {}
