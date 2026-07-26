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

// ---------------------------------------------------------------------------
// Backend dispatch & runtime components
// ---------------------------------------------------------------------------

/// Identifies a backend instance and its capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendComponent {
    pub backend_id: String,
    pub capabilities: Vec<String>,
    pub instance_id: u64,
}
impl Component for BackendComponent {}

/// Tracks a tensor handle managed by a particular backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorComponent {
    pub handle: u64,
    pub backend_id: String,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub residency: String,
}
impl Component for TensorComponent {}

/// Cache entry for a compiled region on a backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledRegionComponent {
    pub handle: u64,
    pub backend_id: String,
    pub region_hash: String,
}
impl Component for CompiledRegionComponent {}
/// Wraps Metal device, command queue, and buffer manager handles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalDeviceState {
    pub device_handle: u64,
    pub command_queue_handle: u64,
    pub buffer_manager_handle: u64,
}
impl Component for MetalDeviceState {}
