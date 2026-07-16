// Local backend modules — implementations live in this directory.
// Types (DType, TensorHandle, TensorBackend, etc.) are re-exported from
// prism-ecs-backend at the bottom.

pub mod accelerate;
// Re-exports accelerate_ffi as a sibling so sibling modules can `use super::accelerate_ffi::*;`
pub mod accelerate_ffi;
pub mod accelerate_lane;

#[cfg(feature = "amd-rocm")]
pub mod amd_megakernel;
pub mod amd_rocm;
pub mod ane;
pub mod authority;
pub mod completion;
pub mod coreai;
pub mod coreai_iosurface;
pub mod coreai_lane;
pub mod cpu_attn;
pub mod evaluation;
pub mod flex_dispatch;
pub mod graph;
pub mod heterogeneous_executor;
pub mod intel_level_zero;
pub mod intel_usm;

#[cfg(target_os = "macos")]
pub mod megakernel_backend;

#[cfg(target_os = "macos")]
pub mod metal;
pub mod metal_consumer;
pub mod metal_iosurface;
pub mod npu;
pub mod placement;
pub mod routing;
pub mod shared_event;
pub mod tensor_registry;
pub mod unified_arena;

// Root-level re-exports from prism-ecs-backend (DType, TensorHandle,
// TensorBackend, etc.)
pub use prism_ecs_backend::*;

/// Create a HeterogeneousExecutor with all available backends registered.
///
/// Registers:
/// - MetalBackend (BackendId(0) = BACKEND_METAL)
/// - AneBackend (BackendId(2) = BACKEND_ANE)
///
/// Returns the executor with an empty operation registry (populated
/// from the cimage plan).
#[cfg(target_os = "macos")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub fn create_heterogeneous_executor(
) -> Result<crate::ecs::backend::heterogeneous_executor::HeterogeneousExecutor, String> {
    use crate::ecs::backend::ane::AneBackend;
    use crate::ecs::backend::heterogeneous_executor::HeterogeneousExecutor;
    use crate::ecs::backend::metal::MetalBackend;

    let mut executor = HeterogeneousExecutor::new();

    let metal = MetalBackend::new()?;
    executor.register(Box::new(metal));

    let ane = AneBackend::new();
    executor.register(Box::new(ane));

    Ok(executor)
}

/// Create a HeterogeneousExecutor with all backends for inference.
/// Loads the cimage and registers MegakernelBackend as the primary decode path.
#[cfg(target_os = "macos")]
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub fn create_inference_executor(
    cimage_path: impl AsRef<std::path::Path>,
    batch_size: u32,
    int4_mode: bool,
) -> Result<crate::ecs::backend::heterogeneous_executor::HeterogeneousExecutor, String> {
    use crate::ecs::backend::ane::AneBackend;
    use crate::ecs::backend::heterogeneous_executor::HeterogeneousExecutor;
    use crate::ecs::backend::megakernel_backend::MegakernelBackend;
    use crate::ecs::backend::metal::MetalBackend;

    let mut executor = HeterogeneousExecutor::new();

    // Register Megakernel fused decode backend (primary inference path)
    let megakernel = MegakernelBackend::from_cimage(cimage_path, batch_size, int4_mode)?;
    executor.register(Box::new(megakernel));

    // Register ANE backend (prefill, attention offload)
    let ane = AneBackend::new();
    executor.register(Box::new(ane));

    // Register Metal per-op backend (auxiliary ops)
    let metal = MetalBackend::new()?;
    executor.register(Box::new(metal));

    Ok(executor)
}
