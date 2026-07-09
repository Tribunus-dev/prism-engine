//! CImage runtime bridge — translates loaded cimage artifacts into runtime
//! execution plans, manages Metal buffer allocation, and runs MLP regions.
//!
//! This module bridges the cimage format (compute-core/src/cimage/) with the
//! existing execution_plan types (ScheduledKernelOp, ExecutionRegion, etc.)
//! and the Metal runtime (metal_runtime/).

pub mod error;
pub mod receipts;
pub mod resolver;
pub mod tensor_store;

#[cfg(feature = "metal-dispatch")]
pub mod bitnet_layer_resolver;

// Implementation modules — gated behind macos + metal-dispatch.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod lower_decoder;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod lower_mlp;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod metal_buffers;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod region_runner;

pub use error::{CImageRuntimeError, CImageRuntimeResult};
pub use receipts::{
    BandwidthEstimate, CImageBindingReceipt, CImageKernelBindingInfo, CImageLayerTiming,
    CImageLayerValidationReceipt, CImageModelExecutionReceipt, CImageRegionExecutionReceipt,
    DispatchSegmentTiming, PerKernelFamilyTiming,
};
pub use resolver::{CImageRuntimeResolver, CpuReferenceBundle, ResolvedMlpShardRuntime};
#[allow(unused_imports)]
pub use tensor_store::{
    MlpRegionExecutionMode, RuntimeTensor, RuntimeTensorPayload, RuntimeTensorStore,
};

#[cfg(feature = "metal-dispatch")]
pub use bitnet_layer_resolver::BitNetLayerTensorResolver;

// Re-export platform-specific items only when available.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub use lower_decoder::{CImageDecoderRegionPlan, DecoderShardRegionBuilder};
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub use lower_mlp::{CImageMlpRegionPlan, MlpShardRegionBuilder};
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub use metal_buffers::MetalCImageBufferStore;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub use region_runner::CImageMetalRegionRunner;
