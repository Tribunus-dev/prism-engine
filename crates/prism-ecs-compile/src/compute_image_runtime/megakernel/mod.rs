//! Megakernel fusion — pure data types and pure algorithms for
//! megakernel fusion (concatenated multi-layer GPU kernel).
//!
//! The engine-coupled implementations (the actual Metal kernel
//! pipeline and the `Megakernel` runtime executor) stay engine-side
//! at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/megakernel/`.

pub mod gather_kernel;
pub mod kernels;
pub mod kv;
pub mod pipeline;

pub use gather_kernel::{GatherKernelInputs, GatherKernelOutput, GatherKernelStats};
pub use kernels::{
    compile_layer_library, CompileLayerLibraryStats, HIDDEN_DIM, KernelBuffers, LAYERS,
    MAX_CONTEXT, MAX_DRAFT_CANDIDATES, NUM_KV_HEADS, NUM_MTP_HEADS, TapMode,
};
pub use kv::{KvCacheSlot, KvSlotState};
pub use pipeline::{Megakernel, MegakernelPipelineConfig, MegakernelStage};
