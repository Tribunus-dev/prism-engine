//! Per-model-family bindings — pure data types and pure algorithms for
//! per-model-family schema, inspection, and graph construction.
//!
//! The engine-coupled implementations (real GGUF / safetensors
//! parsing, real model-graph construction against the canonical
//! model IR) stay engine-side at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/model_family/`.

pub mod gemma4_inspect;
pub mod gemma4_mtp_graph;
pub mod gemma4_unified;
pub mod qwen25_omni;

pub use gemma4_inspect::{
    inspect_gemma4_checkpoint, Gemma4Inspection, ModelConfig, SerializedSchema, SourceIdentity,
    SourceShardStream, TensorClassSummary, TensorEntry, TensorInventory,
};
pub use gemma4_mtp_graph::{MTPExecutionGraph, MTPEdge, MTPPhase, MTPShareContract};
pub use gemma4_unified::{Gemma4UnifiedSchema, TensorClassification};
pub use qwen25_omni::{Qwen25OmniSchema, Qwen25OmniTensorClass};
