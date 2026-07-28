//! `prism_ecs_constitutional::config` — product-shape configuration.
//!
//! This module owns the canonical authority for the engine's
//! `ecs::config/` subsystem: model architectures (text, vision, audio),
//! the per-layer execution plan, the per-operation backend route,
//! compile-time quantization modes, the hardware target, the
//! planning-stage limits, the `config.json` parser, the server
//! `config.toml` shape, and the compile-target `CimageManifest`.
//!
//! Engine-coupled code (the safetensors I/O that backs
//! `resolve_namespace`, the engine-side re-exports, the engine
//! binaries that consume the parser output) stays engine-internal at
//! `compute-core/src/ecs/legacy_config/`. The constitutional surface
//! is the cross-platform, dependency-free home for the data types
//! and pure transformations.
//!
//! Submodules:
//! - [`architecture`] — text/vision/audio architecture types, attention /
//!   rope / MoE / diffusion / quantization configuration.
//! - [`compile_quant_mode`] — `CompileQuantMode` enumeration + parse/format.
//! - [`hardware_target`] — `HardwareTarget` enumeration + detection.
//! - [`model_execution_plan`] — `ModelExecutionPlan`, `ProloguePlan`,
//!   `LayerPlan`, `EpiloguePlan`, `FusedOperation`, ANE-island detection,
//!   speculative-decoding config.
//! - [`layer_plan`] — per-layer compile plan (`ExecutionSpec`,
//!   `LayerSpec`, `TensorBinding`, `TensorRole`, `PackedLinearShapes`)
//!   and the `compile` + `build_execution_plan` + `filter_spec_to_existing`
//!   pure-Rust routines.
//! - [`operation_route`] — per-operation backend routing.
//! - [`namespace_binding`] — the `NamespaceBinding` data type and the
//!   pure-Rust `resolve_namespace` routine (engine-internal I/O is the
//!   caller's responsibility).
//! - [`network`] — `ServerConfig` + section types + CLI/env/TOML
//!   merging + per-backend fusion plan generation.
//! - [`limits`] — `TensorDisposition`, `PlannedTensor`,
//!   `PlannedSegment`, `CompilationPlan`.
//! - [`parser`] — `parse_config` + `ModelManifest` + `CimageManifest`.
//! - [`error`] — `ConfigError` + `ConfigResult` (thiserror, no anyhow).
//!
//! # Authority
//!
//! The `config` surface is the canonical home for product-shape
//! configuration. The legacy engine surface at
//! `compute-core/src/ecs/legacy_config/` re-exports these types and
//! hosts the engine-coupled adapter code.

pub mod architecture;
pub mod compile_quant_mode;
pub mod error;
pub mod hardware_target;
pub mod layer_plan;
pub mod limits;
pub mod model_execution_plan;
pub mod namespace_binding;
pub mod network;
pub mod operation_route;
pub mod parser;

pub use error::{ConfigError, ConfigResult};
