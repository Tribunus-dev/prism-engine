//! Prism ECS server — RPCS3 Cell/B.E. absorption modules.
//!
//! Absorbs SPU mailbox/DMA/workgroup patterns into Prism's NTB/Tenstorrent
//! architecture:
//!
//! * `mailbox` — bounded message queues with blocking push/pop and waiter
//!   signalling, adapted from Cell SPU channels.
//! * `ntb_dma` — DMA transfer components and NoC dispatch system, adapted
//!   from Cell MFC commands (dma_put, dma_get, dma_sync).
//! * `workgroup` — SPU-style workgroup scheduling with work-unit lifecycle,
//!   adapted from lv2_spu_group.

/// Prism inference engine — model loading, inference dispatch, streaming,
/// multimodal, tokenization, and measured evaluation.
pub mod engine;
pub mod mailbox;
pub mod ntb_dma;
pub mod workgroup;

/// Runtime-facing tool surface (port of the engine's
/// `compute-core/src/ecs/tools/`; constitutional home — see
/// `changelogs/2026-07-27-engine-subsystem-deletion-tools.md`).
pub mod tools;

/// CImage binary format types (ported from compute-core).
pub mod cimage_types;

/// KV cache and scheduler types (ported from compute-core).
pub mod inference;
/// Per-image, per-session, and per-step inference state types
/// (constitutional home for the engine's
/// `compute-core/src/ecs/inference/*` module — see
/// `changelogs/2026-07-27-engine-subsystem-deletion-inference.md`).
pub mod inference_state;
pub mod kv_cache;
/// LLM inference server types (ported from src/llm/server.rs).
pub mod llm_server;
/// Tensor residency tracking for layer streaming.
pub mod residency;
/// LLM inference runtime subsystem — session lifecycle, weight residency,
/// KV-cache management, lane dispatch, scheduling, cancellation,
/// memory pressure monitoring, receipt storage, and HTTP serving.
pub mod runtime;
/// HuggingFace tokenizer wrapper.
pub mod tokenizer;
