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

pub mod mailbox;
pub mod ntb_dma;
pub mod workgroup;

/// CImage binary format types (ported from compute-core).
pub mod cimage_types;

/// KV cache and scheduler types (ported from compute-core).
pub mod inference;
pub mod kv_cache;
/// LLM inference server types (ported from src/llm/server.rs).
pub mod llm_server;
/// LLM inference runtime subsystem — session lifecycle, weight residency,
/// KV-cache management, lane dispatch, scheduling, cancellation,
/// memory pressure monitoring, receipt storage, and HTTP serving.
pub mod runtime;
/// HuggingFace tokenizer wrapper.
pub mod tokenizer;
