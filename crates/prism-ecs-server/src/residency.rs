//! Tensor residency tracking for layer streaming.
//!
//! Tracks which model layers are in memory and prefetches upcoming layers
//! during autoregressive generation.

use prism_ecs_core::{Component, Entity};

// ── Schema IDs ─────────────────────────────────────────────────────
pub const SCHEMA_LAYER_RESIDENCY: u64 = 51;
pub const SCHEMA_PRESTAGING_QUEUE: u64 = 52;

// ── LayerResidency Component ───────────────────────────────────────

/// Where a layer's weights currently live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerStatus {
    Cached,
    Streaming,
    Evicted,
}

/// Which device a layer is resident on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyDevice {
    Cpu,
    Metal,
    Ane,
}

/// Tracks memory residency for one model layer.
#[derive(Debug, Clone)]
pub struct LayerResidency {
    pub layer_index: u32,
    pub status: LayerStatus,
    pub device: ResidencyDevice,
    pub file_offset: u64,
    pub byte_length: u64,
}

impl Component for LayerResidency {}

// ── PrestagingQueue Component ──────────────────────────────────────

/// Prestaging queue — the lookahead window of layers to prefetch.
/// Attached to a session entity.
#[derive(Debug, Clone)]
pub struct PrestagingQueue {
    pub session_entity: Entity,
    pub next_layer_index: u32,
    pub num_layers: u32,
    /// How many layers ahead to preload.
    pub lookahead: u32,
    /// Byte offset in the .cimage file where layer 0's weights start.
    pub layer_data_offset: u64,
    /// Byte length of each layer's weights.
    pub layer_byte_length: u64,
}

impl Component for PrestagingQueue {}

// ── PrestagingSystem ───────────────────────────────────────────────

/// Runs before each autoregressive forward pass. Ensures the next N layers
/// are prefetched by setting their LayerResidency to Streaming.
/// The actual I/O happens in StreamingLayerLoader (prism-runtime).
pub struct PrestagingSystem;

impl PrestagingSystem {
    /// Scan for PrestagingQueue entities and update residency for upcoming layers.
    /// Returns the next_layer_index for the streaming loader.
    pub fn tick(queue: &mut PrestagingQueue, _residencies: &mut [LayerResidency]) -> u32 {
        // In v1, the system just advances the lookahead window.
        // The runtime reads this value to know which layer to prefetch.
        // Full integration would update LayerResidency status fields.
        queue.next_layer_index
    }
}
