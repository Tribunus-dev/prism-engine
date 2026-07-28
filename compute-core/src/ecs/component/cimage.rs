use crate::ecs::Component;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CImage component types — pure data structures for entity-bound cimage state.
// These mirror the format types in ecs/cimage but are ECS-component-safe
// (no file I/O dependencies, no memmap, no mutable constructors).
// ---------------------------------------------------------------------------

/// Fixed-size header fields extracted from a cimage V0 file header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageHeader {
    pub magic: u32,
    pub version: u32,
    pub num_tensors: u32,
}
impl Component for CImageHeader {}

/// Manifest of tensor entries for a cimage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageManifest {
    pub entries: Vec<CImageTensorEntry>,
    pub created_at: u64,
}
impl Component for CImageManifest {}

/// An execution or validation receipt bound to a cimage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CImageReceipt {
    pub receipt_type: String,
    pub data: Vec<u8>,
    pub fingerprint: String,
}
impl Component for CImageReceipt {}

/// Physical tile layout — describes how a tensor is tiled on the backend.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhysicalTileLayoutData {
    pub tile_m: u32,
    pub tile_n: u32,
    pub group_size: u32,
}
impl Component for PhysicalTileLayoutData {}

/// A payload that is pending write into a cimage artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingPayload {
    pub payload_id: String,
    pub bytes: Vec<u8>,
}
impl Component for PendingPayload {}

// ---------------------------------------------------------------------------
// Re-export the linked CImageTensorEntry type from the cimage module
// so consumers can reference it without deep path imports.
// ---------------------------------------------------------------------------

pub use crate::ecs::legacy_cimage::manifest::CImageTensorEntry;
