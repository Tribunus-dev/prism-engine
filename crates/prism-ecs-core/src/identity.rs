//! Canonical identity types.
//!
//! Every tensor, kernel, engram, candidate, and compiler artifact in the
//! system is uniquely identified by one of these newtype wrappers. The
//! identity hierarchy mirrors the provenance chain:
//!
//!   Source (ModelSourceId)
//!     → logical tensors
//!       → quantized representations
//!         → packed segments
//!           → kernel semantics
//!             → concrete implementations
//!               → engrams
//!                 → generations (GenerationId)
//!                   → receipts (ReceiptId)
//!
//! CompilerIdentity, HardwareProfileId, and Timestamp provide cross-cutting
//! identity for compilers, hardware targets, and temporal ordering.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SourceFormat { Raw, Gguf, SafeTensors, Onnx, Pytorch, Mlx }

/// Digest of the original model source and relevant sidecars.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelSourceId(pub String);

/// Digest of parent generation plus the complete promoted change set.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenerationId(pub String);

/// Digest of canonical receipt content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ReceiptId(pub String);

/// Compiler identity — name, version, and build metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompilerIdentity {
    pub name: String,
    pub version: String,
    pub build_hash: Option<String>,
    pub build_timestamp: Option<String>,
}

/// Hardware profile identifier.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct HardwareProfileId(pub String);

/// ISO 8601 timestamp wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub String);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorInfo { pub name: String, pub shape: Vec<usize>, pub dtype: String, pub size_bytes: u64 }

pub trait TensorReader: Send { fn read_chunk(&mut self, buffer: &mut [u8]) -> Result<usize, String>; fn shape(&self) -> &[usize]; fn size_bytes(&self) -> u64; }
pub trait TensorProvider: Send + Sync { fn list_tensors(&self) -> Result<Vec<TensorInfo>, String>; fn open_tensor(&self, name: &str) -> Result<Box<dyn TensorReader>, String>; }

pub trait GraphProvider: Send + Sync {
    fn import_graph(&self, world: &mut crate::world::World) -> Result<(), String>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceManifest { pub source_id: ModelSourceId, pub format: SourceFormat, pub path: Option<String> }
