//! CImage V0 proof format — constitutional surface.
//!
//! This module owns the canonical authority for the engine-independent
//! data types of the CImage V0 proof format: the fixed-size header and
//! footer, the typed error enum, the canonical-JSON helper, the payload
//! directory types, and the receipt directory types.
//!
//! # Authority boundary
//!
//! Higher-level CImage operations (writer, loader, validator, streaming
//! writer, shard builders, sealed v1, generation store, manifest types,
//! dashboard, privacy, durability, compatibility) depend on engine-
//! internal types (`PrecisionPlan`, `PrivacyContract`,
//! `CompiledKernelArtifact`, `CimageGeneration`, `GenerationApi`, etc.)
//! and live in the engine-internal home at
//! `compute-core/src/ecs/legacy_cimage/`. The engine-internal home
//! re-exports these constitutional primitives so existing engine
//! callers can use them.
//!
//! # Module layout
//!
//! - [`error`] owns the typed error enum and result alias.
//! - [`header`] owns the fixed-size V0 header and footer.
//! - [`payload`] owns the V0 payload directory types.
//! - [`receipts`] owns the V0 receipt directory types.
//! - [`canonical`] owns the canonical-JSON serialization helper.
//!
//! The split follows the constitutional module-cohesion rule: one
//! authority per file, named in the file's module doc.

pub mod canonical;
pub mod error;
pub mod header;
pub mod payload;
pub mod receipts;

// Re-export the public API surface.

pub use error::{CImageError, CImageResult};
pub use header::{CImageFooterV0, CImageHeaderV0, CIMAGE_FORMAT_VERSION, CIMAGE_MAGIC};
pub use payload::{
    CImagePayloadDirectoryV0, CImagePayloadEntry, CImagePayloadKind, PendingPayload,
    PendingReceipt,
};
pub use receipts::{
    CImageLoadReceipt, CImageProofKind, CImageReceiptDirectoryV0, CImageReceiptEntry,
    CImageShardValidationReceipt, CImageValidationStatus, CImageWriteReceipt, EvidenceReceiptV0,
    ReceiptEvidenceKind,
};
