//! CImage module — CImage V0 proof format: writing, loading, validating, and
//! executing synthetic shards.
//!
//! This module is an independent crate-level module within compute-core.
//! It does not depend on `compute_image` or `compile` internals.

pub mod canonical;
pub mod compatibility;
pub mod dashboard;
pub mod error;
pub mod generation_api;
pub mod generation_store;
pub mod header;
pub mod manifest;
pub mod mlp_reference;
pub mod payload;
pub mod privacy;
pub mod receipts;

// Implementation modules — written after type definitions.
pub mod durability;
pub mod loader;
pub mod shard_builder;
pub mod streaming_writer;
pub mod validate;
pub mod writer;

// Public API surface.

pub use error::{CImageError, CImageResult};
pub use header::{CImageFooterV0, CImageHeaderV0, CIMAGE_FORMAT_VERSION, CIMAGE_MAGIC};
pub use loader::{CImageLoader, LoadedCImageV0};
pub use manifest::{
    AssistantGraphPayloadRef, CImageArtifactKind, CImageManifestV0, CImagePayloadRef,
    CImageReceiptRef, CImageTensorEntry, ModelExecutionPlanSummary, PhysicalTileLayout,
    StateStoreSchemaPayloadRef,
};
pub use mlp_reference::{
    compute_cosine_similarity, compute_max_abs_error, compute_nrmse,
    run_decoder_layer_rawf32_reference, run_mlp_rawf32_reference, run_mlp_reconstructed_reference,
    validate_decoder_layer_shard, LoadedMlpShardTensors,
};
pub use payload::{
    CImagePayloadDirectoryV0, CImagePayloadEntry, CImagePayloadKind, PendingPayload, PendingReceipt,
};
pub use receipts::{
    CImageLoadReceipt, CImageProofKind, CImageReceiptDirectoryV0, CImageReceiptEntry,
    CImageShardValidationReceipt, CImageValidationStatus, CImageWriteReceipt, EvidenceReceiptV0,
    ReceiptEvidenceKind,
};
pub use shard_builder::{
    DecoderLayerShardBuilder, MlpShardBuilder, PendingCImageShard, SyntheticDecoderLayerConfig,
    SyntheticDecoderPolicy, SyntheticMlpShardConfig, SyntheticShardPolicy,
};
pub use validate::CImageValidator;
pub use writer::CImageWriter;

/// Convenience function: emit a synthetic MLP shard cimage and validate it.
///
/// Equivalent to running `cimage emit-synthetic-mlp` followed by
/// `cimage validate` in sequence.
pub fn emit_and_validate_synthetic_mlp(
    path: &std::path::Path,
    config: SyntheticMlpShardConfig,
) -> CImageResult<(
    CImageWriteReceipt,
    CImageLoadReceipt,
    CImageShardValidationReceipt,
)> {
    // 1. Build the pending shard.
    let pending = MlpShardBuilder::build_synthetic_mlp_shard(config)?;

    // 2. Write cimage to disk.
    let write_receipt =
        CImageWriter::write_v0(path, pending.manifest, pending.payloads, pending.receipts)?;

    // 3. Load it back.
    let loaded = CImageLoader::load_v0(path)?;

    // 4. Validate.
    let load_receipt = CImageValidator::validate_loaded(&loaded)?;
    if load_receipt.validation_status == CImageValidationStatus::Invalid {
        return Err(CImageError::Other(format!(
            "cimage validation failed: {}",
            load_receipt.errors.join("; ")
        )));
    }

    // 5. Run numerical validation.
    let shard_validation =
        mlp_reference::validate_mlp_shard(&loaded, &write_receipt.cimage_digest)?;

    Ok((write_receipt, load_receipt, shard_validation))
}
