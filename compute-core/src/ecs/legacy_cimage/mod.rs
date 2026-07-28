//! Engine-internal execution-plane home for the CImage V0 proof format.
//!
//! The constitutional surface for the engine-independent V0 file format
//! data types (header, footer, payload directory, receipt directory,
//! error, canonical-JSON helper) lives in `prism_ecs_compile::cimage_v0`.
//! This module is the engine-internal home for the higher-level CImage
//! operations (writer, loader, streaming_writer, manifest, sealed_v1,
//! validator, shard_builder, MLP reference, generation store, dashboard,
//! privacy, durability, compatibility, generation_api) that depend on
//! engine-internal types (`PrecisionPlan`, `PrivacyContract`,
//! `CompiledKernelArtifact`, `CimageGeneration`, `GenerationApi`, etc.).
//!
//! # Re-exports
//!
//! The five engine-agnostic data-type modules are re-exported from the
//! constitutional surface so engine callers can read them through the
//! `legacy_cimage` path. New code should prefer the
//! `prism_ecs_compile::cimage_v0::...` import path; the re-exports here
//! are the migration bridge.

pub mod canonical;
pub mod compatibility;
pub mod dashboard;
pub mod durability;
pub mod error;
pub mod generation_api;
pub mod generation_store;
pub mod header;
pub mod loader;
pub mod manifest;
pub mod mlp_reference;
pub mod payload;
pub mod privacy;
pub mod receipts;
pub mod sealed_v1;
pub mod shard_builder;
pub mod streaming_writer;
pub mod validate;
pub mod writer;

// Re-exports of the constitutional data types (see
// `prism_ecs_compile::cimage_v0`). Existing engine callers that import
// `crate::ecs::legacy_cimage::CImageError` etc. continue to work; new
// code should prefer the constitutional path.
pub use prism_ecs_compile::cimage_v0::{
    canonical_json_bytes, CImageError, CImageFooterV0, CImageHeaderV0, CImageLoadReceipt,
    CImagePayloadDirectoryV0, CImagePayloadEntry, CImagePayloadKind, CImageProofKind,
    CImageReceiptDirectoryV0, CImageReceiptEntry, CImageResult, CImageShardValidationReceipt,
    CImageValidationStatus, CImageWriteReceipt, EvidenceReceiptV0, PendingPayload, PendingReceipt,
    ReceiptEvidenceKind, CIMAGE_FORMAT_VERSION, CIMAGE_MAGIC,
};

// Re-exports of the engine-internal types (defined in submodules of
// this module).
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
pub use sealed_v1::{
    AbiIdentity, CanonicalManifest, KernelArtifactIdentity, SealedCimageBuilder,
    SealedCimageHeader, SealedCimageV1, SectionEntry, SectionKind, SegmentIdentity,
    TokenizerIdentity, ValidatedSealedCimage, SEALED_CIMAGE_MAGIC, SEALED_CIMAGE_VERSION,
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
