//! Multimodal assembly receipt for reproducibility and evidence.

use crate::ecs::legacy_compute_image_core::multimodal::{
    MultimodalAssemblyReceipt, ProjectionBackend, ProjectionPrecision,
};

impl MultimodalAssemblyReceipt {
    /// Create a receipt from the results of multimodal prompt assembly.
    pub fn new(
        session_id: u64,
        prompt_digest: [u8; 32],
        processor_contract_digest: [u8; 32],
        modality_mask: u32,
        image_count: u32,
        image_soft_token_counts: Vec<u32>,
        assembled_sequence_len: u32,
        embedding_digest: [u8; 32],
        projection_backend: ProjectionBackend,
        projection_precision: ProjectionPrecision,
        elapsed_ns: u64,
    ) -> Self {
        Self {
            session_id,
            prompt_digest,
            processor_contract_digest,
            modality_mask,
            image_count,
            image_soft_token_counts,
            assembled_sequence_len,
            embedding_digest,
            projection_backend,
            projection_precision,
            elapsed_ns,
        }
    }

    /// Create a text-only receipt with no modality activity.
    pub fn text_only(session_id: u64, sequence_len: u32) -> Self {
        Self {
            session_id,
            prompt_digest: [0u8; 32],
            processor_contract_digest: [0u8; 32],
            modality_mask: 0,
            image_count: 0,
            image_soft_token_counts: Vec::new(),
            assembled_sequence_len: sequence_len,
            embedding_digest: [0u8; 32],
            projection_backend: ProjectionBackend::None,
            projection_precision: ProjectionPrecision::Unknown,
            elapsed_ns: 0,
        }
    }

    /// Returns true if this receipt involves multimodal processing.
    pub fn is_multimodal(&self) -> bool {
        self.modality_mask != 0
    }
}
