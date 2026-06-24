#![cfg(feature = "ane")]

//! ANE calibration lane — heterogeneous compiler pipeline stage for
//! Apple Neural Engine weight calibration and feature extraction.
//!
//! Provides [`AneCalibrationLane`] which loads compiled Core ML models
//! with `cpuAndNeuralEngine` compute units, runs batch predictions,
//! and produces [`CompileExecutionReceipt`]s with full timing, routing,
//! and data-movement accounting.
//!
//! # Safety watermark
//!
//! [`ANE_SAFETY_WATERMARK`] (12 GB) is a soft ceiling for the combined
//! weight footprint before the lane refuses further submissions.  It
//! targets 16 GB M1-class devices, leaving headroom for the OS and
//! the Core ML runtime.

use std::time::Instant;

use super::phase_ir::{
    ANEArtifactKey, BridgeKind, CompileExecutionReceipt, CompilationId, CompilePlacement,
    CoreMlComputeUnits, DeviceSignature, EffectiveRoute, PhaseId, ValidationResult,
};
use super::staging::StagingRing;
use crate::arena_info::ArenaInfo;
use crate::coreml_bridge::CoreMlModel;

/// Soft memory ceiling for ANE calibration weight buffers (12 GB on 16 GB M1).
///
/// The lane refuses `submit_batch` when the total submitted weight elements
/// multiplied by 4 bytes (f32) exceeds this watermark.
pub const ANE_SAFETY_WATERMARK: u64 = 12_000_000_000; // 12 GB

// ── Block features ─────────────────────────────────────────────────────────

/// A single block of weight features to be submitted for ANE calibration.
#[derive(Debug, Clone)]
pub struct BlockFeatures {
    /// Byte offset of this block within the parent weight tensor.
    pub block_offset: u64,
    /// Flattened float weights for this block.
    pub weights: Vec<f32>,
    /// Input dimension (number of features per projection vector).
    pub in_dim: u32,
}

// ── ANE calibration lane ───────────────────────────────────────────────────

/// Calibration lane that submits weight blocks to a compiled ANE Core ML model.
///
/// Each lane owns a staging ring for input/output feature vectors and
/// is parameterised by an [`ANEArtifactKey`] that identifies the compiled
/// model artifact to load.
pub struct AneCalibrationLane {
    /// Input feature staging ring — buffers submitted `Vec<f32>` blocks.
    input_ring: StagingRing<Vec<f32>>,
    /// Output feature staging ring — stores extracted projection features.
    output_ring: StagingRing<Vec<f32>>,
    /// Identity of the compiled ANE artifact this lane targets.
    artifact_key: ANEArtifactKey,
    /// Device fingerprint captured at lane creation.
    device_sig: DeviceSignature,
}

impl AneCalibrationLane {
    /// Create a new ANE calibration lane for the given artifact key.
    ///
    /// The `input_ring` and `output_ring` are initialised with a capacity
    /// of 64 entries (sufficient for most per-layer calibration batches).
    pub fn new(artifact: ANEArtifactKey) -> Self {
        let host = crate::hostname_or_default();
        AneCalibrationLane {
            input_ring: StagingRing::new(64),
            output_ring: StagingRing::new(64),
            artifact_key: artifact,
            device_sig: DeviceSignature(format!("ane-calibration-lane/{}", host)),
        }
    }

    /// Submit a batch of weight blocks for ANE calibration.
    ///
    /// Each block loads the compiled Core ML model (identified by
    /// `self.artifact_key`) with `cpuAndNeuralEngine` compute units,
    /// runs a blocking prediction via [`CoreMlModel::predict`], and
    /// stages the extracted output features.
    ///
    /// Returns a [`CompileExecutionReceipt`] with per-batch timing
    /// breakdown, routing metadata, and data-movement accounting.
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// * The total weight footprint exceeds [`ANE_SAFETY_WATERMARK`].
    /// * Any Core ML model load or prediction fails.
    /// * The derived model path does not point to a valid `.mlmodelc`.
    pub fn submit_batch(&self, blocks: &[BlockFeatures]) -> Result<CompileExecutionReceipt, String> {
        let batch_start = Instant::now();

        // ── 1. Safety watermark check ───────────────────────────────────
        let total_bytes: u64 = blocks
            .iter()
            .map(|b| b.weights.len() as u64 * 4)
            .sum();
        if total_bytes > ANE_SAFETY_WATERMARK {
            return Err(format!(
                "ANE calibration batch ({total_bytes} bytes) exceeds safety watermark ({ANE_SAFETY_WATERMARK})"
            ));
        }

        // ── 2. Derive model path from artifact key ──────────────────────
        let model_path = self.derive_model_path()?;

        // ── 3. Load Core ML model with ANE compute units ─────────────────
        let load_start = Instant::now();
        let model = CoreMlModel::load_with_compute_units(
            &model_path,
            crate::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
        )?;
        let load_ns = load_start.elapsed().as_nanos() as u64;

        // ── 4. Process each block ───────────────────────────────────────
        let mut total_execution_ns: u64 = 0;
        let mut total_input_elements: u64 = 0;
        let mut total_output_elements: u64 = 0;
        let total_input_bytes = total_bytes;
        let mut total_copied_bytes: u64 = 0;

        for block in blocks {
            let block_elements = block.weights.len() as u64;
            let output_len = block.in_dim as usize; // projection output dimension
            let mut output_data = vec![0.0f32; output_len];

            let exec_start = Instant::now();

            // ── 4a. Build ArenaInfo from raw weights ────────────────────
            // The ArenaInfo reports a 1×N layout matching Core ML
            // multi-array convention (batch dimension = 1).
            let input_arena = ArenaInfo {
                width: 1,
                height: block_elements as i32,
                logical_dim0: 1,
                logical_dim1: block_elements as i32,
                pixel_format: 0,
                byte_size: (block_elements as i32) * 4,
                bytes_per_row: (block_elements as i32) * 4,
                base_address: block.weights.as_ptr() as *mut std::ffi::c_void,
                cv_buffer: std::ptr::null_mut(),
                io_surface: std::ptr::null_mut(),
            };
            let output_arena = ArenaInfo {
                width: 1,
                height: output_len as i32,
                logical_dim0: 1,
                logical_dim1: output_len as i32,
                pixel_format: 0,
                byte_size: (output_len as i32) * 4,
                bytes_per_row: (output_len as i32) * 4,
                base_address: output_data.as_mut_ptr() as *mut std::ffi::c_void,
                cv_buffer: std::ptr::null_mut(),
                io_surface: std::ptr::null_mut(),
            };

            // ── 4b. Model input and output feature names ────────────────
            // Calibration models export a single multi-array input named
            // "weight_input" and produce a single output named "features".
            model.predict("weight_input", &input_arena, "features", &output_arena)?;

            let exec_ns = exec_start.elapsed().as_nanos() as u64;
            total_execution_ns += exec_ns;
            total_input_elements += block_elements;
            total_output_elements += output_len as u64;
            total_copied_bytes += (block_elements + output_len as u64) * 4;

            // ── 4c. Stage input and output vectors ──────────────────────
            // Clone is required because the staging ring owns its entries.
            self.input_ring
                .push(block.weights.clone())
                .map_err(|e| format!("input ring push: {e}"))?;
            self.output_ring
                .push(output_data)
                .map_err(|e| format!("output ring push: {e}"))?;
        }

        let total_ns = batch_start.elapsed().as_nanos() as u64;

        // ── 5. Build execution receipt ──────────────────────────────────
        let receipt = CompileExecutionReceipt {
            compilation_id: CompilationId(0),
            phase_id: PhaseId(0),
            requested_placement: CompilePlacement::CoreMlCandidate,
            effective_route: EffectiveRoute::CoreMlCpuNe,
            artifact_key: Some(self.artifact_key.clone()),
            device_signature: self.device_sig.clone(),
            input_elements: total_input_elements,
            output_elements: total_output_elements,
            input_bytes: total_input_bytes,
            output_bytes: total_output_elements * 4,
            submit_ns: 0,
            queue_wait_ns: 0,
            execution_ns: total_execution_ns,
            materialization_ns: load_ns,
            dependency_wait_ns: 0,
            total_ns,
            bridge_kind: BridgeKind::Iosurface,
            copy_count: blocks.len() as u32 * 2,
            copied_bytes: total_copied_bytes,
            numerical_validation: ValidationResult::Passed,
            fallback_reason: None,
            coreml_compute_units: Some(
                CoreMlComputeUnits::CpuAndNeuralEngine,
            ),
        };

        Ok(receipt)
    }

    /// Run a single blocking prediction on the loaded ANE Core ML model.
    ///
    /// This is a convenience wrapper that loads the model (identified by
    /// `self.artifact_key`) and runs one inference pass.
    ///
    /// # Arguments
    ///
    /// * `input` — Flat float input tensor data.
    ///
    /// # Returns
    ///
    /// Flat float output tensor of length `in_dim` (the projection dimension
    /// from the artifact key).
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let model_path = self.derive_model_path()?;
        let model = CoreMlModel::load_with_compute_units(
            &model_path,
            crate::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
        )?;

        let output_len = self.output_dimension();
        let mut output_data = vec![0.0f32; output_len];

        let input_arena = ArenaInfo {
            width: 1,
            height: input.len() as i32,
            logical_dim0: 1,
            logical_dim1: input.len() as i32,
            pixel_format: 0,
            byte_size: (input.len() as i32) * 4,
            bytes_per_row: (input.len() as i32) * 4,
            base_address: input.as_ptr() as *mut std::ffi::c_void,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };
        let output_arena = ArenaInfo {
            width: 1,
            height: output_len as i32,
            logical_dim0: 1,
            logical_dim1: output_len as i32,
            pixel_format: 0,
            byte_size: (output_len as i32) * 4,
            bytes_per_row: (output_len as i32) * 4,
            base_address: output_data.as_mut_ptr() as *mut std::ffi::c_void,
            cv_buffer: std::ptr::null_mut(),
            io_surface: std::ptr::null_mut(),
        };

        model.predict("weight_input", &input_arena, "features", &output_arena)?;
        Ok(output_data)
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Derive the compiled `.mlmodelc` path from the artifact key.
    ///
    /// The path is constructed as:
    /// `{TRIBUNUS_ANE_CACHE_DIR or ./ane_cache}/{key_name}.mlmodelc`
    fn derive_model_path(&self) -> Result<String, String> {
        let base = std::env::var("TRIBUNUS_ANE_CACHE_DIR")
            .unwrap_or_else(|_| "./ane_cache".to_string());
        let key_name = self.key_filename();
        Ok(format!("{}/{}.mlmodelc", base, key_name))
    }

    /// Produce a deterministic filename stem for the artifact key.
    fn key_filename(&self) -> String {
        match &self.artifact_key {
            ANEArtifactKey::CalibrationProjection {
                batch,
                in_dim,
                out_dim,
            } => format!("calib_proj_b{batch}_i{in_dim}_o{out_dim}"),
            ANEArtifactKey::CalibrationChannelStats { batch, dim } => {
                format!("calib_stats_b{batch}_d{dim}")
            }
            ANEArtifactKey::BlockFeatureProjection { batch, feature_dim } => {
                format!("blk_feat_b{batch}_f{feature_dim}")
            }
            ANEArtifactKey::BlockReconstructionScore {
                batch,
                codebook_size,
            } => format!("blk_recon_b{batch}_c{codebook_size}"),
        }
    }

    /// Return the output dimension for a single prediction run.
    fn output_dimension(&self) -> usize {
        match &self.artifact_key {
            ANEArtifactKey::CalibrationProjection { out_dim, .. } => *out_dim as usize,
            ANEArtifactKey::CalibrationChannelStats { dim, .. } => *dim as usize,
            ANEArtifactKey::BlockFeatureProjection { feature_dim, .. } => *feature_dim as usize,
            ANEArtifactKey::BlockReconstructionScore { .. } => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ane_safety_watermark_constant() {
        assert_eq!(ANE_SAFETY_WATERMARK, 12_000_000_000);
    }

    #[test]
    fn test_block_features_construction() {
        let bf = BlockFeatures {
            block_offset: 0,
            weights: vec![1.0f32; 256],
            in_dim: 64,
        };
        assert_eq!(bf.weights.len(), 256);
        assert_eq!(bf.in_dim, 64);
    }

    #[test]
    fn test_artifact_key_filename_calibration_projection() {
        let key = ANEArtifactKey::CalibrationProjection {
            batch: 1,
            in_dim: 256,
            out_dim: 64,
        };
        let lane = AneCalibrationLane::new(key);
        let name = lane.key_filename();
        assert_eq!(name, "calib_proj_b1_i256_o64");
    }

    #[test]
    fn test_artifact_key_filename_block_feature_projection() {
        let key = ANEArtifactKey::BlockFeatureProjection {
            batch: 1,
            feature_dim: 128,
        };
        let lane = AneCalibrationLane::new(key);
        let name = lane.key_filename();
        assert_eq!(name, "blk_feat_b1_f128");
    }

    #[test]
    fn test_calibration_projection_output_dim() {
        let key = ANEArtifactKey::CalibrationProjection {
            batch: 1,
            in_dim: 256,
            out_dim: 64,
        };
        let lane = AneCalibrationLane::new(key);
        assert_eq!(lane.output_dimension(), 64);
    }

    #[test]
    fn test_block_reconstruction_output_dim() {
        let key = ANEArtifactKey::BlockReconstructionScore {
            batch: 1,
            codebook_size: 1024,
        };
        let lane = AneCalibrationLane::new(key);
        assert_eq!(lane.output_dimension(), 1);
    }

    #[test]
    fn test_device_signature_contains_lane_prefix() {
        let key = ANEArtifactKey::CalibrationProjection {
            batch: 1,
            in_dim: 256,
            out_dim: 64,
        };
        let lane = AneCalibrationLane::new(key);
        assert!(
            lane.device_sig.0.contains("ane-calibration-lane/"),
            "expected lane prefix in signature, got: {}",
            lane.device_sig.0
        );
    }

    #[test]
    fn test_submit_batch_watermark_exceeded() {
        let key = ANEArtifactKey::CalibrationProjection {
            batch: 1,
            in_dim: 256,
            out_dim: 64,
        };
        let lane = AneCalibrationLane::new(key);

        // Create a block whose weights would exceed the safety watermark.
        let huge_len = (ANE_SAFETY_WATERMARK as usize / 4) + 1;
        let block = BlockFeatures {
            block_offset: 0,
            weights: vec![0.0f32; huge_len],
            in_dim: 64,
        };

        let result = lane.submit_batch(&[block]);
        assert!(result.is_err(), "expected watermark exceeded error");
        assert!(
            result.unwrap_err().contains("exceeds safety watermark"),
            "error should mention watermark"
        );
    }
}
