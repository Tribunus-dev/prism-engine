//! Model loading helpers.
//!
//! Loads pre-compiled .mlmodelc models from the `.cimage` auxiliary
//! section (embedded model bytes) into CoreMlModel instances.

use crate::arena::DataType;
use super::Orchestrator;
use crate::arena::Arena;
use crate::compute_image::compaction;
use crate::coreml_bridge::CoreMlModel;
use metal::*;

impl Orchestrator {
    /// Load a pre-compiled compaction model from embedded .mlmodelc bytes.
    ///
    /// Writes the embedded bytes to a temporary `.mlmodelc` directory,
    /// loads via Core ML, and leaks the temp directory (OS cleans up on
    /// reboot). Falls back to JIT compilation if the embedded load fails.
    pub(crate) fn load_compaction_model(
        bytes: Option<&Vec<u8>>,
        num_kv_heads: u32,
        global_head_dim: u32,
        max_context: u32,
    ) -> Option<CoreMlModel> {
        if let Some(bytes) = bytes {
            // Write embedded model bytes to a temp .mlmodelc directory and load.
            // The temp dir is leaked (mem::forget) to keep the directory alive
            // for the model's lifetime — the OS cleans up on reboot.
            let load_embedded = || -> Option<CoreMlModel> {
                let tmp_dir = tempfile::TempDir::new().ok()?;
                let modelc_dir = tmp_dir.path().join("compaction.mlmodelc");
                std::fs::create_dir_all(&modelc_dir).ok()?;
                let model_path = modelc_dir.join("model.mlmodel");
                std::fs::write(&model_path, bytes).ok()?;
                let model = CoreMlModel::load_with_compute_units(
                    &modelc_dir.to_string_lossy(),
                    crate::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
                )
                .ok()?;
                eprintln!(
                    "[orchestrator] Loaded embedded compaction model ({} bytes)",
                    bytes.len()
                );
                std::mem::forget(tmp_dir); // keep dir alive for model lifetime
                Some(model)
            };
            match load_embedded() {
                Some(m) => Some(m),
                None => {
                    // Fall through to JIT compilation
                    compaction::compile_compaction_model_optimized(
                        num_kv_heads,
                        global_head_dim,
                        max_context,
                        compaction::DEFAULT_TARGET_COUNT,
                    )
                    .map(Some)
                    .unwrap_or_else(|e| {
                        eprintln!("[orchestrator] Compaction model fallback failed: {e}");
                        None
                    })
                }
            }
        } else {
            compaction::compile_compaction_model_optimized(
                num_kv_heads,
                global_head_dim,
                max_context,
                compaction::DEFAULT_TARGET_COUNT,
            )
            .map(Some)
            .unwrap_or_else(|e| {
                eprintln!("[orchestrator] Compaction model JIT compilation failed: {e}");
                None
            })
        }
    }

    /// Load a pre-compiled prefill model from embedded .mlmodel bytes.
    ///
    /// Writes the embedded bytes to a temporary `.mlmodelc` directory
    /// and loads via Core ML, leaking the temp directory for the model's
    /// lifetime.
    pub(crate) fn load_prefill_model(bytes: &[u8]) -> Option<CoreMlModel> {
        let tmp_dir = tempfile::TempDir::new().ok()?;
        let modelc_dir = tmp_dir.path().join("prefill.mlmodelc");
        std::fs::create_dir_all(&modelc_dir).ok()?;
        let model_path = modelc_dir.join("model.mlmodel");
        std::fs::write(&model_path, bytes).ok()?;
        let model = CoreMlModel::load_with_compute_units(
            &modelc_dir.to_string_lossy(),
            crate::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
        )
        .ok()?;
        eprintln!(
            "[orchestrator] Loaded compiled prefill model ({} bytes)",
            bytes.len()
        );
        std::mem::forget(tmp_dir);
        Some(model)
    }

    /// Pre-allocate compaction arenas given a loaded compaction model.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn allocate_compaction_arenas(
        compaction_model: &Option<CoreMlModel>,
        num_kv_heads: u32,
        global_head_dim: u32,
        max_context: u32,
    ) -> (
        Option<Arena>,
        Option<Arena>,
        Option<Arena>,
        Option<Arena>,
        Option<Arena>,
    ) {
        if compaction_model.is_none() {
            return (None, None, None, None, None);
        }

        // Aligned dimensions for [B, C, 1, S] NCHW layout with 64-byte alignment.
        let c = compaction::align_dim(num_kv_heads * global_head_dim, 2);
        let s_in = compaction::align_dim(max_context, 2);
        let s_out = compaction::align_dim(compaction::DEFAULT_TARGET_COUNT, 2);

        // Indices arena: a raw byte buffer sized for target_count Int32 values.
        let indices_byte_count = (compaction::DEFAULT_TARGET_COUNT as usize) * 4;
        let indices_arena = Arena::new_bytes(indices_byte_count as u32)
            .map_err(|e| eprintln!("[orchestrator] Failed to allocate indices arena: {e}"))
            .ok();

        // FP16 input arenas for per-layer KV data fed to the compaction model.
        // Shape: [B=1, C, 1, S_in] flattened = C * S_in FP16 elements.
        let k_in = Arena::new(1, c * s_in, DataType::Float16)
            .map_err(|e| eprintln!("[orchestrator] K input arena: {e}"))
            .ok();
        let v_in = Arena::new(1, c * s_in, DataType::Float16)
            .map_err(|e| eprintln!("[orchestrator] V input arena: {e}"))
            .ok();

        // FP16 output arenas for compacted KV.
        // Shape: [B=1, C, 1, S_out] flattened = C * S_out FP16 elements.
        let k_out = Arena::new(1, c * s_out, DataType::Float16)
            .map_err(|e| eprintln!("[orchestrator] compacted K arena: {e}"))
            .ok();
        let v_out = Arena::new(1, c * s_out, DataType::Float16)
            .map_err(|e| eprintln!("[orchestrator] compacted V arena: {e}"))
            .ok();

        (indices_arena, k_in, v_in, k_out, v_out)
    }
}
