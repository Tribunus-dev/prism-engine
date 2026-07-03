//! Level 2 Core ML teacher bridge — stateless teacher region dispatch via Core ML.
//!
//! Loads .mlmodelc bundles (compiled by the Phase 2 MIL → .mlmodelc pipeline)
//! and dispatches forward predictions using Core ML's `.cpuAndNeuralEngine`
//! compute unit policy. The GPU remains reserved for the student candidate
//! pipeline; Core ML uses CPU and Neural Engine exclusively.
//!
//! Each teacher region is stateless: the compiled model is cached by content
//! digest and reused across microbatches. On failure, the bridge produces a
//! receipt with `failure_reason` set and `actual_route = "Level1-Metal-fallback"`
//! so the scheduler can retry through the Level 1 dense Metal path.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use crate::arena_info::ArenaInfo;
use crate::coreai_bridge::{CoreAiComputeUnits, CoreAiModel};

use super::super::arena::StorageRoute;
use super::super::receipt::BridgeReceipt;

// ── CoreMLTeacher ────────────────────────────────────────────────────────────

/// The Level 2 Core ML teacher forward executor.
///
/// Loads a compiled .mlmodelc bundle for each teacher region (identified by
/// content digest) and dispatches Core ML prediction using `.cpuAndNeuralEngine`
/// compute units. Compiled models are cached in `model_cache` so repeated
/// forward passes for the same region do not reload from disk.
///
/// # Design notes
///
/// * Core ML output is treated as `CoreMLManaged` storage — the bridge never
///   assumes the buffer is Metal-aliasable.
/// * The scheduler asks a bridge provider to export Core ML output into a
///   Metal-readable student input route before consumption.
/// * `zero_copy_verified` defaults to `false` — Core ML bridge routes must
///   be explicitly verified before being promoted to zero-copy.
pub struct CoreMLTeacher {
    /// Compiled models keyed by content digest (hex string).
    model_cache: HashMap<String, CoreAiModel>,
    /// Base directory containing .mlmodelc bundles.
    model_dir: PathBuf,
}

impl CoreMLTeacher {
    /// Create a new Core ML teacher that loads models from `model_dir`.
    pub fn new(model_dir: impl Into<PathBuf>) -> Self {
        CoreMLTeacher {
            model_cache: HashMap::new(),
            model_dir: model_dir.into(),
        }
    }

    /// Return the `.mlmodelc` path for a given region digest.
    fn modelc_path(&self, digest: &str) -> PathBuf {
        self.model_dir.join(format!("{}.mlmodelc", digest))
    }

    /// Ensure the compiled model for `digest` is loaded and cached.
    ///
    /// Uses `CoreAiComputeUnits::CpuAndNeuralEngine` — permits CPU and Neural
    /// Engine execution while excluding the GPU. Core ML still decides what it
    /// can compile and may use CPU fallback for unsupported portions.
    fn load_model(&mut self, digest: &str) -> Result<&CoreAiModel, String> {
        // Fast path — already cached.
        if !self.model_cache.contains_key(digest) {
            let modelc_path = self.modelc_path(digest);
            let path_str = modelc_path
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 model path: {}", modelc_path.display()))?;

            let model = CoreAiModel::load_with_compute_units(
                path_str,
                CoreAiComputeUnits::CpuAndNeuralEngine,
            )
            .map_err(|e| format!("CoreMLTeacher: load failed for {}: {}", digest, e))?;

            self.model_cache.insert(digest.to_string(), model);
        }
        // Safe: we just inserted or it existed.
        Ok(self.model_cache.get(digest).unwrap())
    }

    /// Execute a teacher forward pass for the given region via Core ML.
    ///
    /// * `digest` — content digest identifying the teacher region / .mlmodelc.
    /// * `input_info` — `ArenaInfo` describing the input tensor (e.g. hidden
    ///   states from the calibration frontier).
    /// * `output_info` — `ArenaInfo` describing the output tensor buffer
    ///   (pre-allocated by the scheduler, Core ML writes into it).
    ///
    /// Returns a `BridgeReceipt` recording the execution. On failure, the
    /// receipt carries `actual_route = "Level1-Metal-fallback"` with a
    /// `failure_reason` so the scheduler can retry through dense Metal.
    pub fn forward(
        &mut self,
        digest: &str,
        input_name: &str,
        input_info: &ArenaInfo,
        output_name: &str,
        output_info: &ArenaInfo,
    ) -> BridgeReceipt {
        let start = Instant::now();
        let requested_route = "CoreML-cpuAndNeuralEngine".to_string();

        // Try to load the model (cached).
        let model = match self.load_model(digest) {
            Ok(m) => m,
            Err(e) => {
                return BridgeReceipt {
                    source_slot: 0,
                    destination_slot: 0,
                    requested_route,
                    actual_route: "Level1-Metal-fallback".into(),
                    materialized_bytes: 0,
                    cpu_copy_bytes: 0,
                    gpu_copy_bytes: 0,
                    bridge_latency_ns: start.elapsed().as_nanos() as u64,
                    zero_copy_verified: false,
                    verification_method: "none".into(),
                    failure_reason: Some(format!("model load failed: {}", e)),
                };
            }
        };

        // Run prediction.
        if let Err(e) = model.predict(input_name, input_info, output_name, output_info) {
            return BridgeReceipt {
                source_slot: 0,
                destination_slot: 0,
                requested_route,
                actual_route: "Level1-Metal-fallback".into(),
                materialized_bytes: 0,
                cpu_copy_bytes: 0,
                gpu_copy_bytes: 0,
                bridge_latency_ns: start.elapsed().as_nanos() as u64,
                zero_copy_verified: false,
                verification_method: "none".into(),
                failure_reason: Some(format!("prediction failed: {}", e)),
            };
        }

        let elapsed = start.elapsed().as_nanos() as u64;
        BridgeReceipt {
            source_slot: 0,
            destination_slot: 0,
            requested_route,
            actual_route: "CoreML-cpuAndNeuralEngine".into(),
            materialized_bytes: 0,
            cpu_copy_bytes: 0,
            gpu_copy_bytes: 0,
            bridge_latency_ns: elapsed,
            zero_copy_verified: false, // default — not assumed zero-copy
            verification_method: "default".into(),
            failure_reason: None,
        }
    }

    /// Produce a fallback receipt indicating a retry through Level 1 Metal.
    ///
    /// The scheduler uses this when Core ML is unavailable, model compilation
    /// failed, or the teacher region cannot be represented by Core ML.
    pub fn fallback_to_level1(reason: &str) -> BridgeReceipt {
        BridgeReceipt {
            source_slot: 0,
            destination_slot: 0,
            requested_route: "CoreML-cpuAndNeuralEngine".into(),
            actual_route: "Level1-Metal-fallback".into(),
            materialized_bytes: 0,
            cpu_copy_bytes: 0,
            gpu_copy_bytes: 0,
            bridge_latency_ns: 0,
            zero_copy_verified: false,
            verification_method: "none".into(),
            failure_reason: Some(reason.to_string()),
        }
    }

    /// Return the storage route for Core ML managed output.
    pub const fn output_storage_route() -> StorageRoute {
        StorageRoute::CoreMLManaged
    }
}

impl Default for CoreMLTeacher {
    fn default() -> Self {
        Self::new("models")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_receipt() {
        let receipt = CoreMLTeacher::fallback_to_level1("Core ML unavailable");
        assert_eq!(receipt.actual_route, "Level1-Metal-fallback");
        assert!(!receipt.zero_copy_verified);
        assert!(receipt.failure_reason.is_some());
    }

    #[test]
    fn test_default_model_dir() {
        let teacher = CoreMLTeacher::default();
        assert!(teacher.model_cache.is_empty());
    }
}
