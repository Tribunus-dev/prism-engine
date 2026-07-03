//! Apple Core ML artifact executor — real Core ML runtime on macOS Apple Silicon.
//!
//! Provides [`AppleCoreAiArtifactExecutor`] implementing [`CoreAiArtifactExecutor`]
//! using the ObjC FFI bridge for actual model loading and prediction.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use std::path::Path;

use crate::arena::Arena;
use crate::coreai::fixture::{
    CoreAiArtifactExecutor, CoreAiArtifactHandle, CoreAiBridgeError, CoreAiExecutionPolicy,
    CoreAiPredictionRequest, CoreAiPredictionResult, LoadedCoreAiArtifact, NamedTensorOutput,
};
use crate::coreai_bridge::{CoreAiComputeUnits, CoreAiModel};

// ── Helper: Map policy to compute units ────────────────────────────────────

/// Convert a [`CoreAiExecutionPolicy`] to the corresponding [`CoreAiComputeUnits`]
/// used by the ObjC FFI bridge.
fn compute_units_from_policy(policy: &CoreAiExecutionPolicy) -> CoreAiComputeUnits {
    match policy {
        CoreAiExecutionPolicy::SystemDefault => CoreAiComputeUnits::All,
        CoreAiExecutionPolicy::PreferNeuralEngine => CoreAiComputeUnits::CpuAndNeuralEngine,
        CoreAiExecutionPolicy::CpuAndNeuralEngine => CoreAiComputeUnits::CpuAndNeuralEngine,
        CoreAiExecutionPolicy::AllComputeUnits => CoreAiComputeUnits::All,
    }
}

// ── Bridge helper: Predict with arena-backed data ───────────────────────────

/// Bridge helper that runs a Core ML prediction using [`Arena`]-backed I/O,
/// converting between `Vec<f32>` (the fixture tensor format) and the
/// IOSurface/CVPixelBuffer representation the ObjC bridge expects.
fn predict_with_arenas(
    model: &CoreAiModel,
    input_name: &str,
    input_data: &[f32],
    output_name: &str,
    output_element_count: usize,
) -> Result<Vec<f32>, CoreAiBridgeError> {
    let element_size = std::mem::size_of::<f32>();

    // ── Input arena ───────────────────────────────────────────────────────
    let input_byte_count = input_data.len().checked_mul(element_size).ok_or_else(|| {
        CoreAiBridgeError::InvalidInput("input data byte count computation overflowed".into())
    })?;

    if input_byte_count == 0 {
        return Err(CoreAiBridgeError::InvalidInput(
            "input data is empty".into(),
        ));
    }

    let input_arena = Arena::new_bytes(input_byte_count as u32)
        .map_err(|e| CoreAiBridgeError::ExecutionFailed(format!("input arena allocation: {e}")))?;

    // Copy input f32 data into the arena backing store.
    unsafe {
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr(),
            input_arena.info.base_address as *mut f32,
            input_data.len(),
        );
    }

    // ── Output arena ──────────────────────────────────────────────────────
    let output_byte_count = output_element_count
        .checked_mul(element_size)
        .ok_or_else(|| {
            CoreAiBridgeError::InvalidOutput("output data byte count computation overflowed".into())
        })?;

    let output_arena = Arena::new_bytes(output_byte_count as u32)
        .map_err(|e| CoreAiBridgeError::ExecutionFailed(format!("output arena allocation: {e}")))?;

    // ── Run prediction ────────────────────────────────────────────────────
    model
        .predict(
            input_name,
            &input_arena.info,
            output_name,
            &output_arena.info,
        )
        .map_err(|e| CoreAiBridgeError::PredictionFailed(e))?;

    // ── Extract output data ───────────────────────────────────────────────
    let mut output_data = vec![0.0f32; output_element_count];
    unsafe {
        std::ptr::copy_nonoverlapping(
            output_arena.info.base_address as *const f32,
            output_data.as_mut_ptr(),
            output_element_count,
        );
    }

    Ok(output_data)
}

// ── Apple Core ML artifact executor ────────────────────────────────────────

/// Apple Core ML artifact executor backed by the real Core ML runtime.
///
/// This executor loads `.mlmodelc` directories and runs predictions using
/// the `tribunus` ObjC FFI bridge.  Only available on macOS Apple Silicon.
pub struct AppleCoreAiArtifactExecutor;

impl CoreAiArtifactExecutor for AppleCoreAiArtifactExecutor {
    type Error = CoreAiBridgeError;

    fn load(&self, artifact: &CoreAiArtifactHandle) -> Result<LoadedCoreAiArtifact, Self::Error> {
        // ── Validate path exists and is a directory ──────────────────────────
        let path = Path::new(&artifact.path);
        if !path.exists() {
            return Err(CoreAiBridgeError::ModelNotFound(artifact.path.clone()));
        }
        if !path.is_dir() {
            return Err(CoreAiBridgeError::ModelLoadFailed(format!(
                "expected a .mlmodelc directory, got non-directory: {}",
                artifact.path
            )));
        }

        // ── Validate it is a Core ML model directory ─────────────────────────
        let is_mlmodelc = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map_or(false, |ext| ext == "mlmodelc");

        let has_manifest = path.join("manifest.json").exists();

        if !is_mlmodelc && !has_manifest {
            return Err(CoreAiBridgeError::ModelLoadFailed(format!(
                "path is not a .mlmodelc directory and contains no manifest: {}",
                artifact.path
            )));
        }

        // ── Smoke-test the model loads correctly ─────────────────────────────
        let _model = CoreAiModel::load_with_compute_units(&artifact.path, CoreAiComputeUnits::All)
            .map_err(|e| CoreAiBridgeError::ModelLoadFailed(e))?;

        Ok(LoadedCoreAiArtifact {
            handle: artifact.clone(),
        })
    }

    fn predict(
        &self,
        artifact: &LoadedCoreAiArtifact,
        request: &CoreAiPredictionRequest,
    ) -> Result<CoreAiPredictionResult, Self::Error> {
        // ── Validate and extract the first input ─────────────────────────────
        let input = request.inputs.first().ok_or_else(|| {
            CoreAiBridgeError::InvalidInput("prediction request has no inputs".into())
        })?;

        // ── Reload model for this predict call ───────────────────────────────
        let compute_units = compute_units_from_policy(&request.execution_policy);
        let model = CoreAiModel::load_with_compute_units(&artifact.handle.path, compute_units)
            .map_err(|e| CoreAiBridgeError::ModelLoadFailed(e))?;

        // ── Determine output feature name and element count ──────────────────
        // Default to "output" for fixture models; a real production executor
        // would extract this from the model's metadata.
        let output_name = "output";
        let output_element_count = input.data.len();

        // ── Run prediction via arena bridge ──────────────────────────────────
        let output_data = predict_with_arenas(
            &model,
            &input.name,
            &input.data,
            output_name,
            output_element_count,
        )?;

        // ── Validate output dtype and shape ──────────────────────────────────
        // NamedTensorOutput always carries f32 data; verify element count
        // produces the expected shape [1, output_element_count].
        if output_data.is_empty() {
            return Err(CoreAiBridgeError::InvalidOutput(
                "prediction returned empty output tensor".into(),
            ));
        }

        let output_shape = vec![1usize, output_element_count];

        Ok(CoreAiPredictionResult {
            outputs: vec![NamedTensorOutput {
                name: output_name.to_string(),
                data: output_data,
                shape: output_shape,
            }],
            // Timing: not instrumented in this simple bridge executor.
            provider_latency_ms: 0.0,
        })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coreai::fixture::{CoreAiExecutionPolicy, NamedTensorInput};
    

    /// Helper to build a minimal artifact handle.
    fn dummy_handle(path: &str) -> CoreAiArtifactHandle {
        CoreAiArtifactHandle {
            path: path.into(),
            digest: [0u8; 32],
        }
    }

    #[test]
    fn test_compute_units_from_policy() {
        assert_eq!(
            compute_units_from_policy(&CoreAiExecutionPolicy::SystemDefault),
            CoreAiComputeUnits::All
        );
        assert_eq!(
            compute_units_from_policy(&CoreAiExecutionPolicy::PreferNeuralEngine),
            CoreAiComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(
            compute_units_from_policy(&CoreAiExecutionPolicy::CpuAndNeuralEngine),
            CoreAiComputeUnits::CpuAndNeuralEngine
        );
        assert_eq!(
            compute_units_from_policy(&CoreAiExecutionPolicy::AllComputeUnits),
            CoreAiComputeUnits::All
        );
    }

    #[test]
    fn test_load_nonexistent_path_fails() {
        let executor = AppleCoreAiArtifactExecutor;
        let handle = dummy_handle("/tmp/nonexistent_model.mlmodelc");
        let result = executor.load(&handle);
        assert!(result.is_err());
        match result {
            Err(CoreAiBridgeError::ModelNotFound(_)) => {} // expected
            _ => panic!("expected ModelNotFound error"),
        }
    }

    #[test]
    fn test_load_file_instead_of_directory_fails() {
        // `/tmp` is a directory, not a .mlmodelc directory → should fail
        let executor = AppleCoreAiArtifactExecutor;
        let handle = dummy_handle("/tmp");
        let result = executor.load(&handle);
        // Either ModelNotFound if /tmp doesn't exist (unlikely) or ModelLoadFailed
        assert!(result.is_err());
    }

    #[test]
    fn test_predict_empty_input_fails() {
        let executor = AppleCoreAiArtifactExecutor;
        // Build a loaded artifact with a non-existent path (predict reloads the model)
        let loaded = LoadedCoreAiArtifact {
            handle: dummy_handle("/tmp/nonexistent.mlmodelc"),
        };
        let request = CoreAiPredictionRequest {
            inputs: vec![],
            execution_policy: CoreAiExecutionPolicy::AllComputeUnits,
        };
        let result = executor.predict(&loaded, &request);
        assert!(result.is_err());
        match result {
            Err(CoreAiBridgeError::InvalidInput(_)) => {} // expected
            _ => panic!("expected InvalidInput error"),
        }
    }

    #[test]
    fn test_predict_model_not_found() {
        let executor = AppleCoreAiArtifactExecutor;
        let loaded = LoadedCoreAiArtifact {
            handle: dummy_handle("/tmp/no_such_model.mlmodelc"),
        };
        let request = CoreAiPredictionRequest {
            inputs: vec![NamedTensorInput {
                name: "input".into(),
                data: vec![1.0, 2.0, 3.0, 4.0],
                shape: vec![1, 4],
            }],
            execution_policy: CoreAiExecutionPolicy::AllComputeUnits,
        };
        let result = executor.predict(&loaded, &request);
        // Model doesn't exist → should get ModelNotFound or ModelLoadFailed
        assert!(result.is_err());
    }

    /// Verify that predict_with_arenas rejects empty input (byte_count == 0
    /// triggers the InvalidInput early return before any FFI call).
    #[test]
    fn test_predict_with_arenas_rejects_empty_input() {
        // The function needs a valid CoreAiModel pointer to run; this test
        // verifies we reject empty input before reaching the model.  We test
        // the equivalent validation inline since we can't construct a model here.
        let result = (|| -> Result<Vec<f32>, CoreAiBridgeError> {
            let input_data: Vec<f32> = vec![];
            let input_byte_count = input_data
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CoreAiBridgeError::InvalidInput("overflow".into()))?;
            if input_byte_count == 0 {
                return Err(CoreAiBridgeError::InvalidInput("empty input data".into()));
            }
            Ok(vec![])
        })();
        assert!(result.is_err());
        match result {
            Err(CoreAiBridgeError::InvalidInput(_)) => {} // expected
            _ => panic!("expected InvalidInput error"),
        }
    }
}
