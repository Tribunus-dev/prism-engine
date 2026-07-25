//! Runtime model — multi-model registry accessors.
//!
//! This module owns the canonical authority for resolving a
//! specialised model from the namespaced [`crate::model_manifest`]
//! embedded in a [`RuntimeModel`]. The methods here are pure views
//! over the model manifest; they perform no dispatch and no I/O.

use prism_spatial_ir::execution_plan::FusedScheduleStep;

use super::super::RuntimeError;
use super::RuntimeModel;

impl RuntimeModel {
    /// Resolve a specialised model before dispatching any of its programs.
    pub fn select_model(
        &self,
        modality: crate::model_manifest::ModelModality,
    ) -> Result<&crate::model_manifest::ModelManifest, RuntimeError> {
        self.model_manifest
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
            })?
            .select_modality(modality)
            .map_err(RuntimeError::UnsupportedMode)
    }

    pub fn validate_model_io(
        &self,
        model_id: &str,
        inputs: &[&str],
        outputs: &[&str],
    ) -> Result<(), RuntimeError> {
        self.model_manifest
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
            })?
            .validate_io(model_id, inputs, outputs)
            .map_err(RuntimeError::UnsupportedMode)
    }

    pub fn validate_model_hardware(
        &self,
        model_id: &str,
        available: crate::model_manifest::HardwareCapabilities,
    ) -> Result<(), RuntimeError> {
        self.model_manifest
            .as_ref()
            .ok_or_else(|| {
                RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
            })?
            .validate_hardware(model_id, available)
            .map_err(RuntimeError::UnsupportedMode)
    }

    pub fn model_for_fused_step(
        &self,
        step: &FusedScheduleStep,
    ) -> Result<Option<&crate::model_manifest::ModelManifest>, RuntimeError> {
        let Some(model_id) = step.model_id.as_deref() else {
            return Ok(None);
        };
        let manifest = self.model_manifest.as_ref().ok_or_else(|| {
            RuntimeError::UnsupportedMode("CImage has no multi-model manifest".into())
        })?;
        manifest.get(model_id).map(Some).ok_or_else(|| {
            RuntimeError::InvalidCImage(format!("unknown fused-step model {model_id:?}"))
        })
    }
}
