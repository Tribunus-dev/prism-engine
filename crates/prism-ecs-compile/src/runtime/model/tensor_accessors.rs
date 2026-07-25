//! Runtime model — per-tensor data, shape, and representation accessors.
//!
//! This module owns the canonical authority for reading the per-tensor
//! payloads and per-tensor representation metadata on a loaded
//! [`RuntimeModel`]. The methods are pure views over the model fields;
//! they do not touch the file system or the backend.

use super::RuntimeModel;

impl RuntimeModel {
    /// Get a tensor's data by name.
    pub fn get_tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|v| v.as_slice())
    }

    /// Get the packed per-group scales for a native ternary tensor.
    pub fn get_tensor_scales(&self, name: &str) -> Option<&[u8]> {
        self.tensor_scales.get(name).map(|v| v.as_slice())
    }

    /// Return the validated MoE placement descriptor for a tensor.
    pub fn moe_descriptor(&self, name: &str) -> Option<&crate::cimage::MoeTensorDescriptor> {
        self.tensor_records
            .get(name)
            .and_then(|record| record.moe.as_ref())
    }

    /// Return the validated multimodal vision descriptor for a tensor.
    pub fn vision_descriptor(&self, name: &str) -> Option<&crate::cimage::VisionTensorDescriptor> {
        self.tensor_records
            .get(name)
            .and_then(|record| record.vision.as_ref())
    }
}
