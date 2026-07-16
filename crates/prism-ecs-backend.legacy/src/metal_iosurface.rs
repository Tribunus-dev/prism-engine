/// Metal IOSurface binding — binds Metal consumers/producers to cimage slot contracts.
///
/// Provides resource views that link Metal kernel execution to IOSurface-backed
/// arena slots, enabling zero-copy data exchange between compute lanes.
use crate::shared_event::SharedEventBinding;

/// Kind of Metal resource
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetalResourceKind {
    /// MTLBuffer — private GPU memory
    MTLBuffer,
    /// IOSurfaceBacked — shared IOSurface memory, accessible across processes/lanes
    IOSurfaceBacked,
    /// Texture — Metal texture
    Texture,
    /// Heap — MTLHeap allocated sub-resources
    Heap,
}

/// Metal resource format descriptor
#[derive(Debug, Clone)]
pub struct MetalResourceFormat {
    /// Data type string e.g. "float16", "float32"
    pub data_type: String,
    /// Pixel format for IOSurface, e.g. "r16Float"
    pub pixel_format: Option<String>,
    /// Whether the format is sRGB
    pub is_srgb: bool,
}

/// Metal resource view — binds a Metal kernel to an IOSurface slot
#[derive(Debug, Clone)]
pub struct MetalResourceView {
    /// Slot identifier within the arena
    pub slot_id: u32,
    /// Kind of Metal resource backing this view
    pub resource_kind: MetalResourceKind,
    /// Format descriptor for this resource
    pub resource_format: MetalResourceFormat,
    /// Byte offset into the underlying allocation
    pub byte_offset: u64,
    /// Length in bytes of the resource
    pub length: u64,
    /// Layout digest — opaque hash verifying the buffer layout contract
    pub layout_digest: String,
}

impl MetalResourceView {
    /// Verify that the layout digest matches an expected value.
    ///
    /// Returns `Ok(())` on match, or `Err` with a descriptive message
    /// detailing the slot, expected, and actual digests.
    pub fn verify_layout(&self, expected_digest: &str) -> Result<(), String> {
        if self.layout_digest != expected_digest {
            return Err(format!(
                "layout digest mismatch for slot {}: expected {}, got {}",
                self.slot_id, expected_digest, self.layout_digest
            ));
        }
        Ok(())
    }
}

/// Metal executable — a Metal kernel bound to specific arena slots.
///
/// Carries input and output resource views that describe how the kernel
/// accesses memory via IOSurface-backed slots.
pub struct MetalExecutable {
    /// Artifact identifier for the compiled Metal function
    pub artifact_id: String,
    /// Name of the Metal function within the artifact
    pub function_name: String,
    /// Digest of the pipeline state
    pub pipeline_digest: String,
    /// Input resource views
    pub input_views: Vec<MetalResourceView>,
    /// Output resource views
    pub output_views: Vec<MetalResourceView>,
    /// Shared-event waits/signals that guard IOSurface slot handoff.
    pub shared_event_bindings: Vec<SharedEventBinding>,
}

impl MetalExecutable {
    /// Create a new MetalExecutable with the given artifact, function, and pipeline digest.
    pub fn new(artifact_id: &str, function_name: &str, pipeline_digest: &str) -> Self {
        Self {
            artifact_id: artifact_id.to_string(),
            function_name: function_name.to_string(),
            pipeline_digest: pipeline_digest.to_string(),
            input_views: Vec::new(),
            output_views: Vec::new(),
            shared_event_bindings: Vec::new(),
        }
    }

    /// Add an input resource view.
    pub fn add_input_view(&mut self, view: MetalResourceView) {
        self.input_views.push(view);
    }

    /// Add an output resource view.
    pub fn add_output_view(&mut self, view: MetalResourceView) {
        self.output_views.push(view);
    }

    /// Attach one shared-event wait or signal to this executable.
    pub fn add_shared_event_binding(&mut self, binding: SharedEventBinding) {
        self.shared_event_bindings.push(binding);
    }

    /// Verify that all input views match the expected layout digests.
    ///
    /// Each entry in `expected_digests` is a `(slot_id, expected_digest)` pair.
    pub fn verify_inputs(&self, expected_digests: &[(u32, &str)]) -> Result<(), String> {
        for (slot_id, expected) in expected_digests {
            let view = self
                .input_views
                .iter()
                .find(|v| v.slot_id == *slot_id)
                .ok_or_else(|| format!("no input view for slot {}", slot_id))?;
            view.verify_layout(expected)?;
        }
        Ok(())
    }

    /// Verify that all output views match expected layout digests.
    ///
    /// Each entry in `expected_digests` is a `(slot_id, expected_digest)` pair.
    pub fn verify_outputs(&self, expected_digests: &[(u32, &str)]) -> Result<(), String> {
        for (slot_id, expected) in expected_digests {
            let view = self
                .output_views
                .iter()
                .find(|v| v.slot_id == *slot_id)
                .ok_or_else(|| format!("no output view for slot {}", slot_id))?;
            view.verify_layout(expected)?;
        }
        Ok(())
    }

    /// Bind the canonical NF4Tile640 triplet ABI for a shared-lane kernel.
    pub fn bind_nf4_tile640_triplet(
        &mut self,
        weight_slot: u32,
        weight_byte_offset: u64,
        scale_slot: u32,
        scale_byte_offset: u64,
        bias_slot: u32,
        bias_byte_offset: u64,
        weight_byte_length: u64,
        metadata_byte_length: u64,
        layout_digest: &str,
    ) {
        self.add_input_view(MetalResourceView {
            slot_id: weight_slot,
            resource_kind: MetalResourceKind::IOSurfaceBacked,
            resource_format: MetalResourceFormat {
                data_type: "uint8".into(),
                pixel_format: None,
                is_srgb: false,
            },
            byte_offset: weight_byte_offset,
            length: weight_byte_length,
            layout_digest: layout_digest.into(),
        });
        for (slot_id, byte_offset) in [
            (scale_slot, scale_byte_offset),
            (bias_slot, bias_byte_offset),
        ] {
            self.add_input_view(MetalResourceView {
                slot_id,
                resource_kind: MetalResourceKind::IOSurfaceBacked,
                resource_format: MetalResourceFormat {
                    data_type: "float32".into(),
                    pixel_format: None,
                    is_srgb: false,
                },
                byte_offset,
                length: metadata_byte_length,
                layout_digest: layout_digest.into(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_resource_view_layout_verify() {
        let format = MetalResourceFormat {
            data_type: "float16".to_string(),
            pixel_format: Some("r16Float".to_string()),
            is_srgb: false,
        };
        let view = MetalResourceView {
            slot_id: 0,
            resource_kind: MetalResourceKind::IOSurfaceBacked,
            resource_format: format,
            byte_offset: 0,
            length: 4096,
            layout_digest: "abc123".to_string(),
        };

        assert!(view.verify_layout("abc123").is_ok());
    }

    #[test]
    fn test_metal_resource_view_mismatch_rejected() {
        let format = MetalResourceFormat {
            data_type: "float32".to_string(),
            pixel_format: None,
            is_srgb: false,
        };
        let view = MetalResourceView {
            slot_id: 7,
            resource_kind: MetalResourceKind::MTLBuffer,
            resource_format: format,
            byte_offset: 256,
            length: 8192,
            layout_digest: "digest_a".to_string(),
        };

        let result = view.verify_layout("digest_b");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("slot 7"));
        assert!(err.contains("digest_b"));
        assert!(err.contains("digest_a"));
    }

    #[test]
    fn test_metal_executable_input_output_verification() {
        let format = MetalResourceFormat {
            data_type: "float16".to_string(),
            pixel_format: Some("r16Float".to_string()),
            is_srgb: false,
        };

        let mut exec = MetalExecutable::new("art_1", "kernel_main", "pipeline_digest_1");

        exec.add_input_view(MetalResourceView {
            slot_id: 0,
            resource_kind: MetalResourceKind::IOSurfaceBacked,
            resource_format: format.clone(),
            byte_offset: 0,
            length: 4096,
            layout_digest: "in_digest_0".to_string(),
        });
        exec.add_input_view(MetalResourceView {
            slot_id: 1,
            resource_kind: MetalResourceKind::IOSurfaceBacked,
            resource_format: format.clone(),
            byte_offset: 0,
            length: 4096,
            layout_digest: "in_digest_1".to_string(),
        });
        exec.add_output_view(MetalResourceView {
            slot_id: 3,
            resource_kind: MetalResourceKind::IOSurfaceBacked,
            resource_format: format,
            byte_offset: 0,
            length: 4096,
            layout_digest: "out_digest_3".to_string(),
        });

        // All input views match
        assert!(exec
            .verify_inputs(&[(0, "in_digest_0"), (1, "in_digest_1")])
            .is_ok());

        // All output views match
        assert!(exec.verify_outputs(&[(3, "out_digest_3")]).is_ok());

        // Missing slot in inputs
        let err = exec.verify_inputs(&[(99, "whatever")]).unwrap_err();
        assert!(err.contains("no input view for slot 99"));

        // Mismatched digest in inputs
        let err = exec.verify_inputs(&[(0, "wrong_digest")]).unwrap_err();
        assert!(err.contains("layout digest mismatch for slot 0"));

        // Missing slot in outputs
        let err = exec.verify_outputs(&[(7, "whatever")]).unwrap_err();
        assert!(err.contains("no output view for slot 7"));

        // Mismatched digest in outputs
        let err = exec.verify_outputs(&[(3, "bad_out")]).unwrap_err();
        assert!(err.contains("layout digest mismatch for slot 3"));
    }

    #[test]
    fn test_bind_nf4_tile640_triplet() {
        let mut exec = MetalExecutable::new("art_nf4", "fused_gemv_nf4_tile640_fp32", "pipe");
        exec.bind_nf4_tile640_triplet(4, 32, 5, 96, 6, 160, 320, 40, "nf4tile640-v1");

        assert_eq!(exec.input_views.len(), 3);
        assert_eq!(exec.input_views[0].slot_id, 4);
        assert_eq!(exec.input_views[0].resource_format.data_type, "uint8");
        assert_eq!(exec.input_views[0].byte_offset, 32);
        assert_eq!(exec.input_views[1].byte_offset, 96);
        assert_eq!(exec.input_views[2].byte_offset, 160);
        assert_eq!(exec.input_views[1].length, 40);
        assert_eq!(exec.input_views[2].length, 40);
    }
}
