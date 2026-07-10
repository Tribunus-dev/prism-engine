//! VisionEncoderResource — ECS singleton wrapping the VisionEncoder for
//! multimodal image encoding during the Prefill stage.
//!
//! Loaded from a compiled .cimage's vision encoder weight segments.
//! Systems check for this resource to determine whether vision inference
//! is available.

use crate::ecs::runtime::scheduling::component_id::{ResourceId, SchedulableResource};
use crate::vision::encoder::VisionEncoder;

/// Stable resource ID for VisionEncoderResource (21+ — next after NPU).
pub const VISION_ENCODER_RESOURCE: ResourceId = 21;

/// ECS resource that holds the loaded VisionEncoder.
///
/// Inserted into the World after a vision-model ComputeImage is loaded
/// and its vision encoder weights are parsed.  Systems targeting image
/// input check for this resource and short-circuit when it is absent.
pub struct VisionEncoderResource {
    /// The loaded vision encoder (None if loading failed or a text-only model).
    pub encoder: Option<VisionEncoder>,
}

impl VisionEncoderResource {
    /// Create a new resource wrapping an optional VisionEncoder.
    pub fn new(encoder: Option<VisionEncoder>) -> Self {
        Self { encoder }
    }
}

// ---------------------------------------------------------------------------
// SchedulableResource impl
// ---------------------------------------------------------------------------

impl SchedulableResource for VisionEncoderResource {
    const RESOURCE_ID: ResourceId = VISION_ENCODER_RESOURCE;
    const NAME: &'static str = "VisionEncoderResource";
}
