//! ECS resource types for audio inference.
//!
//! [`AudioEncoderResource`] holds the loaded [`AudioEncoder`] for ASR
//! prefill processing at runtime.
//!
//! [`AudioEncoder`]: crate::audio::AudioEncoder

use crate::audio::AudioEncoder;
use crate::runtime::scheduling::component_id::{ResourceId, SchedulableResource};

/// Stable resource ID for the audio encoder singleton.
pub const AUDIO_ENCODER_RESOURCE: ResourceId = 21;

/// ECS resource wrapping an optional [`AudioEncoder`].
///
/// Inserted into the World once during initialization when an audio
/// model is loaded.  The [`AudioInferenceSystem`] reads this resource
/// during the Prefill stage to encode mel spectrograms into feature
/// tokens.
///
/// [`AudioInferenceSystem`]: crate::runtime::systems::audio::inference::AudioInferenceSystem
pub struct AudioEncoderResource {
    /// The loaded audio encoder, or `None` if no audio model is active.
    pub encoder: Option<AudioEncoder>,
}

impl AudioEncoderResource {
    /// Create a new resource with no encoder loaded.
    pub fn new() -> Self {
        Self { encoder: None }
    }

    /// Create a new resource wrapping a loaded encoder.
    pub fn with_encoder(encoder: AudioEncoder) -> Self {
        Self {
            encoder: Some(encoder),
        }
    }
}

impl Default for AudioEncoderResource {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulableResource for AudioEncoderResource {
    const RESOURCE_ID: ResourceId = AUDIO_ENCODER_RESOURCE;
    const NAME: &'static str = "AudioEncoderResource";
}
