//! ECS resource type for text-to-speech inference.
//!
//! [`TextToSpeechResource`] wraps an optional [`TextToSpeechGenerator`]
//! for TTS synthesis during the compute stage.
//!
//! [`TextToSpeechGenerator`]: crate::generation::text_to_speech::TextToSpeechGenerator

use crate::runtime::scheduling::component_id::{ResourceId, SchedulableResource};

/// Stable resource ID for the TTS generator singleton.
pub const TEXT_TO_SPEECH_RESOURCE: ResourceId = 22;

/// The concrete TTS generator type, defined only when the feature is active.
#[cfg(feature = "generation-tts")]
type TtsGenerator = crate::generation::text_to_speech::TextToSpeechGenerator;

/// ECS resource wrapping an optional [`TextToSpeechGenerator`].
///
/// Inserted into the World once during initialization when a TTS model
/// is loaded.  The [`AudioInferenceSystem`] reads this resource during
/// the Compute stage to synthesize audio from tokenized text.
///
/// [`AudioInferenceSystem`]: crate::runtime::systems::audio::inference::AudioInferenceSystem
pub struct TextToSpeechResource {
    /// The loaded TTS generator, or `None` if no TTS model is active.
    #[cfg(feature = "generation-tts")]
    pub generator: Option<TtsGenerator>,
    /// Placeholder field when TTS feature is disabled.
    #[cfg(not(feature = "generation-tts"))]
    pub generator: Option<()>,
}

impl TextToSpeechResource {
    /// Create a new resource with no generator loaded.
    pub fn new() -> Self {
        Self { generator: None }
    }

    /// Create a new resource wrapping a loaded generator.
    #[cfg(feature = "generation-tts")]
    pub fn with_generator(gen: TtsGenerator) -> Self {
        Self {
            generator: Some(gen),
        }
    }
}

impl Default for TextToSpeechResource {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulableResource for TextToSpeechResource {
    const RESOURCE_ID: ResourceId = TEXT_TO_SPEECH_RESOURCE;
    const NAME: &'static str = "TextToSpeechResource";
}
