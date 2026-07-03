//! AudioInferenceSystem — ECS system driving audio encoding and TTS synthesis.
//!
//! Registered as ID 111 in the scheduler; runs during `Stage::Prefill`.
//!
//! Two processing paths:
//!
//! * **ASR** — reads raw audio path from `WorkerRequest::payload`,
//!   preprocesses to a mel spectrogram, runs the `AudioEncoder` forward pass
//!   to produce feature tokens, and stores the resulting MLX array handle
//!   in the [`AudioFeatures`] component for downstream text-model injection.
//!
//! * **TTS** — reads tokenized text from `WorkerRequest::payload`,
//!   invokes `TextToSpeechGenerator::synthesize()` to produce PCM float32
//!   samples, encodes them as WAV bytes, and writes the audio into
//!   `WorkerStream` for transport.

use lazy_static::lazy_static;

use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::{Entity, World};
use crate::runtime::components::{
    WorkerRequest, WorkerStream,
    worker_lifecycle::{WorkerLifecycle, WorkerRequestPhase},
    worker_request::RequestClass,
};
use crate::runtime::resources::audio::AudioEncoderResource;
use crate::runtime::resources::text_to_speech::TextToSpeechResource;

// ---------------------------------------------------------------------------
// AudioFeatures — marker component for entities with encoded audio features
// ---------------------------------------------------------------------------

/// Marker component emitted by `AudioInferenceSystem` after ASR encoding.
///
/// Carries the MLX array handle for the encoded feature tokens so that
/// downstream systems (e.g. text-model prefill) can inject them via
/// [`inject_audio_features`].
///
/// [`inject_audio_features`]: crate::audio::inject_audio_features
#[derive(Debug, Clone)]
pub struct AudioFeatures {
    /// Handle into `ARRAY_REGISTRY` for the encoded feature tensor.
    pub features_handle: crate::bridge::ArrayHandle,
    /// Number of audio frames (sequence length of the feature tensor).
    pub num_frames: usize,
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// ECS system that performs audio inference for streaming entities.
///
/// During the Prefill stage, this system scans entities that carry a
/// `WorkerRequest` with class [`AudioGen`](RequestClass::AudioGen).
/// For each such entity:
///
/// 1. If an `AudioEncoderResource` is present, the payload (ASR audio path)
///    → mel spectrogram → feature tokens, storing the result in an
///    [`AudioFeatures`] component.
///
/// 2. If a `TextToSpeechResource` is present (and ASR was not explicitly
///    requested), the payload (TTS text) → synthesize PCM → WAV bytes →
///    appended to [`WorkerStream`].
///
/// After processing, the entity's lifecycle transitions from `Queued` to
/// `AwaitingFirstEvent` so the downstream event-drain and streaming
/// pipeline continues.
pub struct AudioInferenceSystem {
    _private: (),
}

impl AudioInferenceSystem {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for AudioInferenceSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpec for AudioInferenceSystem {
    type Reads = (WorkerRequest, WorkerLifecycle);
    type Writes = (WorkerStream, WorkerLifecycle);
    type ReadResources = (AudioEncoderResource, TextToSpeechResource);
    type WriteResources = ();

    const NAME: &'static str = "audio_inference";
    const ID: SystemId = SystemId(111);
    const STAGE: Stage = Stage::Prefill;
    const ORDER: i32 = 0;
    const AFTER: &'static [SystemId] = &[];
    const BEFORE: &'static [SystemId] = &[];
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::Reject;
}

// ---------------------------------------------------------------------------
// Static metadata
// ---------------------------------------------------------------------------

lazy_static! {
    static ref AUDIO_INFERENCE_METADATA: SystemMetadata =
        match <AudioInferenceSystem as SystemSpec>::metadata() {
            Ok(m) => m,
            Err(e) => {
                panic!("AudioInferenceSystem metadata construction failed: {e}")
            }
        };
}

impl ErasedSystem for AudioInferenceSystem {
    fn metadata(&self) -> &SystemMetadata {
        &AUDIO_INFERENCE_METADATA
    }

    fn run(
        &mut self,
        world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // Check audio resources — skip if neither encoder nor TTS is loaded.
        let has_encoder = world
            .get_resource::<AudioEncoderResource>()
            .map(|r| r.encoder.is_some())
            .unwrap_or(false);
        #[cfg(feature = "generation-tts")]
        let has_tts = world
            .get_resource::<TextToSpeechResource>()
            .map(|r| r.generator.is_some())
            .unwrap_or(false);
        #[cfg(not(feature = "generation-tts"))]
        let has_tts = false;

        if !has_encoder && !has_tts {
            return SystemResult::Ok;
        }

        // Collect candidate entities: those with a WorkerRequest of class
        // AudioGen and a WorkerLifecycle in Queued phase.
        let candidates: Vec<Entity> = world
            .iter_entities_with::<WorkerRequest>()
            .filter(|entity| {
                let req = world.get::<WorkerRequest>(*entity);
                let lc = world.get::<WorkerLifecycle>(*entity);

                match (req, lc) {
                    (Some(r), Some(l)) => {
                        r.request_class == RequestClass::AudioGen
                            && l.phase == WorkerRequestPhase::Queued
                    }
                    _ => false,
                }
            })
            .collect();

        if candidates.is_empty() {
            return SystemResult::Ok;
        }

        // Process each candidate entity.
        for entity in candidates {
            let result = Self::process_entity(world, entity, has_encoder, has_tts);

            // On error, transition to Failed and continue with next entity.
            if let Err(msg) = result {
                if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                    let _ = lc.transition_to(WorkerRequestPhase::Failed);
                }
                eprintln!(
                    "[audio_inference] entity {} failed: {msg}",
                    entity.0
                );
            }
        }

        SystemResult::Ok
    }
}

impl AudioInferenceSystem {
    /// Process a single audio-gen entity.
    ///
    /// Returns `Ok(())` on success, or `Err(String)` with a description
    /// of what went wrong.
    fn process_entity(
        world: &mut World,
        entity: Entity,
        has_encoder: bool,
        has_tts: bool,
    ) -> Result<(), String> {
        // Clone payload to avoid borrowing world while mutating.
        let payload = world
            .get::<WorkerRequest>(entity)
            .ok_or_else(|| "WorkerRequest missing".to_string())?
            .payload
            .clone();

        // Determine processing path from payload prefix convention:
        //   "ASR:" prefix → audio encoding (Whisper/ASR)
        //   "TTS:" prefix → text-to-speech synthesis
        //   No prefix → ASR if encoder available, else TTS.
        let is_asr = if payload.starts_with(b"ASR:") {
            true
        } else if payload.starts_with(b"TTS:") {
            false
        } else if has_encoder {
            true
        } else if has_tts {
            false
        } else {
            return Err("no suitable audio resource available".to_string());
        };

        if is_asr {
            Self::run_asr(world, entity, &payload)?;
        } else {
            #[cfg(feature = "generation-tts")]
            {
                Self::run_tts(world, entity, &payload)?;
            }
            #[cfg(not(feature = "generation-tts"))]
            {
                return Err("TTS support not available (generation-tts feature disabled)".to_string());
            }
        }

        // Transition to AwaitingFirstEvent so the downstream pipeline
        // streams the result.
        if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
            lc.transition_to(WorkerRequestPhase::AwaitingFirstEvent)
                .map_err(|e| format!("lifecycle transition: {e}"))?;
        }

        Ok(())
    }

    /// Run ASR: preprocess audio → encode → write feature handle to
    /// [`AudioFeatures`] component.
    fn run_asr(
        world: &mut World,
        entity: Entity,
        payload: &[u8],
    ) -> Result<(), String> {
        // Extract the audio input from payload.
        let audio_input = if payload.starts_with(b"ASR:") {
            std::str::from_utf8(&payload[4..])
                .map_err(|e| format!("invalid ASR payload UTF-8: {e}"))?
                .trim()
                .to_owned()
        } else {
            std::str::from_utf8(payload)
                .map_err(|e| format!("invalid payload UTF-8: {e}"))?
                .trim()
                .to_owned()
        };

        // Borrow audio encoder resource (immutable borrow only).
        let encoder = world
            .get_resource::<AudioEncoderResource>()
            .and_then(|r| r.encoder.as_ref())
            .ok_or_else(|| "AudioEncoderResource not available".to_string())?;

        // Preprocess audio → mel spectrogram.
        let mel_spec = crate::audio::preprocess_audio(&audio_input, &encoder.config)
            .map_err(|e| format!("preprocess_audio: {e}"))?;

        // Encode mel → feature tokens.
        let features = encoder
            .encode(&mel_spec)
            .map_err(|e| format!("audio encode: {e}"))?;

        // Drop the encoder borrow before mutating world.
        let _ = encoder;

        // Store the feature tensor in the global array registry so downstream
        // text-model systems can retrieve it via AudioFeatures.handle.
        let num_frames = features.shape().get(0).copied().unwrap_or(0) as usize;
        let handle = crate::bridge::ARRAY_REGISTRY
            .write()
            .insert(features, None);

        // Insert AudioFeatures component onto the entity.
        let audio_feat = AudioFeatures {
            features_handle: handle,
            num_frames,
        };
        world.insert(entity, audio_feat);

        Ok(())
    }

    /// Run TTS: tokenize text → synthesize audio → write PCM to
    /// [`WorkerStream`].
    #[cfg(feature = "generation-tts")]
    fn run_tts(
        world: &mut World,
        entity: Entity,
        payload: &[u8],
    ) -> Result<(), String> {
        // Extract the text from payload.
        let text = if payload.starts_with(b"TTS:") {
            std::str::from_utf8(&payload[4..])
                .map_err(|e| format!("invalid TTS payload UTF-8: {e}"))?
                .trim()
                .to_owned()
        } else {
            std::str::from_utf8(payload)
                .map_err(|e| format!("invalid payload UTF-8: {e}"))?
                .trim()
                .to_owned()
        };

        // Borrow the TTS generator resource.
        let tts = world
            .get_resource::<TextToSpeechResource>()
            .and_then(|r| r.generator.as_ref())
            .ok_or_else(|| "TextToSpeechResource not available".to_string())?
            .clone();

        // Synthesize audio (blocking call on a tokio runtime).
        let rt = tokio::runtime::Handle::current();
        let (sample_rate, pcm_samples) = rt
            .block_on(tts.synthesize(&text, None))
            .map_err(|e| format!("TTS synthesize: {e}"))?;

        // Encode PCM as WAV bytes.
        let wav_bytes =
            crate::generation::text_to_speech::pcm_to_wav(&pcm_samples, sample_rate);

        // Write audio data to WorkerStream via record_output.
        if let Some(stream) = world.get_mut::<WorkerStream>(entity) {
            stream.record_output(None, wav_bytes.len() as u64);
        }

        Ok(())
    }
}
