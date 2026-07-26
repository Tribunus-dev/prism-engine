//! Audio inference ECS systems — ASR encoding and TTS synthesis.
//!
//! These systems run during the Prefill stage, reading `WorkerRequest`
//! payloads and writing encoded features or synthesized audio into
//! `WorkerStream` for downstream consumption.

pub mod inference;

pub use inference::AudioInferenceSystem;
