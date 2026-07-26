//! TTS (Text-To-Speech) module for Qwen3-TTS integration.
//!
//! Provides the full TTS pipeline:
//! - Talker: 28-layer AR decoder (reuses megakernel pattern)
//! - Code Predictor: 5-layer transformer for RVQ completion
//! - Mimi Codec: causal ConvNet decoder -> PCM waveform
//!
//! Qwen3-TTS is Apache 2.0 licensed.

pub mod code_predictor;
pub mod codec;
pub mod pipeline;
pub mod talker;
