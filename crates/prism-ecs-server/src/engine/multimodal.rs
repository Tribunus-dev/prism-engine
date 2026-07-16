//! Multimodal streaming inference types.
//!
//! Ported from PrismBridgeFFI's StreamEvent/StreamCallback pattern.
//! Supports text, image, video, and audio streaming events.

use serde::Serialize;

/// Events streamed back during multimodal generation.
#[derive(Debug, Clone, Serialize)]
pub enum StreamEvent {
    Text {
        token: String,
        index: u64,
        metrics: TokenMetrics,
    },
    ImageFrame {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
    },
    VideoFrame {
        bytes: Vec<u8>,
        width: u32,
        height: u32,
        timestamp_ns: u64,
    },
    AudioChunk {
        bytes: Vec<u8>,
        sample_rate: u32,
        channels: u32,
    },
    Done {
        reason: String,
    },
    Error {
        message: String,
    },
}

/// Per-token metrics for streaming display.
#[derive(Debug, Clone, Serialize)]
pub struct TokenMetrics {
    pub tokens_per_sec: f64,
    pub time_ms: f64,
    pub layer: u64,
}

/// Input to multimodal generation.
#[derive(Debug, Clone)]
pub struct MultimodalInput {
    pub text: String,
    pub images: Vec<ImageInput>,
    pub audio: Option<AudioInput>,
}

#[derive(Debug, Clone)]
pub struct ImageInput {
    pub data: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone)]
pub struct AudioInput {
    pub data: Vec<u8>,
    pub sample_rate: u32,
    pub channels: u32,
}

/// Trait for receiving streaming multimodal events.
pub trait MultimodalCallback: Send {
    fn on_event(&mut self, event: StreamEvent);
    fn on_done(&mut self);
    fn on_error(&mut self, error: String);
}
