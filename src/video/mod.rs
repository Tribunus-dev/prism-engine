pub mod conv3d;
pub mod frame_scheduler;
pub mod temporal_attention;
pub mod vae_3d;

/// FrameIndex alias
pub type FrameIndex = u32;

/// New operation kinds for video generation
#[derive(Debug, Clone)]
pub enum VideoOp {
    // 3D convolution (spatial + temporal)
    Conv3d {
        in_ch: u32,
        out_ch: u32,
        kernel: (u32, u32, u32), // (temporal, height, width)
        stride: (u32, u32, u32),
        padding: (u32, u32, u32),
    },

    // Temporal attention (across frames at same spatial position)
    TemporalAttention {
        dim: u32,
        heads: u32,
        causal: bool,
    },

    // Joint spatial-temporal attention (full 3D self-attention)
    SpatialTemporalAttention {
        dim: u32,
        heads: u32,
    },

    // Frame interpolation
    FrameInterpolation {
        scale: u32,
    },

    // 3D VAE decode (video latent → video frames)
    VideoVaeDecode {
        latent_channels: u32,
        num_frames: u32,
    },

    // Frame-wise schedule for long video generation
    FrameSchedule {
        keyframe_interval: u32,   // generate keyframe every N frames
        interpolation_steps: u32, // interpolate between keyframes
    },
}

#[derive(Debug, Clone)]
pub struct VideoPipeline {
    /// Total frames to generate
    pub num_frames: u32,
    /// Frames processed per denoising step (sliding window)
    pub window_size: u32, // typically 4-8 frames
    /// Overlap between windows (for temporal consistency)
    pub overlap: u32, // typically 1-2 frames
    /// Generated keyframes (every keyframe_interval frames)
    pub keyframes: Vec<FrameIndex>,
    /// Interpolation between keyframes
    pub interpolate: bool,
}

#[derive(Debug, Clone)]
pub struct FrameSchedule {
    pub keyframe_interval: u32,
    pub interpolation_steps: u32,
}

#[derive(Debug, Clone)]
pub struct VideoGenPipeline {
    pub pipeline: VideoPipeline,
}

// ── Prism Video facade ───────────────────────────────────────────────────

/// Parameters for text-to-video generation.
#[derive(Clone, Debug)]
pub struct VideoParams {
    /// Number of frames to generate.
    pub num_frames: u32,
    /// Target frame rate (for metadata / downstream encoding).
    pub fps: u32,
    /// Base seed for deterministic frame sequences.
    pub seed: u64,
}

/// Receipt for a completed video-generation request.
#[derive(Debug)]
pub struct VideoGenerationReceipt {
    /// Generated frames in temporal order.
    /// Each entry is `(width, height, flat RGBA8888 pixel data)`.
    pub frames: Vec<(u32, u32, Vec<u8>)>,
    /// Wall-clock compute time in milliseconds.
    pub compute_ms: f64,
}

/// Errors that can occur during video generation.
#[derive(Debug, thiserror::Error)]
pub enum PrismVideoError {
    #[error("video generation feature not enabled (add generation-video feature)")]
    MissingFeature,
    #[error("model not found at {0}")]
    ModelNotFound(String),
    #[error("generation failed: {0}")]
    GenerationFailed(String),
}

/// Generate a video from a text prompt.
///
/// This is the main entry point for text-to-video generation. It is always
/// available; when the `generation-video` feature is disabled it returns
/// [`PrismVideoError::MissingFeature`].
pub fn generate_video(
    model_path: &str,
    prompt: &str,
    params: VideoParams,
) -> Result<VideoGenerationReceipt, PrismVideoError> {
    #[cfg(feature = "generation-video")]
    {
        generate_via_compute_core(model_path, prompt, params)
    }

    #[cfg(not(feature = "generation-video"))]
    {
        let _ = (model_path, prompt, params);
        Err(PrismVideoError::MissingFeature)
    }
}

/// Delegate to the compute-core [`VideoGenerationProvider`].
///
/// Only compiled when the `generation-video` feature is enabled.
#[cfg(feature = "generation-video")]
fn generate_via_compute_core(
    model_path: &str,
    prompt: &str,
    params: VideoParams,
) -> Result<VideoGenerationReceipt, PrismVideoError> {
    use tribunus_compute_core::video_provider::{
        VideoGenerationProvider as _, VideoGenerationRequest, VideoProvider,
    };

    let provider = VideoProvider::new(model_path)
        .map_err(|e| PrismVideoError::ModelNotFound(e.to_string()))?;

    let request = VideoGenerationRequest {
        model_path: model_path.to_string(),
        prompt: prompt.to_string(),
        num_frames: params.num_frames,
        fps: params.fps,
        seed: params.seed,
    };

    let result = provider
        .generate_text_to_video(request)
        .map_err(|e| PrismVideoError::GenerationFailed(e.to_string()))?;

    Ok(VideoGenerationReceipt {
        frames: result.frames,
        compute_ms: result.compute_ms,
    })
}
