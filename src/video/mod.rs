pub mod conv3d;
pub mod temporal_attention;
pub mod vae_3d;
pub mod frame_scheduler;
pub mod svd;
pub mod cogvideo;
pub mod mochi;
pub mod animatediff;

/// FrameIndex alias
pub type FrameIndex = u32;

/// New operation kinds for video generation
#[derive(Debug, Clone)]
pub enum VideoOp {
    // 3D convolution (spatial + temporal)
    Conv3d {
        in_ch: u32,
        out_ch: u32,
        kernel: (u32, u32, u32),  // (temporal, height, width)
        stride: (u32, u32, u32),
        padding: (u32, u32, u32),
    },
    
    // Temporal attention (across frames at same spatial position)
    TemporalAttention { dim: u32, heads: u32, causal: bool },
    
    // Joint spatial-temporal attention (full 3D self-attention)
    SpatialTemporalAttention { dim: u32, heads: u32 },
    
    // Frame interpolation
    FrameInterpolation { scale: u32 },
    
    // 3D VAE decode (video latent → video frames)
    VideoVaeDecode { latent_channels: u32, num_frames: u32 },
    
    // Frame-wise schedule for long video generation
    FrameSchedule {
        keyframe_interval: u32,     // generate keyframe every N frames
        interpolation_steps: u32,   // interpolate between keyframes
    },
}

#[derive(Debug, Clone)]
pub struct VideoPipeline {
    /// Total frames to generate
    pub num_frames: u32,
    /// Frames processed per denoising step (sliding window)
    pub window_size: u32,  // typically 4-8 frames
    /// Overlap between windows (for temporal consistency)
    pub overlap: u32,      // typically 1-2 frames
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
