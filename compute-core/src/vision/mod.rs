//! Multi-modal (vision) support for the inference engine.
//!
//! Provides image preprocessing, a ViT-style vision encoder, and
//! cross-attention injection for fusing vision features into the
//! text model's hidden state.
//!
//! ## Architecture
//!
//! ```text
//! Image ──► Preprocess ──► VisionEncoder ──► inject_vision_features ──► Text model
//!                │                │                      │
//!           resize,         patch embed,           cross-attn
//!           normalize       transformer            between vision
//!                           encoder layers          & text tokens
//! ```

pub mod cross_attn;
pub mod direct_projector;
pub mod encoder;
pub mod live_capture;
pub mod preprocess;

pub use cross_attn::{inject_vision_features, CrossAttentionLayer};
pub use direct_projector::project_image_with_loaded_model;
pub use encoder::VisionEncoder;
pub use live_capture::{prism_inject_live_frame_buffer, VisionProjectionConfiguration};
pub use preprocess::preprocess_image;
