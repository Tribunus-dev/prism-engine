//! Modality adapter trait and model-family-specific implementations.

#![allow(dead_code)]

<<<<<<<< HEAD:compute-core/src/ecs/compute_image/legacy_compute_image_runtime/multimodal/adapter.rs
use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::{InputModality, ModalityError};
|||||||| e64c7d94:compute-core/src/ecs/compute_image/multimodal/adapter.rs
use crate::ecs::compute_image::multimodal::{InputModality, ModalityError};
========
use crate::ecs::legacy_compute_image_core::multimodal::{InputModality, ModalityError};
>>>>>>>> migrate/ci-core:compute-core/src/ecs/legacy_compute_image_core/multimodal/adapter.rs

/// Result of modality preparation — raw tensors ready for projection.
pub struct PreparedModality {
    pub modality: InputModality,
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

/// Result of modality projection — decoder-width embedding sequence.
pub struct EmbeddedModality {
    pub modality: InputModality,
    pub embeddings: Vec<f32>,
    pub sequence_len: u32,
    pub soft_token_count: u32,
}

/// Adapter that converts raw input into decoder-width embeddings.
pub trait ModalityAdapter: Send + Sync {
    fn modality(&self) -> InputModality;
    fn prepare(&self, input: &ModalityInput) -> Result<PreparedModality, ModalityError>;
    fn project(&self, prepared: &PreparedModality) -> Result<EmbeddedModality, ModalityError>;
    fn contract_digest(&self) -> [u8; 32];
}

/// Raw input to a modality adapter.
pub enum ModalityInput {
    Text {
        token_ids: Vec<u32>,
    },
    Image {
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        channels: u8,
    },
    Audio {
        samples: Vec<f32>,
        sample_rate: u32,
    },
}

// ── TokenEmbeddingAdapter ──────────────────────────────────────────

/// Adapter that looks up token embeddings from the vocabulary table.
/// This is the standard text path.
pub struct TokenEmbeddingAdapter {
    pub hidden_size: u32,
}

impl ModalityAdapter for TokenEmbeddingAdapter {
    fn modality(&self) -> InputModality {
        InputModality::Text
    }

    fn prepare(&self, input: &ModalityInput) -> Result<PreparedModality, ModalityError> {
        match input {
            ModalityInput::Text { token_ids } => Ok(PreparedModality {
                modality: InputModality::Text,
                data: Vec::new(),
                shape: vec![token_ids.len(), self.hidden_size as usize],
            }),
            _ => Err(ModalityError::UnsupportedModality(InputModality::Text)),
        }
    }

    fn project(&self, prepared: &PreparedModality) -> Result<EmbeddedModality, ModalityError> {
        Ok(EmbeddedModality {
            modality: InputModality::Text,
            embeddings: prepared.data.clone(),
            sequence_len: prepared.shape[0] as u32,
            soft_token_count: 0,
        })
    }

    fn contract_digest(&self) -> [u8; 32] {
        [0u8; 32]
    }
}

// ── LegacyVisionEncoderProjectorAdapter ────────────────────────────

/// Adapter for PaliGemma/LLaVA/Pixtral-class models that use a separate
/// ViT vision encoder followed by a projector.
pub struct LegacyVisionEncoderProjectorAdapter {
    pub hidden_size: u32,
    pub patch_size: u32,
    pub image_size: u32,
}

impl ModalityAdapter for LegacyVisionEncoderProjectorAdapter {
    fn modality(&self) -> InputModality {
        InputModality::Image
    }

    fn prepare(&self, input: &ModalityInput) -> Result<PreparedModality, ModalityError> {
        match input {
            ModalityInput::Image {
                pixels: _,
                width,
                height,
                channels: _,
            } => {
                let num_patches_w = *width / self.patch_size;
                let num_patches_h = *height / self.patch_size;
                Ok(PreparedModality {
                    modality: InputModality::Image,
                    data: Vec::new(),
                    shape: vec![
                        (num_patches_w * num_patches_h) as usize,
                        self.hidden_size as usize,
                    ],
                })
            }
            _ => Err(ModalityError::UnsupportedModality(InputModality::Image)),
        }
    }

    fn project(&self, prepared: &PreparedModality) -> Result<EmbeddedModality, ModalityError> {
        Ok(EmbeddedModality {
            modality: InputModality::Image,
            embeddings: Vec::new(),
            sequence_len: prepared.shape[0] as u32,
            soft_token_count: 0,
        })
    }

    fn contract_digest(&self) -> [u8; 32] {
        [0u8; 32]
    }
}

// ── Gemma4DirectImageProjectionAdapter ─────────────────────────────

/// Adapter for Gemma 4 Unified encoder-free image processing.
///
/// Stages:
///   1. Aspect-preserving resize to patch budget
///   2. Patchification
///   3. Direct patch embedding / projection
///   4. Learned 2D positional contribution
///   5. Pooling / soft-token compaction
///   6. Decoder-width embedding sequence output
pub struct Gemma4DirectImageProjectionAdapter {
    pub hidden_size: u32,
    pub patch_size: u32,
    pub pooling_kernel: u32,
    pub min_soft_tokens: u32,
    pub default_soft_tokens: u32,
    pub max_soft_tokens: u32,
    pub max_patch_count: u32,
    pub width_divisibility: u32,
    pub height_divisibility: u32,
}

impl Gemma4DirectImageProjectionAdapter {
    /// Create a new adapter with Gemma 4 12B Unified defaults.
    pub fn gemma4_12b_defaults() -> Self {
        Self {
            hidden_size: 3584,
            patch_size: 14,
            pooling_kernel: 2,
            min_soft_tokens: 64,
            default_soft_tokens: 280,
            max_soft_tokens: 1024,
            max_patch_count: 4096,
            width_divisibility: 48,
            height_divisibility: 48,
        }
    }
}

impl ModalityAdapter for Gemma4DirectImageProjectionAdapter {
    fn modality(&self) -> InputModality {
        InputModality::Image
    }

    fn prepare(&self, input: &ModalityInput) -> Result<PreparedModality, ModalityError> {
        match input {
            ModalityInput::Image {
                pixels: _,
                width,
                height,
                channels: _,
            } => {
                // Validate divisibility
                if width % self.width_divisibility != 0 {
                    return Err(ModalityError::AssemblyFailed(format!(
                        "image width {} not divisible by {}",
                        width, self.width_divisibility
                    )));
                }
                if height % self.height_divisibility != 0 {
                    return Err(ModalityError::AssemblyFailed(format!(
                        "image height {} not divisible by {}",
                        height, self.height_divisibility
                    )));
                }

                let patches_w = width / self.patch_size;
                let patches_h = height / self.patch_size;
                let num_patches = patches_w * patches_h;

                if num_patches > self.max_patch_count {
                    return Err(ModalityError::AssemblyFailed(format!(
                        "patch count {} exceeds budget {}",
                        num_patches, self.max_patch_count
                    )));
                }

                Ok(PreparedModality {
                    modality: InputModality::Image,
                    data: Vec::new(), // Raw pixels — projection handles conversion
                    shape: vec![
                        num_patches as usize,
                        (self.patch_size * self.patch_size * 3) as usize,
                    ],
                })
            }
            _ => Err(ModalityError::UnsupportedModality(InputModality::Image)),
        }
    }

    fn project(&self, prepared: &PreparedModality) -> Result<EmbeddedModality, ModalityError> {
        let _num_patches = prepared.shape[0] as u32;
        let soft_tokens = self
            .default_soft_tokens
            .min(self.max_soft_tokens)
            .max(self.min_soft_tokens);

        // Stub: actual projection requires Metal compute or CPU fallback.
        // Returns zero-filled decoder-width embeddings for now.
        let embedding_len = soft_tokens as usize * self.hidden_size as usize;

        Ok(EmbeddedModality {
            modality: InputModality::Image,
            embeddings: vec![0.0f32; embedding_len],
            sequence_len: soft_tokens,
            soft_token_count: soft_tokens,
        })
    }

    fn contract_digest(&self) -> [u8; 32] {
        [0u8; 32]
    }
}

// ── Gemma4DirectAudioProjectionAdapter ─────────────────────────────

/// Adapter for Gemma 4 Unified encoder-free audio processing (feature-gated).
pub struct Gemma4DirectAudioProjectionAdapter {
    pub hidden_size: u32,
    pub sample_rate: u32,
    pub frame_size_ms: u32,
    pub hop_size_ms: u32,
}

impl Gemma4DirectAudioProjectionAdapter {
    pub fn gemma4_12b_defaults() -> Self {
        Self {
            hidden_size: 3584,
            sample_rate: 16000,
            frame_size_ms: 25,
            hop_size_ms: 10,
        }
    }
}

impl ModalityAdapter for Gemma4DirectAudioProjectionAdapter {
    fn modality(&self) -> InputModality {
        InputModality::Audio
    }

    fn prepare(&self, input: &ModalityInput) -> Result<PreparedModality, ModalityError> {
        match input {
            ModalityInput::Audio {
                samples,
                sample_rate: _,
            } => {
                let frame_samples = (self.sample_rate * self.frame_size_ms / 1000) as usize;
                let num_frames = samples.len().div_ceil(frame_samples.max(1));
                Ok(PreparedModality {
                    modality: InputModality::Audio,
                    data: Vec::new(),
                    shape: vec![num_frames, frame_samples],
                })
            }
            _ => Err(ModalityError::FeatureGated(InputModality::Audio)),
        }
    }

    fn project(&self, prepared: &PreparedModality) -> Result<EmbeddedModality, ModalityError> {
        let num_frames = prepared.shape[0] as u32;
        let embedding_len = num_frames as usize * self.hidden_size as usize;
        Ok(EmbeddedModality {
            modality: InputModality::Audio,
            embeddings: vec![0.0f32; embedding_len],
            sequence_len: num_frames,
            soft_token_count: 0,
        })
    }

    fn contract_digest(&self) -> [u8; 32] {
        [0u8; 32]
    }
}
