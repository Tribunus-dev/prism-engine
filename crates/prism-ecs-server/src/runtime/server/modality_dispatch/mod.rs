//! Modality dispatch — image, audio, video, embeddings, and multimodal routes.
//!
//! **Single authority:** This directory owns the canonical modality-routing
//! surface of the HTTP API: the `POST /v1/images/generate`,
//! `POST /v1/audio/speech`, `POST /v1/video/generate`,
//! `POST /v1/embeddings`, and `POST /v1/multimodal/generate` handlers, the
//! multimodal-request plan resolver and file-kind / manifest validators, the
//! `capture_live_media` envelope, and the vision-encoder matmul typed port
//! (`make_vision_matmul_provider`).
//!
//! **Sub-module authorities:**
//!
//! | Sub-module | Authority | Classification |
//! |---|---|---|
//! | [`image`] | image + video generation HTTP handlers and vision-encoder support | canonical |
//! | [`audio`] | text-to-speech HTTP handlers | canonical |
//! | [`embeddings`] | text-embedding HTTP handlers | canonical |
//! | [`multimodal`] | mixed-modality routing, plan resolution, capture envelope, manifest validation | canonical |
//!
//! **Canonical-vs-execution-boundary:** All types and functions in this
//! directory are canonical. The Metal matmul kernel that
//! `make_vision_matmul_provider` dispatches to when the `metal-dispatch`
//! feature is enabled is execution-boundary; the provider itself is the
//! typed port interface that bridges the canonical plan resolver to the
//! engine's `crate::engine::metal::dispatch_fp16_matmul`.

#[cfg(feature = "server")]
use crate::runtime::PrismInferenceServer;

#[cfg(feature = "server")]
pub(crate) type AppState = std::sync::Arc<PrismInferenceServer>;

#[cfg(feature = "server")]
pub(super) mod audio;
#[cfg(feature = "server")]
pub(super) mod embeddings;
#[cfg(feature = "server")]
pub(super) mod image;
#[cfg(feature = "server")]
pub(super) mod multimodal;

// Re-exports preserve the previous public surface of the flat
// `modality_dispatch.rs` so that the router in
// `super::request_handling` (and any external caller) keeps working
// without churn.
#[cfg(feature = "server")]
pub use audio::generate_audio;
#[cfg(feature = "server")]
pub use embeddings::generate_embeddings;
#[cfg(feature = "server")]
pub use image::{generate_image, generate_video};
#[cfg(feature = "server")]
pub use multimodal::generate_multimodal;

// =====================================================================
//  Vision-encoder matmul typed port (canonical, directory-level)
// =====================================================================

/// Canonical vision-encoder matmul provider.
///
/// **Typed port interface** to the Metal matmul kernel. The closure falls
/// back to a CPU implementation; when the `metal-dispatch` feature is
/// enabled it forwards to `crate::engine::metal::dispatch_fp16_matmul`,
/// which is the execution-boundary Metal kernel.
///
/// Kept at the directory level (this file) so the typed port is
/// discoverable from `modality_dispatch::*` without descending into the
/// image sub-module.
#[cfg(feature = "server")]
pub(crate) fn make_vision_matmul_provider()
-> prism_multimodal::multimodal::vision_encoder::MatmulProvider
{
    prism_multimodal::multimodal::vision_encoder::MatmulProvider {
        matmul: Box::new(|input, weight, dim_m, dim_n| {
            let m = dim_m as usize;
            let n = dim_n as usize;
            if input.len() != n || weight.len() < n * m {
                return Err("vision matmul dimension mismatch".into());
            }
            #[cfg(all(feature = "metal-dispatch", target_os = "macos"))]
            {
                let fp16_weights = weight
                    .iter()
                    .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
                    .collect::<Vec<_>>();
                if let Ok(output) = crate::engine::metal::dispatch_fp16_matmul(
                    "vision_encoder",
                    input,
                    &fp16_weights,
                    dim_m,
                    dim_n,
                ) {
                    return Ok(output);
                }
            }
            Ok((0..m)
                .map(|j| (0..n).map(|i| input[i] * weight[j * n + i]).sum())
                .collect())
        }),
    }
}
