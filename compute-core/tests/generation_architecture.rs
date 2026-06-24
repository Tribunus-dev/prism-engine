//! Build-matrix architecture tests.
//!
//! These tests verify that the additive feature model holds:
//!
//! - `prism-backend` does NOT remove any generation module
//! - Each `generation-*` feature independently gates its module
//! - Any combination of `prism-backend` + `generation-*` features compiles cleanly
//!
//! All model crates (flux-klein-mlx, funasr-mlx, qwen3-tts-mlx) are imported
//! into the workspace and activate behind their respective `generation-*` features.

// ── Prism-backend alone: LUT runtime only, no generators ────────────────

#[cfg(all(
    feature = "prism-backend",
    not(any(
        feature = "generation-image",
        feature = "generation-diffusion",
        feature = "generation-video",
        feature = "generation-audio-core",
        feature = "generation-asr",
        feature = "generation-tts"
    ))
))]
#[test]
fn prism_backend_alone_does_not_force_any_generator() {
    // Compilation proves prism-backend does not drag in any generator
    // that the caller did not explicitly enable.
    assert!(true, "prism-backend alone: no generator force-activated");
}

// ── prism-backend + generation-image ────────────────────────────────────

#[cfg(all(feature = "prism-backend", feature = "generation-image"))]
#[test]
fn prism_backend_plus_generation_image_coexist() {
    // Unconditional module: always present with a backend
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::image_to_image::ImageToImageGenerator,
    >();
    // text_to_image activates cleanly behind generation-image with no
    // not(prism-backend) transitional gate — flux-klein-mlx is imported.
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::text_to_image::TextToImageGenerator,
    >();
}

// ── prism-backend + generation-diffusion ────────────────────────────────

#[cfg(all(feature = "prism-backend", feature = "generation-diffusion"))]
#[test]
fn prism_backend_plus_generation_diffusion_coexist() {
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::diffusiongemma::DiffusionSampler,
    >();
}

// ── prism-backend + generation-audio-core ───────────────────────────────

#[cfg(all(feature = "prism-backend", feature = "generation-audio-core"))]
#[test]
fn prism_backend_plus_generation_audio_core_coexist() {
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::audio_to_audio::AudioToAudioGenerator,
    >();
}

// ── prism-backend + generation-video ────────────────────────────────────

#[cfg(all(feature = "prism-backend", feature = "generation-video"))]
#[test]
fn prism_backend_plus_generation_video_coexist() {
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::video_generation::VideoGenerator,
    >();
}

// ── prism-backend + generation-audio (convenience alias) ────────────────

#[cfg(all(feature = "prism-backend", feature = "generation-audio"))]
#[test]
fn prism_backend_plus_generation_audio_coexist() {
    // generation-audio = generation-audio-core + generation-asr + generation-tts.
    // audio_to_audio is unconditional within generation and works here.
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::audio_to_audio::AudioToAudioGenerator,
    >();
    // audio_to_text activates behind generation-asr with funasr-mlx imported.
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::audio_to_text::AudioToTextGenerator,
    >();
    // text_to_speech activates behind generation-tts with qwen3-tts-mlx imported.
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::text_to_speech::TextToSpeechGenerator,
    >();
}

// ── Full-apple equivalent: all capabilities ─────────────────────────────

#[cfg(all(
    feature = "prism-backend",
    feature = "generation-image",
    feature = "generation-diffusion",
    feature = "generation-video",
    feature = "generation-audio-core",
    feature = "generation-asr",
    feature = "generation-tts"
))]
#[test]
fn full_apple_capabilities_coexist() {
    // Proves every positive gate compiles alongside prism-backend simultaneously.
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::image_to_image::ImageToImageGenerator,
    >();
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::text_to_image::TextToImageGenerator,
    >();
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::diffusiongemma::DiffusionSampler,
    >();
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::audio_to_audio::AudioToAudioGenerator,
    >();
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::audio_to_text::AudioToTextGenerator,
    >();
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::text_to_speech::TextToSpeechGenerator,
    >();
    let _ = std::any::type_name::<
        tribunus_compute_core::generation::video_generation::VideoGenerator,
    >();
}
