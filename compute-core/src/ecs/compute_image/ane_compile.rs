#![cfg(all(target_os = "macos", any(feature = "mlx-backend", feature = "prism-backend")))]
//! Standalone ANE subgraph compilation for multimodal and MTP (multi-token
//! prediction) draft subgraphs.
//!
//! Each function builds a MIL program for a specific stateless ANE subgraph,
//! serialises it as an `.mlpackage`, and invokes `coremlcompiler` to produce
//! a `.mlmodelc` bundle ready for IOSurface-backed inference at runtime.
//!
//! Unlike the general-purpose `compile::coreai::compile_subgraph` which
//! dispatches on op-name strings, these functions accept concrete weights
//! and dimensions — they are direct MIL-level wrappers for the multimodal
//! encoder projections (vision, audio) and MTP draft decoder projections /
//! decoder layers.

use std::path::{Path, PathBuf};

use crate::ecs::compute_image::subgraph_mil::{build_draft_layer_mil, build_matmul_mil};
use crate::coreai_pipeline;
use crate::mlpackage::{self, ModelMeta};
use coreml_proto::proto::mil_spec;

// ═══════════════════════════════════════════════════════════════════════════
// Canonical SSA output names
// ═══════════════════════════════════════════════════════════════════════════

/// Canonical SSA output names for multimodal / MTP subgraphs.
///
/// These MUST match the actual SSA names produced by the corresponding
/// builder functions in [`crate::ecs::compute_image::subgraph_mil`].
/// `build_matmul_mil` always emits `"matmul_1"` as the sole output.
/// `build_draft_layer_mil` (phase2) is expected to emit `"draft_out"`.
mod ssa_names {
    /// Output SSA name for every `build_matmul_mil` program.
    pub const MATMUL_OUT: &str = "matmul_1";
    /// Output SSA name for `build_draft_layer_mil` programs.
    ///
    /// **MUST** match whatever `build_draft_layer_mil` emits as its
    /// final operation output.
    pub const DRAFT_LAYER_OUT: &str = "draft_out";
}

// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Write a MIL program to an `.mlpackage` and compile it via `coremlcompiler`
/// into a `.mlmodelc` bundle.
///
/// This is the shared compile pipeline for all multimodal/MTP subgraphs.
/// The caller specifies the exact input and output tensor names/shapes
/// so the `ModelMeta` accurately reflects the subgraph's interface.
fn compile_program(
    program: mil_spec::Program,
    name: &str,
    output_dir: &Path,
    input_name: &str,
    input_shape: &[i64],
    output_name: &str,
    output_shape: &[i64],
) -> Result<PathBuf, String> {
    // ── Write .mlpackage ───────────────────────────────────────────
    let mlpackage_dir = output_dir.join(format!("{}.mlpackage", name));
    let _ = std::fs::create_dir_all(&mlpackage_dir);

    let meta = ModelMeta {
        model_name: format!("tribunus-subgraph-{}", name),
        function_name: name.to_string(),
        short_description: format!("Multimodal/MTP ANE subgraph: {}", name),
        version: "1.0".into(),
        author: "Tribunus Compute".into(),
        output_name: output_name.to_string(),
        inputs: vec![(input_name.to_string(), input_shape.to_vec())],
        outputs: vec![(output_name.to_string(), output_shape.to_vec())],
    };

    let written_path = mlpackage::write_mlpackage(program, &mlpackage_dir, &meta)
        .map_err(|e| format!("mlpackage write failed for '{}': {}", name, e))?;

    // ── Compile via coremlcompiler ─────────────────────────────────
    let receipt = coreai_pipeline::compile_mlpackage(
        &written_path,
        output_dir,
        name,
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("coremlcompiler failed for '{}': {}", name, e))?;

    Ok(PathBuf::from(receipt.compiled_modelc_path))
}

// ═══════════════════════════════════════════════════════════════════════════
// Multimodal encoder projections
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a vision patch embedding ANE subgraph (stateless).
///
/// `patch_dense`: matmul `[N_patches, 3840] x [3840, 6912] -> [N_patches, 6912]`
///
/// Weights are supplied at compile time as `weights` (length `3840 * 6912`)
/// but bound as IOSurfaces at runtime (stateless).
pub fn compile_vision_patch_embed_ane(
    weights: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let program = build_matmul_mil(
        "image_patches",
        "patch_dense_weight",
        "patch_features",
        1,    // m: batch dimension (dynamic at runtime via symbolic)
        3840, // k: input dimension
        6912, // n: output dimension
        weights,
        true, // stateless
    )
    .map_err(|e| format!("vision patch embed MIL: {}", e))?;

    compile_program(
        program,
        "vision_patch_embed",
        output_dir,
        "image_patches",
        &[1, 3840],
        ssa_names::MATMUL_OUT,
        &[1, 6912],
    )
}

/// Compile a vision final projection ANE subgraph (stateless).
///
/// `embedding_projection`: matmul `[N_patches, 3840] x [3840, 3840] -> [N_patches, 3840]`
pub fn compile_vision_projection_ane(
    weights: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let program = build_matmul_mil(
        "patch_features",
        "embedding_projection_weight",
        "projected_features",
        1,    // m
        3840, // k
        3840, // n
        weights,
        true, // stateless
    )
    .map_err(|e| format!("vision projection MIL: {}", e))?;

    compile_program(
        program,
        "vision_projection",
        output_dir,
        "patch_features",
        &[1, 3840],
        ssa_names::MATMUL_OUT,
        &[1, 3840],
    )
}

/// Compile an audio frame embedding ANE subgraph (stateless).
///
/// `audio_frame_embed`: matmul `[N_frames, 128] x [128, 2560] -> [N_frames, 2560]`
pub fn compile_audio_embed_ane(
    weights: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let program = build_matmul_mil(
        "audio_frames",
        "audio_embed_weight",
        "encoded_frames",
        1,    // m
        128,  // k
        2560, // n
        weights,
        true, // stateless
    )
    .map_err(|e| format!("audio embed MIL: {}", e))?;

    compile_program(
        program,
        "audio_frame_embed",
        output_dir,
        "audio_frames",
        &[1, 128],
        ssa_names::MATMUL_OUT,
        &[1, 2560],
    )
}

/// Compile an audio projection ANE subgraph (stateless).
///
/// `audio_proj`: matmul `[N_frames, 2560] x [2560, 3840] -> [N_frames, 3840]`
pub fn compile_audio_projection_ane(
    weights: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let program = build_matmul_mil(
        "encoded_frames",
        "audio_proj_weight",
        "projected_frames",
        1,    // m
        2560, // k
        3840, // n
        weights,
        true, // stateless
    )
    .map_err(|e| format!("audio projection MIL: {}", e))?;

    compile_program(
        program,
        "audio_projection",
        output_dir,
        "encoded_frames",
        &[1, 2560],
        ssa_names::MATMUL_OUT,
        &[1, 3840],
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// MTP draft subgraphs
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a MTP draft pre-projection ANE subgraph (stateless).
///
/// `pre_projection`: matmul `[1, 1024] x [1024, 3840] -> [1, 3840]`
///
/// Maps the draft hidden state into the main model's hidden dimension
/// so that draft logits can be compared with main-model logits.
pub fn compile_draft_pre_proj_ane(
    weights: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let program = build_matmul_mil(
        "draft_hidden",
        "pre_proj_weight",
        "main_space_hidden",
        1,    // m
        1024, // k
        3840, // n
        weights,
        true, // stateless
    )
    .map_err(|e| format!("draft pre-proj MIL: {}", e))?;

    compile_program(
        program,
        "draft_pre_proj",
        output_dir,
        "draft_hidden",
        &[1, 1024],
        ssa_names::MATMUL_OUT,
        &[1, 3840],
    )
}

/// Compile a MTP draft post-projection ANE subgraph (stateless).
///
/// `post_projection`: matmul `[1, 3840] x [3840, 1024] -> [1, 1024]`
///
/// Maps the main-model hidden state back into the draft's hidden dimension
/// so the draft decoder can predict tokens in its own latent space.
pub fn compile_draft_post_proj_ane(
    weights: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    let program = build_matmul_mil(
        "main_hidden",
        "post_proj_weight",
        "draft_space_hidden",
        1,    // m
        3840, // k
        1024, // n
        weights,
        true, // stateless
    )
    .map_err(|e| format!("draft post-proj MIL: {}", e))?;

    compile_program(
        program,
        "draft_post_proj",
        output_dir,
        "main_hidden",
        &[1, 3840],
        ssa_names::MATMUL_OUT,
        &[1, 1024],
    )
}

/// Compile a MTP draft decoder layer ANE subgraph (stateless).
///
/// One layer of the 4-layer draft decoder. Accepts all RMSNorm, QKV,
/// gate/up/down weight tensors and compiles the full decoder block
/// (RMSNorm -> QKV -> attention -> FFN) for the ANE.
///
/// Draft dimensions: hidden=1024, n_heads=8, n_kv_heads=8, head_dim=128.
pub fn compile_draft_layer_ane(
    layer_idx: u32,
    rms_w: &[f32],
    q_w: &[f32],
    k_w: &[f32],
    v_w: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    _scales: &[f32],
    output_dir: &Path,
) -> Result<PathBuf, String> {
    const HIDDEN: u32 = 1024;
    const N_HEADS: u32 = 8;
    const N_KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 128;

    let program = build_draft_layer_mil(
        "draft_hidden",
        HIDDEN,
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        rms_w,
        q_w,
        k_w,
        v_w,
        gate_w,
        up_w,
        down_w,
        true, // stateless
    )
    .map_err(|e| format!("draft layer {} MIL: {}", layer_idx, e))?;

    let subgraph_name = format!("draft_layer_{}", layer_idx);
    compile_program(
        program,
        &subgraph_name,
        output_dir,
        "draft_hidden",
        &[1, HIDDEN as i64],
        ssa_names::DRAFT_LAYER_OUT,
        &[1, HIDDEN as i64],
    )
}
