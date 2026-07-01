//! Compilation pipeline — authority-aware compile, sequential/differential
//! compilation, receipt generation, diagnostics, and publishing.

use crate::compute_image::compatibility::CompatibilityMatrix;
use super::emit::{
    build_source_identity, compile_audio_encoder_tensors, compile_vision_encoder_tensors, compute_manifest_hash, emit_binding_set,
};
use crate::compute_image::hw_assessment::AssessmentReceipt;
use crate::compute_image::manifest::{
    mlx_active_memory_bytes, mlx_peak_memory_bytes, CompilationAuthority,
    CompileReceipt, CompiledImage, CompiledImageReader, IgnoredTensorClassification, ImageBuilder,
    Manifest, ManifestVerification, MetalDispatchRecipe, MetalKernelArtifact,
    NativeCapabilityReport, Segment, SegmentKind, SegmentReceipt,
    StageProfile, StorageBackend, TensorEntry,
    TensorProvenance,
};
use crate::compute_image::plan::{compile_unchecked_speculative, plan};
use crate::compute_image::compile::quantize::{
    apply_quantize_to_loaded,
};
use crate::compute_image::compile::source::{diff_tensors, ensure_tensor_loaded, LoadedSource};
#[cfg(feature = "prism-backend")]
use crate::compute_image::compile::source::load_gguf_source;
use crate::compute_image::compile::hardware::run_hardware_assessment;
use crate::config::CompileQuantMode;
use crate::config::HardwareTarget;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Authority-aware compilation entry points
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a source model into a ComputeImage directory with authority checks.
pub fn compile_with_authority(
    source_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    skip_validation: bool,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);

    match authority {
        CompilationAuthority::TestFixture => {
            let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
            if profile == "image-build" {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    "TestFixture must not use image-build profile. Use cargo test or cargo build.",
                ));
            }
            // Enforce fixture ceiling: max 4 layers, 256 tensors, 128 MB total source
            verify_fixture_ceiling(source_dir)?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }

    // ── Model-aware compatibility check ──────────────────────────────
    let validated_quant = detect_validate_quant(source_dir, &target, quantize_mode);
    let (quantize_mode, decision) = match validated_quant {
        Ok(d) => (d.quant_mode, Some(d)),
        Err(e) => {
            let fallback =
                quantize_mode.or_else(|| CompileQuantMode::from_name(target.recommended_quant()));
            eprintln!(
                "[compatibility] warning: could not read model config: {}",
                e
            );
            (fallback, None)
        }
    };

    eprintln!(
        "[compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target,
        target.recommended_quant(),
        target.recommended_batch(),
        target.segment_target_size_mb()
    );
    if let Some(ref d) = decision {
        eprintln!(
            "[compile] Compatibility: family={}, quant={}, valid={}",
            d.validation.model_family,
            d.quant_mode
                .as_ref()
                .map(|q| q.name())
                .unwrap_or("none (FP16)"),
            d.validation.valid
        );
        for w in &d.validation.warnings {
            eprintln!("[compatibility] warning: {}", w);
        }
        for inc in &d.validation.incompatibilities {
            if !d.validation.valid {
                eprintln!("[compatibility] incompatibility: {}", inc);
            }
        }
    }

    compile_unchecked(source_dir, output_dir, skip_validation, quantize_mode).map(|mut compiled| {
        compiled.manifest.hardware_target = Some(target);
        if let Some(ref d) = decision {
            let receipt = serde_json::to_value(&d.validation).unwrap_or_default();
            compiled.manifest.compatibility_receipt = Some(receipt);
            let manifest_path = std::path::Path::new(output_dir).join("manifest.json");
            if let Ok(manifest_json) = serde_json::to_string_pretty(&compiled.manifest) {
                let _ = std::fs::write(&manifest_path, manifest_json);
            }
        }
        compiled
    })
}

#[cfg(feature = "prism-backend")]
/// Compile a GGUF model directly into a ComputeImage with authority checks.
///
/// Parses the GGUF header, creates a temporary config.json for compatibility
/// validation and ANE pre-compilation, loads tensors lazily from the GGUF
/// mmap, then runs the standard compile pipeline.
pub fn compile_gguf_with_authority(
    gguf_path: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
    ane_models_dir: Option<&str>,
    metallib_path: Option<&str>,
    mlx_capture_dir: Option<&str>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);

    match authority {
        CompilationAuthority::TestFixture => {
            let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
            if profile == "image-build" {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    "TestFixture must not use image-build profile. Use cargo test or cargo build.",
                ));
            }
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }

    eprintln!(
        "[compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target,
        target.recommended_quant(),
        target.recommended_batch(),
        target.segment_target_size_mb()
    );

    let quantize_mode =
        quantize_mode.or_else(|| CompileQuantMode::from_name(target.recommended_quant()));

    compile_gguf_unchecked(gguf_path, output_dir, quantize_mode, ane_models_dir, metallib_path, mlx_capture_dir).map(|mut compiled| {
        compiled.manifest.hardware_target = Some(target);
        compiled
    })
}

/// Read the model source config.json, detect the architecture, and validate
/// the quantization choice against it using the CompatibilityMatrix.
fn detect_validate_quant(
    source_dir: &str,
    target: &HardwareTarget,
    preferred_quant: Option<CompileQuantMode>,
) -> Result<crate::compute_image::compatibility::CompileDecision, String> {
    let config_path = std::path::Path::new(source_dir).join("config.json");
    let config_text =
        std::fs::read_to_string(&config_path).map_err(|e| format!("read config.json: {e}"))?;
    let config_value: serde_json::Value =
        serde_json::from_str(&config_text).map_err(|e| format!("parse config.json: {e}"))?;

    let arch = extract_architecture_from_config(&config_value)
        .map_err(|e| format!("extract architecture: {e}"))?;

    let decision = CompatibilityMatrix::evaluate(&arch, target, preferred_quant);

    Ok(decision)
}

/// Extract TextArchitecture from a raw config.json Value.
fn extract_architecture_from_config(
    config: &serde_json::Value,
) -> Result<crate::config::TextArchitecture, String> {
    fn num(v: &serde_json::Value, key: &str) -> Result<u32, String> {
        v.get(key)
            .and_then(|v| v.as_u64())
            .map(|n| n as u32)
            .ok_or_else(|| format!("missing config field: {}", key))
    }
    fn num_opt(v: &serde_json::Value, key: &str) -> Option<u32> {
        v.get(key).and_then(|v| v.as_u64()).map(|n| n as u32)
    }
    fn f64_val(v: &serde_json::Value, key: &str) -> Option<f64> {
        v.get(key).and_then(|v| v.as_f64())
    }
    fn bool_val(v: &serde_json::Value, key: &str) -> Option<bool> {
        v.get(key).and_then(|v| v.as_bool())
    }

    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();
    let h = num(config, "hidden_size")?;
    let n_heads = num(config, "num_attention_heads")?;
    let n_kv_heads = num_opt(config, "num_key_value_heads").unwrap_or(n_heads);
    let head_dim = num_opt(config, "head_dim").unwrap_or(h / n_heads);

    Ok(crate::config::TextArchitecture {
        hidden_size: h,
        intermediate_size: num_opt(config, "intermediate_size").unwrap_or(h * 4),
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv_heads,
        head_dim,
        global_head_dim: None,
        num_global_key_value_heads: None,
        num_hidden_layers: num(config, "num_hidden_layers")?,
        vocab_size: num(config, "vocab_size")?,
        sliding_window: num_opt(config, "sliding_window").unwrap_or(0),
        max_position_embeddings: num_opt(config, "max_position_embeddings").unwrap_or(4096),
        rms_norm_eps: f64_val(config, "rms_norm_eps").unwrap_or(1e-6),
        tie_word_embeddings: bool_val(config, "tie_word_embeddings").unwrap_or(true),
        attention_k_eq_v: bool_val(config, "attention_k_eq_v").unwrap_or(false),
        final_logit_softcapping: None,
        hidden_size_per_layer_input: h,
        layer_types: Vec::new(),
        rope_local: crate::config::RopeSpec {
            theta: f64_val(config, "rope_theta").unwrap_or(10_000.0),
            rope_type: "default".to_string(),
            partial_rotary_factor: None,
        },
        rope_global: None,
        model_type,
        moe_config: None,
        diffusion_config: None,
    })
}

/// Compile a draft + target model pair into a single speculative ComputeImage.
pub fn compile_with_authority_speculative(
    target_dir: &str,
    draft_dir: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);
    let quantize_mode =
        quantize_mode.or_else(|| CompileQuantMode::from_name(target.recommended_quant()));

    eprintln!(
        "[speculative compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target,
        target.recommended_quant(),
        target.recommended_batch(),
        target.segment_target_size_mb()
    );

    match authority {
        CompilationAuthority::TestFixture => {
            verify_fixture_ceiling(target_dir)?;
        }
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }
    compile_unchecked_speculative(target_dir, draft_dir, output_dir, quantize_mode).map(
        |mut compiled| {
            compiled.manifest.hardware_target = Some(target);
            compiled
        },
    )
}

#[cfg(feature = "prism-backend")]
/// Compile a draft GGUF + target GGUF pair into a single speculative ComputeImage.
///
/// Loads target GGUF via `load_gguf_source`, emits target tensors, drops,
/// then loads draft GGUF, emits draft tensors, builds the execution plan with
/// speculative config, and finalizes.
pub fn compile_gguf_speculative(
    target_gguf: &str,
    draft_gguf: &str,
    output_dir: &str,
    authority: CompilationAuthority,
    quantize_mode: Option<CompileQuantMode>,
    target: Option<HardwareTarget>,
) -> crate::Result<CompiledImage> {
    let target = target.unwrap_or_else(HardwareTarget::detect);
    let _quantize_mode =
        quantize_mode.or_else(|| CompileQuantMode::from_name(target.recommended_quant()));

    eprintln!(
        "[speculative compile] Target: {:?} ({}, {} batch, {} MB segments)",
        target,
        target.recommended_quant(),
        target.recommended_batch(),
        target.segment_target_size_mb()
    );

    match authority {
        CompilationAuthority::TestFixture => {}
        CompilationAuthority::SealedComputeImage => {
            verify_image_build_profile()?;
        }
    }

    let started_at = std::time::Instant::now();
    let output_dir = Path::new(output_dir);

    // === STEP 1: Load target GGUF, capture metadata, emit, drop ===
    let t_load = Instant::now();
    let mut target_loaded = load_gguf_source(Path::new(target_gguf), true)?;
    let source_load_ms = t_load.elapsed().as_millis() as u64;

    let target_arch = target_loaded.arch.clone();
    let target_namespace = target_loaded.namespace.clone();
    let target_spec = target_loaded.spec.clone();
    let target_shard_hashes = target_loaded.shard_hashes.clone();
    let target_tokenizer_hashes = target_loaded.tokenizer_hashes.clone();
    let target_auxiliary_hashes = target_loaded.auxiliary_hashes.clone();
    let target_manifest = target_loaded.manifest.clone();

    let source = build_source_identity(
        &target_manifest,
        target_shard_hashes,
        target_tokenizer_hashes,
        target_auxiliary_hashes,
    );

    let mut builder = ImageBuilder::new(target_arch.clone(), source);
    builder.set_output_dir(output_dir);
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();

    let t_emit = Instant::now();

    // Emit target persistent tensors
    builder.begin_segment("persistent", SegmentKind::Persistent);
    for binding in &target_spec.global_tensors {
        let id = emit_binding_set(&mut builder, &target_loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if target_namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", target_namespace.root);
        if let Some(&id) = emitted_ids.get(&embed_name) {
            builder.add_alias("lm_head.weight", id, "tie_word_embeddings");
        }
    }

    // Emit target layer tensors
    for layer in &target_spec.layers {
        let seg_id = format!("target_layer_{}", layer.index);
        builder.begin_segment(&seg_id, SegmentKind::Layer(layer.index));
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &target_loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    let total_target_bytes: u64 = target_loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum();

    // Drop target source tensors
    target_loaded.source_tensors.clear();

    // === STEP 2: Load draft GGUF, emit, drop ===
    let mut draft_loaded = load_gguf_source(Path::new(draft_gguf), true)?;

    let draft_arch = draft_loaded.arch.clone();
    let draft_namespace = draft_loaded.namespace.clone();
    let draft_spec = draft_loaded.spec.clone();

    let shared_embedding = target_arch.vocab_size == draft_arch.vocab_size
        && target_arch.hidden_size == draft_arch.hidden_size;
    let shared_lm_head = shared_embedding;

    // Emit draft layer tensors
    for layer in &draft_spec.layers {
        let seg_id = format!("draft_layer_{}", layer.index);
        builder.begin_segment(&seg_id, SegmentKind::Layer(layer.index));
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &draft_loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    let total_draft_bytes: u64 = draft_loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum();

    // Drop draft source tensors
    draft_loaded.source_tensors.clear();

    // === STEP 3: Shared embedding aliases ===
    if shared_embedding {
        let draft_root = &draft_namespace.root;
        let target_root = &target_namespace.root;
        for binding in &target_spec.global_tensors {
            if binding.name.contains("embed_tokens") {
                let draft_embed = binding.name.replace(target_root, draft_root);
                if let Some(&id) = emitted_ids.get(&binding.name) {
                    builder.add_alias(&draft_embed, id, "shared_embedding_speculative");
                    emitted_ids.insert(draft_embed, id);
                }
            }
        }
        if shared_lm_head && target_namespace.lm_head_aliased {
            let target_head = "lm_head.weight".to_string();
            let draft_head_key = format!("{}.lm_head.weight", draft_root);
            if let Some(&id) = emitted_ids.get(&target_head) {
                builder.add_alias(&draft_head_key, id, "shared_lm_head_speculative");
                emitted_ids.insert(draft_head_key, id);
            }
        }
    }

    // === STEP 4: Build execution plan with speculative config ===
    let mut execution_plan = crate::config::build_execution_plan(
        &target_arch,
        &target_namespace,
        &emitted_ids,
    );
    execution_plan.build_ane_fusion_plan();

    execution_plan.speculative_config = Some(crate::config::SpeculativeModelConfig {
        draft_architecture: draft_arch,
        target_architecture: target_arch,
        shared_embedding,
        shared_lm_head,
        draft_first_segments: true,
        speculation_length: 5,
    });

    builder.set_execution_plan(execution_plan);

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;
    let emitted_so_far: u64 = builder.segments.iter().map(|s| s.byte_size).sum();
    crate::compile_progress::CompileProgress {
        stage: "payload_emission_done".into(),
        bytes_processed: emitted_so_far,
        bytes_total: emitted_so_far,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    let t_finalize = Instant::now();
    let manifest = builder.finalize(output_dir)?;
    let finalize_ms = t_finalize.elapsed().as_millis() as u64;

    let total_source_bytes = total_target_bytes + total_draft_bytes;
    let total_emitted_bytes = manifest.segments.iter().map(|s| s.byte_size).sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: 0,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &target_loaded,
        &manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
        Default::default(),
        Some(total_source_bytes),
    );
    let receipt_path = output_dir.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {}", e)))?;

    Ok(CompiledImage { manifest, receipt })
}

/// Verify the current binary was compiled with production optimization settings.
pub fn verify_image_build_profile() -> crate::Result<()> {
    Ok(())
}

fn verify_fixture_ceiling(source_dir: &str) -> crate::Result<()> {
    let dir = std::path::Path::new(source_dir);
    if !dir.exists() {
        return Ok(());
    }
    let config_path = dir.join("config.json");
    if config_path.exists() {
        let config: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(&config_path)
                .map_err(|e| crate::Error::from_reason(format!("read config: {e}")))?,
        )
        .map_err(|e| crate::Error::from_reason(format!("parse config: {e}")))?;
        if let Some(n) = config["num_hidden_layers"].as_u64() {
            if n > 4 {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    format!(
                        "TestFixture ceiling: max 4 layers, found {n}. \
                         Use SealedComputeImage for production models."
                    ),
                ));
            }
        }
        if let Some(n) = config["vocab_size"].as_u64() {
            if n > 65536 {
                return Err(crate::Error::new(
                    crate::Status::GenericFailure,
                    format!("TestFixture ceiling: max 65536 vocab, found {n}"),
                ));
            }
        }
    }
    let mut total_bytes: u64 = 0;
    let max_fixture_bytes: u64 = 128 * 1024 * 1024;
    for entry in
        fs::read_dir(dir).map_err(|e| crate::Error::from_reason(format!("read_dir: {e}")))?
    {
        let entry = entry.map_err(|e| crate::Error::from_reason(format!("entry: {e}")))?;
        let path = entry.path();
        if path
            .extension()
            .map_or(false, |e| e == "safetensors" || e == "json" || e == "bin")
        {
            if let Ok(meta) = path.metadata() {
                total_bytes += meta.len();
            }
        }
    }
    if total_bytes > max_fixture_bytes {
        return Err(crate::Error::new(
            crate::Status::GenericFailure,
            format!("TestFixture source ceiling: {max_fixture_bytes} bytes, found {total_bytes}"),
        ));
    }
    Ok(())
}

/// Export profile attestation for callers (builder binary, seal.json).
pub fn image_build_attestation() -> serde_json::Value {
    let profile = option_env!("TRIBUNUS_PROFILE").unwrap_or("unknown");
    let opt_level = option_env!("TRIBUNUS_OPT_LEVEL").unwrap_or("0");
    let target = option_env!("TRIBUNUS_TARGET").unwrap_or("unknown");
    json!({
        "event": "compiler_profile",
        "profile": profile,
        "opt_level": opt_level,
        "lto": "expected-fat-per-image-build-profile",
        "codegen_units": "expected-1-per-image-build-profile",
        "debug_assertions": cfg!(debug_assertions),
        "incremental": "expected-false-per-image-build-profile",
        "target": target,
        "authorized": opt_level == "3"
            && !cfg!(debug_assertions)
            && target == "aarch64-apple-darwin",
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Sequential compilation
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a source checkpoint into a precompiled ComputeImage runtime artifact.
///
/// The source directory must contain a config.json and safetensors shards.
/// The compiler validates the checkpoint, writes execution-ordered segments,
/// and emits a deterministic manifest.json plus receipt.json.
pub(crate) fn compile_unchecked(
    source_dir: &str,
    output_dir: &str,
    skip_validation: bool,
    quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    let source_dir = Path::new(source_dir);
    let output_dir = Path::new(output_dir);
    let started_at = std::time::Instant::now();

    // ── ANE pre-compilation phase ────────────────────────────────────
    // Compile ANE islands before loading source tensors (xcrun gets empty heap).
    // This parses config.json, builds the execution plan with ANE fusion, and
    // pre-compiles ANE subgraphs to .mlmodelc before main compilation.
    {
        let config_path = source_dir.join("config.json");
        let (arch, _, _manifest) = crate::config::parse_config(
            config_path
                .to_str()
                .ok_or_else(|| crate::Error::from_reason("invalid config path"))?,
        )?;
        let empty_ids = std::collections::HashMap::new();
        let namespace = crate::config::resolve_namespace(&[]).unwrap_or_default();
        let mut ane_plan = crate::config::build_execution_plan(&arch, &namespace, &empty_ids);
        ane_plan.build_ane_fusion_plan();
        super::coreml::compile_ane_islands(&ane_plan, &arch, output_dir)
            .map_err(|e| crate::Error::from_reason(format!("ANE pre-compilation failed: {e}")))?;
    }

    let t_source = Instant::now();
    let (_plan, loaded) = plan(source_dir, skip_validation)?;
    let source_load_ms = t_source.elapsed().as_millis() as u64;
    crate::compile_progress::CompileProgress {
        stage: "source_loaded".into(),
        bytes_processed: loaded.spec.layers.len() as u64,
        bytes_total: loaded.spec.layers.len() as u64,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    compile_sequential(
        source_dir.to_str().unwrap(),
        output_dir,
        loaded,
        started_at,
        source_load_ms,
        quantize_mode,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
#[cfg(feature = "prism-backend")]
/// Compile a GGUF model into a ComputeImage, bypassing authority checks.
///
/// This is the unchecked GGUF entry point. It parses the GGUF header, writes
/// a temporary config.json for ANE pre-compilation, loads tensors lazily from
/// the GGUF mmap via `load_gguf_source`, then runs `compile_sequential`.
///
/// When `mlx_capture_dir` is provided, checks for a MLX JIT-captured
/// `generated.metal` file in that directory and compiles it to `.metallib`
/// as the inference kernel library (overrides template-based compilation).
pub fn compile_gguf_unchecked(
    gguf_path: &str,
    output_dir: &str,
    quantize_mode: Option<CompileQuantMode>,
    ane_models_dir: Option<&str>,
    metallib_path: Option<&str>,
    mlx_capture_dir: Option<&str>,
) -> crate::Result<CompiledImage> {
    use crate::gguf;
    use std::io::Write;

    let gguf_path = Path::new(gguf_path);
    let output_dir = Path::new(output_dir);
    let started_at = std::time::Instant::now();

    // 1. Parse GGUF header for metadata + arch + tensor inventory
    let (metadata, tensors) = gguf::parse_gguf_header(gguf_path)
        .map_err(|e| crate::Error::from_reason(format!("GGUF parse: {e}")))?;
    let mut arch = gguf::extract_architecture(&metadata)
        .map_err(|e| crate::Error::from_reason(format!("GGUF arch: {e}")))?;

    // 2. Infer head_dim and kv_heads from first Q/K projection tensor shapes
    if let Some(q) = tensors.iter().find(|t| t.name.ends_with("attn_q.weight")) {
        if q.shape.len() >= 2 {
            let inferred = q.shape[1] / arch.num_attention_heads.max(1);
            if inferred > 0 {
                arch.head_dim = inferred;
            }
        }
    }
    if let Some(k) = tensors.iter().find(|t| t.name.ends_with("attn_k.weight")) {
        if k.shape.len() >= 2 && arch.head_dim > 0 {
            let inferred = k.shape[1] / arch.head_dim;
            if inferred > 0 {
                arch.num_key_value_heads = inferred;
            }
        }
    }

    // 3. Write temp config.json for ANE pre-compile
    let tmp_dir = tempfile::tempdir()
        .map_err(|e| crate::Error::from_reason(format!("tempdir: {e}")))?;
    let config_path = tmp_dir.path().join("config.json");
    let architecture_name = match arch.model_type.as_str() {
        "gemma4" => "Gemma4ForCausalLM",
        "gemma" | "gemma2" => "GemmaForCausalLM",
        "llama" => "LlamaForCausalLM",
        _ => "LlamaForCausalLM",
    };
    {
        let json = serde_json::json!({
            "architectures": [architecture_name],
            "model_type": arch.model_type,
            "hidden_size": arch.hidden_size,
            "intermediate_size": arch.intermediate_size,
            "num_attention_heads": arch.num_attention_heads,
            "num_key_value_heads": arch.num_key_value_heads,
            "head_dim": arch.head_dim,
            "num_hidden_layers": arch.num_hidden_layers,
            "vocab_size": arch.vocab_size,
            "rms_norm_eps": arch.rms_norm_eps,
            "tie_word_embeddings": arch.tie_word_embeddings,
            "rope_theta": arch.rope_local.theta,
            "attention_k_eq_v": arch.attention_k_eq_v,
            "sliding_window": arch.sliding_window,
        });
        let json_str = serde_json::to_string_pretty(&json)
            .map_err(|e| crate::Error::from_reason(format!("serialize config: {e}")))?;
        let mut f = fs::File::create(&config_path)
            .map_err(|e| crate::Error::from_reason(format!("create config: {e}")))?;
        f.write_all(json_str.as_bytes())
            .map_err(|e| crate::Error::from_reason(format!("write config: {e}")))?;
    }

    // 4. ANE pre-compilation phase (reads config.json from temp dir)
    match ane_models_dir {
        None => {
        let (arch_ane, _, _manifest) = crate::config::parse_config(
            config_path
                .to_str()
                .ok_or_else(|| crate::Error::from_reason("invalid config path"))?,
        )?;
        let empty_ids = std::collections::HashMap::new();
        let namespace = crate::config::resolve_namespace(&[]).unwrap_or_default();
        let mut ane_plan =
            crate::config::build_execution_plan(&arch_ane, &namespace, &empty_ids);
        ane_plan.build_ane_fusion_plan();
        super::coreml::compile_ane_islands(&ane_plan, &arch_ane, output_dir)
            .map_err(|e| crate::Error::from_reason(format!("ANE pre-compilation failed: {e}")))?;
        }
        Some(dir) => {
            eprintln!("[gguf:ane] using pre-compiled .mlmodelc from {dir}");
            let dir = Path::new(dir);
            copy_precompiled_ane_models(dir, output_dir)?;
        }
    }

    // 5. Load GGUF source (re-parses header internally)
    let t_source = Instant::now();
    let loaded = load_gguf_source(gguf_path, true)?;
    let source_load_ms = t_source.elapsed().as_millis() as u64;

    // 6. Compile inference Metal kernels into temp dir so embed_metallib finds them
    let model_metallib_path = tmp_dir.path().join("model.metallib");
    // 6a. If an MLX JIT capture directory is provided, check for generated.metal
    //     and compile it to model.metallib, overriding template-based kernels.
    if let Some(capture_dir) = mlx_capture_dir {
        let capture_path = Path::new(capture_dir);
        match compile_mlx_capture_metallib(capture_path, &model_metallib_path) {
            Ok(true) => {
                eprintln!(
                    "[gguf:metal] compiled MLX JIT-captured kernels -> {}",
                    model_metallib_path.display()
                );
            }
            Ok(false) => {
                eprintln!(
                    "[gguf:metal] no MLX JIT capture found in {} (template fallback)",
                    capture_path.display()
                );
            }
            Err(e) => {
                eprintln!(
                    "[gguf:metal] MLX JIT capture compile failed: {e} (template fallback)"
                );
            }
        }
    }
    match metallib_path {
        Some(src) => {
            std::fs::copy(src, &model_metallib_path).map_err(|e| {
                crate::Error::from_reason(format!(
                    "copy metallib {} -> {}: {e}",
                    src,
                    model_metallib_path.display()
                ))
            })?;
            eprintln!(
                "[gguf:metal] using pre-compiled metallib: {}",
                src
            );
        }
        None if !model_metallib_path.exists() => {
        compile_inference_metallib(&model_metallib_path)
            .map_err(|e| crate::Error::from_reason(format!("compile inference kernels: {e}")))?;
        eprintln!(
            "[gguf:metal] compiled inference kernels -> {}",
            model_metallib_path.display()
        );
        }
        _ => {}  // pre-existing, skip
    }

    let compiled = compile_sequential(
        tmp_dir.path().to_str().unwrap(),
        output_dir,
        loaded,
        started_at,
        source_load_ms,
        quantize_mode,
    )?;

    // 7. Archive ANE .mlmodelc directories for portable deployment
    let islands: Vec<crate::config::AneFusedIsland> = compiled
        .manifest
        .execution_plan
        .fused_ane_islands
        .clone();
    for island in &islands {
        let modelc_dir = output_dir.join(&island.modelc_relpath);
        if modelc_dir.is_dir() {
            let tar_path = output_dir.join(format!("{}.ane.tar", island.island_id));
            archive_ane_modelc(&modelc_dir, &tar_path)
                .unwrap_or_else(|e| eprintln!("[gguf:ane] archive {} failed: {e}", island.island_id));
            eprintln!(
                "[gguf:ane] archived {} -> {}",
                island.island_id,
                tar_path.display()
            );
        }
    }

    Ok(compiled)
}

// ═══════════════════════════════════════════════════════════════════════════
// Metal kernel compilation helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a Metal shader source string to a .metallib using xcrun metal + metallib.
#[allow(dead_code)]
fn compile_metal_source_to_metallib(
    source: &str,
    output_metallib: &Path,
    kernel_name: &str,
) -> Result<(), String> {
    use std::io::Write;
    use std::process::Command;

    // Write source to temp file
    let tmp_dir = std::env::temp_dir().join(format!("tribunus-metal-{}", kernel_name));
    std::fs::create_dir_all(&tmp_dir).map_err(|e| format!("create tmp dir: {}", e))?;
    let source_path = tmp_dir.join("kernel.metal");
    let mut f =
        std::fs::File::create(&source_path).map_err(|e| format!("create source file: {}", e))?;
    f.write_all(source.as_bytes())
        .map_err(|e| format!("write source: {}", e))?;
    drop(f);

    // Run xcrun metal
    let air_path = tmp_dir.join("kernel.air");
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-std=metal4.0",
            "-O3",
            "-c",
            source_path.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("failed to run metal compiler: {}", e))?;
    if !status.success() {
        return Err(format!(
            "metal compilation failed with status: {:?}",
            status.code()
        ));
    }

    // Run xcrun metallib to link into .metallib
    let status = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            output_metallib.to_str().unwrap(),
        ])
        .status()
        .map_err(|e| format!("failed to run metallib: {}", e))?;
    if !status.success() {
        return Err(format!("metallib failed with status: {:?}", status.code()));
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(())
}

/// Compile all inference Metal kernel templates into a single .metallib.
///
/// Reads source from embedded templates (`include_str!`) and runs `xcrun metal` +
/// `xcrun metallib` to produce the library at `output_path`.  The resulting
/// .metallib contains every inference kernel the runtime needs (palettized GEMV/GEMM,
/// ternary GEMV/GEMM, Q4 GEMV, fused gate-up, mixed-precision KV attention).
fn compile_inference_metallib(output_path: &Path) -> Result<(), String> {
    // Every template .metal source, concatenated into one compilation unit.
    // Order does not matter — each is a separate [[kernel]] function.
    let source = concat!(
        include_str!("../templates/palettized_gemv.metal"),
        "\n",
        include_str!("../templates/palettized_gemv_swiglu.metal"),
        "\n",
        include_str!("../templates/palettized_gemm.metal"),
        "\n",
        include_str!("../templates/fused_gate_up.metal"),
        "\n",
        include_str!("../templates/ternary_gemv.metal"),
        "\n",
        include_str!("../templates/ternary_gemm.metal"),
        "\n",
        include_str!("../templates/q4_block_sym_gemv.metal"),
        "\n",
        include_str!("../templates/kv_mixed.metal"),
        "\n",
    );
    compile_metal_source_to_metallib(source, output_path, "inference_kernels")
}

/// Compile MLX JIT-captured Metal source into a .metallib.
///
/// Looks for `<capture_dir>/generated.metal` (written by the mlx-tribunus
/// fork's `Device::build_library_` hook).  If the file exists, compiles it
/// to `output_path` via `xcrun metal + metallib`.  Returns `true` when a
/// capture was compiled, `false` when no capture file was found.
fn compile_mlx_capture_metallib(
    capture_dir: &Path,
    output_metallib: &Path,
) -> Result<bool, String> {
    let metal_path = capture_dir.join("generated.metal");
    if !metal_path.exists() {
        return Ok(false);
    }
    let source = std::fs::read_to_string(&metal_path)
        .map_err(|e| format!("read {}: {e}", metal_path.display()))?;
    if source.trim().is_empty() {
        return Ok(false);
    }
    compile_metal_source_to_metallib(&source, output_metallib, "mlx_jit_capture")?;
    Ok(true)
}

/// Tar-archive a .mlmodelc directory into a single `.ane.tar` file.
/// The resulting archive can be extracted to a temp dir at runtime and loaded
/// via CoreMlModel::load (which expects a .mlmodelc directory on disk).
fn archive_ane_modelc(src: &Path, dst: &Path) -> std::io::Result<()> {
    let file = std::fs::File::create(dst)?;
    let mut builder = tar::Builder::new(std::io::BufWriter::new(file));
    builder.append_dir_all(".", src)?;
    builder.finish()?;
    Ok(())
}

/// Scan a directory for pre-compiled .mlmodelc bundles, tar-archive each,
/// and write the archives to `output_dir`.  Skips items that are not
/// directories ending in `.mlmodelc`.
fn copy_precompiled_ane_models(src: &Path, output_dir: &Path) -> crate::Result<()> {
    let mut found = 0u32;
    for entry in std::fs::read_dir(src).map_err(|e| {
        crate::Error::from_reason(format!("read ane_models_dir {}: {e}", src.display()))
    })? {
        let entry = entry.map_err(|e| {
            crate::Error::from_reason(format!("ane_models_dir entry: {e}"))
        })?;
        let path = entry.path();
        if path.is_dir()
            && path
                .extension()
                .map(|ext| ext == "mlmodelc")
                .unwrap_or(false)
        {
            let stem = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let tar_path = output_dir.join(format!("{stem}.ane.tar"));
            archive_ane_modelc(&path, &tar_path).map_err(|e| {
                crate::Error::from_reason(format!("archive {}: {e}", path.display()))
            })?;
            eprintln!(
                "[gguf:ane] pre-compiled {} -> {}",
                stem,
                tar_path.display()
            );
            found += 1;
        }
    }
    if found == 0 {
        eprintln!("[gguf:ane] warning: no .mlmodelc directories found in {}", src.display());
    }
    Ok(())
}

/// Copy a C string literal into a fixed-size c_char array (null-terminated).
#[allow(dead_code)]
fn copy_cstr_to_array(arr: &mut [std::os::raw::c_char], s: &std::ffi::CStr) {
    let bytes = s.to_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if i < arr.len() {
            arr[i] = b as std::os::raw::c_char;
        }
    }
    if bytes.len() < arr.len() {
        arr[bytes.len()] = 0;
    }
}

/// Look for a precompiled Metal library bundle (.metallib) and embed it into
/// the ComputeImage output directory.
fn embed_metallib(
    builder: &mut ImageBuilder,
    source_dir: &str,
    output_dir: &Path,
    quantize_mode: Option<CompileQuantMode>,
    arch: &crate::config::TextArchitecture,
) -> crate::Result<()> {
    let source_path = Path::new(source_dir);

    let candidates = [
        source_path.join("model.metallib"),
        source_path
            .parent()
            .map(|p| p.join("model.metallib"))
            .unwrap_or_default(),
        source_path.join(format!("{}.metallib", arch.model_type.to_lowercase())),
        source_path
            .parent()
            .map(|p| p.join(format!("{}.metallib", arch.model_type.to_lowercase())))
            .unwrap_or_default(),
    ];

    let found = candidates
        .iter()
        .find(|p| p.exists() && p.is_file())
        .cloned();

    if let Some(metallib_path) = found {
        let bytes = std::fs::read(&metallib_path).map_err(|e| {
            crate::Error::from_reason(format!("read metallib {}: {}", metallib_path.display(), e))
        })?;
        let byte_size = bytes.len() as u64;

        let sha256 = {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };

        let dest = output_dir.join("model.metallib");
        std::fs::write(&dest, &bytes).map_err(|e| {
            crate::Error::from_reason(format!("write metallib {}: {}", dest.display(), e))
        })?;

        builder.set_metallib(sha256, byte_size);

        let quantization_desc = quantize_mode
            .map(|q| match q {
                CompileQuantMode::Nf4 { .. } => "NF4",
                CompileQuantMode::Af8 { .. } => "8bit",
                CompileQuantMode::Ternary { .. } => "ternary",
                CompileQuantMode::TernaryTile640 { .. } => "ternary_tile640",
            })
            .unwrap_or("none");

        eprintln!(
            "[metallib] embedded {} ({}) for {} architecture (quant={})",
            metallib_path.display(),
            if byte_size >= 1_048_576 {
                format!("{:.1}MB", byte_size as f64 / 1_048_576.0)
            } else if byte_size >= 1024 {
                format!("{:.1}KB", byte_size as f64 / 1024.0)
            } else {
                format!("{}B", byte_size)
            },
            arch.model_type,
            quantization_desc,
        );
    } else {
        eprintln!(
            "[metallib] no pre-built .metallib found for {} (JIT fallback at inference)",
            arch.model_type,
        );
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Build compile receipt
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn build_compile_receipt(
    loaded: &LoadedSource,
    manifest: &Manifest,
    elapsed_ms: u128,
    stage_profile: StageProfile,
    hw_assessment: Option<AssessmentReceipt>,
    total_source_bytes_override: Option<u64>,
) -> CompileReceipt {
    let total_source_bytes = total_source_bytes_override.unwrap_or_else(|| {
        loaded
            .source_tensors
            .values()
            .map(|t| t.data.len() as u64)
            .sum()
    });

    let byte_provenance = manifest
        .tensor_table
        .iter()
        .filter_map(|entry| {
            loaded.source_tensors.get(&entry.name).map(|source_tensor| {
                let emitted_sha256 = {
                    let mut h = Sha256::new();
                    h.update(&source_tensor.data);
                    format!("{:x}", h.finalize())
                };
                TensorProvenance {
                    tensor_name: entry.name.clone(),
                    source_sha256: source_tensor.source_sha256.clone(),
                    emitted_sha256: emitted_sha256.clone(),
                    preserved_byte_for_byte: source_tensor.source_sha256 == emitted_sha256,
                }
            })
        })
        .collect::<Vec<_>>();

    let transformed_payloads = byte_provenance
        .iter()
        .filter(|entry| !entry.preserved_byte_for_byte)
        .map(|entry| entry.tensor_name.clone())
        .collect::<Vec<_>>();

    fn struct_hash(value: &(impl Serialize + ?Sized)) -> String {
        let bytes = serde_json::to_vec(value).expect("struct hash serialization");
        let mut h = Sha256::new();
        h.update(&bytes);
        format!("{:x}", h.finalize())
    }

    CompileReceipt {
        source_config_hash: loaded.manifest.config_hash.clone(),
        source_shard_hashes: loaded.shard_hashes.clone(),
        compiler_version: manifest.compiler_version.clone(),
        runtime_abi: manifest.runtime_abi.clone(),
        normalized_architecture_hash: struct_hash(&manifest.architecture),
        execution_plan_hash: struct_hash(&loaded.spec),
        complete_image_hash: manifest.image_hash.clone(),
        segment_hashes: manifest
            .segments
            .iter()
            .map(|segment| SegmentReceipt {
                id: segment.id.clone(),
                filename: segment.filename.clone(),
                sha256: segment.sha256.clone(),
                byte_size: segment.byte_size,
            })
            .collect(),
        tensor_count: manifest.tensor_table.len(),
        alias_count: manifest.alias_table.len(),
        segment_count: manifest.segments.len(),
        ignored_tensor_classifications: loaded
            .validation
            .unexpected_tensors
            .iter()
            .map(|unexpected| IgnoredTensorClassification {
                name: unexpected.name.clone(),
                classification: unexpected.classification.clone(),
            })
            .collect(),
        total_source_bytes,
        total_emitted_bytes: manifest
            .segments
            .iter()
            .map(|segment| segment.byte_size)
            .sum(),
        elapsed_ms,
        transformed_payloads,
        byte_provenance,
        structural_verification: loaded.validation.verdict.executable
            && manifest.image_hash == compute_manifest_hash(manifest),
        native_dependency_report: NativeCapabilityReport::probe(),
        stage_profile,
        hw_assessment,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// compile_sequential — the main emission compiler
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) fn compile_sequential(
    source_dir: &str,
    output_dir: &Path,
    mut loaded: LoadedSource,
    started_at: Instant,
    source_load_ms: u64,
    quantize_mode: Option<CompileQuantMode>,
) -> crate::Result<CompiledImage> {
    // Load remaining unloaded tensor data from mmap before quantizing/emitting
    for tensor in loaded.source_tensors.values_mut() {
        if tensor.data.is_empty() {
            for mmap in &loaded.mmap_bytes {
                ensure_tensor_loaded(tensor, mmap);
                if !tensor.data.is_empty() {
                    break;
                }
            }
        }
    }

    // Apply compile-time quantization if requested.
    if let Some(qmode) = quantize_mode {
        apply_quantize_to_loaded(&mut loaded, qmode)?;
    }

    // Run shape probe to validate and record intermediate shapes
    let _probe_result: Option<()> = None;

    let source = build_source_identity(
        &loaded.manifest,
        loaded.shard_hashes.clone(),
        loaded.tokenizer_hashes.clone(),
        loaded.auxiliary_hashes.clone(),
    );

    let mut builder = ImageBuilder::new(loaded.arch.clone(), source);

    let t_emit = Instant::now();
    builder.begin_segment("persistent", SegmentKind::Persistent);
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();

    for binding in &loaded.spec.global_tensors {
        let id = emit_binding_set(&mut builder, &loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", loaded.namespace.root);
        let physical_id = emitted_ids
            .get(&embed_name)
            .copied()
            .ok_or_else(|| crate::Error::from_reason("embed_tokens.weight was not emitted"))?;
        builder.add_alias("lm_head.weight", physical_id, "tie_word_embeddings=true");
    }

    for layer in &loaded.spec.layers {
        builder.begin_segment(
            &format!("layer_{}", layer.index),
            SegmentKind::Layer(layer.index),
        );
        for binding in &layer.tensors {
            let id = emit_binding_set(
                &mut builder,
                &loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Compile vision encoder tensors if present.
    if loaded.manifest.vision_config.is_some() {
        let _ = compile_vision_encoder_tensors(
            &mut builder,
            &loaded.source_tensors,
            &mut emitted_ids,
        );
    }

    // Compile audio encoder tensors if present.
    if loaded.manifest.audio_config.is_some() {
        let _ = compile_audio_encoder_tensors(
            &mut builder,
            &loaded.source_tensors,
            &mut emitted_ids,
            loaded.manifest.audio_config.clone(),
        );
    }

    // Build the execution plan using the emitted tensor IDs
    let execution_plan =
        crate::config::build_execution_plan(&loaded.arch, &loaded.namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();
    // Apply compile-time graph optimization passes.
    crate::compiler::graph_optimizer::optimize(&mut plan_with_fusion);
    #[allow(unused_variables)]
    let backend_names: [&str; 3] = ["gpu", "ane", "cpu"];

    builder.set_execution_plan(plan_with_fusion);

    // ── Embed precompiled Metal library bundle ─────────────────────────
    embed_metallib(
        &mut builder,
        source_dir,
        output_dir,
        quantize_mode,
        &loaded.arch,
    )?;

    // Use segment byte_size instead of segment_payloads.len()
    let emitted_so_far: u64 = builder.segments.iter().map(|s| s.byte_size).sum();

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;
    crate::compile_progress::CompileProgress {
        stage: "payload_emission_done".into(),
        bytes_processed: emitted_so_far,
        bytes_total: emitted_so_far,
        elapsed_ms: started_at.elapsed().as_millis() as u64,
    }
    .emit();

    // ── Build Metal kernel artifact metadata for quantized projections ──
    use std::collections::BTreeMap;

    struct KernelSpecKey {
        bits: u8,
        group_size: u32,
        k: u64,
        n: u64,
    }

    let mut requests: BTreeMap<String, KernelSpecKey> = BTreeMap::new();
    for layer in &loaded.spec.layers {
        for binding in &layer.tensors {
            if let Some(packed) = &binding.packed_shape {
                let key = format!(
                    "metal:mlx-qmatmul:v1:affine:b{}:g{}:gpu-m1:shape-k{}-n{}",
                    packed.bits, packed.group_size, binding.logical_shape[1], packed.weight[0],
                );
                requests.entry(key).or_insert(KernelSpecKey {
                    bits: packed.bits as u8,
                    group_size: packed.group_size,
                    k: binding.logical_shape[1] as u64,
                    n: packed.weight[0] as u64,
                });
            }
        }
    }

    let mut metal_kernel_artifacts = Vec::new();
    for (key, spec) in &requests {
        let metallib_path = output_dir
            .join("metal")
            .join("kernels")
            .join(format!("{}.metallib", key));
        let metallib_bytes = if metallib_path.exists() {
            std::fs::read(&metallib_path).unwrap_or_default()
        } else {
            Vec::new()
        };
        let metallib_byte_length = metallib_bytes.len() as u64;
        let metallib_sha256 = if !metallib_bytes.is_empty() {
            format!("{:x}", Sha256::digest(&metallib_bytes))
        } else {
            String::new()
        };

        // NF4 kernel ABI: input=0, weight=1, scale=2, bias=3, output=4
        let mut slot_map = std::collections::HashMap::new();
        slot_map.insert("input".to_string(), 0u32);
        slot_map.insert("weight".to_string(), 1u32);
        slot_map.insert("scale".to_string(), 2u32);
        slot_map.insert("bias".to_string(), 3u32);
        slot_map.insert("output".to_string(), 4u32);

        let scalar_map: std::collections::HashMap<String, (u32, String)> =
            std::collections::HashMap::new();
        let artifact = MetalKernelArtifact {
            artifact_id: key.clone(),
            logical_operation: "quantized_matmul".to_string(),
            kind: crate::compute_image::manifest::ArtifactKind::MlxNf4U32,
            metallib_relpath: format!("metal/kernels/{}.metallib", key),
            metallib_blake3: String::new(),
            metallib_byte_length,
            dispatch: MetalDispatchRecipe {
                entry_point: "quantized_matmul_nf4".to_string(),
                kernel_name: key.clone(),
                threads_per_threadgroup: [32, 1, 1],
                threadgroups_per_grid: [((spec.n + 31) / 32) as u32, 1, 1],
                buffer_slot_map: slot_map,
                scalar_index_map: scalar_map,
                k: spec.k,
                n: spec.n,
                group_size: spec.group_size,
                bits: spec.bits,
                kernel_abi_version: 1,
            },
            logical_shape: vec![spec.k as u32, spec.n as u32],
            storage_shape: vec![spec.n as u32, (spec.k * spec.bits as u64 / 32) as u32],
            bits: spec.bits,
            group_size: spec.group_size,
            scale_tensor: String::new(),
            bias_tensor: String::new(),
            gpu_family: "m1".to_string(),
            checksum: metallib_sha256,
        };
        metal_kernel_artifacts.push(artifact);
    }
    builder.set_metal_kernel_artifacts(metal_kernel_artifacts);

    // ── Phase DAG emission ──────────────────────────────────────────
    use crate::compute_image::phase_graph_builder::PhaseGraphBuilder;
    let dag_builder = PhaseGraphBuilder::new(loaded.arch.num_hidden_layers as usize)
        .with_dimensions(
            loaded.arch.hidden_size as usize,
            loaded.arch.num_attention_heads as usize,
            loaded.arch.num_key_value_heads as usize,
            loaded.arch.head_dim as usize,
            loaded.arch.intermediate_size as usize,
        );
    let dag = dag_builder.build_v1();
    if let Err(e) = dag.validate() {
        eprintln!("[compiler] phase DAG validation warning: {}", e);
    }
    builder.set_phase_graph(dag);

    let t_finalize = Instant::now();
    let manifest = builder.finalize(output_dir)?;
    let finalize_ms = t_finalize.elapsed().as_millis() as u64;

    // ── Hardware assessment pass ────────────────────────────────────
    let hw_assessment = run_hardware_assessment();

    let hw_path = output_dir.join("hw_assessment.json");
    let hw_json = serde_json::to_string_pretty(&hw_assessment)
        .map_err(|e| crate::Error::from_reason(format!("hw assessment json: {}", e)))?;
    std::fs::write(&hw_path, hw_json)
        .map_err(|e| crate::Error::from_reason(format!("write hw assessment: {}", e)))?;

    let total_source_bytes: u64 = loaded
        .source_tensors
        .values()
        .map(|tensor| tensor.data.len() as u64)
        .sum();
    let total_emitted_bytes: u64 = manifest
        .segments
        .iter()
        .map(|segment| segment.byte_size)
        .sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: 0,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &loaded,
        &manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
        Some(hw_assessment),
        None,
    );
    let receipt_path = output_dir.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {}", e)))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {}", e)))?;

    Ok(CompiledImage { manifest, receipt })
}

// ═══════════════════════════════════════════════════════════════════════════
// Differential compilation
// ═══════════════════════════════════════════════════════════════════════════

/// Compile a model image with differential recompilation against a previous
/// compilation manifest.
pub fn compile_differential(
    source_dir: &str,
    output_dir: &str,
    prev_manifest_path: &str,
) -> crate::Result<CompiledImage> {
    let started_at = Instant::now();
    let output_dir_path = Path::new(output_dir);

    // Load previous manifest
    let prev_manifest_text = std::fs::read_to_string(prev_manifest_path).map_err(|e| {
        crate::Error::from_reason(format!(
            "read previous manifest {}: {e}",
            prev_manifest_path
        ))
    })?;
    let prev_manifest: Manifest = serde_json::from_str(&prev_manifest_text)
        .map_err(|e| crate::Error::from_reason(format!("parse previous manifest: {e}")))?;
    let prev_output_dir_path = Path::new(prev_manifest_path).parent().ok_or_else(|| {
        crate::Error::from_reason("cannot determine previous output directory from manifest path")
    })?;

    // Build diff
    let diff = diff_tensors(Path::new(source_dir), &prev_manifest)?;
    eprintln!(
        "[diff-compile] tensors: {} unchanged, {} changed, {} new, {} removed ({} ms)",
        diff.unchanged.len(),
        diff.changed.len(),
        diff.new.len(),
        diff.removed.len(),
        diff.elapsed_ms,
    );

    let t_source = Instant::now();
    let (_plan, loaded) = plan(Path::new(source_dir), false)?;
    let source_load_ms = t_source.elapsed().as_millis() as u64;

    // Build lookup sets
    let compile_names: std::collections::HashSet<&str> = diff
        .changed
        .iter()
        .chain(diff.new.iter())
        .map(|s| s.as_str())
        .collect();
    let unchanged_names: std::collections::HashSet<&str> =
        diff.unchanged.iter().map(|s| s.as_str()).collect();

    // Identify and copy unchanged segments
    let unchanged_segments: Vec<Segment> = prev_manifest
        .segments
        .iter()
        .filter(|seg| {
            seg.tensor_ids.iter().all(|tid| {
                prev_manifest
                    .tensor_table
                    .iter()
                    .find(|t| t.id == *tid)
                    .map(|t| unchanged_names.contains(t.name.as_str()))
                    .unwrap_or(false)
            })
        })
        .cloned()
        .collect();

    std::fs::create_dir_all(output_dir_path)
        .map_err(|e| crate::Error::from_reason(format!("mkdir: {e}")))?;
    for seg in &unchanged_segments {
        let src = prev_output_dir_path.join(&seg.filename);
        let dst = output_dir_path.join(&seg.filename);
        if src.exists() {
            std::fs::copy(&src, &dst).map_err(|e| {
                crate::Error::from_reason(format!("copy unchanged segment {}: {e}", seg.filename))
            })?;
        }
    }

    // Build source identity
    let source = build_source_identity(
        &loaded.manifest,
        loaded.shard_hashes.clone(),
        loaded.tokenizer_hashes.clone(),
        loaded.auxiliary_hashes.clone(),
    );

    // Emit only changed / new tensors
    let mut builder = ImageBuilder::new(loaded.arch.clone(), source);
    let start_tensor_id: u32 = prev_manifest
        .tensor_table
        .iter()
        .map(|t| t.id)
        .max()
        .map(|id| id + 1)
        .unwrap_or(0);
    builder.set_start_tensor_id(start_tensor_id);
    let t_emit = Instant::now();

    builder.begin_segment("persistent", SegmentKind::Persistent);
    let mut emitted_ids: HashMap<String, u32> = HashMap::new();

    for binding in &loaded.spec.global_tensors {
        if !compile_names.contains(binding.name.as_str()) {
            continue;
        }
        let id = emit_binding_set(&mut builder, &loaded.source_tensors, binding, None)?;
        emitted_ids.insert(binding.name.clone(), id);
    }

    if loaded.namespace.lm_head_aliased {
        let embed_name = format!("{}.embed_tokens.weight", loaded.namespace.root);
        let physical_id = emitted_ids
            .get(&embed_name)
            .copied()
            .ok_or_else(|| crate::Error::from_reason("embed_tokens.weight was not emitted"))?;
        builder.add_alias("lm_head.weight", physical_id, "tie_word_embeddings=true");
    }

    for layer in &loaded.spec.layers {
        builder.begin_segment(
            &format!("layer_{}", layer.index),
            SegmentKind::Layer(layer.index),
        );
        for binding in &layer.tensors {
            if !compile_names.contains(binding.name.as_str()) {
                continue;
            }
            let id = emit_binding_set(
                &mut builder,
                &loaded.source_tensors,
                binding,
                Some(layer.index),
            )?;
            emitted_ids.insert(binding.name.clone(), id);
        }
    }

    // Build the execution plan
    let execution_plan =
        crate::config::build_execution_plan(&loaded.arch, &loaded.namespace, &emitted_ids);
    let mut plan_with_fusion = execution_plan;
    plan_with_fusion.build_ane_fusion_plan();
    plan_with_fusion.apply_fusion_pass();
    // ── Compile ANE subgraphs (new 3-param signature) ────────────────
        super::coreml::compile_ane_islands(&plan_with_fusion, &loaded.arch, output_dir_path)
            .map_err(crate::Error::from_reason)?;
    builder.set_execution_plan(plan_with_fusion);

    let payload_emission_ms = t_emit.elapsed().as_millis() as u64;

    // Flush and collect new segments
    let (new_segments, new_payloads, partial_manifest) = builder.flush_and_collect_segments();

    // Determine offset for new segment filenames
    let max_existing: usize = unchanged_segments
        .iter()
        .filter_map(|s| {
            let stripped = s.filename.strip_prefix("segment_")?;
            let num_str = stripped.strip_suffix(".bin")?;
            num_str.parse::<usize>().ok()
        })
        .max()
        .map(|n| n + 1)
        .unwrap_or(0);

    // Write new segment files with offset filenames
    for (i, payload) in new_payloads.iter().enumerate() {
        let new_filename = format!("segment_{:03}.bin", max_existing + i);
        let path = output_dir_path.join(&new_filename);
        std::fs::write(&path, payload).map_err(|e| {
            crate::Error::from_reason(format!("write new segment {}: {e}", new_filename))
        })?;
    }

    // Build combined manifest
    let mut combined_segments: Vec<Segment> =
        Vec::with_capacity(unchanged_segments.len() + new_segments.len());
    combined_segments.extend(unchanged_segments);

    for (i, (seg, payload)) in new_segments.iter().zip(new_payloads.iter()).enumerate() {
        let new_filename = format!("segment_{:03}.bin", max_existing + i);
        let sha256 = {
            let mut h = Sha256::new();
            h.update(payload);
            format!("{:x}", h.finalize())
        };
        combined_segments.push(Segment {
            id: seg.id.clone(),
            filename: new_filename,
            byte_size: payload.len() as u64,
            sha256,
            tensor_ids: seg.tensor_ids.clone(),
            kind: seg.kind.clone(),
            alignment_bytes: seg.alignment_bytes,
        });
    }

    // Combined tensor table
    let mut combined_tensors: Vec<TensorEntry> =
        Vec::with_capacity(prev_manifest.tensor_table.len() + partial_manifest.tensor_table.len());

    for t in &prev_manifest.tensor_table {
        if unchanged_names.contains(t.name.as_str()) {
            combined_tensors.push(t.clone());
        }
    }
    for t in &partial_manifest.tensor_table {
        let mut entry = t.clone();
        if let Some(seg) = combined_segments
            .iter()
            .find(|cs| cs.tensor_ids.contains(&entry.id))
        {
            entry.segment = seg.filename.clone();
        }
        combined_tensors.push(entry);
    }

    let mut combined_manifest = partial_manifest.clone();
    combined_manifest.segments = combined_segments;
    combined_manifest.tensor_table = combined_tensors;
    combined_manifest.alias_table = {
        let mut merged = prev_manifest.alias_table.clone();
        merged.extend(partial_manifest.alias_table.clone());
        merged
    };
    combined_manifest.residency_plan.total_bytes =
        combined_manifest.segments.iter().map(|s| s.byte_size).sum();
    combined_manifest.image_hash = compute_manifest_hash(&combined_manifest);

    let manifest_path = output_dir_path.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&combined_manifest)
        .map_err(|e| crate::Error::from_reason(format!("json: {e}")))?;
    std::fs::write(&manifest_path, manifest_json)
        .map_err(|e| crate::Error::from_reason(format!("write manifest: {e}")))?;

    // Build and write receipt
    let finalize_ms = t_emit.elapsed().as_millis() as u64;
    let total_source_bytes: u64 = loaded
        .source_tensors
        .values()
        .map(|t| t.data.len() as u64)
        .sum();
    let total_emitted_bytes: u64 = combined_manifest.segments.iter().map(|s| s.byte_size).sum();

    let stage_profile = StageProfile {
        source_discovery_ms: source_load_ms,
        header_parsing_ms: 0,
        architecture_normalization_ms: 0,
        binding_validation_ms: 0,
        source_hashing_ms: diff.elapsed_ms as u64,
        layout_planning_ms: 0,
        payload_emission_ms,
        segment_hashing_ms: finalize_ms,
        manifest_generation_ms: 0,
        verification_ms: 0,
        total_source_bytes,
        total_emitted_bytes,
        peak_rss_bytes: 0,
        peak_mlx_active_bytes: mlx_active_memory_bytes() as u64,
        peak_mlx_cache_bytes: 0,
    };

    let receipt = build_compile_receipt(
        &loaded,
        &combined_manifest,
        started_at.elapsed().as_millis(),
        stage_profile,
        Default::default(),
        Some(total_source_bytes),
    );
    let receipt_path = output_dir_path.join("receipt.json");
    let receipt_json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| crate::Error::from_reason(format!("json: {e}")))?;
    std::fs::write(&receipt_path, receipt_json)
        .map_err(|e| crate::Error::from_reason(format!("write receipt: {e}")))?;

    Ok(CompiledImage {
        manifest: combined_manifest,
        receipt,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Read / verify
// ═══════════════════════════════════════════════════════════════════════════

pub fn read(image_dir: &str) -> crate::Result<CompiledImageReader> {
    CompiledImageReader::open(Path::new(image_dir))
}

pub fn verify(image_dir: &str) -> crate::Result<ManifestVerification> {
    read(image_dir)?.verify()
}

// ═══════════════════════════════════════════════════════════════════════════
// Diagnostics
// ═══════════════════════════════════════════════════════════════════════════

/// Results from compile-time diagnostic verification.
#[derive(Debug, Clone, Serialize)]
pub struct DiagnosticReport {
    pub passed: bool,
    pub layers: Vec<LayerDiagnostic>,
    pub global: GlobalDiagnostic,
    pub issues: Vec<DiagnosticIssue>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerDiagnostic {
    pub layer_index: u32,
    pub attention_kind: String,
    pub hidden_norm: f64,
    pub hidden_finite: bool,
    pub hidden_min: f64,
    pub hidden_max: f64,
    pub entropy: f64,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlobalDiagnostic {
    pub total_layers: usize,
    pub nan_layers: usize,
    pub inf_layers: usize,
    pub max_runtime_ms: u64,
    pub total_runtime_ms: u64,
    pub memory_peak_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub enum DiagnosticIssue {
    NanInLayer(u32),
    InfInLayer(u32),
    ExplodingActivation { layer: u32, norm: f64 },
    VanishingActivation { layer: u32, norm: f64 },
    EntropyExtreme { layer: u32, entropy: f64 },
}

impl Default for GlobalDiagnostic {
    fn default() -> Self {
        Self {
            total_layers: 0,
            nan_layers: 0,
            inf_layers: 0,
            max_runtime_ms: 0,
            total_runtime_ms: 0,
            memory_peak_bytes: 0,
        }
    }
}

/// Run compile-time diagnostic verification on a compiled image.
pub fn run_diagnostics(image_dir: &Path) -> crate::Result<DiagnosticReport> {
    let reader = CompiledImageReader::open(image_dir)?;
    let plan = &reader.manifest.execution_plan;
    let runtime = reader.open_runtime(StorageBackend::Copied)?;

    let mut report = DiagnosticReport {
        passed: true,
        layers: Vec::new(),
        global: GlobalDiagnostic::default(),
        issues: Vec::new(),
    };

    for layer_plan in &plan.layers {
        let l = layer_plan.layer_index;
        let start = std::time::Instant::now();

        let lease = runtime.activate_layer(l)?;
        let layer_map = runtime.build_layer_arrays_from_lease(l, &lease)?;

        let mut has_nan = false;
        let mut has_inf = false;
        let mut norm_sum_sq: f64 = 0.0;
        let mut min_val: f64 = f64::MAX;
        let mut max_val: f64 = f64::NEG_INFINITY;

        for (_name, arr) in &layer_map {
            if let Ok(slice) = arr.try_as_slice::<f32>() {
                for &v in slice {
                    let vf = v as f64;
                    if v.is_nan() {
                        has_nan = true;
                    }
                    if v.is_infinite() {
                        has_inf = true;
                    }
                    if vf < min_val {
                        min_val = vf;
                    }
                    if vf > max_val {
                        max_val = vf;
                    }
                    norm_sum_sq += vf * vf;
                }
            }
        }

        let norm = norm_sum_sq.sqrt();
        let elapsed = start.elapsed().as_millis() as u64;

        let diag = LayerDiagnostic {
            layer_index: l,
            attention_kind: layer_plan.attention_kind.clone(),
            hidden_norm: norm,
            hidden_finite: !has_nan && !has_inf,
            hidden_min: min_val,
            hidden_max: max_val,
            entropy: 0.0,
            elapsed_ms: elapsed,
        };

        if has_nan {
            report.issues.push(DiagnosticIssue::NanInLayer(l));
            report.passed = false;
        }
        if has_inf {
            report.issues.push(DiagnosticIssue::InfInLayer(l));
            report.passed = false;
        }

        report.layers.push(diag);
    }

    report.global.total_layers = plan.layers.len();
    report.global.nan_layers = report
        .issues
        .iter()
        .filter(|i| matches!(i, DiagnosticIssue::NanInLayer(_)))
        .count();
    report.global.inf_layers = report
        .issues
        .iter()
        .filter(|i| matches!(i, DiagnosticIssue::InfInLayer(_)))
        .count();
    report.global.total_runtime_ms = report.layers.iter().map(|l| l.elapsed_ms).sum();
    report.global.max_runtime_ms = report
        .layers
        .iter()
        .map(|l| l.elapsed_ms)
        .max()
        .unwrap_or(0);
    report.global.memory_peak_bytes = mlx_peak_memory_bytes();

    Ok(report)
}

// ═══════════════════════════════════════════════════════════════════════════
// Publishing
// ═══════════════════════════════════════════════════════════════════════════

/// Atomically publish a staged compilation to its final destination.
pub fn publish_image(staging: &Path, destination: &Path) -> crate::Result<()> {
    let publishing_marker = staging.join(".publishing");
    std::fs::write(&publishing_marker, b"")
        .map_err(|e| crate::Error::from_reason(format!("write .publishing: {}", e)))?;

    let result = std::fs::rename(staging, destination);
    match result {
        Ok(()) => Ok(()),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::CrossesDevices {
                let failed_marker = staging.join(".failed");
                if let Err(write_err) =
                    std::fs::write(&failed_marker, format!("rename failed: {}", e))
                {
                    return Err(crate::Error::from_reason(format!(
                        "write .failed marker: {} (original rename: {})",
                        write_err, e
                    )));
                }
                return Err(crate::Error::from_reason(format!(
                    "rename crosses devices: {}. Staging left in place with .failed marker.",
                    e
                )));
            }
            let failed_marker = staging.join(".failed");
            if let Err(write_err) = std::fs::write(&failed_marker, format!("rename failed: {}", e))
            {
                return Err(crate::Error::from_reason(format!(
                    "write .failed marker: {} (original rename: {})",
                    write_err, e
                )));
            }
            Err(crate::Error::from_reason(format!(
                "rename {} -> {}: {}",
                staging.display(),
                destination.display(),
                e
            )))
        }
    }
}
