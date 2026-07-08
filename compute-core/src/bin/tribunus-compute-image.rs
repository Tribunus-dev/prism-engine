//! tribunus-compute-image — CLI for building and verifying ComputeImage directories.
//!
//! Commands:
//!   build  --source <dir> --output <dir>
//!   verify --image <dir> [--expected-hash <hash>] [--full]

use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use tribunus_compute_core::compute_image;
use tribunus_compute_core::config::CompileQuantMode;
use tribunus_compute_core::config::HardwareTarget;
use tribunus_compute_core::kv_cache::KvCache;
use tribunus_compute_core::profiled_executor::{LoadedProfiledModel, ProfiledInferenceSession};
use tribunus_compute_core::nf4tile640::{
    nf4_dequantize, pack_int8_weights, pack_nf4_tile_with_group_size,
    pack_nf4_weights, pack_nf4_weights_awls, pack_symmetric_int4_tile,
    unpack_int8_weights, unpack_nf4_weights,
};
use tribunus_compute_core::quantization::admission::compute_weight_nrmse;
use tribunus_compute_core::quantization::substitution::SubstitutionCandidate;
use tribunus_compute_core::quantization::substitution::SubstitutionContext;
use tribunus_compute_core::quantization::substitution_pass::try_all_candidates;
use tribunus_compute_core::quantization::embed_cluster::{
    pack_ternary_weights, unpack_ternary_weights,
};

// ═══════════════════════════════════════════════════════════════════════════
// Entry point
// ═══════════════════════════════════════════════════════════════════════════

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  tribunus-compute-image build --source <dir> --output <dir>");
        eprintln!("       source can be a local path or hf:org/model[@revision]");
        eprintln!("       [--draft-model <dir>] [--diagnostic] [--quantize <mode>]");
        eprintln!("       [--diff <manifest.json>]");
        eprintln!("       [--target <target>]");
        eprintln!("    quantize modes: nf4, nf4-128, nf4tile640, 8bit");
        eprintln!("    quantize modes: nf4, nf4-128, nf4tile640, 8bit, none (default: hardware auto-detect)");
        eprintln!("    targets: m1, m1pro, m2, m2ultra, m3ultra (default: auto-detect)");
        eprintln!(
            "  tribunus-compute-image verify --image <dir> [--expected-hash <hash>] [--full]"
        );
        eprintln!("  tribunus-compute-image infer --image <dir>");
        eprintln!("  tribunus-compute-image decode-one --image <dir>");
        eprintln!("  tribunus-compute-image emit-v0 --output-dir <dir> [--allow-contract-only-kv]");
        eprintln!("  tribunus-compute-image verify-v0 --image <dir>");
        eprintln!("  tribunus-compute-image build-ecs --source <dir> [--draft-source <dir>] [--tts-source <dir>] --output <dir>");
        eprintln!("       [--substitution <mode>]  (mode: try)");
        eprintln!("  tribunus-compute-image quant-sweep --source <dir> --output <out> [--tensor-regex <re>] [--max-candidates <n>]");
        std::process::exit(1);
    }

    let result = match args[1].as_str() {
        "build" => cmd_build(&args[2..]),
        "build-ecs" => cmd_build_ecs(&args[2..]),
        "verify" => cmd_verify(&args[2..]),
        "infer" => cmd_infer(&args[2..]),
        "decode-one" => cmd_decode_one(&args[2..]),
        "emit-v0" => cmd_emit_v0(&args[2..]),
        "verify-v0" => cmd_verify_v0(&args[2..]),
        "quant-sweep" => cmd_quant_sweep(&args[2..]),
        other => {
            tribunus_compute_core::log_error!("unknown command: {other}");
            std::process::exit(1);
        }
    };

    if let Err(e) = result {
        tribunus_compute_core::log_error!("error: {}", e);
        tribunus_compute_core::log_error!("error: {}", e);
        std::process::exit(1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Argument helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Look up `--key` in `args` and return the following value, or `None`.
fn get_opt<'a>(args: &'a [String], key: &str) -> Option<&'a str> {
    args.windows(2).find_map(|w| {
        if w[0] == key {
            Some(w[1].as_str())
        } else {
            None
        }
    })
}

/// Return `true` if `--flag` appears anywhere in `args`.
fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

// ═══════════════════════════════════════════════════════════════════════════
// build command
/// ═══════════════════════════════════════════════════════════════════════════

fn cmd_build(args: &[String]) -> Result<(), String> {
    let source = get_opt(args, "--source").ok_or_else(|| "--source is required".to_string())?;
    let output = get_opt(args, "--output").ok_or_else(|| "--output is required".to_string())?;
    let diff_manifest = get_opt(args, "--diff");
    let draft_model = get_opt(args, "--draft-model");
    let diagnostic = has_flag(args, "--diagnostic");
    let quantize_mode = get_opt(args, "--quantize")
        .map(|q| match q {
            "nf4" => Ok(CompileQuantMode::Nf4 { group_size: 64 }),
            "nf4-128" => Ok(CompileQuantMode::Nf4 { group_size: 128 }),
            "nf4tile640" | "nf4-tile640" | "nftile640" => {
                Ok(CompileQuantMode::Nf4Tile640 { group_size: 128 })
            }
            "8bit" => Ok(CompileQuantMode::Af8 { group_size: 64 }),
            "none" => Ok(CompileQuantMode::Nf4 { group_size: 64 }),
            other => Err(format!(
                "unknown quantize mode: '{other}'. Expected nf4, nf4-128, nf4tile640, 8bit, or none"
            )),
        })
        .transpose()?;

    let target = get_opt(args, "--target")
        .map(|t| match t.to_lowercase().as_str() {
            "m1" => Ok(HardwareTarget::M1),
            "m1pro" => Ok(HardwareTarget::M1Pro),
            "m2" => Ok(HardwareTarget::M2),
            "m2ultra" => Ok(HardwareTarget::M2Ultra),
            "m3ultra" => Ok(HardwareTarget::M3Ultra),
            other => Err(format!(
                "unknown target: '{other}'. Expected m1, m1pro, m2, m2ultra, or m3ultra"
            )),
        })
        .transpose()?;

    let output_path = Path::new(output);

    // Refuse to overwrite an existing output directory.
    if output_path.exists() {
        return Err(format!(
            "output directory already exists. Refusing to overwrite sealed image."
        ));
    }

    // Profile attestation — print before compiling
    let attestation = compute_image::image_build_attestation();
    println!("{}", serde_json::to_string(&attestation).unwrap());

    // Create staging directory.
    let uuid = Uuid::new_v4();
    let staging = format!("{output}.build-{uuid}");
    let staging_path = Path::new(&staging);

    fs::create_dir_all(staging_path).map_err(|e| format!("create staging dir {staging}: {e}"))?;

    // Compile into staging.
    let compile_start = Instant::now();
    // Resolve source: if --source starts with "hf:", stream from HuggingFace.
    let (_hf_download_dir, compile_source, seal_source) =
        if let Some(hf_source) = source.strip_prefix("hf:") {
            let parts: Vec<&str> = hf_source.splitn(2, '@').collect();
            let hub_id = parts[0];
            let revision = parts.get(1).copied().unwrap_or("main");

            tribunus_compute_core::log_info!(
                "[build] streaming from HuggingFace: hub={hub_id}, revision={revision}"
            );

            let download_dir =
                tempfile::tempdir().map_err(|e| format!("create HF download dir: {e}"))?;
            let download_path: PathBuf = download_dir.path().to_path_buf();

            compute_image::download_hf_model(hub_id, revision, &download_path, None)
                .map_err(|e| format!("HF download failed: {e}"))?;

            let compile_source = download_path
                .to_str()
                .ok_or_else(|| "invalid download path".to_string())?
                .to_string();
            let seal_source = source.to_string();
            (Some(download_dir), compile_source, seal_source)
        } else {
            let compile_source = source.to_string();
            let seal_source = source.to_string();
            (None, compile_source, seal_source)
        };

    let compiled = if let Some(draft) = draft_model {
        tribunus_compute_core::log_info!(
            "[build] speculative compile: target={} draft={}",
            compile_source,
            draft
        );
        compute_image::compile_with_authority_speculative(
            &compile_source,
            draft,
            &staging,
            compute_image::CompilationAuthority::SealedComputeImage,
            quantize_mode,
            target,
        )
        .map_err(|e| format!("speculative compilation failed: {e}"))?
    } else if let Some(prev) = diff_manifest {
        tribunus_compute_core::log_info!("[build] differential compile against {}", prev);
        compute_image::compile_differential(&compile_source, &staging, prev)
            .map_err(|e| format!("differential compilation failed: {e}"))?
    } else {
        compute_image::compile_with_authority(
            &compile_source,
            &staging,
            compute_image::CompilationAuthority::SealedComputeImage,
            false,
            quantize_mode,
            target,
        )
        .map_err(|e| format!("compilation failed: {e}"))?
    };
    let compile_ns = compile_start.elapsed().as_nanos() as u64;
    let compile_duration_s = compile_ns as f64 / 1_000_000_000.0;

    // Extract fields from the compiled output.
    let image_hash = compiled.manifest.image_hash.clone();
    let segment_count = compiled.manifest.segments.len();
    let tensor_count = compiled.manifest.tensor_table.len();
    let storage_abi = compiled.manifest.required_storage_abi.clone();
    let runtime_abi = compiled.manifest.runtime_abi.clone();

    // Reopen and validate with CompiledImageReader.
    let reader =
        compute_image::read(&staging).map_err(|e| format!("reopen staging image failed: {e}"))?;

    // Validate execution plan.
    let plan_errors = reader.manifest.execution_plan.validate();
    if let Err(errs) = plan_errors {
        let joined = errs.join("; ");
        return Err(format!("execution plan validation failed: {joined}"));
    }

    // Verify all segment files exist on disk. Full hash verification is a
    // separate concern handled by the verify command.
    for seg in &reader.manifest.segments {
        let seg_path = staging_path.join(&seg.filename);
        if !seg_path.exists() {
            return Err(format!("missing segment file: {}", seg.filename));
        }
    }

    // Write seal.json.
    let compiler_commit = env!("CARGO_PKG_VERSION");
    let builder_sha256 = {
        let exe_path = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
        let mut file = File::open(&exe_path).map_err(|e| format!("open {:?}: {e}", exe_path))?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 65536];
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| format!("read {:?}: {e}", exe_path))?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        format!("{:x}", hasher.finalize())
    };
    // Compute artifact root hash from all segment files (parallel with rayon)
    tribunus_compute_core::log_info!(
        "[build] computing artifact root hash (parallel, {} segments)...",
        compiled.manifest.segments.len()
    );
    let mut root_hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    for seg in &compiled.manifest.segments {
        let sp = staging_path.join(&seg.filename);
        let mut file =
            File::open(&sp).map_err(|e| format!("open segment {}: {}", seg.filename, e))?;
        loop {
            let n = file
                .read(&mut buf)
                .map_err(|e| format!("read segment {}: {}", seg.filename, e))?;
            if n == 0 {
                break;
            }
            root_hasher.update(&buf[..n]);
        }
    }
    let artifact_root_hash = format!("{:x}", root_hasher.finalize());
    tribunus_compute_core::log_info!(
        "[build] artifact_root_hash: {}...",
        &artifact_root_hash[..16]
    );

    let sealed_at = format_iso8601(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs(),
    );

    let seal = json!({
        "status": "sealed",
        "image_hash": image_hash,
        "artifact_root_hash": artifact_root_hash,
        "manifest_image_hash": image_hash,
        "builder_sha256": builder_sha256,
        "segment_count": segment_count,
        "tensor_count": tensor_count,
        "compile_duration_s": compile_duration_s,
        "storage_abi": storage_abi,
        "runtime_abi": runtime_abi,
        "source_dir": &seal_source,
        "compiler_commit": compiler_commit,
        "sealed_at": sealed_at,
    });

    let seal_path = staging_path.join("seal.json");
    let seal_json =
        serde_json::to_string_pretty(&seal).map_err(|e| format!("serialize seal.json: {e}"))?;
    fs::write(&seal_path, &seal_json).map_err(|e| format!("write seal.json: {e}"))?;

    // Flush all files.
    sync_dir(staging_path)?;

    // Atomic rename: staging -> output.
    fs::rename(staging_path, output_path)
        .map_err(|e| format!("rename {staging} -> {output}: {e}"))?;

    // Print success JSON.
    let out = json!({
        "status": "sealed",
        "image_dir": output,
        "image_hash": image_hash,
        "segment_count": segment_count,
        "tensor_count": tensor_count,
        "compile_ns": compile_ns,
        "storage_abi": storage_abi,
        "runtime_abi": runtime_abi,
    });
    println!("{}", serde_json::to_string(&out).unwrap());

    // Run compile-time diagnostics if requested.
    if diagnostic {
        tribunus_compute_core::log_info!("Running compile-time diagnostic verification...");
        match compute_image::run_diagnostics(output_path) {
            Ok(diag_report) => {
                // Write diagnostic.json to the output directory.
                let diag_json = serde_json::to_string_pretty(&diag_report)
                    .map_err(|e| format!("serialize diagnostic.json: {e}"))?;
                let diag_path = output_path.join("diagnostic.json");
                fs::write(&diag_path, &diag_json)
                    .map_err(|e| format!("write diagnostic.json: {e}"))?;

                let passed_str = if diag_report.passed {
                    "PASSED"
                } else {
                    "FAILED"
                };
                tribunus_compute_core::log_info!("=== Compile-time Diagnostics ===");
                tribunus_compute_core::log_info!(
                    "Layers: {}/{} checked",
                    diag_report.layers.len(),
                    diag_report.global.total_layers
                );
                tribunus_compute_core::log_info!("NaN layers: {}", diag_report.global.nan_layers);
                tribunus_compute_core::log_info!("Inf layers: {}", diag_report.global.inf_layers);
                tribunus_compute_core::log_info!("Issues: {}", diag_report.issues.len());
                tribunus_compute_core::log_info!(
                    "Max activation norm: {:.3}",
                    diag_report
                        .layers
                        .iter()
                        .map(|l| l.hidden_norm)
                        .fold(0.0_f64, f64::max)
                );
                tribunus_compute_core::log_info!(
                    "Max layer runtime: {} ms",
                    diag_report.global.max_runtime_ms
                );
                tribunus_compute_core::log_info!("Total: {passed_str}");
            }
            Err(e) => {
                tribunus_compute_core::log_warn!("warning: diagnostics failed: {e}");
            }
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// build-ecs command — stage-graph ECS compilation pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Experimental: compile a model using the stage-graph ECS pipeline.
/// Produces one .cimage file per stage in the output directory.
fn cmd_build_ecs(args: &[String]) -> Result<(), String> {
    let source = get_opt(args, "--source").ok_or_else(|| "--source is required".to_string())?;
    let draft_source = get_opt(args, "--draft-source");
    let tts_source = get_opt(args, "--tts-source");
    let output = get_opt(args, "--output").ok_or_else(|| "--output is required".to_string())?;
    let substitution_mode = get_opt(args, "--substitution");

    use std::path::Path;
    use std::fs;
    use std::fs::File;
    use safetensors::SafeTensors;
    use memmap2::Mmap;
    use tribunus_compute_core::runtime::compilation_systems::{
        compile_stage, TensorInput, ModelConfig,
    };
    use tribunus_compute_core::runtime::stage_graph::{
        StageConfig, ComponentType, StageQuantizationConfig,
    };
    use tribunus_compute_core::compute_image::compile::capability_registry::CapabilityRegistry;
    use tribunus_compute_core::quantization::contract::{CanonicalShape, BackendKind, WeightValidationReport};
    use tribunus_compute_core::quantization::contract::QuantizationValidationProfile;
    use tribunus_compute_core::quantization::validation::validate_weight_space;
    use tribunus_compute_core::quantization::admission::compute_weight_nrmse;

    let output_dir = Path::new(output);
    fs::create_dir_all(output_dir).map_err(|e| format!("create output dir: {e}"))?;

    // Source 0: main model. Source 1 (optional): draft. Source 2 (optional): TTS.
    let source_dirs: [(&str, Option<&str>); 3] = [
        (source, None),
        (draft_source.unwrap_or(""), Some("MtpDraft")),
        (tts_source.unwrap_or(""), Some("AudioEncoder")),
    ];

    // ── Types and helpers ─────────────────────────────────────────────────
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum TensorGroup {
        Embedding, Decoder, LmHead, Norm, VisionEncoder, AudioEncoder, MtpDraft, Other,
    }
    fn classify_key(name: &str) -> TensorGroup {
        let n = name;
        if n.contains("optimizer") || n.contains("momentum") || n.contains("_cache")
            || n.contains("adam_") || n.contains("rmsprop") { return TensorGroup::Other; }
        if n.contains("mtp") || n.contains("draft") || n.contains("speculative")
            || n.contains("proposal") || n.contains("dspark") || n.contains("confidence_head")
            || n.contains("dflash") || n.contains("eagle") { return TensorGroup::MtpDraft; }
        if n.contains("multimodal_image") || n.contains("mm_image")
            || n.contains("vision_") || n.contains("vision.")
            || n.contains("image_") || n.contains("image.")
            || n.contains("patch") || (n.contains("projection") && n.contains("layers")) {
            return TensorGroup::VisionEncoder;
        }
        if n.contains("multimodal_audio") || n.contains("mm_audio")
            || n.contains("audio_") || n.contains("audio.")
            || n.contains("waveform") || n.contains("speech") {
            return TensorGroup::AudioEncoder;
        }
        if n.contains("self_attn") || n.contains("mlp.")
            || n.contains("input_layernorm") || n.contains("post_attention_layernorm")
            || (n.contains(".layers.") && (n.ends_with(".weight") || n.ends_with(".bias"))) {
            return TensorGroup::Decoder;
        }
        if n.contains("embed_tokens") || n.contains("embed.") || n.contains("wte")
            || n.contains("tok_embeddings") { return TensorGroup::Embedding; }
        if n.contains("lm_head") || n.contains("output.") || n.contains("embed_out")
            || n.contains("head.") { return TensorGroup::LmHead; }
        if n.contains("norm.") || n.contains("final_layernorm") || n.contains("ln_f") {
            return TensorGroup::Norm;
        }
        TensorGroup::Other
    }

    /// Metadata-only tensor record (no weight data loaded).
    struct TensorMeta {
        key: String,
        shape: Vec<usize>,
        group: TensorGroup,
        layer: u32,  // 0 for non-layer tensors
    }

    // ── Phase 1: Scan tensor metadata only (no weight data loaded) ──────
    let mut hidden_dim = 0u32;
    let mut num_layers = 0u32;
    let mut num_heads = 0u32;
    let mut head_dim = 0u32;
    let mut intermediate_dim = 0u32;
    let mut vocab_size = 0u32;
    let mut tensor_meta: Vec<TensorMeta> = Vec::new();

    for (source_dir, group_override) in &source_dirs {
        if source_dir.is_empty() { continue; }
        let source_dir = Path::new(source_dir);
        for entry in fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))? {
            let entry = entry.map_err(|e| format!("entry: {e}"))?;
            let path = entry.path();
            if !path.extension().map_or(false, |e| e == "safetensors") { continue; }
            let file = File::open(&path).map_err(|e| format!("open {path:?}: {e}"))?;
            let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
            let tensors = SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;

            for (key, view) in tensors.tensors() {
                let shape: Vec<usize> = view.shape().to_vec();
                let group = match group_override {
                    Some(s) if *s == "MtpDraft" => TensorGroup::MtpDraft,
                    Some(s) if *s == "AudioEncoder" => TensorGroup::AudioEncoder,
                    _ => classify_key(&key),
                };
                // Extract layer number from "layers.N" in the key
                let layer = key.split('.')
                    .skip_while(|s| *s != "layers")
                    .nth(1)
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0);
                tensor_meta.push(TensorMeta {
                    key: key.clone(),
                    shape: shape.clone(),
                    group: group,
                    layer,
                });

                // Infer model dimensions from keys/shapes
                if shape.len() >= 2 && vocab_size == 0 {
                    if key.contains("embed_tokens") || key.contains("lm_head") {
                        vocab_size = shape[0] as u32;
                        hidden_dim = shape[1] as u32;
                    }
                }
                if key.contains("self_attn.q_proj") && num_heads == 0 && shape.len() >= 2 {
                    hidden_dim = shape[1] as u32;
                    head_dim = 256;
                }
                if key.contains("mlp.gate_proj") && intermediate_dim == 0 && !shape.is_empty() {
                    intermediate_dim = shape[0] as u32;
                }
                if key.contains("layers.") {
                    let n = key.split('.').filter_map(|s| s.parse::<u32>().ok()).next().unwrap_or(0);
                    num_layers = num_layers.max(n + 1);
                }
            }
        }
    }
    if tensor_meta.is_empty() {
        return Err("No safetensor files found in source directory".into());
    }
    eprintln!("scanned {} tensors, {} layers, hidden={}, vocab={}",
        tensor_meta.len(), num_layers, hidden_dim, vocab_size);

    // ── Diagnose a single tensor ─────────────────────────────────────────
    if let Some(diag_key) = get_opt(args, "--diagnose-tensor") {
        let meta = tensor_meta.iter().find(|m| m.key == diag_key)
            .ok_or_else(|| format!("tensor not found: {diag_key}"))?;

        if meta.shape.len() != 2 {
            return Err(format!("diagnose-tensor requires a 2D tensor, got {:?} dims", meta.shape.len()));
        }
        let in_features = meta.shape[0];
        let out_features = meta.shape[1];

        // Reload safetensors to get the actual f32 data for this single tensor
        let mut source_f32: Option<Vec<f32>> = None;
        for (source_dir, _group_override) in &source_dirs {
            if source_dir.is_empty() { continue; }
            let source_dir = Path::new(source_dir);
            for entry in fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))? {
                let entry = entry.map_err(|e| format!("entry: {e}"))?;
                let path = entry.path();
                if !path.extension().map_or(false, |e| e == "safetensors") { continue; }
                let file = File::open(&path).map_err(|e| format!("open {path:?}: {e}"))?;
                let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
                let tensors = SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;
                for (key, view) in tensors.tensors() {
                    if key != diag_key { continue; }
                    let dtype = view.dtype();
                    let data = view.data().to_vec();
                    source_f32 = Some(match dtype {
                        safetensors::Dtype::F32 => data.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        safetensors::Dtype::BF16 => data.chunks_exact(2)
                            .map(|c| {
                                let bits = ((c[0] as u32) << 16) | ((c[1] as u32) << 24);
                                f32::from_bits(bits)
                            })
                            .collect(),
                        _ => return Err(format!("unsupported dtype {:?} for diagnose-tensor", dtype)),
                    });
                }
            }
            if source_f32.is_some() { break; }
        }
        let source = source_f32.ok_or_else(|| format!("could not load tensor data for: {diag_key}"))?;
        let total_elements = source.len() as f64;

        // Create a default profile for weight-space validation reporting.
        let diag_profile = QuantizationValidationProfile {
            tensor_class: tribunus_compute_core::quantization::contract::TensorClass::DecoderAttentionProjection,
            phase: tribunus_compute_core::quantization::contract::ProfilePhase::Promotion,
            max_weight_nrmse: f64::MAX,
            investigation_nrmse_ceiling: f64::MAX,
            max_zero_collapse_ratio: f64::MAX,
            max_operator_nrmse: f32::MAX,
            min_mean_cosine: 0.0,
            min_worst_cosine: 0.0,
            max_norm_ratio_drift: f32::MAX,
        };

        // Variant 1: NF4 max-abs
        let (codes1, scales1, biases1, _, _) = pack_nf4_weights(&source, in_features, out_features);
        let recon1 = unpack_nf4_weights(&codes1, &scales1, &biases1, in_features, out_features);
        let wr1 = validate_weight_space(&source, &recon1, &diag_profile);
        let nrmse1_legacy = compute_weight_nrmse(&source, &recon1);

        // Variant 2: NF4 AffineUniform (all-ones activation weights, 8 iters)
        let (codes2, scales2, biases2, _, _) = pack_nf4_weights_awls(&source, in_features, out_features, None, 8);
        let recon2 = unpack_nf4_weights(&codes2, &scales2, &biases2, in_features, out_features);
        let wr2 = validate_weight_space(&source, &recon2, &diag_profile);
        let nrmse2_legacy = compute_weight_nrmse(&source, &recon2);

        // Variant 3: INT8
        let (codes3, scales3, biases3) = pack_int8_weights(&source, in_features, out_features);
        let recon3 = unpack_int8_weights(&codes3, &scales3, &biases3, in_features, out_features);
        let wr3 = validate_weight_space(&source, &recon3, &diag_profile);
        let nrmse3_legacy = compute_weight_nrmse(&source, &recon3);

        // Variant 4: Ternary
        let (codes4, scales4, biases4) = pack_ternary_weights(&source, in_features, out_features);
        let recon4 = unpack_ternary_weights(&codes4, &scales4, &biases4, in_features, out_features);
        let wr4 = validate_weight_space(&source, &recon4, &diag_profile);
        let nrmse4_legacy = compute_weight_nrmse(&source, &recon4);

        let source_shape: Vec<usize> = meta.shape.clone();
        

        // ── Codec Sweep: NF4 codebook × group sizes ────────────────────────
        const SWEEP_GROUP_SIZES: [usize; 3] = [32, 64, 128];
        let mut sweep_variants: Vec<serde_json::Value> = Vec::new();

        // Helper closure: pack entire weight matrix at tile level with custom pack function.
        // Tiles are output-axis (contiguous row slices of 640).
        let do_sweep = |pack_tile: &dyn Fn(&[f32; 640], usize) -> (Vec<u8>, Vec<f32>, Vec<f32>),
                        is_nf4: bool, gs: usize|
        {
            let padded_cols = if out_features % 640 == 0 { out_features } else { (out_features.div_ceil(640)) * 640 };
            let tiles_per_row = padded_cols / 640;
            let total_tiles = in_features * tiles_per_row;
            let groups_per_tile = 640 / gs;
            let bpgs = gs / 2; // bytes per group of packed codes

            let mut packed = vec![0u8; total_tiles * groups_per_tile * bpgs];
            let mut scales = vec![0.0f32; total_tiles * groups_per_tile];
            let mut biases = vec![0.0f32; total_tiles * groups_per_tile];
            let mut recon = vec![0.0f32; in_features * out_features];

            for row in 0..in_features {
                for tile_in_row in 0..tiles_per_row {
                    let tile_idx = row * tiles_per_row + tile_in_row;
                    let col_start = tile_in_row * 640;
                    let mut tile_vals = [0.0f32; 640];
                    for i in 0..640 {
                        let c = col_start + i;
                        tile_vals[i] = if c < out_features { source[row * out_features + c] } else { 0.0 };
                    }
                    let (codes_tile, scale_tile, bias_tile) = pack_tile(&tile_vals, gs);

                    let codes_off = tile_idx * groups_per_tile * bpgs;
                    packed[codes_off..codes_off + groups_per_tile * bpgs].copy_from_slice(&codes_tile);

                    let scale_off = tile_idx * groups_per_tile;
                    scales[scale_off..scale_off + groups_per_tile].copy_from_slice(&scale_tile);
                    biases[scale_off..scale_off + groups_per_tile].copy_from_slice(&bias_tile);

                    // Dequantize tile for validation
                    for g in 0..groups_per_tile {
                        let s = scale_tile[g];
                        let b = bias_tile[g];
                        for j in 0..gs / 2 {
                            let byte = codes_tile[g * bpgs + j];
                            let code0 = byte & 0x0F;
                            let code1 = byte >> 4;
                            let abs_col0 = col_start + g * gs + 2 * j;
                            let abs_col1 = abs_col0 + 1;
                            let (v0, v1) = if is_nf4 {
                                (nf4_dequantize(code0) * s + b, nf4_dequantize(code1) * s + b)
                            } else {
                                let dq = |idx: u8| { ((idx as i8) - 7) as f32 * s + b };
                                (dq(code0), dq(code1))
                            };
                            if abs_col0 < out_features { recon[row * out_features + abs_col0] = v0; }
                            if abs_col1 < out_features { recon[row * out_features + abs_col1] = v1; }
                        }
                    }
                }
            }

            let wr = validate_weight_space(&source, &recon, &diag_profile);
            (wr, recon)
        };

        // NF4 sweep
        for &gs in &SWEEP_GROUP_SIZES {
            let pack_nf4 = |vals: &[f32; 640], gs: usize| pack_nf4_tile_with_group_size(vals, gs);
            let (wr, rcon) = do_sweep(&pack_nf4, true, gs);
            let legacy = compute_weight_nrmse(&source, &rcon);
            let groups_per_tile = 640 / gs;
            let padded_cols = if out_features % 640 == 0 { out_features } else { (out_features.div_ceil(640)) * 640 };
            let tiles_per_row = padded_cols / 640;
            let total_tiles = in_features * tiles_per_row;
            sweep_variants.push(serde_json::json!({
                "format": "Nf4Tile640Base",
                "policy": format!("MaxAbs_group{}", gs),
                "code_bytes": total_tiles * groups_per_tile * (gs / 2),
                "scale_count": total_tiles * groups_per_tile,
                "bias_count": total_tiles * groups_per_tile,
                "weight_nrmse": (wr.nrmse * 10000.0).round() / 10000.0,
                "nrmse_legacy": (legacy * 10000.0).round() / 10000.0,
                "zero_collapse_ratio": (wr.zero_collapse_ratio * 10000.0).round() / 10000.0,
                "rmse": (wr.rmse * 10000.0).round() / 10000.0,
                "max_abs_error": (wr.max_abs_error * 10000.0).round() / 10000.0,
            }));
        }

        // SymmetricInt4 sweep
        for &gs in &SWEEP_GROUP_SIZES {
            let pack_sym = |vals: &[f32; 640], gs: usize| pack_symmetric_int4_tile(vals, gs);
            let (wr, rcon) = do_sweep(&pack_sym, false, gs);
            let legacy = compute_weight_nrmse(&source, &rcon);
            let groups_per_tile = 640 / gs;
            let padded_cols = if out_features % 640 == 0 { out_features } else { (out_features.div_ceil(640)) * 640 };
            let tiles_per_row = padded_cols / 640;
            let total_tiles = in_features * tiles_per_row;
            sweep_variants.push(serde_json::json!({
                "format": "SymInt4Tile640Base",
                "policy": format!("Symmetric_group{}", gs),
                "code_bytes": total_tiles * groups_per_tile * (gs / 2),
                "scale_count": total_tiles * groups_per_tile,
                "bias_count": total_tiles * groups_per_tile,
                "weight_nrmse": (wr.nrmse * 10000.0).round() / 10000.0,
                "nrmse_legacy": (legacy * 10000.0).round() / 10000.0,
                "zero_collapse_ratio": (wr.zero_collapse_ratio * 10000.0).round() / 10000.0,
                "rmse": (wr.rmse * 10000.0).round() / 10000.0,
                "max_abs_error": (wr.max_abs_error * 10000.0).round() / 10000.0,
            }));
        }
let mut receipt = serde_json::json!({
            "tensor_key": diag_key,
            "source_shape": source_shape,
            "in_features": in_features,
            "out_features": out_features,
            "variants": [
                {
                    "format": "Nf4Tile640Base",
                    "policy": "MaxAbs",
                    "code_bytes": codes1.len(),
                    "scale_count": scales1.len(),
                    "bias_count": biases1.len(),
                    "weight_nrmse": (wr1.nrmse * 10000.0).round() / 10000.0,
                    "nrmse_legacy": (nrmse1_legacy * 10000.0).round() / 10000.0,
                    "zero_collapse_ratio": (wr1.zero_collapse_ratio * 10000.0).round() / 10000.0,
                    "rmse": (wr1.rmse * 10000.0).round() / 10000.0,
                    "max_abs_error": (wr1.max_abs_error * 10000.0).round() / 10000.0,
                },
                {
                    "format": "Nf4Tile640Base",
                    "policy": "AffineUniform",
                    "code_bytes": codes2.len(),
                    "scale_count": scales2.len(),
                    "bias_count": biases2.len(),
                    "weight_nrmse": (wr2.nrmse * 10000.0).round() / 10000.0,
                    "nrmse_legacy": (nrmse2_legacy * 10000.0).round() / 10000.0,
                    "zero_collapse_ratio": (wr2.zero_collapse_ratio * 10000.0).round() / 10000.0,
                    "rmse": (wr2.rmse * 10000.0).round() / 10000.0,
                    "max_abs_error": (wr2.max_abs_error * 10000.0).round() / 10000.0,
                },
                {
                    "format": "Int8Tile640Base",
                    "policy": "Symmetric",
                    "code_bytes": codes3.len(),
                    "scale_count": scales3.len(),
                    "bias_count": biases3.len(),
                    "weight_nrmse": (wr3.nrmse * 10000.0).round() / 10000.0,
                    "nrmse_legacy": (nrmse3_legacy * 10000.0).round() / 10000.0,
                    "zero_collapse_ratio": (wr3.zero_collapse_ratio * 10000.0).round() / 10000.0,
                    "rmse": (wr3.rmse * 10000.0).round() / 10000.0,
                    "max_abs_error": (wr3.max_abs_error * 10000.0).round() / 10000.0,
                },
                {
                    "format": "TernaryTile640Base",
                    "policy": "BlockTernary",
                    "code_bytes": codes4.len(),
                    "scale_count": scales4.len(),
                    "bias_count": biases4.len(),
                    "weight_nrmse": (wr4.nrmse * 10000.0).round() / 10000.0,
                    "nrmse_legacy": (nrmse4_legacy * 10000.0).round() / 10000.0,
                    "zero_collapse_ratio": (wr4.zero_collapse_ratio * 10000.0).round() / 10000.0,
                    "rmse": (wr4.rmse * 10000.0).round() / 10000.0,
                    "max_abs_error": (wr4.max_abs_error * 10000.0).round() / 10000.0,
                },
            ],
        });
        // Merge codec sweep variants into the output
        if let Some(variants) = receipt["variants"].as_array_mut() {
            variants.extend(sweep_variants);
        }
        // ── Substitution pass: ranked codec candidates ────────────────────
        // Build substitution context from tensor metadata and policy hints
        let is_audio_encoder = meta.group == TensorGroup::AudioEncoder;
        let ctx = SubstitutionContext {
            rawf32_required: is_audio_encoder,
            disallowed_codecs: if is_audio_encoder {
                vec!["Ternary".into(), "SymInt4".into(), "NF4".into()]
            } else {
                Vec::new()
            },
            hardware_available: false,
            rollout_available: false,
            operator_backend: "synthetic_cpu_probe".into(),
        };

        if substitution_mode == Some("try") {
            let mut candidates = vec![
                SubstitutionCandidate::ternary(),
                SubstitutionCandidate::sym_int4_g32(),
                SubstitutionCandidate::nf4_bnb_g32(),
                SubstitutionCandidate::int8_g128(),
                SubstitutionCandidate::fp16(),
            ];
            // Filter out candidates disallowed by policy
            candidates.retain(|c| !ctx.disallowed_codecs.contains(&c.name));
            let primary_bytes = (in_features as u64) * (out_features as u64) * 4;
            let attempts = try_all_candidates(
                &source,
                in_features as u32,
                out_features as u32,
                &candidates,
                primary_bytes,
                &ctx,
            );
            let attempt_jsons: Vec<serde_json::Value> = attempts.iter().map(|a| {
                eprintln!("  substitution {}: {:?}", a.candidate, a.outcome);
                serde_json::to_value(a).unwrap_or_else(|_| serde_json::Value::Null)
            }).collect();
            receipt["substitution_attempts"] = serde_json::json!(attempt_jsons);
        } else {
            receipt["substitution_attempts"] = serde_json::json!([]);
        }
        eprintln!("{}", serde_json::to_string_pretty(&receipt)
            .unwrap_or_else(|_| receipt.to_string()));
        return Ok(());
    }

    // Build group index from metadata (no weight data yet)
    let mut grouped: Vec<(TensorGroup, Vec<&TensorMeta>)> = Vec::new();
    for meta in &tensor_meta {
        if meta.group == TensorGroup::Other { continue; }
        match grouped.iter().position(|(g, _)| *g == meta.group) {
            Some(idx) => grouped[idx].1.push(meta),
            None => grouped.push((meta.group.clone(), vec![meta])),
        }
    }
    eprintln!("grouped tensors:");
    for (group, metas) in &grouped {
        eprintln!("  {:?}: {} tensors", group, metas.len());
    }

    let model_config = ModelConfig {
        num_layers,
        num_heads: num_heads.max(8),
        head_dim: head_dim.max(128),
        hidden_dim: hidden_dim.max(1024),
        intermediate_dim: intermediate_dim.max(4096),
        vocab_size: vocab_size.max(32000),
        quantization_schema: 1,
        draft_num_layers: 0,
        num_experts: 0,
        num_shared_experts: 0,
        top_k: 0,
        expert_intermediate_dim: 0,
    };

    // ── Phase 2: For each group, stream-load tensors and compile ─────────
    const LAYERS_PER_BATCH: u32 = 4;
    let mut matrix_id: u32 = 0;
    for (group, metas) in &grouped {
        // Special case: Decoder group is ~40+ GB, batch by layer
        if *group == TensorGroup::Decoder {
            // Group metas by layer number
            let mut per_layer: std::collections::BTreeMap<u32, Vec<&&TensorMeta>> =
                std::collections::BTreeMap::new();
            for meta in metas {
                per_layer.entry(meta.layer).or_default().push(meta);
            }
            let layer_nums: Vec<u32> = per_layer.keys().copied().collect();
            eprintln!("decoder: {} layers, processing in batches of {}",
                layer_nums.len(), LAYERS_PER_BATCH);

            for batch_chunk in layer_nums.chunks(LAYERS_PER_BATCH as usize) {
                // Collect keys for this batch's layers
                let batch_keys: std::collections::HashSet<String> = batch_chunk
                    .iter()
                    .flat_map(|l| per_layer[l].iter())
                    .map(|m| m.key.clone())
                    .collect();

                let mut batch_tensors: Vec<TensorInput> = Vec::with_capacity(batch_keys.len());
                // Load only tensors matching keys in this batch
                for (_dir_idx, (sd, _)) in source_dirs.iter().enumerate() {
                    if sd.is_empty() { continue; }
                    let source_dir = Path::new(sd);
                    for entry in fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))? {
                        let entry = entry.map_err(|e| format!("entry: {e}"))?;
                        let path = entry.path();
                        if !path.extension().map_or(false, |e| e == "safetensors") { continue; }
                        eprintln!("  loading {} for decoder layers {:?}", path.display(), batch_chunk);
                        let file = File::open(&path).map_err(|e| format!("open {path:?}: {e}"))?;
                        let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
                        let tensors = SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;
                        for (key, view) in tensors.tensors() {
                            if !batch_keys.contains(&key) { continue; }
                            let dtype = view.dtype();
                            let shape: Vec<usize> = view.shape().to_vec();
                            let data = view.data().to_vec();
                            let f32_data = match dtype {
                                safetensors::Dtype::F32 => data.chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect(),
                                safetensors::Dtype::BF16 => data.chunks_exact(2)
                                    .map(|c| {
                                        let bits = ((c[0] as u32) << 16) | ((c[1] as u32) << 24);
                                        f32::from_bits(bits)
                                    })
                                    .collect(),
                                _ => { continue; }
                            };
                            let (in_f, out_f) = if shape.len() >= 2 {
                                (shape[0] as u32, shape[1] as u32)
                            } else if shape.len() == 1 {
                                (shape[0] as u32, 1u32)
                            } else { continue; };
                            batch_tensors.push(TensorInput {
                                matrix_id,
                                weights: f32_data,
                                shape: CanonicalShape { in_features: in_f, out_features: out_f, rank: shape.len() as u16 },
                            });
                            matrix_id += 1;
                        }
                    }
                }

                if batch_tensors.is_empty() { continue; }

                let stage_config = StageConfig {
                    stage_id: 1, component: ComponentType::DecoderLayer,
                    tensor_key_patterns: vec![],
                    quantization: StageQuantizationConfig::decoder_default(),
                    backend: BackendKind::Metal, gpu_memory_utilization: 0.6, tensor_parallel_size: 1,
                };
                let batch_label = format!("decoder_layers_{}_{}", batch_chunk[0], batch_chunk.last().unwrap());
                eprintln!("compiling {} with {} tensors...", batch_label, batch_tensors.len());
                let (stage_result, bindings) = compile_stage(
                    batch_tensors, stage_config, model_config, CapabilityRegistry::default_metal_v1(),
                );
                let path = output_dir.join(format!("stage_1_{}.cimage", batch_label));
                fs::write(&path, &stage_result.cimage)
                    .map_err(|e| format!("write {}: {e}", path.display()))?;
                eprintln!("  wrote {}: {} bytes, {} bindings",
                    path.display(), stage_result.cimage.len(), bindings.len());
            }
            continue;
        }

        let mut group_tensors: Vec<TensorInput> = Vec::with_capacity(metas.len());

        // Re-open safetensors files and load only tensors matching this group
        for (_dir_idx, (sd, _)) in source_dirs.iter().enumerate() {
            if sd.is_empty() { continue; }
            let source_dir = Path::new(sd);
            for entry in fs::read_dir(source_dir).map_err(|e| format!("read source dir: {e}"))? {
                let entry = entry.map_err(|e| format!("entry: {e}"))?;
                let path = entry.path();
                if !path.extension().map_or(false, |e| e == "safetensors") { continue; }
                eprintln!("  loading {} for stage {:?}", path.display(), group);
                let file = File::open(&path).map_err(|e| format!("open {path:?}: {e}"))?;
                let mmap = unsafe { Mmap::map(&file).map_err(|e| format!("mmap: {e}"))? };
                let tensors = SafeTensors::deserialize(&mmap).map_err(|e| format!("deserialize: {e}"))?;

                for (key, view) in tensors.tensors() {
                    // Load only if the key matches a tensor in this group
                    if !metas.iter().any(|m| m.key == key) { continue; }

                    let dtype = view.dtype();
                    let shape: Vec<usize> = view.shape().to_vec();
                    let data = view.data().to_vec();
                    let f32_data = match dtype {
                        safetensors::Dtype::F32 => data.chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect(),
                        safetensors::Dtype::BF16 => data.chunks_exact(2)
                            .map(|c| {
                                let bits = ((c[0] as u32) << 16) | ((c[1] as u32) << 24);
                                f32::from_bits(bits)
                            })
                            .collect(),
                        _ => { continue; }
                    };
                    let (in_f, out_f) = if shape.len() >= 2 {
                        (shape[0] as u32, shape[1] as u32)
                    } else if shape.len() == 1 {
                        (shape[0] as u32, 1u32)
                    } else {
                        continue;
                    };
                    group_tensors.push(TensorInput {
                        matrix_id,
                        weights: f32_data,
                        shape: CanonicalShape { in_features: in_f, out_features: out_f, rank: shape.len() as u16 },
                    });
                    matrix_id += 1;
                }
            }
        }

        if group_tensors.is_empty() {
            eprintln!("  no tensors loaded for {:?}, skipping", group);
            continue;
        }

        let stage_config = match group {
            TensorGroup::Embedding => StageConfig {
                stage_id: 0, component: ComponentType::TextEmbedding,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::projection_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.3, tensor_parallel_size: 1,
            },
            TensorGroup::Decoder => StageConfig {
                stage_id: 1, component: ComponentType::DecoderLayer,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::decoder_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.6, tensor_parallel_size: 1,
            },
            TensorGroup::LmHead => StageConfig {
                stage_id: 2, component: ComponentType::LmHead,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::projection_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.1, tensor_parallel_size: 1,
            },
            TensorGroup::Norm => StageConfig {
                stage_id: 3, component: ComponentType::Norm,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::projection_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.1, tensor_parallel_size: 1,
            },
            TensorGroup::VisionEncoder => StageConfig {
                stage_id: 4, component: ComponentType::VisionEncoder,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::encoder_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.4, tensor_parallel_size: 1,
            },
            TensorGroup::AudioEncoder => StageConfig {
                stage_id: 5, component: ComponentType::AudioEncoder,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::encoder_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.4, tensor_parallel_size: 1,
            },
            TensorGroup::MtpDraft => StageConfig {
                stage_id: 6, component: ComponentType::MtpDraft,
                tensor_key_patterns: vec![],
                quantization: StageQuantizationConfig::decoder_default(),
                backend: BackendKind::Metal, gpu_memory_utilization: 0.3, tensor_parallel_size: 1,
            },
            _ => continue,
        };

        let stage_id = stage_config.stage_id;
        let stage_label = stage_config.component.as_str().to_string();
        eprintln!("compiling stage {} ({}) with {} tensors...",
            stage_id, stage_label, group_tensors.len());

        let (stage_result, bindings) = compile_stage(
            group_tensors,  // moved — freed after compile
            stage_config,
            model_config,
            CapabilityRegistry::default_metal_v1(),
        );

        let path = output_dir.join(format!("stage_{}_{}.cimage", stage_id, stage_label));
        fs::write(&path, &stage_result.cimage)
            .map_err(|e| format!("write {}: {e}", path.display()))?;
        eprintln!("  wrote {}: {} bytes, {} bindings",
            path.display(), stage_result.cimage.len(), bindings.len());
    }

    eprintln!("done — {} groups compiled", grouped.len());
    Ok(())
}

fn cmd_verify(args: &[String]) -> Result<(), String> {
    let image = get_opt(args, "--image").ok_or_else(|| "--image is required".to_string())?;
    let expected_hash = get_opt(args, "--expected-hash");
    let full = has_flag(args, "--full");

    let image_path = Path::new(image);

    // Image dir must exist with seal.json.
    let seal_path = image_path.join("seal.json");
    if !image_path.exists() || !seal_path.exists() {
        return Err(format!(
            "image directory '{image}' does not exist or seal.json is missing"
        ));
    }

    // Read seal.json.
    let seal_text = fs::read_to_string(&seal_path).map_err(|e| format!("read seal.json: {e}"))?;
    let seal: serde_json::Value =
        serde_json::from_str(&seal_text).map_err(|e| format!("parse seal.json: {e}"))?;
    let stored_hash = seal["image_hash"]
        .as_str()
        .ok_or_else(|| "seal.json missing image_hash".to_string())?
        .to_string();

    // If --expected-hash provided, compare.
    if let Some(expected) = expected_hash {
        if expected != stored_hash {
            tribunus_compute_core::log_error!(
                "hash mismatch: expected={expected} stored={stored_hash}"
            );
            return Err("image hash mismatch".to_string());
        }
    }

    // Open image (triggers full verification internally).
    let reader =
        compute_image::read(image).map_err(|e| format!("image verification failed: {e}"))?;

    // Validate execution plan.
    let plan_errors = reader.manifest.execution_plan.validate();
    if let Err(errs) = plan_errors {
        let joined = errs.join("; ");
        return Err(format!("execution plan validation failed: {joined}"));
    }

    // Verify all segment files exist.
    for seg in &reader.manifest.segments {
        let seg_path = image_path.join(&seg.filename);
        if !seg_path.exists() {
            return Err(format!("missing segment file: {}", seg.filename));
        }
    }

    // If --full: verify every segment SHA-256 against manifest, then verify
    // artifact root hash against seal.json using a streaming read.
    if full {
        tribunus_compute_core::log_info!(
            "[verify] full: hashing {} segments...",
            reader.manifest.segments.len()
        );
        let mut mismatches: Vec<String> = Vec::new();
        let mut verified = 0usize;
        let mut root_hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        for seg in &reader.manifest.segments {
            let sp = image_path.join(&seg.filename);
            let mut file =
                File::open(&sp).map_err(|e| format!("open segment {}: {}", seg.filename, e))?;
            let mut seg_hasher = Sha256::new();
            loop {
                let n = file
                    .read(&mut buf)
                    .map_err(|e| format!("read segment {}: {}", seg.filename, e))?;
                if n == 0 {
                    break;
                }
                seg_hasher.update(&buf[..n]);
                root_hasher.update(&buf[..n]);
            }
            let computed = format!("{:x}", seg_hasher.finalize());
            if computed == seg.sha256 {
                verified += 1;
            } else {
                mismatches.push(format!("{}: hash mismatch", seg.filename));
            }
        }
        if !mismatches.is_empty() {
            return Err(format!(
                "segment hash mismatches ({}/{} verified):\n{}",
                verified,
                reader.manifest.segments.len(),
                mismatches.join("\n")
            ));
        }
        tribunus_compute_core::log_info!(
            "[verify] segments: {}/{} verified",
            verified,
            reader.manifest.segments.len()
        );

        let recomputed_root = format!("{:x}", root_hasher.finalize());
        // Compare against seal.json artifact_root_hash
        let expected_root = seal
            .get("artifact_root_hash")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| stored_hash.clone());
        if recomputed_root != expected_root {
            return Err(format!(
                "artifact root hash mismatch: seal={} recomputed={}",
                &expected_root[..16],
                &recomputed_root[..16]
            ));
        }
        tribunus_compute_core::log_info!("[verify] artifact root hash: match");
    }

    let segment_count = reader.manifest.segments.len();
    let tensor_count = reader.manifest.tensor_table.len();
    let storage_abi = reader.manifest.required_storage_abi.clone();
    let image_hash = reader.manifest.image_hash.clone();

    let out = json!({
        "status": "verified",
        "segments_verified": segment_count,
        "image_hash": image_hash,
        "artifact_root_hash": seal["artifact_root_hash"].as_str().unwrap_or(&image_hash).to_string(),
        "segment_count": segment_count,
        "tensor_count": tensor_count,
        "storage_abi": storage_abi,
    });
    println!("{}", serde_json::to_string(&out).unwrap());

    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Sync (fsync) an open directory. Falls back to a no-op on platforms where
/// File::open on a directory is unsupported.
fn sync_dir(path: &Path) -> Result<(), String> {
    match fs::File::open(path) {
        Ok(file) => file.sync_all().map_err(|e| format!("sync dir failed: {e}")),
        Err(_) => Ok(()),
    }
}

/// Format a Unix timestamp (whole seconds since epoch) as an ISO 8601 UTC
/// string.
fn format_iso8601(secs: u64) -> String {
    // Days since epoch.
    let days = secs / 86400;
    let day_secs = secs % 86400;
    let hour = day_secs / 3600;
    let min = (day_secs % 3600) / 60;
    let sec = day_secs % 60;

    let (year, month, day) = civil_from_days(days as i64);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month as u32, day as u32, hour, min, sec,
    )
}

/// Convert a days-from-epoch value to (year, month, day) in the Gregorian
/// civil calendar.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    // Shamelessly adapted from Howard Hinnant's public-domain algorithm.
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // day-of-era
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}
fn cmd_decode_one(args: &[String]) -> Result<(), String> {
    tribunus_compute_core::log_info!(
        "[experimental diagnostic] Running compute-native decode-one diagnostic verification"
    );

    let mut image: Option<String> = None;
    let mut prompt_str: Option<String> = None;
    let mut sliding_capacity: u32 = 1024;
    let mut full_capacity: u32 = 8;
    let mut steps: usize = 1;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                i += 1;
                if i < args.len() {
                    image = Some(args[i].clone());
                }
            }
            "--prompt" => {
                i += 1;
                if i < args.len() {
                    prompt_str = Some(args[i].clone());
                }
            }
            "--sliding-capacity" => {
                i += 1;
                if i < args.len() {
                    sliding_capacity = args[i]
                        .parse::<u32>()
                        .map_err(|e| format!("parse sliding capacity: {e}"))?;
                }
            }
            "--full-capacity" => {
                i += 1;
                if i < args.len() {
                    full_capacity = args[i]
                        .parse::<u32>()
                        .map_err(|e| format!("parse full capacity: {e}"))?;
                }
            }
            "--steps" => {
                i += 1;
                if i < args.len() {
                    steps = args[i]
                        .parse::<usize>()
                        .map_err(|e| format!("parse steps: {e}"))?;
                }
            }
            _ => {
                return Err(format!("unknown flag: {}", args[i]));
            }
        }
        i += 1;
    }
    let image_dir = image.ok_or("missing --image")?;
    let image_path = Path::new(&image_dir);

    // Parse prompt
    let prompt: Vec<u32> = if let Some(p_str) = prompt_str {
        p_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| format!("parse token '{s}': {e}"))
            })
            .collect::<Result<Vec<u32>, String>>()?
    } else {
        vec![2, 42, 100, 500] // default fallback
    };

    tribunus_compute_core::log_info!("Opening sealed image: {}", image_dir);
    let reader = compute_image::read(&image_dir).map_err(|e| format!("read image: {e}"))?;
    let plan = &reader.manifest.execution_plan;

    // Build KV caches (one per layer) using parsed capacities
    let kv_caches: Vec<KvCache> = plan
        .layers
        .iter()
        .map(|lp| {
            let is_sliding = lp.attention_kind == "sliding_attention";
            let capacity: u32 = if is_sliding {
                sliding_capacity
            } else {
                full_capacity
            };
            let (n_kv_heads, head_dim) = if lp.attention_kind == "full_attention" {
                (
                    lp.n_global_kv_heads.unwrap_or(1) as u32,
                    lp.global_head_dim.unwrap_or(512) as u32,
                )
            } else {
                (lp.n_kv_heads as u32, lp.head_dim as u32)
            };
            KvCache::new(capacity, n_kv_heads, head_dim, is_sliding)
        })
        .collect();

    // Build the profiled model
    let model = LoadedProfiledModel::new(image_path).map_err(|e| format!("load model: {e}"))?;
    let mut session = ProfiledInferenceSession::new("decode-one".into(), kv_caches);

    // Prefill with prompt
    tribunus_compute_core::log_info!("Prefill with {} tokens...", prompt.len());
    let t0 = std::time::Instant::now();
    let prefill_token = session
        .prefill(&prompt, &model)
        .map_err(|e| format!("prefill: {e}"))?;
    let prefill_elapsed = t0.elapsed().as_secs_f64();
    tribunus_compute_core::log_info!(
        "GATE: prefill_token={} elapsed={:.2}s",
        prefill_token,
        prefill_elapsed
    );

    // Decode one token
    tribunus_compute_core::log_info!("Decode {} tokens...", steps);
    let t0 = std::time::Instant::now();
    let mut next_token = prefill_token;
    for _ in 0..steps {
        next_token = session
            .decode_one(next_token, &model)
            .map_err(|e| format!("decode at step: {e}"))?;
    }
    let decode_token = next_token;
    let decode_elapsed = t0.elapsed().as_secs_f64();
    tribunus_compute_core::log_info!(
        "GATE: decode_token={} elapsed={:.2}s",
        decode_token,
        decode_elapsed
    );

    // Verify KV caches are committed correctly
    let expected_committed = (prompt.len() + steps) as u32;
    for (l, kvc) in session.kv_caches.iter().enumerate() {
        let committed = kvc.committed_len;
        if committed != expected_committed {
            tribunus_compute_core::log_warn!(
                "WARN: layer {} has {} committed positions (expected {})",
                l,
                committed,
                expected_committed
            );
        }
    }

    let out = serde_json::json!({
        "status": "decoded",
        "image_hash": model.reader.manifest.image_hash,
        "prefill_token": prefill_token,
        "decode_token": decode_token,
        "prefill_elapsed_s": prefill_elapsed,
        "decode_elapsed_s": decode_elapsed,
        "layers": plan.layers.len(),
        "experimental_receipt": {
            "label": "experimental diagnostic",
            "prompt_tokens": prompt,
            "sliding_capacity": sliding_capacity,
            "full_capacity": full_capacity,
            "kv_cache_committed_positions": expected_committed,
        }
    });
    println!("{}", serde_json::to_string(&out).unwrap());
    Ok(())
}

fn cmd_infer(args: &[String]) -> Result<(), String> {
    let mut image: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--image" => {
                i += 1;
                if i < args.len() {
                    image = Some(args[i].clone());
                }
            }
            _ => {
                return Err(format!("unknown flag: {}", args[i]));
            }
        }
        i += 1;
    }
    let image_dir = image.ok_or("missing --image")?;
    let image_path = Path::new(&image_dir);
    if !image_path.join("manifest.json").exists() {
        return Err("not a ComputeImage directory (missing manifest.json)".into());
    }

    tribunus_compute_core::log_info!("Opening sealed image: {}", image_dir);
    let reader = compute_image::read(&image_dir).map_err(|e| format!("read: {e}"))?;

    let plan = &reader.manifest.execution_plan;
    let plan_errors = plan.validate();
    if let Err(errs) = plan_errors {
        return Err(format!("plan validation failed: {}", errs.join("; ")));
    }

    let start = std::time::Instant::now();
    let mut runtime = reader
        .open_runtime(compute_image::StorageBackend::Copied)
        .map_err(|e| format!("open runtime: {e}"))?;

    tribunus_compute_core::log_info!("Running 48-layer forward pass...");
    let token = runtime
        .run_full_model(&[2i32])
        .map_err(|e| format!("run_full_model: {e}"))?;
    let elapsed = start.elapsed();
    let elapsed_s = elapsed.as_secs_f64();

    let out = serde_json::json!({
        "status": "inferred",
        "image_hash": reader.manifest.image_hash,
        "output_token": token,
        "elapsed_s": elapsed_s,
        "layers": plan.layers.len(),
    });
    println!("{}", serde_json::to_string(&out).unwrap());

    tribunus_compute_core::log_info!("GATE PASSED: token={} elapsed={:.1}s", token, elapsed_s);
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// emit-v0 and verify-v0 commands
// ═══════════════════════════════════════════════════════════════════════════

fn cmd_emit_v0(args: &[String]) -> Result<(), String> {
    let output_dir =
        get_opt(args, "--output-dir").ok_or_else(|| "--output-dir is required".to_string())?;
    let allow_contract_only_kv = has_flag(args, "--allow-contract-only-kv");

    let out_path = Path::new(output_dir);
    fs::create_dir_all(out_path).map_err(|e| format!("create output dir: {}", e))?;

    let adapter = tribunus_compute_core::compute_image_v0::evidence::SyntheticFixtureAdapter {
        scenarios: tribunus_compute_core::compute_image_v0::evidence::default_synthetic_fixtures(),
    };

    let mut options = tribunus_compute_core::compute_image_v0::emitter::EmitterOptions::default();
    options.allow_contract_only_kv = allow_contract_only_kv;

    let (image, md) =
        tribunus_compute_core::compute_image_v0::emitter::emit_v0_image(&adapter, options)?;

    let json_path = out_path.join("compute_image_v0.json");
    let md_path = out_path.join("compute_image_v0.md");

    let json_str =
        serde_json::to_string_pretty(&image).map_err(|e| format!("json serialize: {}", e))?;
    fs::write(&json_path, json_str).map_err(|e| format!("write json: {}", e))?;
    fs::write(&md_path, md).map_err(|e| format!("write md: {}", e))?;

    tribunus_compute_core::log_info!("Emitted compute_image_v0.json and .md to {}", output_dir);
    Ok(())
}

fn cmd_verify_v0(args: &[String]) -> Result<(), String> {
    let image_dir = get_opt(args, "--image").ok_or_else(|| "--image is required".to_string())?;

    let json_path = Path::new(image_dir).join("compute_image_v0.json");
    if !json_path.exists() {
        return Err(format!("{} does not exist", json_path.display()));
    }

    let json_str = fs::read_to_string(&json_path).map_err(|e| format!("read json: {}", e))?;
    let image: tribunus_compute_core::compute_image_v0::schema::ComputeImageV0 =
        serde_json::from_str(&json_str).map_err(|e| format!("parse json: {}", e))?;

    let override_dirty = has_flag(args, "--override-dirty");
    let options = tribunus_compute_core::compute_image_v0::verifier::VerifierOptions {
        override_dirty_tree: override_dirty,
    };

    match tribunus_compute_core::compute_image_v0::verifier::verify_v0_image(&image, options) {
        Ok(_) => {
            tribunus_compute_core::log_info!("ComputeImageV0 validation passed.");
            Ok(())
        }
        Err(errors) => Err(format!(
            "ComputeImageV0 verification failed:\n  - {}",
            errors.join("\n  - ")
        )),
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// quant-sweep command — parametric quantization sweep
// ═══════════════════════════════════════════════════════════════════════════
/// Run a parametric quantization sweep across selected tensors.
fn cmd_quant_sweep(args: &[String]) -> Result<(), String> {
    let source = get_opt(args, "--source").ok_or_else(|| "--source is required".to_string())?;
    let output = get_opt(args, "--output").ok_or_else(|| "--output is required".to_string())?;
    let tensor_regex = get_opt(args, "--tensor-regex").unwrap_or(".*");
    let max_candidates: usize = get_opt(args, "--max-candidates")
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let output_path = std::path::PathBuf::from(output);

    use std::path::Path;
    let source_path = Path::new(source);
    if !source_path.is_dir() {
        return Err(format!("source directory not found: {source}"));
    }

    use tribunus_compute_core::quantization::sweep::runner::{
        default_resource_limits, default_scoring_config, default_validation_config,
        run_weight_sweep, write_sweep_output,
    };
    use tribunus_compute_core::quantization::sweep::spec::{
        QuantFamilySweep, QuantSweepSpec, SweepResourceLimits, SweepScoringConfig,
        SweepValidationConfig, TensorSelector,
    };
    use tribunus_compute_core::quantization::sweep::families::{
        int8::create_int8_grid, nf4::create_nf4_grid, sym_int4::create_sym_int4_grid,
        ternary::create_ternary_grid,
    };

    // Build the sweep spec
    let selectors = vec![TensorSelector::Regex(tensor_regex.to_string())];

    let families: Vec<QuantFamilySweep> = vec![
        QuantFamilySweep::Nf4(create_nf4_grid()),
        QuantFamilySweep::SymInt4(create_sym_int4_grid()),
        QuantFamilySweep::Int8(create_int8_grid()),
        QuantFamilySweep::Ternary(create_ternary_grid()),
    ];

    let validation = SweepValidationConfig {
        run_weight_validation: true,
        max_candidates: None,
        max_candidates_per_tensor: max_candidates,
        max_total_candidates: None,
        policy_mode: tribunus_compute_core::quantization::sweep::spec::PolicyMode::ProductionCandidateOnly,
    };
    let scoring = default_scoring_config();
    let resource_limits = default_resource_limits();

    let spec = QuantSweepSpec {
        spec_version: 1,
        tensor_selectors: selectors,
        families,
        validation,
        scoring,
        resource_limits,
        output_dir: output_path.clone(),
    };

    eprintln!("QuantSweep v{} starting...", spec.spec_version);
    eprintln!("  source: {source}");
    eprintln!("  regex: {tensor_regex}");
    eprintln!("  families: {}", spec.families.len());
    eprintln!("  max_candidates: {max_candidates}");
    eprintln!("  output: {output}");

    let result = run_weight_sweep(&spec, source_path)?;

    writeln!(
        std::io::stderr(),
        "Sweep complete: {} tensors, {} candidates in {:.2}s",
        result.num_tensors,
        result.num_candidates,
        result.wall_ms as f64 / 1000.0
    )
    .unwrap();

    if result.per_class_policies.is_empty() {
        eprintln!("  WARNING: no per-class policies generated");
    } else {
        eprintln!("  Per-class policies:");
        for p in &result.per_class_policies {
            eprintln!(
                "    {:?}: {} preferred, fallback={}",
                p.tensor_class,
                p.preferred.len(),
                p.fallback
            );
        }
    }

    write_sweep_output(&output_path, &result)?;
    eprintln!("Output written to {output}");

    // ── Optional ANE operator validation for preferred candidates ──
    #[cfg(all(
        target_os = "macos",
        any(feature = "mlx-backend", feature = "prism-backend"),
    ))]
    if has_flag(args, "--ane-validation") {
        use std::collections::HashMap;
        use tribunus_compute_core::quantization::sweep::ane_validation::validate_operator;

        let opval_path = output_path.join("operator_validation.json");
        let mut opval_results: Vec<serde_json::Value> = Vec::new();

        for policy in &result.per_class_policies {
            if policy.preferred.is_empty() {
                continue;
            }
            // Find a receipt matching this class to get the tensor_key
            let class_receipts: Vec<&_> = result.candidates.iter()
                .filter(|r| r.tensor_class == policy.tensor_class && matches!(r.status, tribunus_compute_core::quantization::sweep::SweepCandidateStatus::Passed))
                .collect();
            if class_receipts.is_empty() {
                continue;
            }
            let r = class_receipts[0];
            let canonical = r.logical_shape;
            let in_f = canonical.in_features;
            let out_f = canonical.out_features;

            eprintln!("  [ANE] validating {:?} on {}...", policy.tensor_class, r.tensor_key);

            // Load the tensor from safetensors
            let src_dir = std::path::Path::new(source);
            let loaded = tribunus_compute_core::quantization::sweep::runner::load_tensor_f32(src_dir, &r.tensor_key)
                .map_err(|e| format!("load {}: {}", r.tensor_key, e))?;

            // Reconstruct the weights — use pack_nf4_weights for NF4 preferred candidates
            // Determine codec parameters from the winning candidate
            let params = &r.parameters;
            let (codes, scales, biases, extra_bytes, recon): (Vec<u8>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<f32>) = if matches!(r.family, tribunus_compute_core::quantization::sweep::QuantFamilyId::Nf4) {
                let codebook_str = params.get("codebook").and_then(|v| v.as_str()).unwrap_or("PrismCurrent");
                use tribunus_compute_core::quantization::sweep::spec::Nf4CodebookId;
                let cb_id = match codebook_str {
                    "BitsAndBytesNf4" => Nf4CodebookId::BitsAndBytesNf4,
                    "SymmetricNormalFloat" => Nf4CodebookId::SymmetricNormalFloat,
                    _ => Nf4CodebookId::PrismCurrent,
                };
                let cb = tribunus_compute_core::nf4tile640::nf4_codebook(cb_id);
                let gs = params.get("group_size").and_then(|v| v.as_u64()).unwrap_or(64) as usize;
                let (c, s, b) = tribunus_compute_core::quantization::sweep::ane_validation::pack_nf4_weights_with_codebook(
                    &loaded, in_f as u32, out_f as u32, cb, gs);
                let recon = tribunus_compute_core::nf4tile640::unpack_nf4_weights_with_group_size_and_codebook(
                    &c, &s, &b, in_f, out_f, gs, cb);
                (c, s, b, vec![], recon)
            } else if matches!(r.family, tribunus_compute_core::quantization::sweep::QuantFamilyId::Int8) {
                let (codes_i8, scales_i8, biases_i8) = pack_int8_weights(&loaded, in_f, out_f);
                let recon_i8 = unpack_int8_weights(&codes_i8, &scales_i8, &biases_i8, in_f, out_f);
                (codes_i8, scales_i8, biases_i8, vec![], recon_i8)
            } else {
                eprintln!("  [ANE] skipping non-NF4/INT8 family");
                continue;
            };

            match validate_operator(&recon, in_f as u32, out_f as u32) {
                Ok(metrics) => {
                    eprintln!("    op_rmse={:.6} op_nrmse={:.6} cosine={:.6} drift={:.6} max_abs={:.6}",
                        metrics.operator_rmse, metrics.operator_nrmse,
                        metrics.cosine_similarity, metrics.norm_ratio_drift, metrics.max_abs_error);
                    opval_results.push(serde_json::json!({
                        "tensor_class": format!("{:?}", policy.tensor_class),
                        "tensor_key": r.tensor_key,
                        "family": format!("{:?}", r.family),
                        "parameters": r.parameters,
                        "operator_rmse": (metrics.operator_rmse * 10000.0).round() / 10000.0,
                        "operator_nrmse": (metrics.operator_nrmse * 10000.0).round() / 10000.0,
                        "cosine_similarity": (metrics.cosine_similarity * 10000.0).round() / 10000.0,
                        "norm_ratio_drift": (metrics.norm_ratio_drift * 10000.0).round() / 10000.0,
                        "max_abs_error": (metrics.max_abs_error * 10000.0).round() / 10000.0,
                    }));
                }
                Err(e) => {
                    eprintln!("    [ANE] validation FAILED: {}", e);
                }
            }
        }

        if !opval_results.is_empty() {
            let opval_json = serde_json::to_string_pretty(&opval_results)
                .map_err(|e| format!("serialize: {}", e))?;
            std::fs::write(&opval_path, &opval_json)
                .map_err(|e| format!("write operator_validation.json: {}", e))?;
            eprintln!("Operator validation written to {}", opval_path.display());
        }
    }

    Ok(())
}
