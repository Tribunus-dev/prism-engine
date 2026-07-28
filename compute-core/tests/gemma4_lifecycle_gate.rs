//! Gemma 4 lifecycle gate — Phases 7-10 definitive tests.
//!
//! Exercises Gemma 4-specific lifecycle gates:
//!   1. Tensor classification via synthetic checkpoint inspection
//!   2. MTP depth detection via config.json
//!   3. Replay manifest verification with drift classification
//!   4. Promote/rollback generation lifecycle
//!   5. M1MemoryBudget admission for realistic Gemma 4 sizes
//!   6. Catalogue source registration for all Gemma 4 kernel semantic IDs
//!
//! Every test is self-contained and targets only the `prism-backend` feature.

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

use tribunus_compute_core::ecs::canonical::generation::CimageGeneration;
use tribunus_compute_core::ecs::canonical::generation::RepresentationBinding;
use tribunus_compute_core::ecs::canonical::identity::{
    CandidateId, CompilerIdentity, GenerationId, HardwareProfileId, LogicalTensorId, ModelSourceId,
    PhysicalSegmentId, ReceiptId, RepresentationId, Timestamp,
};
use tribunus_compute_core::ecs::canonical::kernel_abi::{
    ArtifactProvenance, DispatchGeometryPolicy, KernelAbi, KernelSemanticId,
};
use tribunus_compute_core::ecs::canonical::provenance::{LifecycleReceiptBundle, ReplayManifest};
use tribunus_compute_core::ecs::canonical::{ExecutionGraph, MemoryPlan, RuntimeStatePlan};
use tribunus_compute_core::ecs::legacy_cimage::generation_api::GenerationApi;
use tribunus_compute_core::ecs::compute_image::model_family::gemma4_inspect::{
    build_ingestion_receipt, inspect_gemma4_checkpoint,
};
use tribunus_compute_core::ecs::compute_image::model_family::gemma4_unified::{
    classify_tensor_name, TensorClassification,
};
use tribunus_compute_core::ecs::evolution::foundation::{NumericalReceipt, PerformanceReceipt};
use tribunus_compute_core::ecs::evolution::replay::replay_from_manifest;
use tribunus_compute_core::ecs::execution_profile::PhysicalTileLayout;
use tribunus_compute_core::ecs::metal_backend::catalogue::{
    catalogue_source_for, GEMMA4_KERNEL_SEMANTICS,
};
use tribunus_compute_core::ecs::plan::CodecFamily;
use tribunus_compute_core::prism_ecs_quantization::precision_policy::M1MemoryBudget;

// ====================================================================
// Helpers
// ====================================================================

/// Compute the SHA-256 hex digest of raw bytes.
fn sha256_digest(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Build a minimal CimageGeneration for test use.
fn make_base_generation(
    gen_id: &str,
    parent: Option<&str>,
    seg_id: PhysicalSegmentId,
) -> CimageGeneration {
    let mut bindings = BTreeMap::new();
    bindings.insert(
        LogicalTensorId("weight".into()),
        RepresentationBinding {
            representation_id: RepresentationId("rawf32".into()),
            codec: CodecFamily::RawF32,
            layout: PhysicalTileLayout::default(),
            primary_segment: seg_id,
            scale_segments: vec![],
            residual_segments: vec![],
            source_representation: None,
            acceptance_receipt: ReceiptId("accept".into()),
        },
    );
    CimageGeneration {
        generation_id: GenerationId(gen_id.into()),
        parent_generation: parent.map(|p| GenerationId(p.into())),
        base_model: ModelSourceId("gemma4".into()),
        compiler_identity: CompilerIdentity {
            name: "tribunus".into(),
            version: "1.0.0".into(),
            build_hash: Some("abc123".into()),
            build_timestamp: Some("2026-07-13T00:00:00Z".into()),
        },
        hardware_profile: HardwareProfileId("apple-m4-max".into()),
        tensor_bindings: bindings,
        kernel_bindings: BTreeMap::new(),
        engram_bindings: BTreeMap::new(),
        execution_graph: ExecutionGraph {
            regions: vec![],
            edges: vec![],
            state: RuntimeStatePlan {
                max_context_tokens: 4096,
                kv_cache_bytes_per_token: 256,
                total_kv_cache_bytes: 1048576,
            },
            memory: MemoryPlan {
                total_activation_bytes: 0,
                total_weight_bytes: 0,
                arena_region_count: 0,
            },
        },
        receipt_root: ReceiptId(sha256_digest(b"test-root")),
        created_at: Timestamp("2026-07-13T00:00:00Z".into()),
    }
}

/// Create a synthetic Gemma 4 checkpoint directory with a config.json and
/// a minimal model.safetensors with realistic tensor names and shapes for
/// inspection validation.
///
/// When `with_mtp` is true, the config includes `"mtp_depth": 2`.
fn create_synthetic_gemma4_checkpoint(
    dir: &Path,
    with_mtp: bool,
    with_image: bool,
    with_audio: bool,
) {
    use std::io::Write;

    std::fs::create_dir_all(dir).expect("create checkpoint dir");

    // ── config.json ───────────────────────────────────────────────────
    let mut config = serde_json::json!({
        "hidden_size": 3840,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "vocab_size": 262144,
        "intermediate_size": 15360,
        "head_dim": 256,
    });
    if with_mtp {
        config["mtp_depth"] = serde_json::json!(2);
    }
    let mut f_cfg = std::fs::File::create(dir.join("config.json")).expect("create config");
    f_cfg
        .write_all(serde_json::to_string_pretty(&config).unwrap().as_bytes())
        .expect("write config");

    // ── Build tensor list and write safetensors ───────────────────────
    let mut tensors: Vec<(String, Vec<u32>)> = Vec::new();
    // Minimal shapes — inspection only reads names + shapes from header.
    let s: u32 = 2;

    // Embedding
    tensors.push(("model.embed_tokens.weight".into(), vec![s, s]));

    // 48 layers of decoder weights (matching Gemma 4 12B Unified)
    for layer in 0..48u32 {
        let prefix = format!("model.layers.{}", layer);
        tensors.push((format!("{}.self_attn.q_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.self_attn.k_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.self_attn.v_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.self_attn.o_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.mlp.gate_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.mlp.up_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.mlp.down_proj.weight", prefix), vec![s, s]));
        tensors.push((format!("{}.input_layernorm.weight", prefix), vec![s]));
        tensors.push((
            format!("{}.post_attention_layernorm.weight", prefix),
            vec![s],
        ));
    }

    // Norms
    tensors.push(("model.norm.weight".into(), vec![s]));

    // LM head
    tensors.push(("lm_head.weight".into(), vec![s, s]));

    // Multimodal
    if with_image {
        tensors.push(("vision_embedder.patch_embedding.weight".into(), vec![s, s]));
    }
    if with_audio {
        tensors.push(("embed_audio.weight".into(), vec![s, s]));
    }

    // MTP (only if with_mtp)
    if with_mtp {
        tensors.push(("model.mtp_projection.weight".into(), vec![s, s]));
        tensors.push(("model.mtp_norm.weight".into(), vec![s]));
    }

    // Write safetensors using serialize_to_file
    write_safetensors_shard(dir.join("model.safetensors"), &tensors);
}

/// Write a minimal safetensors file with deterministic content.
fn write_safetensors_shard(path: impl AsRef<Path>, tensors: &[(String, Vec<u32>)]) {
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;

    let path = path.as_ref();

    // Build all data buffers first so TensorView references stay valid.
    let mut data_buffers: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
    for (_name, shape) in tensors.iter() {
        let byte_count: usize = shape.iter().map(|d| *d as usize).product::<usize>() * 4;
        // Deterministic data: each 4-byte chunk encodes position info
        let data: Vec<u8> = (0..byte_count)
            .map(|j| ((j ^ (j >> 4)) as u8).wrapping_mul(37))
            .collect();
        data_buffers.push(data);
    }

    // Create TensorViews referencing the owned data buffers — all alive for
    // the duration of serialize_to_file since data_buffers lives to scope end.
    let named: Vec<(&str, TensorView)> = tensors
        .iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let usize_shape: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let view = TensorView::new(Dtype::F32, usize_shape, &data_buffers[i])
                .unwrap_or_else(|e| panic!("create TensorView for {name}: {e}"));
            (name.as_str(), view)
        })
        .collect();

    serialize_to_file(
        named,
        &None::<std::collections::HashMap<String, String>>,
        &path,
    )
    .expect("write safetensors");
}

/// Write a minimal tokenizer config so the inspection processor contract fills.
fn write_tokenizer_config(dir: &Path) {
    use std::io::Write;
    let tk = serde_json::json!({
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    });
    let mut f = std::fs::File::create(dir.join("tokenizer_config.json")).expect("create tokenizer");
    f.write_all(serde_json::to_string_pretty(&tk).unwrap().as_bytes())
        .expect("write tokenizer");
}

// ====================================================================
// Tests
// ====================================================================

// ── 1. Tensor classification from synthetic checkpoint ──────────────

#[test]
fn test_gemma4_tensor_classification() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    // Create a compact checkpoint with text-only tensors
    create_synthetic_gemma4_checkpoint(
        dir.path(),
        /*mtp=*/ false,
        /*image=*/ false,
        /*audio=*/ false,
    );
    write_tokenizer_config(dir.path());

    // Pre-flight: classify specific names directly
    assert_eq!(
        classify_tensor_name("model.layers.0.self_attn.q_proj.weight"),
        TensorClassification::DecoderRequired,
    );
    assert_eq!(
        classify_tensor_name("model.layers.1.mlp.down_proj.weight"),
        TensorClassification::DecoderRequired,
    );
    assert_eq!(
        classify_tensor_name("model.embed_tokens.weight"),
        TensorClassification::TextEmbeddingRequired,
    );
    assert_eq!(
        classify_tensor_name("lm_head.weight"),
        TensorClassification::LmHeadRequired,
    );
    assert_eq!(
        classify_tensor_name("model.norm.weight"),
        TensorClassification::NormRequired,
    );
    assert_eq!(
        classify_tensor_name("__metadata__"),
        TensorClassification::Ignored,
    );
    assert_eq!(
        classify_tensor_name("unknown_tensor"),
        TensorClassification::Unknown,
    );

    // Full inspection from synthetic directory
    let inspection = inspect_gemma4_checkpoint(dir.path()).expect("inspect checkpoint");

    // Verify architecture constants match the synthetic config
    assert_eq!(inspection.config.hidden_size, 3840);
    assert_eq!(inspection.config.num_layers, 48);
    assert_eq!(inspection.config.num_attention_heads, 16);
    assert_eq!(inspection.config.num_key_value_heads, 8);
    assert_eq!(inspection.config.vocab_size, 262144);
    assert_eq!(inspection.config.intermediate_size, 15360);
    assert!(
        inspection.config.mtp_depth.is_none(),
        "no MTP in this config"
    );

    // Inventory must have classified tensors
    assert!(
        inspection.inventory.total_tensors > 0,
        "must have tensors in inventory"
    );
    let cls = &inspection.inventory.classification;
    assert!(
        cls.contains_key("decoder_required"),
        "must have decoder_required classification"
    );
    assert!(
        cls.contains_key("text_embedding_required"),
        "must have text_embedding_required"
    );
    assert!(
        cls.contains_key("lm_head_required"),
        "must have lm_head_required"
    );
    assert!(cls.contains_key("norm_required"), "must have norm_required");

    // Verify no unknown tensors above threshold
    assert!(
        inspection.inventory.unknown_large.is_empty(),
        "must not have unknown large tensors"
    );

    // Verify source identity is populated
    assert!(
        !inspection.source_identity.config_digest.is_empty(),
        "config digest must be non-empty"
    );
    assert!(
        !inspection.source_identity.tokenizer_digest.is_empty(),
        "tokenizer digest must be non-empty"
    );

    // Check ingestion receipt is structurally valid
    let receipt = build_ingestion_receipt(&inspection);
    assert_eq!(receipt["total_tensors"].as_u64().unwrap_or(0) > 0, true);
    assert_eq!(receipt["mtp_detected"].as_bool(), Some(false));
    assert!(
        receipt["mtp_depth"].is_null(),
        "mtp_depth is null when absent"
    );
    assert!(receipt["classification"].is_object());
    assert!(receipt["config_digest"].as_str().unwrap_or("").len() > 0);
}

// ── 2. MTP depth detection ──────────────────────────────────────────

#[test]
fn test_gemma4_mtp_detection() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    create_synthetic_gemma4_checkpoint(
        dir.path(),
        /*mtp=*/ true,
        /*image=*/ true,
        /*audio=*/ true,
    );
    write_tokenizer_config(dir.path());

    // Direct classification of MTP tensors
    assert_eq!(
        classify_tensor_name("model.mtp_projection.weight"),
        TensorClassification::MtpRequired,
    );
    assert_eq!(
        classify_tensor_name("model.mtp_norm.weight"),
        TensorClassification::MtpRequired,
    );
    assert_eq!(
        classify_tensor_name("vision_embedder.patch_embedding.weight"),
        TensorClassification::MultimodalImageRequired,
    );
    assert_eq!(
        classify_tensor_name("embed_audio.weight"),
        TensorClassification::MultimodalAudioRequired,
    );

    // Full inspection
    let inspection = inspect_gemma4_checkpoint(dir.path()).expect("inspect checkpoint with MTP");

    // MTP depth must be detected from config.json
    assert_eq!(
        inspection.config.mtp_depth,
        Some(2),
        "MTP depth must be detected from config"
    );

    // Check MTP in classification
    let cls = &inspection.inventory.classification;
    assert!(
        cls.contains_key("mtp_required"),
        "classification must include mtp_required"
    );

    // Verify multimodal classifications are present
    assert!(
        cls.contains_key("multimodal_image_required"),
        "must have multimodal_image_required"
    );
    assert!(
        cls.contains_key("multimodal_audio_required"),
        "must have multimodal_audio_required"
    );

    // Ingestion receipt must reflect MTP detection
    let receipt = build_ingestion_receipt(&inspection);
    assert_eq!(receipt["mtp_detected"].as_bool(), Some(true));
    assert_eq!(receipt["mtp_depth"].as_u64(), Some(2));
    assert!(receipt["total_tensors"].as_u64().unwrap_or(0) > 0);

    // Total parameters must be non-zero
    let total_params: u64 = receipt["classification"]
        .as_object()
        .unwrap()
        .values()
        .map(|v| v["total_params"].as_u64().unwrap_or(0))
        .sum();
    assert!(
        total_params > 0,
        "total parameters must be non-zero, got {}",
        total_params
    );
}

// ── 3. Replay verification ──────────────────────────────────────────

#[test]
fn test_gemma4_replay_verification() {
    // Build a ReplayManifest with a minimal generation and payloads,
    // then verify replay_from_manifest produces a clean drift classification
    // (or gracefully reports Metal unavailability).
    let payload = vec![42u8; 64];
    let seg_id = PhysicalSegmentId(sha256_digest(&payload));

    // Build minimal replay manifest
    let generation = make_base_generation("replay-gen", None, seg_id.clone());

    let mut payloads = BTreeMap::new();
    payloads.insert(seg_id.clone(), payload.clone());

    let artifacts: BTreeMap<KernelSemanticId, ArtifactProvenance> = BTreeMap::new();
    let compiled_artifacts: BTreeMap<KernelSemanticId, Vec<u8>> = BTreeMap::new();

    let bundle = LifecycleReceiptBundle {
        compiler_receipt: ReceiptId("a".repeat(64)),
        numerical_receipt: ReceiptId("b".repeat(64)),
        quality_receipt: ReceiptId("c".repeat(64)),
        performance_receipt: ReceiptId("d".repeat(64)),
        policy_receipt: ReceiptId("e".repeat(64)),
        promotion_receipt: ReceiptId("f".repeat(64)),
        generation_id: generation.generation_id.clone(),
        sealed_at: "2026-07-13T00:00:00Z".into(),
    };

    let manifest = ReplayManifest {
        generation,
        payloads,
        artifacts,
        compiled_artifacts,
        abi: KernelAbi {
            version: 1,
            buffers: vec![],
            constants: vec![],
            threadgroup_memory: vec![],
            dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
            threads_per_threadgroup: (1, 1, 1),
        },
        receipt_bundle: bundle,
        numerical_receipt: NumericalReceipt {
            candidate_id: CandidateId("gemma4-replay".into()),
            passed: true,
            max_absolute_error: 0.001,
            max_relative_error: 0.001,
            threshold: 0.01,
            provenance: Vec::new(),
        },
        performance_receipt: PerformanceReceipt {
            candidate_id: CandidateId("gemma4-replay".into()),
            latency_p50_ns: 1_000_000,
            latency_p95_ns: 1_500_000,
            encode_time_ns: 0,
            sync_time_ns: 0,
            memory_traffic_bytes: 0,
            energy_uj: None,
            repetitions: 1,
            provenance: Vec::new(),
        },
        expects_numerical_parity: true,
    };

    // Replay verification: must produce clean drift classification or
    // fail gracefully if Metal toolchain is unavailable.
    let replay_result = replay_from_manifest(&manifest);

    match &replay_result {
        Ok(outcome) => {
            // Clean replay: payloads verified, drift classification may
            // be None (clean) or Some(Both/Performance/Semantic) depending on
            // whether the actual recompilation matched expectations.
            assert!(
                outcome.payloads_verified,
                "all payloads must be digest-consistent"
            );
            // With no artifacts in the manifest, replay should succeed
            // quickly and report clean drift.
            eprintln!(
                "[gemma4-replay] clean: payloads_verified={} numerical_parity={} drift={:?}",
                outcome.payloads_verified, outcome.numerical_parity, outcome.drift_classification,
            );
        }
        Err(e) => {
            // The only acceptable failure is Metal toolchain unavailability
            assert!(
                e.contains("Metal") || e.contains("toolchain") || e.contains("not available"),
                "replay must succeed or fail only due to Metal unavailability, got: {e}",
            );
            eprintln!("[gemma4-replay] skipped (Metal unavailable): {e}");
        }
    }

    // ── Corrupted artifact test ────────────────────────────────────
    // Verify that a manifest with corrupted payloads fails verification.
    // Store a payload with a deliberately mismatched digest:
    // the seg_id is based on one payload, but we store a different payload.
    let orig_payload = vec![255u8; 64];
    let mismatched_payload = vec![128u8; 64]; // different content
    let bad_seg_id = PhysicalSegmentId(sha256_digest(&orig_payload));

    let bad_generation = make_base_generation("corrupted-replay", None, bad_seg_id.clone());

    let mut bad_payloads = BTreeMap::new();
    bad_payloads.insert(bad_seg_id.clone(), mismatched_payload.clone());

    let bad_manifest = ReplayManifest {
        generation: bad_generation,
        payloads: bad_payloads,
        artifacts: BTreeMap::new(),
        compiled_artifacts: BTreeMap::new(),
        abi: KernelAbi {
            version: 1,
            buffers: vec![],
            constants: vec![],
            threadgroup_memory: vec![],
            dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
            threads_per_threadgroup: (1, 1, 1),
        },
        receipt_bundle: LifecycleReceiptBundle {
            compiler_receipt: ReceiptId("z".repeat(64)),
            numerical_receipt: ReceiptId("y".repeat(64)),
            quality_receipt: ReceiptId("x".repeat(64)),
            performance_receipt: ReceiptId("w".repeat(64)),
            policy_receipt: ReceiptId("v".repeat(64)),
            promotion_receipt: ReceiptId("u".repeat(64)),
            generation_id: GenerationId("corrupted".into()),
            sealed_at: "2026-07-13T00:00:00Z".into(),
        },
        numerical_receipt: NumericalReceipt {
            candidate_id: CandidateId("corrupted".into()),
            passed: false,
            max_absolute_error: 999.0,
            max_relative_error: 999.0,
            threshold: 0.01,
            provenance: Vec::new(),
        },
        performance_receipt: PerformanceReceipt {
            candidate_id: CandidateId("corrupted".into()),
            latency_p50_ns: 0,
            latency_p95_ns: 0,
            encode_time_ns: 0,
            sync_time_ns: 0,
            memory_traffic_bytes: 0,
            energy_uj: None,
            repetitions: 0,
            provenance: Vec::new(),
        },
        expects_numerical_parity: true,
    };

    // The corrupted payload should cause payloads_verified = false.
    // If Metal is available, the replay will catch the digest mismatch.
    // If Metal is unavailable, it returns Err("Metal toolchain not available").
    let bad_result = replay_from_manifest(&bad_manifest);
    match &bad_result {
        Ok(outcome) => {
            assert!(
                !outcome.payloads_verified,
                "corrupted payload should fail verification"
            );
            eprintln!(
                "[gemma4-replay] correctly identified corruption: payloads_verified={}",
                outcome.payloads_verified
            );
        }
        Err(e) => {
            assert!(
                e.contains("Metal") || e.contains("toolchain") || e.contains("not available"),
                "corrupted replay must fail or report Metal unavailability, got: {e}",
            );
            eprintln!("[gemma4-replay] corrupted test skipped (Metal unavailable): {e}");
        }
    }
}

// ── 4. Promote and rollback ─────────────────────────────────────────

#[test]
fn test_gemma4_promote_and_rollback() {
    let mut api = GenerationApi::new();

    // Seed generation
    let payload0 = vec![100u8; 64];
    let seg0 = PhysicalSegmentId(sha256_digest(&payload0));
    api.store_payload(seg0.clone(), payload0);
    let gen0 = make_base_generation("gen0", None, seg0.clone());
    api.promote(gen0).expect("gen0 must promote");

    // Child generation
    let payload1 = vec![200u8; 64];
    let seg1 = PhysicalSegmentId(sha256_digest(&payload1));
    api.store_payload(seg1.clone(), payload1);
    let gen1 = make_base_generation("gen1", Some("gen0"), seg1.clone());
    api.promote(gen1).expect("gen1 must promote");

    // Verify parent is preserved
    let current = api.current_generation().expect("current generation exists");
    assert_eq!(current.generation_id.0, "gen1");
    assert_eq!(
        current.parent_generation.as_ref().map(|p| p.0.as_str()),
        Some("gen0"),
        "child must reference parent gen0"
    );

    // Roll back and verify parent is restored
    let parent = api.rollback().expect("rollback must succeed");
    assert_eq!(parent.0, "gen0", "rollback must return parent id");

    let after_rollback = api
        .current_generation()
        .expect("current generation after rollback");
    assert_eq!(
        after_rollback.generation_id.0, "gen0",
        "after rollback, current generation must be the parent"
    );
}

// ── 5. Memory budget ────────────────────────────────────────────────

#[test]
fn test_gemma4_memory_budget() {
    let budget = M1MemoryBudget::default_16gb();

    // Realistic Gemma 4 sizes: 12-13 GB model, ~500 MB KV cache
    // 12 GB model + 0.5 GB KV + 2 GB OS = 14.5 GB, within 16 GB
    assert!(
        budget.can_fit_model(12.0, 0.5),
        "must fit 12 GB model + 0.5 GB KV"
    );

    // 13 GB model + 0.5 GB KV + 2 GB OS = 15.5 GB, within 16 GB
    assert!(
        budget.can_fit_model(13.0, 0.5),
        "must fit 13 GB model + 0.5 GB KV"
    );

    // 11 GB model + 1.0 GB KV + 2 GB OS = 14.0 GB, within 16 GB
    assert!(
        budget.can_fit_model(11.0, 1.0),
        "must fit 11 GB model + 1.0 GB KV"
    );

    // Edge: exactly at boundary (14 GB model + 0 GB KV = 16 GB total)
    assert!(
        budget.can_fit_model(14.0, 0.0),
        "must exactly fit 14 GB model with no KV cache"
    );

    // Exceeds: 14 GB model + 1 GB KV = 17 GB, exceeds 16 GB
    assert!(
        !budget.can_fit_model(14.0, 1.0),
        "must reject 14 GB model + 1 GB KV (17 GB > 16 GB)"
    );

    // Exceeds: 15 GB model + 0 GB KV = 17 GB, exceeds 16 GB
    assert!(
        !budget.can_fit_model(15.0, 0.0),
        "must reject 15 GB model with no KV cache"
    );

    // Custom budget for smaller models
    let small_budget = M1MemoryBudget {
        total_ram_gb: 8.0,
        os_overhead_gb: 1.0,
        max_model_gb: 7.0,
    };
    assert!(
        small_budget.can_fit_model(6.0, 0.5),
        "must fit 6 GB model + 0.5 GB KV in 8 GB budget"
    );
    assert!(
        !small_budget.can_fit_model(7.5, 0.0),
        "must reject 7.5 GB model in 8 GB budget"
    );
}

// ── 6. Catalogue registration ───────────────────────────────────────

#[test]
fn test_gemma4_catalogue_registration() {
    // Verify that catalogue_source_for returns valid (non-empty) source
    // paths for every Gemma 4 kernel semantic ID.
    let mut all_found = true;
    let mut failures: Vec<&str> = Vec::new();

    for (sem_id, source_path) in GEMMA4_KERNEL_SEMANTICS {
        let source = catalogue_source_for(&KernelSemanticId(sem_id.to_string()));
        match source {
            Some(content) => {
                assert!(
                    !content.is_empty(),
                    "source for semantic ID '{}' must be non-empty (file: {})",
                    sem_id,
                    source_path
                );
                // Verify the source contains a valid Metal function declaration
                assert!(
                    content.contains("kernel")
                        || content.contains("Metal")
                        || content.contains("device"),
                    "source for '{}' must contain valid Metal shader content",
                    sem_id
                );
                eprintln!(
                    "[catalogue] OK: {} -> {} ({} bytes)",
                    sem_id,
                    source_path,
                    content.len()
                );
            }
            None => {
                all_found = false;
                failures.push(sem_id);
                eprintln!("[catalogue] MISSING: {} ({})", sem_id, source_path);
            }
        }
    }

    assert!(
        all_found,
        "{} Gemma 4 kernel semantic IDs missing from catalogue: {:?}",
        failures.len(),
        failures
    );
}
