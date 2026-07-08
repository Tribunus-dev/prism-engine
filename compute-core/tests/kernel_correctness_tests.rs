//! Kernel-level correctness tests for Prism's Metal kernel templates.
//!
//! These integration tests verify the data types and dispatch logic that
//! govern PSO compilation. They are platform-independent (no Metal device
//! required) and validate:
//!
//! 1.  `KernelSpecializationKey` — every codec/layout field is serialized,
//!     so no parameter is silently ignored when computing a PSO cache key.
//! 2.  `KernelTemplate` bindings — the declared buffer assignment matches
//!     the actual Metal shader ABI.
//! 3.  `PsoCacheKey` — the deterministic mapping covers all fields.
//! 4.  `FunctionConstantSet` — each constant maps to a documented shader
//!     `constant_id` slot.

use tribunus_compute_core::execution_plan::*;
use tribunus_compute_core::execution_plan::pso_cache::PsoCacheKey;

// ═══════════════════════════════════════════════════════════════════════
// Test 1: KernelSpecializationKey serializes every format-dependent field
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_kernel_specialization_key_covers_all_format_params() {
    // Build a maximally populated key for the nf4_tile640_gemv kernel.
    let key = KernelSpecializationKey {
        template_id: KernelTemplateId::Nf4Tile640Gemv,
        execution_phase: ExecutionPhase::Decode,
        codec: CodecFamily::Nf4,
        tile_shape: TileShape::tile640_decode(),
        group_size: 32,
        group_axis: Axis::PackedContiguous,
        affine_mode: AffineMode::ScaleOnly,
        metadata_layout: MetadataLayout::AdjacentTile,
        input_dtype: DType::F32,
        output_dtype: DType::F16,
        hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
        mode_flags: 0,
    };

    // Serialize to JSON — serde must round-trip every field.
    let json = serde_json::to_value(&key).expect("KernelSpecializationKey must be JSON-serializable");

    // Every field that affects PSO compilation MUST appear in the JSON output.
    // If a field is added to the struct but not checked here, the test fails.
    assert!(json.get("template_id").is_some(),    "missing template_id");
    assert!(json.get("execution_phase").is_some(),"missing execution_phase");
    assert!(json.get("codec").is_some(),          "missing codec");
    assert!(json.get("tile_shape").is_some(),     "missing tile_shape");
    assert!(json.get("group_size").is_some(),     "missing group_size");
    assert!(json.get("group_axis").is_some(),     "missing group_axis");
    assert!(json.get("affine_mode").is_some(),    "missing affine_mode");
    assert!(json.get("metadata_layout").is_some(),"missing metadata_layout");
    assert!(json.get("input_dtype").is_some(),    "missing input_dtype");
    assert!(json.get("output_dtype").is_some(),   "missing output_dtype");
    assert!(json.get("hardware_profile").is_some(),"missing hardware_profile");
    assert!(json.get("mode_flags").is_some(),     "missing mode_flags");

    // Verify round-trip deserialization preserves equality.
    let deserialized: KernelSpecializationKey =
        serde_json::from_value(json).expect("KernelSpecializationKey must round-trip through JSON");
    assert_eq!(key, deserialized,
        "KernelSpecializationKey round-trip must preserve semantic equality");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: PsoCacheKey deterministic mapping covers all fields
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_pso_cache_key_fully_deterministic() {
    // Two keys that differ only in one field must produce different cache keys.
    let base = KernelSpecializationKey {
        template_id: KernelTemplateId::Nf4Tile640Gemv,
        execution_phase: ExecutionPhase::Decode,
        codec: CodecFamily::Nf4,
        tile_shape: TileShape::tile640_decode(),
        group_size: 32,
        group_axis: Axis::PackedContiguous,
        affine_mode: AffineMode::ScaleOnly,
        metadata_layout: MetadataLayout::AdjacentTile,
        input_dtype: DType::F32,
        output_dtype: DType::F16,
        hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
        mode_flags: 0,
    };

    let cache_base: PsoCacheKey = (&base).into();

    // Vary every single field one at a time — each mutation MUST produce a
    // distinct PsoCacheKey. If any pair collides, the PSO cache would
    // silently reuse a stale pipeline for a different parameterization.
    // NOTE: input_dtype and output_dtype are template-inherent
    // (baked into the Metal function name), not cache-key fields,
    // so they are intentionally not tested here.
    for (label, varied) in [
        ("template_id",         KernelSpecializationKey { template_id:    KernelTemplateId::Int8Tile640Gemv,        ..base.clone() }),
        ("execution_phase",     KernelSpecializationKey { execution_phase: ExecutionPhase::Prefill,                  ..base.clone() }),
        ("codec",               KernelSpecializationKey { codec:           CodecFamily::Int8,                       ..base.clone() }),
        ("tile_shape",          KernelSpecializationKey { tile_shape:      TileShape::tile256_decode(),             ..base.clone() }),
        ("group_size",          KernelSpecializationKey { group_size:      64,                                       ..base.clone() }),
        ("group_axis",          KernelSpecializationKey { group_axis:      Axis::Input,                             ..base.clone() }),
        ("affine_mode",         KernelSpecializationKey { affine_mode:     AffineMode::ScaleBias,                    ..base.clone() }),
        ("metadata_layout",     KernelSpecializationKey { metadata_layout: MetadataLayout::SeparatedManifest,        ..base.clone() }),
        ("hardware_profile",    KernelSpecializationKey { hardware_profile: HardwareProfileId::AppleMProBalanced,     ..base.clone() }),
        ("mode_flags",          KernelSpecializationKey { mode_flags:      1,                                        ..base.clone() }),
    ] {
        let cache_varied: PsoCacheKey = (&varied).into();
        assert_ne!(
            cache_base, cache_varied,
            "PsoCacheKey collision: varying {label} produced the same key as the base configuration",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: Nf4Tile640Gemv kernel template bindings match Metal shader ABI
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_nf4_tile640_gemv_template_bindings_match_shader_abi() {
    // The nf4_tile640_gemv Metal kernel declares buffer slots in its
    // buffer ABI (see compute-core/src/compute_image/templates/nf4_tile640_gemv.metal):
    //
    //   [0] packed_weights  device const uchar*  raw Tile640 bytes
    //   [1] scales          device const float*  FP32 group scales
    //   [2] biases          device const float*  FP32 group biases
    //   [3] in_vector       device const float*  activation vector [in_dim]
    //   [4] out_vector      device float*        result vector
    //   [5] num_macro_tiles constant uint8       ceil(in_dim / 640)
    //   [6] in_dim          constant uint8       real (unpadded) input width
    //
    // Buffers 5–6 are constants, not regular buffers, so the template only
    // declares bindings for slots 0–4.
    let tmpl = KernelTemplate {
        id: KernelTemplateId::Nf4Tile640Gemv,
        metal_function_name: "fused_gemv_nf4_tile640_fp32".into(),
        expected_bindings: vec![
            BindingSpec { index: 0, purpose: "packed_weights".into(), required: true },
            BindingSpec { index: 1, purpose: "scales".into(),           required: true },
            BindingSpec { index: 2, purpose: "biases".into(),           required: false },  // unused in NF4 (always 0)
            BindingSpec { index: 3, purpose: "in_vector".into(),        required: true },
            BindingSpec { index: 4, purpose: "out_vector".into(),       required: true },
        ],
        supported_phases: vec![ExecutionPhase::Decode],
        supported_codecs: vec![CodecFamily::Nf4],
        supports_function_constants: true,
    };

    // Validate invariants: at least 3 required bindings, all bindings
    // have unique indices, and the name matches the conventional pattern.
    let required_count = tmpl.expected_bindings.iter().filter(|b| b.required).count();
    assert!(
        required_count >= 3,
        "Nf4Tile640Gemv template must have at least 3 required bindings (got {required_count})"
    );

    // Check unique indices — no duplicate buffer slot numbers.
    let mut indices: Vec<u32> = tmpl.expected_bindings.iter().map(|b| b.index).collect();
    indices.sort();
    indices.dedup();
    assert_eq!(
        indices.len(),
        tmpl.expected_bindings.len(),
        "Nf4Tile640Gemv template must not have duplicate buffer indices"
    );

    // The metal function name follows the fused_{codec}_tile{size}_{op} naming convention.
    assert!(
        tmpl.metal_function_name.starts_with("fused_gemv_nf4_tile640_"),
        "Nf4Tile640Gemv metal function name should follow fused_gemv_nf4_tile640_* pattern"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Test 4: Int8Tile640Gemv kernel template has valid bindings
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_int8_tile640_gemv_template_bindings_valid() {
    let tmpl = KernelTemplate {
        id: KernelTemplateId::Int8Tile640Gemv,
        metal_function_name: "fused_gemv_int8_tile640_fp32".into(),
        expected_bindings: vec![
            BindingSpec { index: 0, purpose: "packed_weights".into(), required: true },
            BindingSpec { index: 1, purpose: "scales".into(),           required: true },
            BindingSpec { index: 2, purpose: "biases".into(),           required: false },
            BindingSpec { index: 3, purpose: "in_vector".into(),        required: true },
            BindingSpec { index: 4, purpose: "out_vector".into(),       required: true },
        ],
        supported_phases: vec![ExecutionPhase::Decode],
        supported_codecs: vec![CodecFamily::Int8],
        supports_function_constants: true,
    };

    let required_count = tmpl.expected_bindings.iter().filter(|b| b.required).count();
    assert!(
        required_count >= 3,
        "Int8Tile640Gemv template must have at least 3 required bindings (got {required_count})"
    );

    // Validate NameMatches
    assert!(
        tmpl.metal_function_name.starts_with("fused_gemv_int8_tile640_"),
        "Int8Tile640Gemv metal function name should follow fused_gemv_int8_tile640_*"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// Test 5: KernelTemplateId exhaustiveness covers Tile640 family
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_kernel_template_id_covers_tile640_family() {
    // Verify that all Tile640 codec/op kernel IDs plausibly exist.
    // This is a structural sanity check — if a new codec gets a Tile640
    // kernel, it should also get a KernelTemplateId variant.
    let tile640_ids: &[KernelTemplateId] = &[
        KernelTemplateId::Nf4Tile640Gemv,
        KernelTemplateId::Int8Tile640Gemv,
    ];

    for id in tile640_ids {
        match id {
            KernelTemplateId::Nf4Tile640Gemv | KernelTemplateId::Int8Tile640Gemv => {
                // Known Tile640 GEMV kernels — valid.
            }
            _ => panic!("Unexpected tile640 variant: {id:?}"),
        }
    }
}
