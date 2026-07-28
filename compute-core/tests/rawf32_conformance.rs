#![cfg(feature = "prism-backend")]

//! RawF32 representation conformance tests (Spec §18).
//!
//! All tests are CPU-only — no Metal or GPU required. They verify byte layout,
//! binding contract validation, source layout normalization, and format dispatch
//! isolation for the RawF32 (RuntimeRepresentationClass::RawF32 = 3) path.

use tribunus_compute_core::compute_image::cimage_loader::validate_binding;
use tribunus_compute_core::compute_image::legacy_compute_image_compile::ternary::MatrixWeightBindingV1;
use tribunus_compute_core::quantization::admission::{pack_candidate, reconstruct_candidate};
use tribunus_compute_core::quantization::contract::{
    validate_source_layout, RuntimeRepresentationClass, SourceMatrixLayout, TileMacroLayout,
};

// ── Test 1: Canonical orientation storage ──────────────────────────────────

#[test]
fn canonical_orientation_storage_test() {
    // Create a 3x4 f32 matrix in Prism canonical layout:
    // W[in_features=3, out_features=4] — row-major, 3 rows × 4 cols.
    let in_f = 3;
    let out_f = 4;
    let mut source = Vec::with_capacity(in_f * out_f);
    for i in 0..in_f {
        for j in 0..out_f {
            source.push((i * out_f + j + 1) as f32);
        }
    }

    // Pack into RawF32 format (flat f32 LE bytes, in-f outermost).
    let (codes, scales, biases, scale_vec) = pack_candidate(
        &source,
        in_f,
        out_f,
        RuntimeRepresentationClass::RawF32,
        None,
    );

    // RawF32 should produce no scales, biases, or scale vector.
    assert!(scales.is_empty(), "RawF32 must not produce scales");
    assert!(biases.is_empty(), "RawF32 must not produce biases");
    assert!(scale_vec.is_none(), "RawF32 must not produce scale vector");

    // Byte length must be in_f * out_f * 4.
    assert_eq!(codes.len(), in_f * out_f * 4);

    // Verify each f32 was stored in little-endian bytes in row-major order.
    for idx in 0..(in_f * out_f) {
        let expected_val = (idx + 1) as f32;
        let le_bytes = expected_val.to_le_bytes();
        let offset = idx * 4;
        assert_eq!(
            codes[offset..offset + 4],
            le_bytes,
            "byte mismatch at element {} (value {})",
            idx,
            expected_val
        );
    }

    // Round-trip through reconstruct_candidate.
    let reconstructed = reconstruct_candidate(
        RuntimeRepresentationClass::RawF32,
        &codes,
        &scales,
        &biases,
        in_f,
        out_f,
        None,
    );
    assert_eq!(reconstructed, source, "RawF32 round-trip must be exact");
}

// ── Test 2: F32 byte order (little-endian) ────────────────────────────────

#[test]
fn f32_byte_order_test() {
    // Known float values and their little-endian byte representations.
    let cases: &[(f32, [u8; 4])] = &[
        (1.0f32, [0x00, 0x00, 0x80, 0x3F]),
        (2.5f32, [0x00, 0x00, 0x20, 0x40]),
        (-3.125f32, [0x00, 0x00, 0x48, 0xC0]),
    ];

    for (val, expected_le) in cases {
        let actual_le = val.to_le_bytes();
        assert_eq!(
            &actual_le, expected_le,
            "LE byte mismatch for value {}",
            val
        );
    }

    // Verify pack_candidate emits the same LE bytes for a multi-element matrix.
    let source = vec![1.0f32, 2.5, -3.125];
    let (codes, _, _, _) = pack_candidate(&source, 1, 3, RuntimeRepresentationClass::RawF32, None);
    assert_eq!(codes.len(), 3 * 4);
    assert_eq!(&codes[0..4], &1.0f32.to_le_bytes());
    assert_eq!(&codes[4..8], &2.5f32.to_le_bytes());
    assert_eq!(&codes[8..12], &(-3.125f32).to_le_bytes());
}

// ── Test 3: RawF32 dispatch isolation — not routed through quantized path ──

#[test]
fn rawf32_dispatch_isolation_test() {
    // Construct a valid RawF32 binding.
    let in_f = 3u32;
    let out_f = 4u32;
    let code_len = (in_f as u64) * (out_f as u64) * 4;

    let valid_binding = MatrixWeightBindingV1::new(
        1,         // binding_wire_version
        0,         // matrix_id
        [0u8; 16], // tensor_id
        3,         // representation = RawF32
        1,         // representation_version
        [0u8; 32], // kernel_abi_digest
        in_f,      // in_features
        out_f,     // out_features
        0,         // reduction_tile_size (must be 0)
        0,         // tiles_per_output_channel (must be 0)
        0,         // tail_reduction_count
        TileMacroLayout::OutputChannelContiguous as u8,
        0,        // tail_handling (skipped for RawF32)
        0,        // code_segment
        0,        // code_offset
        code_len, // code_length
        0,        // code_tile_stride_bytes
        0,        // metadata_segment
        0,        // metadata_offset
        0,        // metadata_length
        0,        // metadata_tile_stride_bytes
        0,        // sidecar_segment
        0,        // sidecar_offset
        0,        // sidecar_length
        0,        // sidecar_kind
        0,        // sidecar_element_format
        0,        // sidecar_count
        0,        // residual_segment
        0,        // residual_offset
        0,        // residual_length
        0,        // required_alignment_bytes
    )
    .expect("valid RawF32 binding must construct");

    // validate_binding accepts it.
    assert!(
        validate_binding(&valid_binding, 1024 * 1024).is_ok(),
        "validate_binding must accept a valid RawF32 binding"
    );

    // A binding with representation=3 and reduction_tile_size != 0 must be rejected.
    // Construct directly (bypass ::new() which would reject it).
    let bad_binding = MatrixWeightBindingV1 {
        binding_wire_version: 1,
        matrix_id: 1,
        tensor_id: [0u8; 16],
        representation: 3,
        representation_version: 1,
        kernel_abi_digest: [0u8; 32],
        in_features: in_f,
        out_features: out_f,
        reduction_tile_size: 640, // illegal for RawF32
        tiles_per_output_channel: 0,
        tail_reduction_count: 0,
        macro_layout: TileMacroLayout::OutputChannelContiguous as u8,
        tail_handling: 0,
        code_segment: 0,
        code_offset: 0,
        code_length: code_len,
        code_tile_stride_bytes: 0,
        metadata_segment: 0,
        metadata_offset: 0,
        metadata_length: 0,
        metadata_tile_stride_bytes: 0,
        sidecar_segment: 0,
        sidecar_offset: 0,
        sidecar_length: 0,
        sidecar_kind: 0,
        sidecar_element_format: 0,
        sidecar_count: 0,
        residual_segment: 0,
        residual_offset: 0,
        residual_length: 0,
        required_alignment_bytes: 0,
    };

    let result = validate_binding(&bad_binding, 1024 * 1024);
    assert!(
        result.is_err(),
        "RawF32 with reduction_tile_size != 0 must be rejected"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("reduction_tile_size"),
        "error must mention reduction_tile_size: {}",
        err
    );
}

// ── Test 4: CheckpointOutByIn transpose (non-square) ──────────────────────

#[test]
fn canonical_source_layout_transpose_test() {
    // Source matrix: 4 rows × 3 cols (CheckpointOutByIn convention).
    // Expected CanonicalShape: in_features=3, out_features=4.
    let result = validate_source_layout(
        4,  // source_rows
        3,  // source_cols
        12, // source_element_count
        3,  // in_features
        4,  // out_features
        SourceMatrixLayout::CheckpointOutByIn,
    );

    let shape = result.expect("validate_source_layout must succeed");
    assert_eq!(
        shape.in_features, 3,
        "CheckpointOutByIn: normalized in_features must be source_cols"
    );
    assert_eq!(
        shape.out_features, 4,
        "CheckpointOutByIn: normalized out_features must be source_rows"
    );
    assert_eq!(shape.rank, 2, "canonical shape rank must be 2");
}

// ── Test 5: Square matrix transpose (orientation matters) ─────────────────

#[test]
fn square_matrix_source_layout_transpose_test() {
    // 4x4 square: same dimensions still require orientation swap via CheckpointOutByIn.
    let result = validate_source_layout(
        4,  // source_rows
        4,  // source_cols
        16, // source_element_count
        4,  // in_features = source_cols (normalized)
        4,  // out_features = source_rows (normalized)
        SourceMatrixLayout::CheckpointOutByIn,
    );

    let shape = result.expect("validate_source_layout must succeed for square");
    assert_eq!(shape.in_features, 4);
    assert_eq!(shape.out_features, 4);
    assert_eq!(shape.rank, 2);

    // Prove orientation matters: wrong in/out assignment fails.
    let wrong = validate_source_layout(
        4,
        4,
        16,
        3, // wrong in_features
        5, // wrong out_features
        SourceMatrixLayout::CheckpointOutByIn,
    );
    assert!(
        wrong.is_err(),
        "must reject mismatched in/out features even for square matrices"
    );
}

// ── Test 6: source_element_count mismatch ─────────────────────────────────

#[test]
fn element_count_mismatch_rejection() {
    // Source claims 4x3 but element_count is wrong.
    let result = validate_source_layout(
        4,  // source_rows
        3,  // source_cols
        13, // source_element_count (wrong! 4x3 = 12)
        3,  // in_features
        4,  // out_features
        SourceMatrixLayout::CheckpointOutByIn,
    );

    assert!(result.is_err(), "element_count mismatch must be rejected");
    let err = result.unwrap_err();
    assert!(
        err.contains("13") && err.contains("12"),
        "error must mention wrong count: {}",
        err
    );
}

// ── Test 7: RawF32 binding zero-tile contract ─────────────────────────────

#[test]
fn binding_zero_tile_contract() {
    let in_f = 3u32;
    let out_f = 4u32;
    let code_len = (in_f as u64) * (out_f as u64) * 4;

    let binding = MatrixWeightBindingV1::new(
        1,
        0,
        [0u8; 16],
        3,
        1,
        [0u8; 32],
        in_f,
        out_f,
        0, // reduction_tile_size
        0, // tiles_per_output_channel
        0, // tail_reduction_count
        TileMacroLayout::OutputChannelContiguous as u8,
        0, // tail_handling
        0, // code_segment
        0, // code_offset
        code_len,
        0, // code_tile_stride_bytes
        0, // metadata_segment
        0, // metadata_offset
        0, // metadata_length
        0, // metadata_tile_stride_bytes
        0, // sidecar_segment
        0, // sidecar_offset
        0, // sidecar_length
        0, // sidecar_kind
        0, // sidecar_element_format
        0, // sidecar_count
        0, // residual_segment
        0, // residual_offset
        0, // residual_length
        0, // required_alignment_bytes
    )
    .expect("valid RawF32 must construct");

    // The core tile-contract fields that distinguish RawF32 from quantized reps.
    assert_eq!(
        binding.reduction_tile_size, 0,
        "RawF32 reduction_tile_size must be 0"
    );
    assert_eq!(
        binding.tiles_per_output_channel, 0,
        "RawF32 tiles_per_output_channel must be 0"
    );
    assert_eq!(
        binding.code_tile_stride_bytes, 0,
        "RawF32 code_tile_stride_bytes must be 0"
    );

    // Validate also enforces these through validate_binding.
    assert!(
        validate_binding(&binding, 1024 * 1024).is_ok(),
        "validate_binding must accept binding with zero-tile contract"
    );
}

// ── Test 8: Binding with non-zero metadata_length rejected ────────────────

#[test]
fn binding_metadata_rejection() {
    let in_f = 3u32;
    let out_f = 4u32;
    let code_len = (in_f as u64) * (out_f as u64) * 4;

    // Construct binding directly with metadata_length != 0 (bypass ::new()
    // which would reject at construction time).
    let bad_binding = MatrixWeightBindingV1 {
        binding_wire_version: 1,
        matrix_id: 0,
        tensor_id: [0u8; 16],
        representation: 3,
        representation_version: 1,
        kernel_abi_digest: [0u8; 32],
        in_features: in_f,
        out_features: out_f,
        reduction_tile_size: 0,
        tiles_per_output_channel: 0,
        tail_reduction_count: 0,
        macro_layout: TileMacroLayout::OutputChannelContiguous as u8,
        tail_handling: 0,
        code_segment: 0,
        code_offset: 0,
        code_length: code_len,
        code_tile_stride_bytes: 0,
        metadata_segment: 0,
        metadata_offset: 0,
        metadata_length: 64, // illegal — RawF32 metadata must be empty
        metadata_tile_stride_bytes: 0,
        sidecar_segment: 0,
        sidecar_offset: 0,
        sidecar_length: 0,
        sidecar_kind: 0,
        sidecar_element_format: 0,
        sidecar_count: 0,
        residual_segment: 0,
        residual_offset: 0,
        residual_length: 0,
        required_alignment_bytes: 0,
    };

    let result = validate_binding(&bad_binding, 1024 * 1024);
    assert!(
        result.is_err(),
        "validate_binding must reject RawF32 with non-zero metadata_length"
    );
    let err = result.unwrap_err();
    assert!(
        err.contains("metadata") || err.contains("metadata_length"),
        "error must mention metadata: {}",
        err
    );
}
