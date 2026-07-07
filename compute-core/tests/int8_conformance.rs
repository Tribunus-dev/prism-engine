//! Prism V1 Runtime Representation conformance: INT8 Tile640.
//!
//! Validates symmetric quantization invariants, code range constraints,
//! payload byte formulas per spec §10.3, full/partial tile reconstruction,
//! and nonzero metadata offset binding validation.

use tribunus_compute_core::compute_image::cimage_loader::validate_binding;
use tribunus_compute_core::compute_image::compile::ternary::MatrixWeightBindingV1;
use tribunus_compute_core::quantization::admission::{pack_candidate, reconstruct_candidate};
use tribunus_compute_core::quantization::contract::*;

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Tolerance for reconstruction RMSE.
const RMSE_TOL: f64 = 5e-3;

/// Tile elements per axis.
const T640: usize = 640;

/// Compute RMSE between two f32 slices.
fn rmse(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().max(1) as f64;
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    (sum / n).sqrt()
}

// ── Test 1: Symmetric zero-bias invariant ──────────────────────────────────

#[test]
fn symmetric_zero_bias_invariant_test() {
    // INT8 uses symmetric quantization: per-tile beta is always 0.0.
    let in_features = 640;
    let out_features = 640;
    let source: Vec<f32> = (0..in_features * out_features)
        .map(|i| ((i as f32) * 0.0078125 - 0.5) * 2.0)
        .collect();

    let (_codes, _scales, biases, _scale_vec) = pack_candidate(
        &source,
        in_features,
        out_features,
        RuntimeRepresentationClass::Int8Tile640Base,
        None,
    );

    // Every bias must be exactly zero (symmetric quantization has no beta).
    assert!(
        biases.iter().all(|b| *b == 0.0),
        "INT8 symmetric quantization: expected all biases == 0.0, got {} non-zero samples",
        biases.iter().filter(|b| **b != 0.0).count()
    );
}

// ── Test 2: Signed code range ─────────────────────────────────────────────

#[test]
fn signed_code_range_test() {
    // INT8 codes are stored as u8 but logically signed i8 in [-127, +127].
    // -128 must never appear (clamp prevents it).
    let in_features = 17;
    let out_features = 640;

    // Fill with values that would produce -1.0*128 = -128 without clamping.
    let source: Vec<f32> = (0..in_features * out_features)
        .map(|i| ((i as f32) * 7.3).sin() * 200.0)
        .collect();

    let (codes, _scales, _biases, _scale_vec) = pack_candidate(
        &source,
        in_features,
        out_features,
        RuntimeRepresentationClass::Int8Tile640Base,
        None,
    );

    // Interpret every code byte as signed i8; none may be -128 (0x80).
    let num_violations = codes.iter().filter(|c| **c == 0x80u8).count();

    assert_eq!(
        num_violations, 0,
        "INT8 signed range violation: {} codes equal -128 (0x80), must be in [-127, +127]",
        num_violations
    );

    // Also verify max/min are in valid signed range.
    let signed_codes: Vec<i8> = codes.iter().map(|c| *c as i8).collect();
    let min_code = *signed_codes.iter().min().unwrap();
    let max_code = *signed_codes.iter().max().unwrap();
    assert!(
        min_code >= -127 && max_code <= 127,
        "INT8 code range [{}, {}] exceeds [-127, 127]",
        min_code,
        max_code
    );
}

// ── Test 3: Exact payload byte formula ─────────────────────────────────────

#[test]
fn exact_payload_byte_formula_test() {
    // Per spec §10.3: INT8_TILE640_CODE_BYTES = 640, INT8_TILE640_METADATA_BYTES = 4.
    // validate_binding checks that code_length == total_tiles * 640 and
    // metadata_length == total_tiles * 4.
    let in_features: u32 = 640;
    let out_features: u32 = 640;
    let total_tiles = (out_features as u64) * ((in_features as u64).div_ceil(T640 as u64));
    let expected_code = total_tiles * INT8_TILE640_CODE_BYTES as u64;
    let expected_meta = total_tiles * INT8_TILE640_METADATA_BYTES as u64;

    let binding = MatrixWeightBindingV1 {
        binding_wire_version: 1,
        matrix_id: 99,
        tensor_id: [0; 16],
        representation: RuntimeRepresentationClass::Int8Tile640Base as u8,
        representation_version: 1,
        kernel_abi_digest: [0; 32],
        in_features,
        out_features,
        reduction_tile_size: T640 as u16,
        tiles_per_output_channel: (in_features as u64).div_ceil(T640 as u64) as u32,
        tail_reduction_count: (in_features % T640 as u32) as u16,
        macro_layout: TileMacroLayout::OutputChannelContiguous as u8,
        tail_handling: TailHandlingContract::ActivationZeroPredicationV1 as u8,
        code_segment: 1,
        code_offset: 0,
        code_length: expected_code,
        code_tile_stride_bytes: INT8_TILE640_CODE_BYTES as u32,
        metadata_segment: 1,
        metadata_offset: expected_code,
        metadata_length: expected_meta,
        metadata_tile_stride_bytes: INT8_TILE640_METADATA_BYTES as u16,
        sidecar_segment: 0,
        sidecar_offset: 0,
        sidecar_length: 0,
        sidecar_kind: 0,
        sidecar_element_format: 0,
        sidecar_count: 0,
        residual_segment: 0,
        residual_offset: 0,
        residual_length: 0,
        required_alignment_bytes: 1,
    };

    let cimage_bytes = (expected_code + expected_meta) as usize;
    validate_binding(&binding, cimage_bytes).expect("INT8 byte formula validation must pass");
}

// ── Test 4: Full tile reconstruction ───────────────────────────────────────

#[test]
fn tile640_full_tile_test() {
    // Pack and reconstruct an exact 640x640 matrix.
    let in_features = 640;
    let out_features = 640;
    let n = in_features * out_features;

    // Deterministic values that exercise the full [-1, 1] range.
    let source: Vec<f32> = (0..n)
        .map(|i| ((i as f64) * 1.618_033_988_749_895).sin() as f32)
        .collect();

    let format = RuntimeRepresentationClass::Int8Tile640Base;
    let (codes, scales, biases, scale_vec) =
        pack_candidate(&source, in_features, out_features, format, None);
    let reconstructed = reconstruct_candidate(
        format,
        &codes,
        &scales,
        &biases,
        in_features,
        out_features,
        scale_vec.as_deref(),
    );

    assert_eq!(reconstructed.len(), n);

    let err_rmse = rmse(&source, &reconstructed);
    assert!(
        err_rmse < RMSE_TOL,
        "INT8 640x640 full tile RMSE {:.6} exceeds {:.6}",
        err_rmse,
        RMSE_TOL
    );
}

// ── Test 5: Partial tail tile reconstruction ───────────────────────────────

#[test]
fn tile640_partial_tail_test() {
    // Pack and reconstruct a 17x31 matrix — exercises partial tail tile.
    let in_features = 17;
    let out_features = 31;
    let n = in_features * out_features;

    let source: Vec<f32> = (0..n)
        .map(|i| ((i as f64) * 2.718_281_828_459_045).cos() as f32)
        .collect();

    let format = RuntimeRepresentationClass::Int8Tile640Base;
    let (codes, scales, biases, scale_vec) =
        pack_candidate(&source, in_features, out_features, format, None);
    let reconstructed = reconstruct_candidate(
        format,
        &codes,
        &scales,
        &biases,
        in_features,
        out_features,
        scale_vec.as_deref(),
    );

    assert_eq!(reconstructed.len(), n);

    let err_rmse = rmse(&source, &reconstructed);
    assert!(
        err_rmse < RMSE_TOL,
        "INT8 17x31 partial tail RMSE {:.6} exceeds {:.6}",
        err_rmse,
        RMSE_TOL
    );
}

// ── Test 6: Nonzero metadata offset ────────────────────────────────────────

#[test]
fn nonzero_metadata_offset_test() {
    // Create a binding where metadata_offset is past code payload but within
    // total cimage size. validate_binding must accept it.
    let in_features: u32 = 640;
    let out_features: u32 = 640;
    let tiles = out_features as u64; // tiles_per_output_channel == 1
    let code_len = tiles * INT8_TILE640_CODE_BYTES as u64;
    let meta_len = tiles * INT8_TILE640_METADATA_BYTES as u64;
    let meta_off = code_len + 1000; // non-zero gap between codes and metadata

    let binding = MatrixWeightBindingV1 {
        binding_wire_version: 1,
        matrix_id: 42,
        tensor_id: [0; 16],
        representation: RuntimeRepresentationClass::Int8Tile640Base as u8,
        representation_version: 1,
        kernel_abi_digest: [0; 32],
        in_features,
        out_features,
        reduction_tile_size: T640 as u16,
        tiles_per_output_channel: 1,
        tail_reduction_count: 0,
        macro_layout: TileMacroLayout::OutputChannelContiguous as u8,
        tail_handling: TailHandlingContract::ActivationZeroPredicationV1 as u8,
        code_segment: 1,
        code_offset: 0,
        code_length: code_len,
        code_tile_stride_bytes: INT8_TILE640_CODE_BYTES as u32,
        metadata_segment: 1,
        metadata_offset: meta_off,
        metadata_length: meta_len,
        metadata_tile_stride_bytes: INT8_TILE640_METADATA_BYTES as u16,
        sidecar_segment: 0,
        sidecar_offset: 0,
        sidecar_length: 0,
        sidecar_kind: 0,
        sidecar_element_format: 0,
        sidecar_count: 0,
        residual_segment: 0,
        residual_offset: 0,
        residual_length: 0,
        required_alignment_bytes: 1,
    };

    let cimage_bytes = (meta_off + meta_len) as usize;
    validate_binding(&binding, cimage_bytes)
        .expect("Binding with nonzero metadata_offset must pass validate_binding");
}
