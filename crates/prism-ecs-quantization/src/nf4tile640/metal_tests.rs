//! Tests for Metal-profile descriptor compatibility.
//! These tests verify the CPU-Metal boundary without requiring GPU hardware.

use super::profile::{
    CodebookDescriptor, QuantizerProfile, PROFILE_ID_CANONICAL_NF4_V1,
    PROFILE_ID_GEMMA_ATTENTION_V1,
};

/// ProfileDescriptor struct layout that MUST match the Metal shader's struct.
/// Metal struct:
///   uint  profile_id       (offset 0, 4 bytes)
///   uint  abi_version      (offset 4, 4 bytes)
///   uint  group_size       (offset 8, 4 bytes)
///   uint  tile_elements    (offset 12, 4 bytes)
///   float codebook[16]     (offset 16, 64 bytes)
///   uint  clipping_policy  (offset 80, 4 bytes)
///   uint  bias_policy      (offset 84, 4 bytes)
///   uint  sidecar_policy   (offset 88, 4 bytes)
///   uint  pad              (offset 92, 4 bytes)
/// Total: 96 bytes
#[repr(C)]
struct ProfileDescriptor {
    profile_id: u32,
    abi_version: u32,
    group_size: u32,
    tile_elements: u32,
    codebook: [f32; 16],
    clipping_policy: u32,
    bias_policy: u32,
    sidecar_policy: u32,
    pad: u32,
}

#[test]
fn test_profile_descriptor_metal_layout() {
    use std::mem::{offset_of, size_of};
    assert_eq!(
        size_of::<ProfileDescriptor>(),
        96,
        "ProfileDescriptor must be 96 bytes to match Metal struct layout"
    );
    // Verify field offsets
    assert_eq!(offset_of!(ProfileDescriptor, profile_id), 0);
    assert_eq!(offset_of!(ProfileDescriptor, abi_version), 4);
    assert_eq!(offset_of!(ProfileDescriptor, group_size), 8);
    assert_eq!(offset_of!(ProfileDescriptor, tile_elements), 12);
    assert_eq!(offset_of!(ProfileDescriptor, codebook), 16);
    assert_eq!(offset_of!(ProfileDescriptor, clipping_policy), 80);
    assert_eq!(offset_of!(ProfileDescriptor, bias_policy), 84);
    assert_eq!(offset_of!(ProfileDescriptor, sidecar_policy), 88);
    assert_eq!(offset_of!(ProfileDescriptor, pad), 92);
}

#[test]
fn test_fallback_codebook_matches_cpu() {
    // The Metal shader fallback values must match CPU's NF4_CODEBOOK exactly.
    let metal_fallback: [f32; 16] = [
        -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.09105, 0.0, 0.0795803,
        0.1609302, 0.2461123, 0.3379152, 0.4407099, 0.562617, 0.7229568, 1.0,
    ];
    let cpu = crate::nf4tile640::NF4_CODEBOOK;
    for i in 0..16 {
        assert!(
            (metal_fallback[i] - cpu[i]).abs() < 1e-6,
            "Mismatch at [{i}]: Metal={:.10}, CPU={:.10}",
            metal_fallback[i],
            cpu[i]
        );
    }
}

#[test]
fn test_canonical_codebook_matches_fallback() {
    let desc = CodebookDescriptor::canonical_nf4();
    let metal_fallback: [f32; 16] = [
        -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.09105, 0.0, 0.0795803,
        0.1609302, 0.2461123, 0.3379152, 0.4407099, 0.562617, 0.7229568, 1.0,
    ];
    for i in 0..16 {
        assert!(
            (desc.values[i] - metal_fallback[i]).abs() < 1e-6,
            "Mismatch at [{i}]: desc={:.10}, fallback={:.10}",
            desc.values[i],
            metal_fallback[i]
        );
    }
}

#[test]
fn test_profile_serialize_roundtrip() {
    let profile = QuantizerProfile::canonical_nf4();
    let json = serde_json::to_string(&profile).unwrap();
    let deser: QuantizerProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(profile, deser);
}

#[test]
fn test_profile_id_zero_uses_fallback() {
    // ProfileId(0) = canonical NF4 — Metal uses fallback (no descriptor)
    // ProfileId(1..=4) = learned — Metal reads from buffer[9]
    let canonical = PROFILE_ID_CANONICAL_NF4_V1;
    let learned = PROFILE_ID_GEMMA_ATTENTION_V1;
    assert_eq!(canonical.0, 0, "ProfileId(0) should use Metal fallback");
    assert!(learned.0 > 0, "Learned profiles need buffer[9]");
}
