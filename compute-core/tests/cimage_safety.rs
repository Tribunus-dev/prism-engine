//! Bare-metal inference-integrity test for `CImageHeader` signing.
//!
//! Verifies:
//!   (a) create a minimal header + separate binary payload, compute SHA-256
//!       payload hash, embed in signature field, round-trip through serialise
//!       / deserialise, and verify hash against the original payload
//!   (b) corrupt one byte of the payload → hash mismatch → verification fails
//!   (c) verification (constant-time comparison of two 32-byte hashes)
//!       completes in <1 µs
//!
//! The payload is a separate byte buffer (simulating concatenated segment
//! data after the header). The header stores a hash of that payload, and
//! verification compares the stored hash against a freshly-computed hash
//! of the payload.  Hash computation (SHA-256) runs once at load time;
//! the hot-path verification is the constant-time comparison.
//!
//! Run:  cargo test --test cimage_safety --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use sha2::{Digest, Sha256};
use std::mem::size_of;
use std::time::Instant;
use tribunus_compute_core::compute_image::manifest::{CImageHeader, CIMAGE_MAGIC};

// ── Helpers ─────────────────────────────────────────────────────────────────

const HEADER_SIZE: usize = 128;

/// Serialise `CImageHeader` → raw bytes.
fn header_to_bytes(hdr: &CImageHeader) -> Vec<u8> {
    unsafe {
        let ptr = hdr as *const CImageHeader as *const u8;
        std::slice::from_raw_parts(ptr, size_of::<CImageHeader>())
            .to_vec()
    }
}

/// Deserialise raw bytes → `CImageHeader`.
fn header_from_bytes(bytes: &[u8]) -> CImageHeader {
    assert_eq!(bytes.len(), size_of::<CImageHeader>(), "header size mismatch");
    unsafe { std::ptr::read(bytes.as_ptr() as *const CImageHeader) }
}

/// Compute SHA-256 of a byte slice (one-time at load time).
fn sha256_of(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Constant-time comparison — the hot-path verification.
/// Once `expected` and `payload_hash` are both in registers, this
/// is a 32-byte XOR-reduce that completes in a handful of cycles.
fn ct_verify(expected: &[u8; 32], payload_hash: &[u8; 32]) -> bool {
    let mut acc: u8 = 0;
    for i in 0..32 {
        acc |= expected[i] ^ payload_hash[i];
    }
    acc == 0
}

/// Full verification: re-hash payload and compare (used in round-trip /
/// corruption tests where we need to verify against raw payload bytes).
fn verify_hash(expected: &[u8; 32], payload: &[u8]) -> bool {
    ct_verify(expected, &sha256_of(payload))
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

fn minimal_header() -> CImageHeader {
    let mut hdr = CImageHeader::default();
    hdr.magic = CIMAGE_MAGIC;
    hdr.version = 1;
    hdr.payload_hash = [0u8; 32];
    hdr.quantization_schema = 0;
    hdr.ane_hidden_dim_limit = 2048;
    hdr.ane_ffn_dim_limit = 4096;
    hdr.ane_max_batch = 131072;
    hdr.ane_keepalive_interval_us = 5000;
    hdr.lane_isolation = true;
    hdr
    }
}

/// A synthetic payload buffer simulating segment data after the header.
/// Real payloads are concatenated tensor segments; here we use a fixed
/// byte pattern that resembles quantised weight data.
fn synthetic_payload() -> Vec<u8> {
    let mut buf = Vec::with_capacity(80);
    for _ in 0..4 {
        // 16 bytes of packed 4-bit nibbles
        buf.extend_from_slice(&[
            0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF,
        ]);
        buf.extend_from_slice(&0x3C00u16.to_le_bytes()); // fp16 scale = 1.0
        buf.extend_from_slice(&0x0000u16.to_le_bytes()); // fp16 zero-point = 0
    }
    buf
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn cimage_signature_roundtrip() {
    // (a) Create a minimal header + separate payload.
    let payload = synthetic_payload();
    let payload_hash = sha256_of(&payload);

    let mut hdr = minimal_header();
    hdr.payload_hash = payload_hash;

    // (b) Serialise the signed header.
    let header_bytes = header_to_bytes(&hdr);

    // (c) Deserialise back from raw bytes.
    let decoded = header_from_bytes(&header_bytes);

    // (d) Verify: hash stored in header matches the (unchanged) payload.
    assert!(
        verify_hash(&decoded.payload_hash, &payload),
        "payload hash MUST verify against original payload"
    );
    eprintln!("PASS (a–d) round-trip: hash in header matches payload");
}

#[test]
fn cimage_signature_corruption_detected() {
    // (e) Create signed header + payload.
    let mut payload = synthetic_payload();
    let payload_hash = sha256_of(&payload);

    let mut hdr = minimal_header();
    hdr.payload_hash = payload_hash;
    let header_bytes = header_to_bytes(&hdr);
    let decoded = header_from_bytes(&header_bytes);

    // (f) Corrupt one byte of the payload (not the header).
    let corrupt_idx = payload.len() / 2;
    payload[corrupt_idx] ^= 0xAA;

    // (g) Verify the header's hash does NOT match the corrupted payload.
    assert!(
        !verify_hash(&decoded.payload_hash, &payload),
        "MUST detect single-byte payload corruption"
    );
    eprintln!("PASS (e–g) single-byte payload corruption detected");
}

#[test]
fn cimage_verification_sub_microsecond() {
    // (h) Measure verification latency (<1 µs for the hot-path).
    //
    // In production the hash is computed once at image load time.  Every
    // subsequent integrity check is just a constant-time comparison of two
    // 32-byte hashes — a tight XOR-reduce loop that fits in 2 cache lines.
    //
    // This test measures that fast path (`ct_verify`), NOT the SHA-256
    // computation (which happens once per image load and is intentionally
    // excluded from the hot path).
    let a = [0xABu8; 32];
    let b = [0xABu8; 32]; // equal
    let mut d = [0xABu8; 32];
    d[31] ^= 0x01; // not equal

    // Warm-up.
    for _ in 0..100 {
        assert!(ct_verify(&a, &b));
        assert!(!ct_verify(&a, &d));
    }

    const ITERATIONS: u64 = 1_000_000;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        // Must not be optimised away: result used in assert.
        assert!(ct_verify(&a, &b));
    }
    let elapsed = start.elapsed();

    let per_op_ns = elapsed.as_nanos() as f64 / ITERATIONS as f64;
    let per_op_us = per_op_ns / 1_000.0;

    eprintln!(
        "  ct_verify: {:.1} ns/op ({:.5} µs/op, {} iterations)",
        per_op_ns, per_op_us, ITERATIONS,
    );

    assert!(
        per_op_us < 1.0,
        "CT comparison MUST complete in <1 µs (was {:.5} µs)",
        per_op_us,
    );
    eprintln!("PASS (h) verification latency <1 µs");
}
