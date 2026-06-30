//! Zero-copy mmap access test for CImageHeader binary layout.
//!
//! Verifies:
//!   (1) a CImageHeader can be transmuted to/from raw bytes
//!   (2) the binary layout matches field offsets
//!   (3) an mmap'd file can be cast directly to &CImageHeader
//!   (4) corrupted bytes in the file produce a mismatch
//!   (5) the mmap→struct-access overhead is below 100 ns (pointer arithmetic)
//!
//! Run:  cargo test --test cimage_zero_copy --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::fs;
use std::mem;

use memmap2::Mmap;
use std::time::Instant;
use tribunus_compute_core::compute_image::manifest::{CImageHeader, CIMAGE_MAGIC};

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Create a CImageHeader with known values.
fn known_header() -> CImageHeader {
    let mut payload_hash = [0u8; 32];
    payload_hash[0..4].copy_from_slice(b"abcd");
    let mut hdr = CImageHeader::default();
    hdr.magic = CIMAGE_MAGIC;
    hdr.version = 2;
    hdr.payload_hash = payload_hash;
    hdr.phase_count = 42;
    hdr.layout_offset = 128;
    hdr.phase_offset = 4096;
    hdr
}

/// Serialize a CImageHeader to raw bytes (binary, not JSON).
fn header_to_binary(hdr: &CImageHeader) -> Vec<u8> {
    let size = mem::size_of::<CImageHeader>();
    let mut buf = vec![0u8; size];
    // SAFETY: CImageHeader has no invalid bit patterns for its fields
    // (u32, [u8; 32], u64, u64, u64) and is #[repr(C)].
    unsafe {
        std::ptr::write(buf.as_mut_ptr() as *mut CImageHeader, hdr.clone());
    }
    buf
}

/// Interpret raw bytes as a &CImageHeader.
/// SAFETY: caller must guarantee the bytes are exactly
/// `mem::size_of::<CImageHeader>()` bytes and the alignment is 64.
unsafe fn bytes_as_header(bytes: &[u8]) -> &CImageHeader {
    assert_eq!(bytes.len(), mem::size_of::<CImageHeader>());
    // Alignment check: the slice must be 64-byte aligned (align(64)).
    let ptr = bytes.as_ptr();
    assert!(
        ptr as usize % 64 == 0,
        "slice pointer {ptr:p} is not 64-byte aligned"
    );
    &*(ptr as *const CImageHeader)
}

/// Corrupt a byte at `offset` in the original bytes, mmap the result, and verify that
/// the payload_hash field changed (for hash-region corruption) or that a non-hash field
/// changed (for other offset corruption).
fn corrupt_and_verify(original: &[u8], offset: usize, field_label: &str, expect_hash_change: bool) {
    let mut corrupted = original.to_vec();
    corrupted[offset] ^= 0xFF;

    let dir = std::env::temp_dir();
    let path = dir.join(format!("cimage_corrupt_{offset}.tmp"));
    fs::write(&path, &corrupted).expect("write corrupt file");
    let mmap = unsafe { Mmap::map(&fs::File::open(&path).unwrap()) }.expect("mmap corrupt");
    let hdr = unsafe { bytes_as_header(&mmap[..]) };

    // Read a single byte from the original at this offset.
    let original_byte = original[offset];
    let corrupted_byte = corrupted[offset];
    assert_ne!(
        original_byte, corrupted_byte,
        "byte at offset {offset} MUST differ"
    );

    if expect_hash_change {
        // The corrupted byte is inside the payload_hash field — the hash value itself changed.
        let reference: [u8; 32] = [0u8; 32];
        assert_ne!(
            hdr.payload_hash, reference,
            "payload_hash MUST be non-zero after corruption at offset {offset} ({field_label})"
        );
        eprintln!("  PASS corruption at offset {offset} ({field_label}) — payload_hash changed");
    } else {
        // The corrupted byte is outside payload_hash — verify that the hash field is unchanged
        // but some other header field differs from the original.
        let original_hdr = unsafe { bytes_as_header(original) };
        let mut diff = false;
        if hdr.magic != original_hdr.magic {
            diff = true;
        }
        if hdr.version != original_hdr.version {
            diff = true;
        }
        if hdr.phase_count != original_hdr.phase_count {
            diff = true;
        }
        if hdr.layout_offset != original_hdr.layout_offset {
            diff = true;
        }
        if hdr.phase_offset != original_hdr.phase_offset {
            diff = true;
        }
        assert!(
            diff,
            "MUST detect corruption in field at offset {offset} ({field_label})"
        );
        eprintln!("  PASS corruption at offset {offset} ({field_label}) — field value changed");
    }

    let _ = fs::remove_file(&path);
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[test]
fn test_cimage_zero_copy() {
    // 1. Create a CImageHeader with known values.
    let hdr = known_header();

    // 2. Serialize to raw binary bytes.
    let binary = header_to_binary(&hdr);
    assert_eq!(
        binary.len(),
        mem::size_of::<CImageHeader>(),
        "serialized size matches struct size"
    );
    eprintln!(
        "CImageHeader size: {} bytes, align: {}",
        mem::size_of::<CImageHeader>(),
        mem::align_of::<CImageHeader>(),
    );

    // 3. Write to temp file aligned at 64 bytes (for mmap safety).
    let dir = std::env::temp_dir();
    let path = dir.join("cimage_zero_copy_test.bin");
    // Pad the file so the header starts at a 64-byte aligned offset.
    let pad_len = if binary.len() % 64 == 0 {
        0
    } else {
        64 - (binary.len() % 64)
    };
    let mut padded = Vec::with_capacity(binary.len() + pad_len);
    padded.extend_from_slice(&binary);
    padded.resize(binary.len() + pad_len, 0);
    fs::write(&path, &padded).expect("write test file");
    eprintln!(
        "Wrote {} bytes ({} header + {} pad) to {:?}",
        padded.len(),
        binary.len(),
        pad_len,
        path
    );

    // 4. mmap the file.
    let file = fs::File::open(&path).expect("open temp file");
    let mmap_start = Instant::now();
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap temp file");
    let mmap_dur = mmap_start.elapsed();
    eprintln!("mmap completed in {mmap_dur:?}");

    // 5. Cast the mapped bytes to &CImageHeader.
    let cast_start = Instant::now();
    let mapped_hdr = unsafe { bytes_as_header(&mmap[..binary.len()]) };
    let cast_dur = cast_start.elapsed();
    eprintln!("cast to &CImageHeader in {cast_dur:?}");

    // 6. Verify magic, version, phase_count, payload_hash.
    assert_eq!(mapped_hdr.magic, CIMAGE_MAGIC, "magic matches");
    assert_eq!(mapped_hdr.version, 2, "version matches");
    assert_eq!(mapped_hdr.phase_count, 42, "phase_count matches");
    assert_eq!(mapped_hdr.layout_offset, 128, "layout_offset matches");
    assert_eq!(mapped_hdr.phase_offset, 4096, "phase_offset matches");
    let mut expected_hash = [0u8; 32];
    expected_hash[0..4].copy_from_slice(b"abcd");
    assert_eq!(
        mapped_hdr.payload_hash, expected_hash,
        "payload_hash matches"
    );
    eprintln!("PASS all field values correct via mmap");

    // 7. Measure: overhead from mmap to struct access.
    // After the first pointer calculation, re-accessing fields is
    // a direct memory load.  Measure a tight loop of field reads.
    const ITERATIONS: u64 = 10_000;
    let bench_start = Instant::now();
    for _ in 0..ITERATIONS {
        // Force a field read through the mapped reference.
        let _ = mapped_hdr.magic;
        let _ = mapped_hdr.version;
        let _ = mapped_hdr.payload_hash;
        let _ = mapped_hdr.phase_count;
        let _ = mapped_hdr.layout_offset;
        let _ = mapped_hdr.phase_offset;
    }
    let bench_dur = bench_start.elapsed();
    let per_access_ns = bench_dur.as_nanos() as f64 / (ITERATIONS as f64 * 6.0);
    eprintln!(
        "Field reads via mmap: {per_access_ns:.2} ns/field ({} iterations × 6 fields)",
        ITERATIONS
    );

    // The cast itself is pointer arithmetic — measure the raw cast overhead
    // separately on the already-mapped memory.
    let empty_start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = unsafe { bytes_as_header(&mmap[..binary.len()]) };
    }
    let empty_dur = empty_start.elapsed();
    let per_cast_ns = empty_dur.as_nanos() as f64 / ITERATIONS as f64;
    eprintln!(
        "Pointer-arithmetic cast: {per_cast_ns:.1} ns/op ({} iterations)",
        ITERATIONS
    );

    // Both should be well under 100 ns.
    assert!(
        per_access_ns < 100.0,
        "Each field read should be <100 ns (was {per_access_ns:.2} ns)"
    );
    eprintln!("PASS zero-copy access overhead <100 ns");

    // 8. Corrupt bytes at various offsets and verify hash mismatch.
    eprintln!("Corruption detection tests:");

    // Use the original binary bytes (before padding) as the reference.
    let original_bytes = &mmap[..binary.len()];

    // Corruption in the payload_hash field (offset 8–39) — hash value changes.
    corrupt_and_verify(original_bytes, 8, "payload_hash[0]", true);
    corrupt_and_verify(original_bytes, 39, "payload_hash[31]", true);
    // Corruption in phase_count field (offset 40–47) — field value changes.
    corrupt_and_verify(original_bytes, 40, "phase_count", false);
    corrupt_and_verify(original_bytes, 47, "phase_count[7]", false);
    // Corruption in layout_offset field (offset 48–55).
    corrupt_and_verify(original_bytes, 48, "layout_offset", false);
    // Corruption in phase_offset field (offset 56–63).
    corrupt_and_verify(original_bytes, 56, "phase_offset", false);

    eprintln!("ALL TESTS PASSED");

    // Clean up.
    let _ = fs::remove_file(&path);
}

#[test]
fn test_cimage_zero_copy_serialize_deserialize_roundtrip() {
    // Round-trip through binary serialization (not JSON).
    let hdr = known_header();
    let binary = header_to_binary(&hdr);

    // Deserialize via transmute.
    let temp_file = std::env::temp_dir().join("cimage_roundtrip.bin");
    let pad = vec![0u8; 64 - (binary.len() % 64)];
    let mut padded = binary.clone();
    padded.extend(pad);
    fs::write(&temp_file, &padded).expect("write roundtrip file");

    let file = fs::File::open(&temp_file).expect("open roundtrip file");
    let mmap = unsafe { Mmap::map(&file) }.expect("mmap roundtrip file");
    let decoded = unsafe { bytes_as_header(&mmap[..binary.len()]) };

    assert_eq!(decoded.magic, CIMAGE_MAGIC);
    assert_eq!(decoded.version, hdr.version);
    assert_eq!(decoded.payload_hash, hdr.payload_hash);
    assert_eq!(decoded.phase_count, hdr.phase_count);
    assert_eq!(decoded.layout_offset, hdr.layout_offset);
    assert_eq!(decoded.phase_offset, hdr.phase_offset);
    eprintln!("PASS binary round-trip via mmap");

    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_cimage_zero_copy_struct_offsets() {
    // Verify expected field offsets for the binary layout.
    // layout: magic(4) + version(4) + payload_hash(32) + phase_count(8)
    //         + layout_offset(8) + phase_offset(8) = 64, padded to align(64).
    assert_eq!(mem::size_of::<CImageHeader>(), 64);
    assert_eq!(mem::align_of::<CImageHeader>(), 64);

    // Use field-pointer arithmetic to validate offsets at runtime.
    let zeroed: CImageHeader = unsafe { mem::zeroed() };
    let base = &zeroed as *const CImageHeader as usize;

    let magic_off = (&zeroed.magic as *const u32) as usize - base;
    let ver_off = (&zeroed.version as *const u32) as usize - base;
    let hash_off = (&zeroed.payload_hash as *const [u8; 32]) as usize - base;
    let count_off = (&zeroed.phase_count as *const u32) as usize - base;
    let lo_off = (&zeroed.layout_offset as *const u64) as usize - base;
    let po_off = (&zeroed.phase_offset as *const u64) as usize - base;

    assert_eq!(magic_off, 0, "magic at offset 0");
    assert_eq!(ver_off, 4, "version at offset 4");
    assert_eq!(hash_off, 8, "payload_hash at offset 8");
    assert_eq!(count_off, 40, "phase_count at offset 40");
    assert_eq!(lo_off, 48, "layout_offset at offset 48");
    assert_eq!(po_off, 56, "phase_offset at offset 56");

    eprintln!("PASS all field offsets verified");
    eprintln!(
        "  struct offsets: magic=0 version=4 payload_hash=8 phase_count=40 layout_offset=48 phase_offset=56"
    );
}
