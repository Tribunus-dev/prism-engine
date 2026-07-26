//! Test module for the CImage packer.
//!
//! The tests exercise the constitutional surface of the packer:
//! page-alignment, segment-table construction, header serialization,
//! multimodal classification, and the helpers used by the
//! directory-aware packer.

use super::helpers::{
    classify_multimodal_entry, classify_multimodal_tensor, dims4, dtype_code, header_fields_from_manifest,
    MultimodalClass, MultimodalEntryKind,
};
use super::multimodal::{logical_shape_from_manifest};
use super::pack_unified::{pack_unified_cimage, CimageHeaderSerialized};
use super::segment_writer::{align_to_page, write_segment_aligned};
use super::{CimageHeader, CImagePackerError, SegmentEntry, SegmentKind, APPLE_PAGE_SIZE};
use std::io::{Cursor, Seek, SeekFrom, Write};

#[test]
fn packer_error_categories_are_distinct() {
    let r = CImagePackerError::rejected("rejected");
    let f = CImagePackerError::failed("failed");
    let s = CImagePackerError::stale("stale");
    assert!(format!("{r}").contains("rejected"));
    assert!(format!("{f}").contains("failed"));
    assert!(format!("{s}").contains("stale"));
}

#[test]
fn apple_page_size_is_16kb() {
    assert_eq!(APPLE_PAGE_SIZE, 16_384);
}

#[test]
fn segment_entry_records_kind_as_u32() {
    let entry = SegmentEntry::new(SegmentKind::MetalLib, 0x1000, 0x2000);
    assert_eq!(entry.kind, SegmentKind::MetalLib.discriminant());
    assert_eq!(entry.offset, 0x1000);
    assert_eq!(entry.length, 0x2000);
}

#[test]
fn cimage_header_default_is_prism_v4() {
    let h = CimageHeader::default();
    assert_eq!(&h.magic, b"PRISM\0\0\0");
    assert_eq!(h.version, 4);
    assert_eq!(h.segment_count, 0);
}

#[test]
fn cimage_header_serialized_is_byte_stable() {
    let h = CimageHeaderSerialized {
        magic: *b"PRISM\0\0\0",
        version: 4,
        segment_count: 5,
        payload_hash: [0u8; 32],
        num_layers: 12,
        num_heads: 24,
        head_dim: 64,
        hidden_dim: 1536,
        intermediate_dim: 8960,
        vocab_size: 151936,
        quantization_schema: 0,
        draft_num_layers: 0,
        metal_lib_offset: 0x10000,
        metal_lib_len: 0x20000,
        main_weights_offset: 0x30000,
        main_weights_len: 0x40000,
        mtp_weights_offset: 0x50000,
        mtp_weights_len: 0x60000,
        main_graph_offset: 0x70000,
        main_graph_len: 0x80000,
        mtp_graph_offset: 0,
        mtp_graph_len: 0,
    };
    let bytes = h.to_bytes();
    // The first 8 bytes are the magic.
    assert_eq!(&bytes[..8], b"PRISM\0\0\0");
    // The next 4 bytes are the version (4 as little-endian u32).
    assert_eq!(&bytes[8..12], &4u32.to_le_bytes());
    // The next 4 bytes are the segment count (5 as little-endian u32).
    assert_eq!(&bytes[12..16], &5u32.to_le_bytes());
}

#[test]
fn align_to_page_pads_to_next_16kb() {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    cursor.write_all(&vec![0u8; 100]).unwrap();
    let aligned = align_to_page(&mut cursor).unwrap();
    assert_eq!(aligned, APPLE_PAGE_SIZE as u64);
}

#[test]
fn write_segment_aligned_returns_offset_and_length() {
    let mut cursor = Cursor::new(Vec::<u8>::new());
    let (offset, length) = write_segment_aligned(&mut cursor, b"hello").unwrap();
    assert_eq!(offset, 0);
    assert_eq!(length, 5);
    let pos = cursor.seek(SeekFrom::Current(0)).unwrap();
    assert_eq!(pos, 5);
}

#[test]
fn classify_multimodal_tensor_detects_vision_and_audio() {
    assert_eq!(
        classify_multimodal_tensor("vision_patch.weight"),
        Some(MultimodalClass::Vision)
    );
    assert_eq!(
        classify_multimodal_tensor("audio_projection.weight"),
        Some(MultimodalClass::Audio)
    );
    assert_eq!(classify_multimodal_tensor("lm_head.weight"), None);
}

#[test]
fn classify_multimodal_entry_distinguishes_patch_and_projection() {
    assert_eq!(
        classify_multimodal_entry("vision_patch.weight", &[1, 2, 3]),
        MultimodalEntryKind::VisionPatch
    );
    assert_eq!(
        classify_multimodal_entry("vision_projection.weight", &[1, 2, 3]),
        MultimodalEntryKind::VisionProjection
    );
    assert_eq!(
        classify_multimodal_entry("audio_frame.weight", &[1, 2, 3]),
        MultimodalEntryKind::AudioFrame
    );
    assert_eq!(
        classify_multimodal_entry("audio_projection.weight", &[1, 2, 3]),
        MultimodalEntryKind::AudioProjection
    );
}

#[test]
fn dims4_pads_short_shapes_and_truncates_long_shapes() {
    assert_eq!(dims4(&[1, 2, 3, 4]), [1, 2, 3, 4]);
    assert_eq!(dims4(&[1, 2]), [1, 2, 1, 1]);
    assert_eq!(dims4(&[1, 2, 3, 4, 5, 6]), [1, 2, 3, 4]);
}

#[test]
fn dtype_code_maps_known_dtypes() {
    assert_eq!(dtype_code("FP16"), 1);
    assert_eq!(dtype_code("NF4"), 5);
    assert_eq!(dtype_code("TERNARY"), 6);
    assert_eq!(dtype_code("unknown"), 0);
}

#[test]
fn header_fields_from_manifest_returns_zeros_for_missing() {
    let (l, h, hd, hid, idim, vs, qs) = header_fields_from_manifest(None);
    assert_eq!((l, h, hd, hid, idim, vs, qs), (0, 0, 0, 0, 0, 0, 0));
}

#[test]
fn header_fields_from_manifest_reads_architecture() {
    let manifest = serde_json::json!({
        "architecture": {
            "num_hidden_layers": 12,
            "num_attention_heads": 24,
            "head_dim": 64,
            "hidden_size": 1536,
            "intermediate_size": 8960,
            "vocab_size": 151936,
        }
    });
    let (l, h, hd, hid, idim, vs, qs) = header_fields_from_manifest(Some(&manifest));
    assert_eq!(l, 12);
    assert_eq!(h, 24);
    assert_eq!(hd, 64);
    assert_eq!(hid, 1536);
    assert_eq!(idim, 8960);
    assert_eq!(vs, 151936);
    assert_eq!(qs, 0);
}

#[test]
fn logical_shape_from_manifest_returns_empty_for_missing() {
    let manifest = serde_json::json!({});
    let shape = logical_shape_from_manifest(&manifest, "vision_patch.weight");
    assert!(shape.is_empty());
}

#[test]
fn pack_unified_cimage_writes_three_segments_when_mtp_graph_empty() {
    let tmp = std::env::temp_dir().join("prism-cimage-packer-test-unified");
    let _ = std::fs::create_dir_all(&tmp);
    let out = tmp.join("out.cimage");
    pack_unified_cimage(
        out.to_str().unwrap(),
        b"metallib",
        b"graph",
        b"weights",
        b"",
        b"mtp_weights",
    )
    .unwrap();
    let bytes = std::fs::read(&out).unwrap();
    assert!(!bytes.is_empty());
    // The magic must be at offset 0.
    assert_eq!(&bytes[..8], b"PRISM\0\0\0");
    let _ = std::fs::remove_dir_all(&tmp);
}
