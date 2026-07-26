//! Per-segment write helpers for the CImage packer.
//!
//! This module owns the canonical authority for the segment-write
//! helpers used by both [`super::pack_unified`] and
//! [`super::pack_from_dir`]. The helpers are *private* to the
//! packer: callers outside the packer do not need them.
//!
//! The pattern is the same as the engine's `AlignedMmapBuilder`:
//! 1. Pad the writer to the next 16 KB page boundary.
//! 2. Write the segment at that offset.
//! 3. Return the (offset, length) recorded in the header.
//!
//! The packer uses these helpers to satisfy the page-alignment
//! invariant that the runtime's IOSurface arena requires for
//! zero-copy `mmap`.

use std::io::{Seek, Write};

use super::APPLE_PAGE_SIZE;

/// Pad the writer to the next 16 KB page boundary and return the
/// resulting stream position.
///
/// `stream_position` is used to discover the post-pad offset so the
/// caller can record it in the header. The helper does not flush
/// between writes — the caller is responsible for that.
pub fn align_to_page<W: Write + Seek>(writer: &mut W) -> std::io::Result<u64> {
    let pos = writer.stream_position()?;
    let page = APPLE_PAGE_SIZE as u64;
    let aligned = (pos + page - 1) & !(page - 1);
    if aligned > pos {
        let padding = vec![0u8; (aligned - pos) as usize];
        writer.write_all(&padding)?;
    }
    writer.stream_position()
}

/// Write a single segment at the next 16 KB-aligned offset and return
/// the recorded (offset, length).
pub fn write_segment_aligned<W: Write + Seek>(
    writer: &mut W,
    data: &[u8],
) -> std::io::Result<(u64, u64)> {
    let offset = align_to_page(writer)?;
    writer.write_all(data)?;
    Ok((offset, data.len() as u64))
}
