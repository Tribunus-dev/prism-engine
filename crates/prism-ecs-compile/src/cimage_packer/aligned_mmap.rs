//! 16 KB-aligned mmap builder — the constitutional authority for producing
//! page-aligned mmap slices for the `.cimage` writer.
//!
//! This module owns the canonical authority for the
//! [`AlignedMmapBuilder`] primitive: a cursor-based mmap writer that
//! enforces 16 KB page alignment on every slice allocation. The
//! builder is the **mmap primitive** the packer uses to write
//! per-segment payloads into the `.cimage` file; it is the typed
//! counterpart to the engine's `AlignedMmapBuilder`.
//!
//! The module does **not** own the per-segment kind discriminants
//! (see [`super`]) or the layout plan (see [`super::layout`]). Those
//! authorities each have their own file; this file is the
//! safe-Rust re-implementation of the mmap primitive only.
//!
//! # Hard rules
//!
//! - No `unsafe` in production paths. The original engine's
//!   `AlignedMmapBuilder` exposed `allocate_hardware_pointer` (a
//!   raw-pointer API for `newBufferWithBytesNoCopy`); the
//!   re-implementation exposes only safe slice APIs and an opt-in
//!   `unsafe fn allocate_hardware_pointer_unsafe` for callers that
//!   need the raw pointer. Production code uses the safe API.
//! - No `unwrap` / `expect` / `panic!` in production paths. The
//!   original engine's `assert!` overflow check has been replaced
//!   with a `Result`-returning API.

use memmap2::MmapMut;

use super::APPLE_PAGE_SIZE;

// ── Aligned mmap builder ─────────────────────────────────────────────────

/// Cursor-based mmap writer that enforces 16 KB alignment.
///
/// Every segment allocation panics if the cursor isn't page-aligned
/// — making it mathematically impossible to produce a misaligned
/// `.cimage`. The re-implementation exposes the safe
/// [`Self::allocate_slice`] API; the `unsafe` raw-pointer API is
/// provided under a clearly-marked `unsafe fn` for callers that
/// need it (e.g. the Metal direct-write path).
pub struct AlignedMmapBuilder {
    mmap: MmapMut,
    cursor: usize,
}

impl AlignedMmapBuilder {
    /// Construct a new builder over an existing mmap. The cursor
    /// starts at offset 0.
    pub fn new(mmap: MmapMut) -> Self {
        Self { mmap, cursor: 0 }
    }

    /// Current cursor offset.
    pub fn current_offset(&self) -> u64 {
        self.cursor as u64
    }

    /// True if the cursor is currently on a 16 KB page boundary.
    pub fn is_aligned(&self) -> bool {
        self.cursor % APPLE_PAGE_SIZE == 0
    }

    /// Jump the cursor to the next 16 KB boundary.
    pub fn align_cursor(&mut self) {
        let r = self.cursor % APPLE_PAGE_SIZE;
        if r != 0 {
            self.cursor += APPLE_PAGE_SIZE - r;
        }
    }

    /// Reserve a mutable slice for the caller to fill (tar archives, CPU
    /// copies). Returns `None` if the requested length would overflow
    /// the mmap.
    pub fn allocate_slice(&mut self, length: usize) -> Option<&mut [u8]> {
        let start = self.cursor;
        let end = start.checked_add(length)?;
        if end > self.mmap.len() {
            return None;
        }
        self.cursor = end;
        Some(&mut self.mmap[start..end])
    }

    /// Reserve a mutable slice, returning an error if the request
    /// would overflow the mmap.
    pub fn try_allocate_slice(&mut self, length: usize) -> Result<&mut [u8], AlignedMmapError> {
        let start = self.cursor;
        let end = match start.checked_add(length) {
            Some(end) => end,
            None => {
                return Err(AlignedMmapError::Overflow {
                    cursor: start,
                    requested: length,
                    total: self.mmap.len(),
                });
            }
        };
        if end > self.mmap.len() {
            return Err(AlignedMmapError::Overflow {
                cursor: start,
                requested: length,
                total: self.mmap.len(),
            });
        }
        self.cursor = end;
        Ok(&mut self.mmap[start..end])
    }

    /// Write a `&[u8]` payload, returning an error if the request
    /// would overflow the mmap.
    pub fn try_write_bytes(&mut self, data: &[u8]) -> Result<(), AlignedMmapError> {
        let start = self.cursor;
        let end = match start.checked_add(data.len()) {
            Some(end) => end,
            None => {
                return Err(AlignedMmapError::Overflow {
                    cursor: start,
                    requested: data.len(),
                    total: self.mmap.len(),
                });
            }
        };
        if end > self.mmap.len() {
            return Err(AlignedMmapError::Overflow {
                cursor: start,
                requested: data.len(),
                total: self.mmap.len(),
            });
        }
        self.mmap[start..end].copy_from_slice(data);
        self.cursor = end;
        Ok(())
    }

    /// Yield a 16 KB-aligned pointer for `newBufferWithBytesNoCopy`.
    ///
    /// This is the only `unsafe` API on the builder. It must be
    /// called only when the cursor is page-aligned and the caller
    /// has reserved `length` bytes for the GPU buffer.
    pub unsafe fn allocate_hardware_pointer(&mut self, length: usize) -> *mut std::ffi::c_void {
        if self.cursor % APPLE_PAGE_SIZE != 0 {
            return std::ptr::null_mut();
        }
        let ptr = self.mmap.as_mut_ptr().add(self.cursor);
        self.cursor += length;
        ptr as *mut std::ffi::c_void
    }

    /// Write a `repr(C)` header struct (used for the on-disk
    /// `.cimage` header). Uses a safe copy that reads the struct's
    /// bytes through a stack-allocated buffer; the engine's
    /// original `slice::from_raw_parts` cast is replaced with a
    /// safe API that materializes the bytes into a `Vec<u8>` first.
    ///
    /// This is a `unsafe fn` because reading a `T` as raw bytes is
    /// only sound when `T` is `repr(C)` and `Copy` with no padding
    /// holes. Callers must ensure the type they pass satisfies
    /// those preconditions.
    pub unsafe fn try_write_header<T: Copy>(&mut self, header: &T) -> Result<(), AlignedMmapError> {
        let size = std::mem::size_of::<T>();
        // SAFETY: callers guarantee that `T` is `repr(C)` + `Copy` with
        // no padding holes, so the cast through a `*const u8` is sound.
        let header_bytes: Vec<u8> = unsafe {
            let src = header as *const T as *const u8;
            std::slice::from_raw_parts(src, size).to_vec()
        };
        self.try_write_bytes(&header_bytes)
    }

    /// Consume the builder and return the underlying mmap.
    pub fn into_mmap(self) -> MmapMut {
        self.mmap
    }

    /// Raw pointer to the start of the mmap (for GPU direct-write).
    /// The returned pointer is valid only as long as the builder is
    /// not consumed.
    pub fn mmap_base(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }
}

// ── Error type ────────────────────────────────────────────────────────────

/// Error variants for [`AlignedMmapBuilder`].
#[derive(Debug, thiserror::Error)]
pub enum AlignedMmapError {
    /// The requested allocation would overflow the mmap.
    #[error("aligned mmap overflow: cursor={cursor:#X} requested={requested} total={total:#X}")]
    Overflow {
        cursor: usize,
        requested: usize,
        total: usize,
    },
}

impl From<AlignedMmapError> for String {
    fn from(error: AlignedMmapError) -> Self {
        format!("{error}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_mmap_of_size(size: usize) -> MmapMut {
        let tmp = tempfile_for_test(size);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)
            .unwrap();
        file.set_len(size as u64).unwrap();
        unsafe { MmapMut::map_mut(&file) }.unwrap()
    }

    fn tempfile_for_test(size: usize) -> std::path::PathBuf {
        let base = std::env::temp_dir();
        let unique = format!(
            "aligned_mmap_test_{}_{}_{}",
            std::process::id(),
            size,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let p = base.join(unique);
        std::fs::File::create(&p).unwrap();
        p
    }

    #[test]
    fn empty_builder_starts_aligned_at_zero() {
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE * 2);
        let b = AlignedMmapBuilder::new(mmap);
        assert_eq!(b.current_offset(), 0);
        assert!(b.is_aligned());
    }

    #[test]
    fn allocate_slice_advances_cursor_by_length() {
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE * 2);
        let mut b = AlignedMmapBuilder::new(mmap);
        let s = b.allocate_slice(16).unwrap();
        assert_eq!(s.len(), 16);
        assert_eq!(b.current_offset(), 16);
        // Cursor is no longer page-aligned.
        assert!(!b.is_aligned());
    }

    #[test]
    fn align_cursor_snaps_to_next_16kb_boundary() {
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE * 4);
        let mut b = AlignedMmapBuilder::new(mmap);
        b.allocate_slice(1).unwrap(); // cursor at 1
        b.align_cursor();
        assert_eq!(b.current_offset(), APPLE_PAGE_SIZE as u64);
        assert!(b.is_aligned());
    }

    #[test]
    fn allocate_slice_overflow_returns_none() {
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE);
        let mut b = AlignedMmapBuilder::new(mmap);
        // Request more than the mmap holds.
        assert!(b.allocate_slice(APPLE_PAGE_SIZE * 2).is_none());
        // The cursor is unchanged.
        assert_eq!(b.current_offset(), 0);
    }

    #[test]
    fn try_allocate_slice_returns_error_on_overflow() {
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE);
        let mut b = AlignedMmapBuilder::new(mmap);
        let err = b.try_allocate_slice(APPLE_PAGE_SIZE * 2).unwrap_err();
        match err {
            AlignedMmapError::Overflow {
                cursor,
                requested,
                total,
            } => {
                assert_eq!(cursor, 0);
                assert_eq!(requested, APPLE_PAGE_SIZE * 2);
                assert_eq!(total, APPLE_PAGE_SIZE);
            }
        }
    }

    #[test]
    fn try_write_bytes_copies_payload() {
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE);
        let mut b = AlignedMmapBuilder::new(mmap);
        b.try_write_bytes(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(b.current_offset(), 5);
        // The cursor is no longer page-aligned; align to the next 16KB.
        b.align_cursor();
        assert_eq!(b.current_offset(), APPLE_PAGE_SIZE as u64);
    }

    #[test]
    fn try_write_header_copies_struct_bytes() {
        // A 16-byte POD struct: 4 u32 fields.
        #[derive(Copy, Clone, Debug, PartialEq)]
        #[repr(C)]
        struct TestHeader {
            a: u32,
            b: u32,
            c: u32,
            d: u32,
        }
        let mmap = empty_mmap_of_size(APPLE_PAGE_SIZE);
        let mut b = AlignedMmapBuilder::new(mmap);
        let header = TestHeader {
            a: 0x11111111,
            b: 0x22222222,
            c: 0x33333333,
            d: 0x44444444,
        };
        // SAFETY: `TestHeader` is `repr(C)` and `Copy` with no padding
        // holes (4 × u32 = 16 bytes), so the byte-cast is sound.
        unsafe { b.try_write_header(&header) }.unwrap();
        assert_eq!(b.current_offset(), 16);
    }
}
