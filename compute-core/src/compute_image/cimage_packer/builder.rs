//! Cursor-based mmap writer that enforces 16 KB alignment.
//!
//! `AlignedMmapBuilder` replaces the old `ImageBuilder` vector
//! accumulation and `DeterministicSegmentWriter`.  Every segment
//! allocation panics if the cursor isn't page-aligned, making it
//! mathematically impossible to produce a misaligned `.cimage`.

use memmap2::MmapMut;
use std::mem::size_of;

const APPLE_PAGE_SIZE: usize = super::APPLE_PAGE_SIZE as usize;

pub struct AlignedMmapBuilder {
    mmap: MmapMut,
    pub(crate) cursor: usize,
}

impl AlignedMmapBuilder {
    pub fn new(mmap: MmapMut) -> Self {
        Self { mmap, cursor: 0 }
    }

    /// Jump the cursor to the next 16 KB boundary.
    pub fn align_cursor(&mut self) {
        let r = self.cursor % APPLE_PAGE_SIZE;
        if r != 0 {
            self.cursor += APPLE_PAGE_SIZE - r;
        }
    }

    /// Reserve a mutable slice for the caller to fill (tar archives, CPU copies).
    pub fn allocate_slice(&mut self, length: usize) -> &mut [u8] {
        let start = self.cursor;
        let end = start + length;
        assert!(
            end <= self.mmap.len(),
            "AlignedMmapBuilder overflow: cursor={:#X} len={} total={:#X}",
            start, length, self.mmap.len()
        );
        self.cursor = end;
        &mut self.mmap[start..end]
    }

    /// Yield a 16 KB-aligned pointer for `newBufferWithBytesNoCopy`.
    pub unsafe fn allocate_hardware_pointer(
        &mut self, length: usize,
    ) -> *mut std::ffi::c_void {
        assert!(
            self.cursor % APPLE_PAGE_SIZE == 0,
            "Alignment fault at offset {:#X}", self.cursor
        );
        let ptr = self.mmap.as_mut_ptr().add(self.cursor);
        self.cursor += length;
        ptr as *mut std::ffi::c_void
    }

    /// Write a C-repr struct (used for the header).
    pub fn write_header<T>(&mut self, header: &T) {
        let bytes = unsafe {
            std::slice::from_raw_parts(header as *const T as *const u8, size_of::<T>())
        };
        self.allocate_slice(bytes.len()).copy_from_slice(bytes);
    }

    pub fn current_offset(&self) -> u64 {
        self.cursor as u64
    }

    pub fn into_mmap(self) -> MmapMut {
        self.mmap
    }

    /// Raw pointer to the start of the mmap (for GPU direct-write).
    pub fn mmap_base(&mut self) -> *mut u8 {
        self.mmap.as_mut_ptr()
    }
}
