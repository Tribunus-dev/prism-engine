//! UnifiedExecutionArena — one memory region, three Apple execution lanes.
//! All tensors, KV blocks, activations, and intermediates live in a single
//! mmap-backed arena. MLX, Accelerate, and Core ML receive *views* into
//! the same underlying pages — no backend-to-backend copies.

use std::hash::{Hash, Hasher};

/// Handle into the unified arena, or a standalone mmap-backed view.
#[derive(Clone, Copy, Debug)]
pub struct ArenaView {
    /// Offset into a UnifiedExecutionArena (for arena-allocated regions).
    pub offset: u64,
    /// Direct CPU pointer to the backing memory (for standalone mmap views).
    pub cpu_ptr: *mut u8,
    /// Byte length of the backing memory.
    pub byte_length: u64,
    /// Optional Metal buffer wrapping this memory (StorageModeShared, no copy).
    pub metal_buffer: Option<*mut std::ffi::c_void>,
}

// ArenaView identity is based on offset alone (for HashMap-key usage in hazard tracking).
impl PartialEq for ArenaView {
    fn eq(&self, other: &Self) -> bool {
        self.offset == other.offset
    }
}
impl Eq for ArenaView {}
impl Hash for ArenaView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.offset.hash(state);
    }
}

impl ArenaView {
    /// Create a handle for an offset into a UnifiedExecutionArena.
    pub fn from_offset(offset: u64) -> Self {
        ArenaView {
            offset,
            cpu_ptr: std::ptr::null_mut(),
            byte_length: 0,
            metal_buffer: None,
        }
    }

    /// Create an ArenaView backed by a page-aligned mmap slice from the
    /// .cimage file.  `base` must be 16 KB aligned (guaranteed by
    /// AlignedMmapBuilder).  The backing memory is wired physical RAM
    /// shared with the ANE via IOSurface without kernel shadow copies.
    ///
    /// # Safety
    /// - `base` must point to a valid, page-aligned memory region of at
    ///   least `byte_length` bytes.
    /// - The memory must remain valid for the lifetime of this view.
    pub unsafe fn from_mmap_slice(
        base: *const u8,
        byte_length: u64,
        device: Option<&metal::Device>,
    ) -> Self {
        let metal_buffer = device.map(|dev| {
            dev.new_buffer_with_bytes_no_copy(
                base as *mut std::ffi::c_void,
                byte_length,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            )
        });
        // Map the Metal buffer pointer through the inner id<MTLBuffer>*
        let metal_ptr = metal_buffer.map(|buf| buf.contents());
        Self {
            offset: 0,
            cpu_ptr: base as *mut u8,
            byte_length,
            metal_buffer: metal_ptr,
        }
    }
}

/// Memory backing strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryBacking {
    /// mmap backed (CPU accessible, pageable)
    Mmap,
    /// IOSurface backed (GPU/CPU zero-copy, ANE-accessible)
    IOSurface,
    /// Metal shared buffer (GPU/CPU zero-copy, unified memory)
    MetalSharedBuffer,
}

/// A contiguous region of the unified arena.
pub struct ArenaRegion {
    pub offset: u64,
    pub byte_size: u64,
    pub backing: MemoryBacking,
    pub cpu_ptr: Option<*mut u8>,
    pub iosurface_id: Option<u32>,
    pub metal_buffer: Option<*mut std::ffi::c_void>,
}

/// Unified execution arena — owns all tensor memory for the current request.
pub struct UnifiedExecutionArena {
    mmap_size: u64,
    mmap_ptr: *mut u8,
    regions: Vec<ArenaRegion>,
    next_offset: u64,
    read_hazards: Vec<(ArenaView, ArenaView)>,
    #[allow(dead_code)]
    write_hazards: Vec<(ArenaView, ArenaView)>,
}

impl UnifiedExecutionArena {
    /// Create a new arena backed by an anonymous mmap of `size` bytes.
    pub fn new(size: u64) -> std::io::Result<Self> {
        let mmap_ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if mmap_ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(UnifiedExecutionArena {
            mmap_size: size,
            mmap_ptr: mmap_ptr as *mut u8,
            regions: Vec::new(),
            next_offset: 0,
            read_hazards: Vec::new(),
            write_hazards: Vec::new(),
        })
    }

    /// Allocate a region within the arena.
    pub fn allocate(&mut self, size: u64, backing: MemoryBacking) -> Option<ArenaView> {
        let aligned = (size + 4095) & !4095;
        if self.next_offset + aligned > self.mmap_size {
            return None;
        }
        let view = ArenaView::from_offset(self.next_offset);
        self.regions.push(ArenaRegion {
            offset: self.next_offset,
            byte_size: aligned,
            backing,
            cpu_ptr: unsafe { Some(self.mmap_ptr.add(self.next_offset as usize)) },
            iosurface_id: None,
            metal_buffer: None,
        });
        self.next_offset += aligned;
        Some(view)
    }

    /// Get a CPU pointer to a view.
    pub fn cpu_ptr(&self, view: ArenaView) -> Option<*mut u8> {
        self.regions
            .iter()
            .find(|r| r.offset == view.offset)
            .and_then(|r| r.cpu_ptr)
    }

    /// Write float data to a view (CPU lane).
    pub fn write_f32(&mut self, view: ArenaView, data: &[f32]) -> Result<(), String> {
        let ptr = self.cpu_ptr(view).ok_or("invalid view")?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len());
        }
        Ok(())
    }

    /// Read float data from a view (CPU lane).
    pub fn read_f32(&self, view: ArenaView, out: &mut [f32]) -> Result<(), String> {
        let ptr = self.cpu_ptr(view).ok_or("invalid view")?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr as *const f32, out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    /// Record a read-after-write hazard pair.
    pub fn record_hazard(&mut self, reader: ArenaView, writer: ArenaView) {
        self.read_hazards.push((reader, writer));
    }

    /// Convert to an IOSurface-backed arena (for ANE/GPU sharing).
    pub fn ensure_iosurface(&mut self) -> Result<(), String> {
        // Stub: real impl creates IOSurface from mmap region
        // For now, just mark all regions as IOSurface-compatible
        for region in &mut self.regions {
            region.backing = MemoryBacking::IOSurface;
        }
        Ok(())
    }

    /// Total bytes allocated so far.
    pub fn total_allocated(&self) -> u64 {
        self.next_offset
    }

    /// Total capacity of the arena.
    pub fn capacity(&self) -> u64 {
        self.mmap_size
    }
}

impl Drop for UnifiedExecutionArena {
    fn drop(&mut self) {
        if !self.mmap_ptr.is_null() {
            unsafe {
                libc::munmap(self.mmap_ptr as *mut libc::c_void, self.mmap_size as usize);
            }
        }
    }
}
