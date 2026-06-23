use crate::ane::arena_info::ArenaInfo;

/// Minimal dtype enum matching the ObjC bridge values.
/// Float16 = 0, Float32 = 1 (matches MLX Dtype ordering).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Float16 = 0,
    Float32 = 1,
}

extern "C" {
    fn tribunus_arena_alloc(
        info: *mut ArenaInfo,
        dim0: i32,
        dim1: i32,
        dtype: i32,
    ) -> i32;
    fn tribunus_arena_free(info: *mut ArenaInfo);
    fn tribunus_arena_lock(info: *const ArenaInfo) -> i32;
    fn tribunus_arena_unlock(info: *const ArenaInfo) -> i32;
}

/// IOSurface-backed memory arena for zero-copy GPU/ANE handoff.
///
/// # Lifecycle
/// - Allocated by `Arena::new`
/// - Borrowed by backends (writer-exclusive or reader-shared)
/// - Freed when dropped (IOSurface + CVPixelBuffer released)
pub struct Arena {
    pub info: ArenaInfo,
    pub dtype: Dtype,
}

// Safety: Arena memory is IOSurface-backed and accessed through Core ML
// APIs that are thread-safe. The raw pointer is valid for the Arena's lifetime.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Allocate a new arena backed by IOSurface + CVPixelBuffer.
    ///
    /// Currently supports FP16 and FP32. The ObjC bridge owns all storage;
    /// Rust merely holds the metadata.
    pub fn new(logical_dim0: u32, logical_dim1: u32, dtype: Dtype) -> Result<Self, String> {
        match dtype {
            Dtype::Float16 | Dtype::Float32 => {}
        }

        let mut info: ArenaInfo = unsafe { std::mem::zeroed() };
        let rc = unsafe {
            tribunus_arena_alloc(
                &mut info,
                logical_dim0 as i32,
                logical_dim1 as i32,
                dtype as i32,
            )
        };
        if rc != 0 {
            return Err(format!("tribunus_arena_alloc failed: {}", rc));
        }
        Ok(Arena { info, dtype })
    }

    /// Lock the arena's IOSurface for CPU access.
    pub fn lock(&self) -> Result<(), String> {
        let rc = unsafe { tribunus_arena_lock(&self.info) };
        if rc != 0 {
            return Err(format!("tribunus_arena_lock failed: {}", rc));
        }
        Ok(())
    }

    /// Unlock the IOSurface after CPU access.
    pub fn unlock(&self) -> Result<(), String> {
        let rc = unsafe { tribunus_arena_unlock(&self.info) };
        if rc != 0 {
            return Err(format!("tribunus_arena_unlock failed: {}", rc));
        }
        Ok(())
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { tribunus_arena_free(&mut self.info) };
    }
}
