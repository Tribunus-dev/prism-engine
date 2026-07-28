//! Pre-computed memory plan data types.
//!
//! This module owns the canonical authority for the engine-independent
//! `MemoryPlan` / `MemoryPlanSlot` data types — an ordered list of
//! `(iosurface_ptr, size)` pairs the MLX Metal allocator consumes
//! instead of allocating new GPU memory. The FFI surface that pushes
//! the plan into the Metal allocator (`mlx_set_memory_plan` /
//! `mlx_clear_memory_plan`) and the `plan_from_scheduled_module`
//! walker stay engine-side at
//! `compute-core/src/ecs/memory_impl/plan_impl.rs` because they
//! depend on engine-internal `Arena` and `ScheduledModule` types.
//!
//! # Thread-safety
//!
//! `MemoryPlanSlot` contains a raw `*mut c_void` and is therefore
//! `!Send + !Sync` by default. The engine-side
//! `memory_impl::plan_impl::SendPlan(MemoryPlan)` newtype adds the
//! `unsafe impl Send + Sync` on the engine side where the
//! hardware-FFI invariant lives. Callers that need to pass a plan
//! across threads must wrap it in the engine-side newtype.

use std::ffi::c_void;

/// Memory plan slot — matches `mlx_memory_plan_slot` in `mlx/c/memory.h`.
///
/// POD (plain old data); no references, no `Drop`. `!Send + !Sync`
/// by default; the engine-side `SendPlan` newtype is the only path
/// for cross-thread ownership transfer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MemoryPlanSlot {
    /// Base address of the pre-assigned IOSurface slice.
    pub iosurface_ptr: *mut c_void,
    /// Expected allocation size in bytes.
    pub size: usize,
}

/// A pre-computed memory plan: an ordered list of `(ptr, size)` pairs
/// that the Metal allocator will use instead of allocating new GPU
/// memory.
#[derive(Debug, Clone)]
pub struct MemoryPlan {
    /// Ordered allocation slots.
    pub slots: Vec<MemoryPlanSlot>,
}

impl MemoryPlan {
    /// Create a new empty memory plan.
    pub fn new() -> Self {
        Self { slots: Vec::new() }
    }

    /// Add a slot: an IOSurface pointer and its expected allocation size.
    pub fn add_slot(&mut self, ptr: *mut c_void, size: usize) {
        self.slots.push(MemoryPlanSlot {
            iosurface_ptr: ptr,
            size,
        });
    }

    /// Number of slots in the plan.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// True if the plan has no slots.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

impl Default for MemoryPlan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_plan_is_empty() {
        let plan = MemoryPlan::new();
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
    }

    #[test]
    fn add_slot_increments_len() {
        let mut plan = MemoryPlan::new();
        let dummy = 0xdeadbeef as *mut c_void;
        plan.add_slot(dummy, 4096);
        assert_eq!(plan.len(), 1);
        assert!(!plan.is_empty());
        assert_eq!(plan.slots[0].iosurface_ptr, dummy);
        assert_eq!(plan.slots[0].size, 4096);
    }

    #[test]
    fn default_matches_new() {
        let plan: MemoryPlan = Default::default();
        assert!(plan.is_empty());
    }
}
