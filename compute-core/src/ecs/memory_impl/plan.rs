//! Memory plan FFI bridge — sends a [`prism_ecs_data::memory::MemoryPlan`]
//! to the MLX Metal allocator via the C FFI bridge, and walks a
//! compiler `ScheduledModule` to build one.
//!
//! The plan *data type* lives in
//! `prism_ecs_data::memory::{MemoryPlan, MemoryPlanSlot}` (E-1).
//! This module owns the engine-internal execution-plane side of
//! the same surface: the `mlx_set_memory_plan` / `mlx_clear_memory_plan`
//! C FFI declarations, the `apply` / `clear` safe wrappers, and the
//! `plan_from_scheduled_module` walker that depends on
//! engine-internal `Arena` and `ScheduledModule` types.
//!
//! # Architecture
//!
//! 1. The compiler produces a [`prism_ecs_data::memory::MemoryPlan`]
//!    with an ordered list of `(iosurface_ptr, size)` pairs — one
//!    per allocation the model needs during a forward pass.
//! 2. Before executing a planned region, the executor calls
//!    [`apply`] which passes the plan to the Metal allocator via
//!    the C FFI bridge.
//! 3. For each `malloc(size)` call during execution, the Metal
//!    allocator checks the next plan slot. If it exists and sizes
//!    match, it wraps the IOSurface pointer as an `MTLBuffer`
//!    instead of allocating new GPU memory.
//! 4. After the region completes, the executor calls [`clear`].

use std::ffi::c_void;

use prism_ecs_data::memory::{MemoryPlan, MemoryPlanSlot};

// ── C-compatible types ────────────────────────────────────────────────────
//
// `MemoryPlanSlot` and `MemoryPlan` are now defined in
// `prism_ecs_data::memory`. They are `#[repr(C)]` POD; the C ABI
// matches the `mlx_memory_plan_slot` C struct.

// ── Send / Sync bridge for the raw-pointer slot ──────────────────────────
//
// `MemoryPlanSlot` contains a raw `*mut c_void` and is therefore
// `!Send + !Sync` by default in the data crate. The MLX Metal
// allocator FFI is thread-safe (it serialises internally), so the
// engine is allowed to add the cross-thread marker here on the
// engine side where the hardware-FFI invariant lives. The
// data crate does not need this — it only carries the data type.
//
// SAFETY: `MemoryPlanSlot` is `#[repr(C)]` POD. The raw pointer is
// a `*mut c_void`; the engine never dereferences it — it only
// hands the slot array to the C FFI. The C allocator takes a
// copy internally (`mlx_set_memory_plan` documents that the
// allocator copies the plan entries). Therefore sending the slot
// across threads is sound: the pointer is treated as an opaque
// handle to IOSurface-backed memory whose lifetime is managed
// by the engine's `Arena`, not by the slot itself.
unsafe impl Send for MemoryPlanSlot {}
unsafe impl Sync for MemoryPlanSlot {}

// ── FFI declarations ──────────────────────────────────────────────────────

extern "C" {
    /// Set the memory plan for the Metal allocator.
    ///
    /// `slots` must remain valid for the duration of the planned region.
    /// The allocator copies the plan entries internally.
    fn mlx_set_memory_plan(num_slots: usize, slots: *const MemoryPlanSlot) -> i32;

    /// Clear the memory plan without consuming remaining slots.
    fn mlx_clear_memory_plan() -> i32;
}

// ── Safe wrappers ─────────────────────────────────────────────────────────

/// Send the plan to the Metal allocator via the C FFI bridge.
///
/// The allocator will use the pre-assigned IOSurface slices for its
/// next `plan.len()` allocations, in order.
///
/// # Safety
///
/// Caller must ensure that:
/// - All `iosurface_ptr` values point to valid, mapped IOSurface memory
/// - The IOSurface memory remains valid until `clear()` is called or
///   all plan slots are consumed
/// - The actual allocations made by MLX during the planned region
///   match the plan slots in both count and size
pub unsafe fn apply(plan: &MemoryPlan) -> Result<(), String> {
    if plan.is_empty() {
        return Ok(());
    }
    let ret = mlx_set_memory_plan(plan.len(), plan.slots.as_ptr());
    if ret != 0 {
        return Err(format!("mlx_set_memory_plan returned {}", ret));
    }
    Ok(())
}

/// Clear the active memory plan from the Metal allocator.
///
/// After calling this, subsequent `malloc`s will use normal Metal
/// buffer allocation (heap/cache) instead of the plan.
///
/// Safe to call when no plan is active (no-op).
pub fn clear() -> Result<(), String> {
    let ret = unsafe { mlx_clear_memory_plan() };
    if ret != 0 {
        return Err(format!("mlx_clear_memory_plan returned {}", ret));
    }
    Ok(())
}

/// Backwards-compatible alias for [`clear`]. Engine callers
/// imported `crate::memory_impl::plan::clear_memory_plan` after
/// the memory deletion; keep the named function available so the
/// migration is a pure import-path swap.
pub fn clear_memory_plan() -> Result<(), String> {
    clear()
}

// ── Integration with the compiler ─────────────────────────────────────────

/// Generate a memory plan from the compiler's `ScheduledModule`.
///
/// Walks the scheduled regions' [`MemoryPlan`] and produces an
/// ordered list of [`MemoryPlanSlot`] entries that the executor
/// passes to the Metal allocator before running the module.
///
/// When `compression_ratio` is `Some(r)`, the estimated KV cache
/// byte sizes are scaled down by `r` (e.g. 4.57 for TurboQuant3:
/// 16 bits / 3.5 bits). Pass `None` for uncompressed FP16 mode
/// (no scaling).
///
/// Returns `None` if the scheduled module has no material
/// allocations that need planning (empty or all in-place).
pub fn plan_from_scheduled_module(
    scheduled: &crate::ecs::compiler::scheduled::ScheduledModule,
    arena: &crate::ecs::arena::Arena,
    compression_ratio: Option<f64>,
) -> Option<MemoryPlan> {
    let mut plan = MemoryPlan::new();
    let ratio = compression_ratio.unwrap_or(1.0);

    for region in &scheduled.regions {
        let scaled = (region.temp_memory_bytes as f64 / ratio) as u64;
        if scaled == 0 {
            continue;
        }

        // The arena pre-allocates a contiguous block. We carve out slices
        // for each planned allocation within it.
        let slice_base = unsafe { arena.base_ptr() as *mut c_void };
        let offset = plan.slots.len() as u64 * 4096; // page-aligned offset
        let ptr = unsafe { (slice_base as *mut u8).add(offset as usize) as *mut c_void };
        let size = scaled as usize;

        plan.add_slot(ptr, size);
    }

    if plan.is_empty() {
        None
    } else {
        Some(plan)
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
