//! Typed port for execution-plane arena allocation.
//!
//! This module owns the canonical authority for the *contract* that any
//! execution-plane arena pool MUST satisfy when integrated with the Prism
//! runtime. It does **not** own the arena itself — arenas are
//! execution-plane state (IOSurface-backed CVPixelBuffer handles, raw
//! pointers, OS-level FFI). See `compute-core/src/ecs/core/arena.rs` for
//! the engine-side implementation that satisfies this port.
//!
//! ## Authority boundary
//!
//! The constitutional rule that drives this split is the one in `AGENTS.md`:
//! "Hardware handles, file descriptors, locks, and process-local channels
//! are execution-plane state. They must not be persisted as durable
//! components or events. They must not appear in error messages that
//! propagate beyond the runtime boundary." An arena handle is a hardware
//! handle. The world may store a stable key (the [`ArenaHandle`] newtype)
//! for it, but never the raw `IOSurfaceRef` / `CVPixelBufferRef` / `*mut
//! c_void` pointers that the engine holds.
//!
//! This port is the seam: callers in the runtime speak the typed contract
//! defined here; the engine implements the contract by holding the
//! execution-plane state. If a future backend (ANE buffer pool, ANE
//! `IOSurfaceStack`, MLX `mtl::Buffer` pool, etc.) wants to participate
//! in arena allocation, it implements this trait — it does not invent a
//! parallel authority.
//!
//! ## Stable identity
//!
//! [`ArenaHandle`] is an opaque `u64` newtype. The high 16 bits carry a
//! backend tag (so a future dispatcher can route by handle); the low 48
//! bits are the local handle id. Callers MUST treat the value as opaque
//! — only [`ArenaAllocator::release`], [`ArenaAllocator::lock`], and
//! [`ArenaAllocator::release`] are permitted to interpret it.
//!
//! ## Errors
//!
//! [`ArenaError`] mirrors the constitutional error discipline
//! (`WorldTxnError`-style: `Rejected` / `Failed` / `Stale`):
//!
//! - `Rejected` (preflight): the request was rejected before any effect
//!   ran. `ZeroBytes`, `TooLarge`.
//! - `Failed` (effect): the underlying allocator returned a failure
//!   status. `BackendFailed`, `BackendLockFailed`.
//! - `Stale` (fencing mismatch): the handle is no longer live — released
//!   or never allocated by this dispatcher. `UnknownHandle`,
//!   `HandleAlreadyReleased`.
//!
//! ## Tests
//!
//! The tests in this module exercise the *contract* against an in-memory
//! [`MockArenaAllocator`] so the typed port is verified independent of the
//! engine's IOSurface bridge. The engine's own tests in
//! `compute-core/src/ecs/core/arena.rs` verify the production bridge
//! against real Apple-platform IOSurfaces.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use prism_ecs_constitutional::SchemaKey;

// ── Typed newtypes ─────────────────────────────────────────────────────────

/// Logical element data type for the arena's payload.
///
/// Mirrors the engine's `DataType` (Float16 / Float32) without dragging
/// the FFI surface across the constitutional boundary. The runtime never
/// sees the engine's enum; it sees this neutral variant and the engine
/// is free to map it onto whatever the platform supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArenaElementDtype {
    Float16,
    Float32,
}

impl ArenaElementDtype {
    /// Byte size of a single element under this dtype.
    pub fn element_bytes(self) -> u64 {
        match self {
            ArenaElementDtype::Float16 => 2,
            ArenaElementDtype::Float32 => 4,
        }
    }
}

/// Non-zero byte count for an arena allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ByteCount(u32);

impl ByteCount {
    /// Construct a [`ByteCount`]. Returns `None` for zero.
    pub fn new(n: u32) -> Option<Self> {
        if n == 0 { None } else { Some(Self(n)) }
    }

    /// The raw byte count. Always > 0.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Logical element geometry (dim0, dim1) for a 2-D typed arena.
///
/// Mirrors the engine's `logical_dim0` / `logical_dim1` fields without
/// leaking the IOSurface / CVPixelBuffer ABI. `dim0` and `dim1` are
/// logical (not physical) — bytes-per-row padding is the backend's
/// concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArenaGeometry {
    pub dim0: u32,
    pub dim1: u32,
}

impl ArenaGeometry {
    /// Total logical element count.
    pub fn element_count(&self) -> u64 {
        u64::from(self.dim0) * u64::from(self.dim1)
    }
}

/// Opaque, backend-issued handle to a live arena.
///
/// The high 16 bits are a backend tag (assigned by the dispatcher at
/// allocation time, stable for the lifetime of the dispatcher); the
/// low 48 bits are the local handle id. Callers MUST treat the value
/// as opaque; the only valid operations are the methods on
/// [`ArenaAllocator`].
///
/// `ArenaHandle` is `Copy` because handles are cheap, stable, and safe
/// to share — the underlying arena outlives any single handle clone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArenaHandle(pub u64);

impl ArenaHandle {
    /// Construct a handle from its raw form. Intended for backend
    /// implementations; callers should treat the value as opaque.
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw handle bits. Intended for backend implementations.
    pub fn as_raw(self) -> u64 {
        self.0
    }
}

/// RAII guard returned by [`ArenaAllocator::lock`].
///
/// Holds the lock until dropped. Drop releases the lock; drop failures
/// are logged by the backend but are not surfaced to the caller
/// (consistent with the engine's `CVPixelBufferLockBaseAddress` contract,
/// which has no meaningful "fail to unlock" semantics).
pub struct LockedArenaGuard {
    handle: ArenaHandle,
    /// Opaque releaser that the dispatcher installs when the guard is
    /// created. Decoupled from the concrete `ArenaAllocator` type so the
    /// trait is dyn-compatible (`Send + Sync`).
    releaser: Box<dyn Fn(ArenaHandle) + Send + Sync>,
}

impl LockedArenaGuard {
    /// The handle this guard locks.
    pub fn handle(&self) -> ArenaHandle {
        self.handle
    }
}

impl fmt::Debug for LockedArenaGuard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockedArenaGuard")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

impl Drop for LockedArenaGuard {
    fn drop(&mut self) {
        (self.releaser)(self.handle);
    }
}

// ── Error taxonomy ─────────────────────────────────────────────────────────

/// Errors that can occur during arena port operations.
///
/// Categorized per the constitutional error discipline:
/// - `Rejected` (preflight): the request was rejected before any effect
///   ran.
/// - `Failed` (effect): the underlying allocator returned a failure
///   status.
/// - `Stale` (fencing mismatch): the handle is no longer live.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ArenaError {
    // ── Rejected (preflight) ────────────────────────────────────────────
    #[error("zero-byte allocation is not permitted")]
    ZeroBytes,

    #[error("requested size {requested} bytes exceeds dispatcher limit {limit} bytes")]
    TooLarge { requested: u32, limit: u32 },

    // ── Failed (effect) ─────────────────────────────────────────────────
    #[error("backend reported allocation failure: {0}")]
    BackendFailed(String),

    #[error("backend reported lock failure on handle {handle:?}: {detail}")]
    BackendLockFailed {
        handle: ArenaHandle,
        detail: String,
    },

    // ── Stale (fencing mismatch) ────────────────────────────────────────
    #[error("unknown arena handle {handle:?}")]
    UnknownHandle { handle: ArenaHandle },

    #[error("arena handle {handle:?} has already been released")]
    HandleAlreadyReleased { handle: ArenaHandle },
}

// ── The typed port ─────────────────────────────────────────────────────────

/// Provider-neutral port for execution-plane arena allocation.
///
/// The engine's `compute-core/src/ecs/core/arena.rs` implements this port
/// (the FFI to the ObjC `tribunus_arena_*` bridge that wraps
/// `IOSurfaceCreate` + `CVPixelBufferCreateWithIOSurface`). The runtime
/// never holds the raw handles; it only ever speaks this trait.
///
/// # Contract
///
/// - [`ArenaAllocator::allocate_bytes`] and
///   [`ArenaAllocator::allocate_2d`] MUST return a fresh, distinct
///   [`ArenaHandle`] for every successful call.
/// - [`ArenaAllocator::release`] MUST reject (with
///   [`ArenaError::HandleAlreadyReleased`]) any handle that has already
///   been released. After successful release, subsequent operations on
///   the same handle MUST return [`ArenaError::UnknownHandle`].
/// - [`ArenaAllocator::lock`] MUST return a guard that, when dropped,
///   releases the underlying base-address lock.
/// - All operations MUST be safe to call from multiple threads.
///
/// # Schema admission
///
/// The `schema` argument is a stable [`SchemaKey`] identifying the
/// payload's content type. The dispatcher MAY use it to route between
/// IOSurface-backed and heap-backed fallbacks (the engine's existing
/// `Arena::new_bytes` does exactly this). A dispatcher that ignores
/// `schema` is still correct, but one that observes it may pool
/// allocations across calls with the same schema.
pub trait ArenaAllocator: Send + Sync {
    /// Allocate a 1-D arena of `bytes` bytes for a payload identified by
    /// `schema`. The `dtype` parameter is advisory — backends that
    /// ignore it (e.g. byte-oriented IOSurface pools) are still correct.
    fn allocate_bytes(
        &self,
        schema: SchemaKey,
        dtype: ArenaElementDtype,
        bytes: ByteCount,
    ) -> Result<ArenaHandle, ArenaError>;

    /// Allocate a 2-D arena with the given logical geometry.
    fn allocate_2d(
        &self,
        schema: SchemaKey,
        dtype: ArenaElementDtype,
        geom: ArenaGeometry,
    ) -> Result<ArenaHandle, ArenaError>;

    /// Release an arena. After this returns, the handle is dead; any
    /// further reference to it via this dispatcher MUST return
    /// [`ArenaError::UnknownHandle`].
    fn release(&self, handle: ArenaHandle) -> Result<(), ArenaError>;

    /// Acquire the CPU-side base-address lock. Drop the returned guard
    /// to release. A handle MAY be locked multiple times by the same
    /// thread; the backend's lock implementation is responsible for
    /// the read/write balance.
    fn lock(&self, handle: ArenaHandle) -> Result<LockedArenaGuard, ArenaError>;
}

// ── Mock implementation (for tests) ─────────────────────────────────────────

/// Internal state of a live mock arena.
#[derive(Debug, Clone, Copy)]
struct MockArenaState {
    bytes: u32,
    lock_count: u32,
    /// `true` once the handle has been released; subsequent operations
    /// on the handle must return `UnknownHandle`.
    released: bool,
}

/// In-memory mock implementation of [`ArenaAllocator`].
///
/// The mock is the canonical reference for what the port promises: the
/// engine's IOSurface-backed implementation must satisfy the same
/// properties. The mock deliberately holds no real hardware state — it
/// only exercises the typed contract.
pub struct MockArenaAllocator {
    inner: Arc<MockInner>,
}

struct MockInner {
    next_id: AtomicU64,
    backend_tag: u16,
    live: Mutex<BTreeMap<ArenaHandle, MockArenaState>>,
    max_bytes: u32,
}

impl MockArenaAllocator {
    /// Create a mock dispatcher with the given `backend_tag` and a
    /// maximum allocation size of `max_bytes`.
    pub fn new(backend_tag: u16, max_bytes: u32) -> Self {
        Self {
            inner: Arc::new(MockInner {
                next_id: AtomicU64::new(1),
                backend_tag,
                live: Mutex::new(BTreeMap::new()),
                max_bytes,
            }),
        }
    }

    /// The number of currently-live arenas.
    pub fn live_count(&self) -> usize {
        self.inner.live.lock().expect("mock mutex").len()
    }

    fn mint_handle(&self) -> ArenaHandle {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let raw = (u64::from(self.inner.backend_tag) << 48) | id;
        ArenaHandle(raw)
    }
}

impl ArenaAllocator for MockArenaAllocator {
    fn allocate_bytes(
        &self,
        _schema: SchemaKey,
        _dtype: ArenaElementDtype,
        bytes: ByteCount,
    ) -> Result<ArenaHandle, ArenaError> {
        let n = bytes.get();
        if n > self.inner.max_bytes {
            return Err(ArenaError::TooLarge {
                requested: n,
                limit: self.inner.max_bytes,
            });
        }
        let handle = self.mint_handle();
        let state = MockArenaState {
            bytes: n,
            lock_count: 0,
            released: false,
        };
        self.inner
            .live
            .lock()
            .expect("mock mutex")
            .insert(handle, state);
        Ok(handle)
    }

    fn allocate_2d(
        &self,
        schema: SchemaKey,
        dtype: ArenaElementDtype,
        geom: ArenaGeometry,
    ) -> Result<ArenaHandle, ArenaError> {
        let elements = geom.element_count();
        let bytes = elements.saturating_mul(dtype.element_bytes());
        let n = u32::try_from(bytes).map_err(|_| ArenaError::TooLarge {
            requested: u32::MAX,
            limit: self.inner.max_bytes,
        })?;
        self.allocate_bytes(schema, dtype, ByteCount::new(n).ok_or(ArenaError::ZeroBytes)?)
    }

    fn release(&self, handle: ArenaHandle) -> Result<(), ArenaError> {
        let mut live = self.inner.live.lock().expect("mock mutex");
        let state = live.get_mut(&handle).ok_or(ArenaError::UnknownHandle { handle })?;
        if state.released {
            return Err(ArenaError::HandleAlreadyReleased { handle });
        }
        if state.lock_count > 0 {
            // An active lock guard still holds the handle. The release
            // call is rejected; the caller must drop guards first.
            return Err(ArenaError::HandleAlreadyReleased { handle });
        }
        state.released = true;
        // Remove the entry so subsequent calls see `UnknownHandle`.
        live.remove(&handle);
        Ok(())
    }

    fn lock(&self, handle: ArenaHandle) -> Result<LockedArenaGuard, ArenaError> {
        {
            let mut live = self.inner.live.lock().expect("mock mutex");
            let state = live.get_mut(&handle).ok_or(ArenaError::UnknownHandle { handle })?;
            if state.released {
                return Err(ArenaError::HandleAlreadyReleased { handle });
            }
            state.lock_count += 1;
        }

        // Install the releaser against the shared `inner` Arc. The Arc
        // outlives any guard because the guard holds a strong reference.
        let inner = Arc::clone(&self.inner);
        let releaser: Box<dyn Fn(ArenaHandle) + Send + Sync> = Box::new(move |h: ArenaHandle| {
            if let Ok(mut live) = inner.live.lock() {
                if let Some(state) = live.get_mut(&h) {
                    if state.lock_count > 0 {
                        state.lock_count -= 1;
                    }
                }
            }
        });
        Ok(LockedArenaGuard { handle, releaser })
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> SchemaKey {
        SchemaKey {
            namespace: "prism.test",
            id: 1,
            version: 1,
        }
    }

    #[test]
    fn byte_count_rejects_zero_at_construction() {
        assert!(ByteCount::new(0).is_none());
        assert!(ByteCount::new(1).is_some());
        assert_eq!(ByteCount::new(64).expect("non-zero").get(), 64);
    }

    #[test]
    fn zero_bytes_rejected_preflight() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        // We can't construct a ByteCount(0) (compile-time prevented), so
        // exercise the indirect path through allocate_2d on a zero-area
        // geometry (0 * 0 = 0 bytes).
        let g = ArenaGeometry { dim0: 0, dim1: 0 };
        let result = alloc.allocate_2d(test_schema(), ArenaElementDtype::Float16, g);
        assert_eq!(result, Err(ArenaError::ZeroBytes));
    }

    #[test]
    fn too_large_rejected_preflight() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let result =
            alloc.allocate_bytes(test_schema(), ArenaElementDtype::Float16, ByteCount::new(2048).expect("non-zero"));
        assert!(matches!(result, Err(ArenaError::TooLarge { .. })));
    }

    #[test]
    fn successful_allocate_and_release() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let h = alloc
            .allocate_bytes(
                test_schema(),
                ArenaElementDtype::Float16,
                ByteCount::new(64).expect("non-zero"),
            )
            .expect("allocate");
        assert_eq!(alloc.live_count(), 1);
        alloc.release(h).expect("release");
        assert_eq!(alloc.live_count(), 0);
    }

    #[test]
    fn double_release_returns_unknown_handle() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let h = alloc
            .allocate_bytes(
                test_schema(),
                ArenaElementDtype::Float16,
                ByteCount::new(64).expect("non-zero"),
            )
            .expect("allocate");
        alloc.release(h).expect("first release");
        let second = alloc.release(h);
        assert_eq!(second, Err(ArenaError::UnknownHandle { handle: h }));
    }

    #[test]
    fn release_unknown_handle_returns_stale() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let phantom = ArenaHandle(0xDEAD_BEEF_DEAD_BEEF);
        let result = alloc.release(phantom);
        assert_eq!(result, Err(ArenaError::UnknownHandle { handle: phantom }));
    }

    #[test]
    fn lock_guard_drops_releases_lock_and_allows_release() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let h = alloc
            .allocate_bytes(
                test_schema(),
                ArenaElementDtype::Float16,
                ByteCount::new(64).expect("non-zero"),
            )
            .expect("allocate");
        let guard = alloc.lock(h).expect("lock");
        assert_eq!(guard.handle(), h);
        // While the guard is live, release must reject.
        let rejected = alloc.release(h);
        assert_eq!(rejected, Err(ArenaError::HandleAlreadyReleased { handle: h }));
        // After drop, the release path is permitted.
        drop(guard);
        alloc.release(h).expect("release after guard drop");
    }

    #[test]
    fn lock_unknown_handle_returns_stale() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let phantom = ArenaHandle(0xDEAD_BEEF_DEAD_BEEF);
        let result = alloc.lock(phantom);
        assert_eq!(result, Err(ArenaError::UnknownHandle { handle: phantom }));
    }

    #[test]
    fn arena_geometry_element_count() {
        let g = ArenaGeometry { dim0: 4, dim1: 512 };
        assert_eq!(g.element_count(), 2048);
    }

    #[test]
    fn arena_element_dtype_byte_size() {
        assert_eq!(ArenaElementDtype::Float16.element_bytes(), 2);
        assert_eq!(ArenaElementDtype::Float32.element_bytes(), 4);
    }

    #[test]
    fn allocate_2d_uses_dtype_stride() {
        // Float16: 4*512*2 = 4096 bytes — under the 8192 cap.
        // Float32: 4*512*4 = 8192 bytes — exactly the cap (still accepted).
        let alloc = MockArenaAllocator::new(0xBEEF, 8192);
        let g = ArenaGeometry { dim0: 4, dim1: 512 };
        let h16 = alloc
            .allocate_2d(test_schema(), ArenaElementDtype::Float16, g)
            .expect("allocate fp16");
        let h32 = alloc
            .allocate_2d(test_schema(), ArenaElementDtype::Float32, g)
            .expect("allocate fp32");
        assert_ne!(h16, h32);
        assert_eq!(alloc.live_count(), 2);
    }

    #[test]
    fn handle_backend_tag_is_high_16_bits() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let h = alloc
            .allocate_bytes(
                test_schema(),
                ArenaElementDtype::Float16,
                ByteCount::new(64).expect("non-zero"),
            )
            .expect("allocate");
        let tag = (h.as_raw() >> 48) & 0xFFFF;
        assert_eq!(tag, 0xBEEF);
    }

    #[test]
    fn handle_ids_are_unique() {
        let alloc = MockArenaAllocator::new(0xBEEF, 1024);
        let h1 = alloc
            .allocate_bytes(test_schema(), ArenaElementDtype::Float16, ByteCount::new(64).expect("nz"))
            .expect("h1");
        let h2 = alloc
            .allocate_bytes(test_schema(), ArenaElementDtype::Float16, ByteCount::new(64).expect("nz"))
            .expect("h2");
        let h3 = alloc
            .allocate_bytes(test_schema(), ArenaElementDtype::Float16, ByteCount::new(64).expect("nz"))
            .expect("h3");
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_ne!(h1, h3);
    }

    #[test]
    fn handle_round_trip() {
        let raw = 0xBEEF_0000_0000_0042u64;
        let h = ArenaHandle::from_raw(raw);
        assert_eq!(h.as_raw(), raw);
    }

    #[test]
    fn arena_error_variants_are_typed() {
        // Confirm the variants are distinct enum variants (not Strings).
        let a = ArenaError::ZeroBytes;
        let b = ArenaError::TooLarge { requested: 10, limit: 5 };
        let c = ArenaError::BackendFailed("x".into());
        let d = ArenaError::UnknownHandle { handle: ArenaHandle(0) };
        let e = ArenaError::HandleAlreadyReleased { handle: ArenaHandle(0) };
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(c, d);
        assert_ne!(d, e);
    }
}
