use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;

pub const STATE_IDLE: u8 = 0;
pub const STATE_PREFETCHING: u8 = 1;
pub const STATE_READY: u8 = 2;
pub const STATE_EXECUTING: u8 = 3;

/// A single agent's execution slot.  Cache-line-aligned to prevent false
/// sharing between the E-core prefetcher and the P-core multiplexer.
#[repr(align(64))]
pub struct AgentSlot {
    pub state: AtomicU8,
    pub surface_id: u32,
    pub weight_offset: usize,
    pub prefetch_phase: u8,
}

impl AgentSlot {
    pub fn new(surface_id: u32, weight_offset: usize) -> Self {
        Self {
            state: AtomicU8::new(STATE_IDLE),
            surface_id,
            weight_offset,
            prefetch_phase: 0,
        }
    }
    pub fn try_transition(&self, expected: u8, target: u8) -> bool {
        self.state.compare_exchange(expected, target, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }
    pub fn load_state(&self) -> u8 { self.state.load(Ordering::Acquire) }
    pub fn store_state(&self, state: u8) { self.state.store(state, Ordering::Release); }
}

unsafe impl Sync for AgentSlot {}

/// The global execution context shared between the E-core prefetch thread
/// and the P-core ANE multiplexer.  All 32 agents are pre-allocated.
///
/// The SLC WriteCombined buffer is pre-allocated once at init.  The pump
/// writes swizzled u8 ternary data into it via `slc_buf_ptr`.  The ANE
/// reads from it after the slot transitions to READY.
pub struct MultiplexerState {
    pub world: RwLock<crate::runtime::world::World>,
    pub cimage_mmap: Option<Arc<memmap2::Mmap>>,
    /// Pre-allocated SLC buffer for swizzled u8 ternary (ANE format).
    /// Never reallocated after init.  Accessed via slc_buf_ptr by the pump.
    pub slc_buf: Option<Vec<u8>>,
    /// Mutable pointer to slc_buf data for the pump thread.
    /// Written once at init; the pump casts to `&mut [u8]` each cycle.
    /// Safe because only one writer (the E-core pump) ever touches it.
    pub slc_buf_ptr: Option<*mut u8>,
    /// Tensor dimensions from cimage: (hidden_dim, intermediate_dim).
    pub tensor_dims: Option<(u32, u32)>,
}

unsafe impl Send for MultiplexerState {}
unsafe impl Sync for MultiplexerState {}

impl MultiplexerState {
    pub fn new() -> Self {
        let mut world = crate::runtime::world::World::with_capacity(32);
        world.register_component::<AgentSlot>();
        world.register_component::<crate::runtime::components::KVCacheRef>();
        world.register_component::<crate::runtime::components::AgentPayload>();
        world.register_component::<crate::runtime::components::ToolRegistry>();
        Self {
            world: RwLock::new(world),
            cimage_mmap: None,
            slc_buf: None,
            slc_buf_ptr: None,
            tensor_dims: None,
        }
    }

    /// Initialise agent slots from the .cimage header, spawning 32 agents
    /// and pre-allocating the SLC WriteCombined buffer.
    pub fn init_from_cimage(
        &mut self,
        mmap: Arc<memmap2::Mmap>,
        header: &crate::compute_image::compile::ternary::PrismCimageHeader,
        hidden_dim: u32,
        intermediate_dim: u32,
    ) {
        self.cimage_mmap = Some(mmap);
        self.tensor_dims = Some((hidden_dim, intermediate_dim));

        // Pre-allocate SLC buffer for the largest layer (intermediate).
        // Each agent processes a row slice; 32 agents share total rows.
        let max_out = intermediate_dim.max(hidden_dim) as usize;
        let rows_per_agent = (max_out + 31) / 32;
        let cols = intermediate_dim.max(hidden_dim) as usize;
        let slc_size = crate::compute_image::compile::ternary::swizzled_buffer_size(rows_per_agent, cols);
        let mut buf = vec![0u8; slc_size];
        self.slc_buf_ptr = Some(buf.as_mut_ptr());
        self.slc_buf = Some(buf);

        let main_bytes = header.main_weights_len as usize;
        let slot_size = main_bytes / 32;
        let mut world = self.world.write();
        for i in 0..32 {
            if let Some(entity) = world.spawn() {
                world.insert(entity, AgentSlot::new(
                    i as u32,
                    (header.main_weights_offset as usize) + i * slot_size,
                ));
                world.insert(entity, crate::runtime::components::KVCacheRef::new(4096));
                world.insert(entity, crate::runtime::components::ToolRegistry::new());
            }
        }
    }

    /// Read-only accessor for the pump thread.
    /// Returns (mmap_ref, slc_mut_ptr, slc_len_bytes, (hidden_dim, intermediate_dim)).
    pub fn cimage_data(&self) -> Option<(&[u8], *mut u8, usize, (u32, u32))> {
        Some((
            self.cimage_mmap.as_ref()?,
            self.slc_buf_ptr?,
            self.slc_buf.as_ref()?.len(),
            self.tensor_dims?,
        ))
    }
}
