//! E-core prefetch pump — ECS system.
//!
//! Phase 1: reads ternary u32 packs from .cimage mmap (GPU tile64 format)
//! and re-packs into 16×16 block-swizzled u8 in the pre-allocated SLC
//! WriteCombined buffer for ANE consumption.
//!
//! Phase 2: during idle cycles, requantizes FP16 KV cache from the ANE's
//! output surface into swizzled u8 ternary format, enabling DRAM-efficient
//! KV storage that the ANE can read back via the gather LUT.
//!
//! State machine: IDLE → PREFETCHING → READY, lock-free atomics.

use std::sync::Arc;
use std::thread::{self, JoinHandle};

use super::agent_slot::{MultiplexerState, STATE_IDLE, STATE_PREFETCHING, STATE_READY};
use crate::runtime::world::Entity;
use crate::runtime::components::AgentSlot;
use crate::compute_image::compile::ternary::{
    swizzled_buffer_size,
    repack_ternary_to_swizzled_u8,
};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StrideDescriptor {
    pub chunk_size_bytes: u32,
    pub prefetch_stride_elements: u32,
    pub alignment_padding_bytes: u32,
    pub tensor_shape_quad: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CImageTopologyTable {
    pub slice_4: StrideDescriptor,
    pub slice_8: StrideDescriptor,
    pub slice_16: StrideDescriptor,
    pub slice_32: StrideDescriptor,
}

/// Spawn the E-core prefetch pump on a dedicated thread.
pub fn spawn_ecore_prefetch_pump(state: Arc<MultiplexerState>) -> JoinHandle<()> {
    thread::spawn(move || loop {
        let world = state.world.read();
        let mut any_idle = false;

        // ── Phase 1: Weight prefetch ─────────────────────────────────
        for i in 0..32u32 {
            let entity = Entity(i);
            if !world.is_alive(entity) { continue; }
            let Some(slot) = world.get::<AgentSlot>(entity) else { continue; };

            if slot.load_state() != STATE_IDLE || slot.weight_offset == 0 { continue; }
            any_idle = true;
            if !slot.try_transition(STATE_IDLE, STATE_PREFETCHING) { continue; }

            let (mmap, slc_ptr, slc_len, (hd, id)) = match state.cimage_data() {
                Some(d) => d,
                None => { slot.store_state(STATE_READY); continue; }
            };
            let offset = slot.weight_offset;
            if offset >= mmap.len() { slot.store_state(STATE_READY); continue; }

            let phase = slot.prefetch_phase;
            let out_dim = if phase == 0 { hd } else { id } as usize;
            let in_dim  = if phase == 0 { hd } else { id } as usize;
            let rows_per = (out_dim + 31) / 32;
            let tile_stride = ((in_dim + 639) / 640) * 32 * 4;
            let avail = (mmap.len() - offset) / tile_stride;
            let rows = rows_per.min(avail);
            if rows > 0 {
                let ternary_data = &mmap[offset..offset + rows * tile_stride];
                let swz_size = swizzled_buffer_size(rows, in_dim);
                if slc_len >= swz_size {
                    let slc = unsafe { std::slice::from_raw_parts_mut(slc_ptr, swz_size) };
                    repack_ternary_to_swizzled_u8(ternary_data, rows, in_dim, slc, in_dim);
                    std::hint::black_box(unsafe { *slc_ptr });
                }
            }
            slot.store_state(STATE_READY);
        }
        drop(world);

        // ── Phase 2: KV requantization (idle cycles) ───────────────
        // All agents have their weights prefetched.  If KV cache data
        // is available from the ANE's output surface, compress it to
        // ternary swizzled u8 for DRAM-efficient storage.
        if !any_idle {
            // Stub: KV requantization is wired here once the ANE output
            // buffer address is available from the KVCacheRef component.
            // Call pattern:
            //   let (kv_ptr, seq_len, kv_dim, slc_out) = get_kv_state(&world, &state);
            //   requantize_kv_to_swizzled_u8(
            //       unsafe { slice::from_raw_parts(kv_ptr, seq_len * kv_dim * 2) },
            //       seq_len, kv_dim,
            //       unsafe { slice::from_raw_parts_mut(slc_out, swizzled_buffer_size(seq_len, kv_dim)) },
            //   );
        }

        std::thread::yield_now();
    })
}
