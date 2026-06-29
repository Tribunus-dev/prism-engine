//! Paged KV cache arena with 64-token pages and a lightweight CPU-side page table.
//! Decouples logical token positions from physical MTLBuffer continuity.
//! Enables 1M+ context windows without 54 GB contiguous allocations.

use metal::{Buffer, Device, MTLResourceOptions};
use std::collections::BTreeMap;

// ── Constants ─────────────────────────────────────────────────────
pub const TOKENS_PER_PAGE: usize = 64;
pub const NUM_KV_HEADS: usize = 8;
pub const BLOCKS_PER_HEAD: usize = 16;  // ceil(512/32)
pub const TERNARY_BLOCK_BYTES: usize = 9;  // 7 trits + 2 scale
pub const OUTLIER_BYTES: usize = 2;       // 1 FP16 per block worst case
pub const BYTES_PER_PAGE: usize = TOKENS_PER_PAGE * NUM_KV_HEADS * BLOCKS_PER_HEAD * (TERNARY_BLOCK_BYTES + OUTLIER_BYTES);

/// Physical page descriptor.
pub struct Page {
    pub page_id: u32,
    pub buffer: Buffer,
    pub offset: u64,        // byte offset within buffer
    pub capacity: u32,      // tokens
    pub filled: u32,        // tokens written
}

/// Page table: maps (layer, kv_head, logical_token_pos) → physical page + offset.
pub struct PageTable {
    pub entries: BTreeMap<(u32, u32, u32), (u32, u32)>,  // (layer, head, token) → (page_id, offset_within_page)
    pub pages: Vec<Page>,
    pub device: Device,
}

impl PageTable {
    pub fn new(device: &Device) -> Self {
        Self {
            entries: BTreeMap::new(),
            pages: Vec::new(),
            device: device.clone(),
        }
    }

    /// Allocate a new physical page. Returns page_id.
    pub fn alloc_page(&mut self) -> Result<u32, String> {
        let buf = self.device.new_buffer(
            BYTES_PER_PAGE as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let page_id = self.pages.len() as u32;
        self.pages.push(Page {
            page_id,
            buffer: buf,
            offset: 0,
            capacity: TOKENS_PER_PAGE as u32,
            filled: 0,
        });
        Ok(page_id)
    }

    /// Map a (layer, head, token) triplet to its physical address.
    /// Allocates a new page if needed.
    pub fn map_token(
        &mut self,
        layer: u32,
        head: u32,
        token: u32,
    ) -> Result<(u32, u32), String> {
        let key = (layer, head, token);
        if let Some(&(page_id, offset)) = self.entries.get(&key) {
            return Ok((page_id, offset));
        }

        // Need to allocate: find an existing page for this layer+head with space
        // or create a new one
        let page_token = token / TOKENS_PER_PAGE as u32;
        let _page_key = (layer, head, page_token);

        // Check if we already have this page
        for (page_id, page) in self.pages.iter().enumerate() {
            if page.filled < page.capacity {
                // This page has space — use it
                let offset = page.filled * BYTES_PER_PAGE as u32 / TOKENS_PER_PAGE as u32;
                let pid = page_id as u32;
                for t in 0..TOKENS_PER_PAGE as u32 {
                    self.entries.insert((layer, head, page_token * TOKENS_PER_PAGE as u32 + t), (pid, offset));
                }
                self.pages[page_id].filled += TOKENS_PER_PAGE as u32;
                return Ok((pid, offset));
            }
        }

        // Need a new page
        let page_id = self.alloc_page()?;
        let offset = 0;
        for t in 0..TOKENS_PER_PAGE as u32 {
            self.entries.insert((layer, head, page_token * TOKENS_PER_PAGE as u32 + t), (page_id, offset));
        }
        self.pages[page_id as usize].filled = TOKENS_PER_PAGE as u32;
        Ok((page_id, offset))
    }

    /// Get the physical address for a (layer, head, token).
    /// Returns None if not mapped.
    pub fn lookup(&self, layer: u32, head: u32, token: u32) -> Option<(&Buffer, u64)> {
        let key = (layer, head, token);
        self.entries.get(&key).and_then(|&(page_id, offset)| {
            self.pages.get(page_id as usize).map(|page| {
                (&page.buffer, page.offset + offset as u64)
            })
        })
    }

    /// Evict tokens from a page (mark as available for reuse).
    pub fn evict_page(&mut self, page_id: u32) {
        if let Some(page) = self.pages.get_mut(page_id as usize) {
            page.filled = 0;
        }
        // Remove entries — scan and remove all entries pointing to this page
        self.entries.retain(|_, &mut (pid, _)| pid != page_id);
    }

    /// Current memory usage in bytes.
    pub fn memory_used(&self) -> u64 {
        self.pages.len() as u64 * BYTES_PER_PAGE as u64
    }
}

/// L1 staging ring: contiguous FP16 buffer for active (~20K) tokens.
pub struct L1StagingRing {
    pub buffer: Buffer,         // FP16, contiguous
    pub capacity_tokens: u32,   // ~20480
    pub active_token_ids: Vec<u32>,  // logical token indices in L1
}

impl L1StagingRing {
    /// Create L1 staging buffer for ANE/GPU fast attention.
    /// size = capacity_tokens × NUM_KV_HEADS × GLOBAL_HEAD_DIM × 2 (FP16)
    pub fn new(device: &Device, capacity_tokens: u32) -> Self {
        let bytes = capacity_tokens as u64 * NUM_KV_HEADS as u64 * 512 * 2;
        let buf = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        Self {
            buffer: buf,
            capacity_tokens,
            active_token_ids: Vec::new(),
        }
    }
}

/// Paged KV cache: combines page table + L1 staging + gather/scatter.
pub struct PagedKVCache {
    pub page_table: PageTable,
    pub l1_k: L1StagingRing,
    pub l1_v: L1StagingRing,
    pub device: Device,
}

impl PagedKVCache {
    pub fn new(device: &Device, l1_capacity: u32) -> Self {
        Self {
            page_table: PageTable::new(device),
            l1_k: L1StagingRing::new(device, l1_capacity),
            l1_v: L1StagingRing::new(device, l1_capacity),
            device: device.clone(),
        }
    }

    /// Stage a set of tokens into L1 by gathering from their paged locations.
    /// Reads ternary pages from L2, decompresses to FP16 in contiguous L1 buffer.
    /// `token_ids`: logical token indices to stage.
    /// `layer`: which layer's KV to gather.
    /// Returns (l1_offset_start, num_tokens_staged).
    pub fn gather_to_l1(
        &mut self,
        token_ids: &[u32],
        layer: u32,
    ) -> Result<(u32, u32), String> {
        // Map each token to its physical page location
        // Then execute GPU gather kernel to read ternary pages + decompress + write FP16 to L1
        // For now: placeholder that just marks tokens as active
        self.l1_k.active_token_ids = token_ids.to_vec();
        Ok((0, token_ids.len() as u32))
    }
}
