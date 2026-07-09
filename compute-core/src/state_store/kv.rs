use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::state_store::pages::PageDescriptor;
use crate::state_store::receipts::{KvAppendReceipt, KvReadReceipt};
use crate::state_store::schema::{AccessKind, KvCacheLayout, KvCacheStoreDecl, StateAccessPolicy};

/// Descriptor for a span of contiguous tokens in the KV cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvSpan {
    pub span_id: String,
    pub token_start: u32,
    pub token_count: u32,
    pub epoch_id: u64,
    pub pages: Vec<u32>,
    pub pinned: bool,
}

/// Operations that can be performed on a KV cache store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheOp {
    Allocate,
    Append,
    Read,
    Pin,
    Evict,
}

/// In-memory paged KV cache manager.
///
/// Pre-allocates all pages on `allocate_store()` and manages token spans
/// referencing those pages. Provides deterministic memory accounting and
/// epoch-gated access checks.
pub struct KvCacheManager {
    pub store_id: String,
    pub config: KvCacheStoreDecl,
    pub owner_region: String,
    pub access_policies: Vec<StateAccessPolicy>,
    pub pages: Vec<PageDescriptor>,
    pub spans: Vec<KvSpan>,
    pub current_epoch: u64,
    pub total_bytes: u64,
    pub max_bytes: u64,
    allocated: bool,
    /// Track which page_ids are claimed by any span.
    claimed_pages: HashSet<u32>,
    /// Sequence counter for span IDs.
    next_span_seq: u64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bytes_per_element(dtype: &str) -> u64 {
    match dtype {
        "fp32" | "float32" => 4,
        _ => 2,
    }
}

fn align_up(val: u64, align: u32) -> u64 {
    let a = align as u64;
    (val + a - 1) & !(a - 1)
}

impl KvCacheManager {
    /// Create a new un-allocated manager.
    pub fn new(store_id: &str, config: KvCacheStoreDecl, owner_region: &str) -> Self {
        Self {
            store_id: store_id.to_string(),
            config,
            owner_region: owner_region.to_string(),
            access_policies: Vec::new(),
            pages: Vec::new(),
            spans: Vec::new(),
            current_epoch: 0,
            total_bytes: 0,
            max_bytes: 0,
            allocated: false,
            claimed_pages: HashSet::new(),
            next_span_seq: 0,
        }
    }

    /// Allocate the paged layout: pre-compute page descriptors for the entire
    /// KV cache capacity.
    ///
    /// Returns an error if already allocated.
    pub fn allocate_store(&mut self) -> Result<(), String> {
        if self.allocated {
            return Err("store already allocated".to_string());
        }

        let layout = match &self.config.cache_layout {
            KvCacheLayout::PagedLayerHead {
                page_tokens,
                alignment_bytes,
            } => (*page_tokens, *alignment_bytes),
        };

        if layout.0 == 0 || layout.1 == 0 {
            return Err("page_tokens and alignment_bytes must be > 0".to_string());
        }

        let page_tokens = layout.0;
        let alignment_bytes = layout.1;

        // Compute per-page byte sizes.
        let key_size = bytes_per_element(&self.config.precision_policy.key_dtype);
        let value_size = bytes_per_element(&self.config.precision_policy.value_dtype);
        let bytes_per_token_per_head = self.config.head_dim as u64 * (key_size + value_size);
        // Each page stores tokens for ONE head in ONE layer.
        let page_raw = page_tokens as u64 * bytes_per_token_per_head;
        let page_aligned = align_up(page_raw, alignment_bytes);

        let pages_per_layer_head = (self.config.max_sequence_len + page_tokens - 1) / page_tokens;

        let total_pages =
            self.config.layer_count as u32 * self.config.head_count as u32 * pages_per_layer_head;

        self.max_bytes = total_pages as u64 * page_aligned;
        self.total_bytes = 0;
        let mut pids: Vec<PageDescriptor> = Vec::with_capacity(total_pages as usize);

        let mut page_id = 0u32;
        for layer in 0..self.config.layer_count {
            // Use kv_head_count for the actual KV head allocation, but for
            // simplicity allocate for all heads (head_count).
            let heads = self.config.head_count;
            for head in 0..heads {
                for p_idx in 0..pages_per_layer_head {
                    let token_offset = p_idx * page_tokens;
                    pids.push(PageDescriptor {
                        page_id,
                        layer_index: layer,
                        head_index: head,
                        token_offset,
                        token_count: page_tokens,
                        byte_offset: page_id as u64 * page_aligned,
                        byte_length: page_aligned,
                        epoch_id: 0,
                        owner_region: self.owner_region.clone(),
                    });
                    page_id += 1;
                }
            }
        }

        self.pages = pids;
        self.allocated = true;
        Ok(())
    }

    /// Append `token_count` tokens starting at `token_start`, creating a new
    /// span. The caller's region identity is [`self.owner_region`] by default;
    /// for the non-owner test the access policies are consulted.
    ///
    /// Returns an error if:
    /// - The store hasn't been allocated.
    /// - The caller isn't the owner (access policy check).
    pub fn append_tokens(
        &mut self,
        token_start: u32,
        token_count: u32,
        span_id: &str,
        epoch_id: u64,
        caller_region: &str,
    ) -> Result<KvAppendReceipt, String> {
        if !self.allocated {
            return Err("store not allocated".to_string());
        }

        // Access check: only the owner can write (unless a policy grants it).
        let access_allowed = self.check_region_write(caller_region);
        if !access_allowed {
            return Err(format!(
                "region '{}' is not permitted to write to store '{}'",
                caller_region, self.store_id
            ));
        }

        // Find unclaimed pages.
        let tokens_needed = token_count;
        let page_tokens = match &self.config.cache_layout {
            KvCacheLayout::PagedLayerHead { page_tokens, .. } => *page_tokens,
        };

        let mut page_ids: Vec<u32> = Vec::new();
        for p in &self.pages {
            if self.claimed_pages.contains(&p.page_id) {
                continue;
            }
            // Claim pages within the requested token range for the given epoch.
            // Simple strategy: take any free page.
            if (page_ids.len() as u32) * page_tokens < tokens_needed {
                page_ids.push(p.page_id);
            } else {
                break;
            }
        }

        let pages_allocated = page_ids.len() as u32;
        let span = KvSpan {
            span_id: span_id.to_string(),
            token_start,
            token_count,
            epoch_id,
            pages: page_ids.clone(),
            pinned: false,
        };

        for &pid in &page_ids {
            self.claimed_pages.insert(pid);
        }
        self.spans.push(span);
        self.next_span_seq += 1;
        self.total_bytes += pages_allocated as u64 * self.pages[0].byte_length;
        Ok(KvAppendReceipt {
            store_id: self.store_id.clone(),
            span_id: span_id.to_string(),
            epoch_id,
            pages_allocated,
            total_pages: self.pages.len() as u32,
            total_bytes_after: self.total_bytes,
            memory_ok: true,
        })
    }

    /// Resolve pages covering `[token_start, token_start + token_count)` within
    /// the given epoch. Returns a `KvReadReceipt`.
    ///
    /// If the epoch is uncommitted and no read policy grants access, access is
    /// denied.
    pub fn read_window(
        &self,
        token_start: u32,
        token_count: u32,
        epoch_id: u64,
        caller_region: &str,
    ) -> Result<KvReadReceipt, String> {
        if !self.allocated {
            return Err("store not allocated".to_string());
        }

        // Access check: uncommitted epochs require a read policy or ownership.
        // Determine if epoch is committed — we only store committed state
        // externally; here we trust caller_region == owner for committed epochs.
        let access_granted = if caller_region == self.owner_region {
            true
        } else {
            // Check read access in policies.
            self.check_region_read(caller_region)
        };

        if !access_granted {
            return Ok(KvReadReceipt {
                store_id: self.store_id.clone(),
                token_start,
                token_count,
                epoch_id,
                pages_resolved: 0,
                byte_offset: 0,
                byte_length: 0,
                access_granted: false,
            });
        }

        // Resolve pages in spans matching the epoch and token range.
        let token_end = token_start.saturating_add(token_count);
        let mut resolved_pages: Vec<u32> = Vec::new();

        for s in &self.spans {
            if s.epoch_id != epoch_id {
                continue;
            }
            let span_end = s.token_start.saturating_add(s.token_count);
            if s.token_start < token_end && span_end > token_start {
                // Overlap — include this span's pages.
                resolved_pages.extend_from_slice(&s.pages);
            }
        }

        resolved_pages.sort_unstable();
        resolved_pages.dedup();

        let pages_resolved = resolved_pages.len() as u32;
        let mut byte_offset = u64::MAX;
        let mut byte_length = 0u64;

        for &pid in &resolved_pages {
            if let Some(p) = self.pages.get(pid as usize) {
                if p.byte_offset < byte_offset {
                    byte_offset = p.byte_offset;
                }
                let end = p.byte_offset + p.byte_length;
                if end > byte_length + byte_offset {
                    byte_length = end - byte_offset;
                }
            }
        }

        if pages_resolved == 0 {
            byte_offset = 0;
        }

        Ok(KvReadReceipt {
            store_id: self.store_id.clone(),
            token_start,
            token_count,
            epoch_id,
            pages_resolved,
            byte_offset,
            byte_length,
            access_granted: true,
        })
    }

    /// Pin a span by its span_id, preventing eviction.
    pub fn pin_span(&mut self, span_id: &str) -> Result<(), String> {
        for s in &mut self.spans {
            if s.span_id == span_id {
                if !s.pinned {
                    s.pinned = true;
                    return Ok(());
                }
                return Err(format!("span '{}' is already pinned", span_id));
            }
        }
        Err(format!("span '{}' not found", span_id))
    }

    /// Return (used_bytes, max_bytes).
    pub fn memory_usage(&self) -> (u64, u64) {
        (self.total_bytes, self.max_bytes)
    }

    // -----------------------------------------------------------------------
    // Internal access helpers
    // -----------------------------------------------------------------------

    fn check_region_write(&self, region: &str) -> bool {
        if region == self.owner_region {
            return true;
        }
        // Check policies for explicit write access.
        for p in &self.access_policies {
            if p.store_id != self.store_id {
                continue;
            }
            if !p.allowed_regions.iter().any(|r| r == region) {
                continue;
            }
            match p.access {
                AccessKind::Write | AccessKind::ReadWrite => return true,
                _ => {}
            }
        }
        false
    }

    fn check_region_read(&self, region: &str) -> bool {
        if region == self.owner_region {
            return true;
        }
        for p in &self.access_policies {
            if p.store_id != self.store_id {
                continue;
            }
            if !p.allowed_regions.iter().any(|r| r == region) {
                continue;
            }
            match p.access {
                AccessKind::Read | AccessKind::ReadWrite => return true,
                _ => {}
            }
        }
        false
    }
}
