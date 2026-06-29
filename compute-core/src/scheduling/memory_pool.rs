use serde::{Deserialize, Serialize};
use half::f16;

// ── Token classification ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenClass {
    /// 4-bit TurboQuant, protected from eviction.
    Reasoning,
    /// 1.58-bit Ternary, evictable after 2K steps.
    Execution,
    /// 2-bit Packed, evictable after 512 steps.
    Transition,
}

impl TokenClass {
    /// Bytes per token for this class's KV cache storage.
    pub fn bytes_per_token(&self) -> u64 {
        match self {
            TokenClass::Reasoning => 98304,   // 4-bit: 48×8×512×0.5
            TokenClass::Execution => 49152,   // 1.58-bit: 48×8×512×0.25
            TokenClass::Transition => 49152,  // 2-bit: 48×8×512×0.25
        }
    }
    /// Predictive delta buffer bytes per token (constant 4-bit across all classes).
    pub fn delta_bytes_per_token() -> u64 {
        98304  // 48×8×512×0.5
    }
}

// ── Virtual memory page ────────────────────────────────────────────

/// One physical IOSurface page capable of holding up to 20K tokens.
#[derive(Debug, Clone)]
pub struct VirtualPage {
    pub surface_id: usize,
    pub active_tokens: usize,
}

/// Descriptor mapping a page range to a slot's allocation.
#[derive(Debug, Clone)]
pub struct PageDescriptor {
    pub surface_index: usize,
    pub token_start: usize,
    pub token_count: usize,
}

/// A single slot's allocation context.
#[derive(Debug, Clone)]
pub struct SlotContext {
    pub slot_id: usize,
    pub virtual_pages: Vec<VirtualPage>,
    pub token_classes: Vec<TokenClass>,
    pub total_context_length: usize,
}

// ── Memory pool allocator ──────────────────────────────────────────

/// Manages the global pool of 8 IOSurface pages (each 20K tokens = 160K total)
/// and provides token-stealing allocation across active slots.
pub struct MemoryPoolAllocator {
    pub max_vram_bytes: u64,
    pub fixed_overhead_bytes: u64,
    /// Track allocation status of the 8 physical IOSurfaces.
    pub global_page_pool: Vec<bool>,
    pub active_slots: Vec<SlotContext>,
}

impl MemoryPoolAllocator {
    pub fn new(max_vram_bytes: u64, fixed_overhead_bytes: u64) -> Self {
        Self {
            max_vram_bytes,
            fixed_overhead_bytes,
            global_page_pool: vec![false; 8],
            active_slots: Vec::new(),
        }
    }

    /// Evaluate whether the active processing window fits within hardware VRAM.
    /// Returns total used bytes on success, or an OOM error string.
    pub fn verify_memory_budget(&self) -> Result<u64, String> {
        let mut total_used = self.fixed_overhead_bytes;
        for slot in &self.active_slots {
            for class in &slot.token_classes {
                total_used += class.bytes_per_token() + TokenClass::delta_bytes_per_token();
            }
        }
        if total_used > self.max_vram_bytes {
            Err(format!(
                "OOM Risk: required {} bytes exceeds system capacity {}",
                total_used, self.max_vram_bytes
            ))
        } else {
            Ok(total_used)
        }
    }

    /// Orchestrate global page redistribution using token-stealing logic.
    /// Grants requested_tokens to slot_id by claiming free pages or stealing
    /// from low-entropy execution contexts.
    pub fn resolve_pool_allocation(
        &mut self,
        slot_id: usize,
        requested_tokens: usize,
    ) -> Result<(), String> {
        const TOKENS_PER_PAGE: usize = 20_000;
        let required_pages = (requested_tokens + TOKENS_PER_PAGE - 1) / TOKENS_PER_PAGE;
        let mut allocated = 0usize;
        let mut mapped_pages: Vec<VirtualPage> = Vec::new();

        // 1. Claim free physical pages.
        for (idx, busy) in self.global_page_pool.iter_mut().enumerate() {
            if !*busy {
                *busy = true;
                mapped_pages.push(VirtualPage { surface_id: idx, active_tokens: 0 });
                allocated += 1;
                if allocated >= required_pages { break; }
            }
        }

        // 2. Token-stealing: harvest from low-entropy execution contexts.
        if allocated < required_pages {
            // Sort active slots by page count (steal from those with most).
            let steal_candidates: Vec<usize> = self.active_slots.iter().enumerate()
                .filter(|(_, s)| s.slot_id != slot_id && s.virtual_pages.len() > 1)
                .map(|(i, _)| i)
                .collect();
            for idx in steal_candidates {
                if let Some(stolen) = self.active_slots[idx].virtual_pages.pop() {
                    mapped_pages.push(VirtualPage {
                        surface_id: stolen.surface_id,
                        active_tokens: 0,
                    });
                    allocated += 1;
                    if allocated >= required_pages { break; }
                }
            }
        }

        if allocated < required_pages {
            return Err("System Saturation: insufficient pages in global pool".to_string());
        }

        // Assign pages to the target slot.
        if let Some(slot) = self.active_slots.iter_mut().find(|s| s.slot_id == slot_id) {
            slot.virtual_pages.extend(mapped_pages);
        }
        Ok(())
    }

    /// Compact a slot's KV cache by selecting which positions to keep
    /// based on their entropy scores.  Returns the selected positions.
    pub fn compact_slot(
        &self,
        slot_id: usize,
        entropy_map: &[f16],
        target_count: usize,
    ) -> Vec<u32> {
        let _slot = match self.active_slots.iter().find(|s| s.slot_id == slot_id) {
            Some(s) => s,
            None => return vec![],
        };
        // Use entropy-guided compaction
        crate::compute_image::compaction::select_entropy_compaction_positions(
            entropy_map,
            target_count,
        )
    }

    /// Classify a token position into a TokenClass based on its entropy.
    /// High entropy (>0.7) → Reasoning (protected)
    /// Medium entropy (0.3-0.7) → Execution (evictable after 2K)
    /// Low entropy (<0.3) → Transition (evictable after 512)
    pub fn classify_token(entropy: f32) -> TokenClass {
        if entropy > 0.7 {
            TokenClass::Reasoning
        } else if entropy > 0.3 {
            TokenClass::Execution
        } else {
            TokenClass::Transition
    }
    }
}
