use serde::{Deserialize, Serialize};

/// Descriptor for a single page in a paged KV cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDescriptor {
    pub page_id: u32,
    pub layer_index: u32,
    pub head_index: u32,
    pub token_offset: u32,
    pub token_count: u32,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub epoch_id: u64,
    pub owner_region: String,
}

/// Table of pages with uniform size/capacity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageTable {
    pub pages: Vec<PageDescriptor>,
    pub page_size_bytes: u64,
    pub page_capacity_tokens: u32,
}

impl PageTable {
    /// Create a new PageTable with the given pages.
    pub fn new(
        pages: Vec<PageDescriptor>,
        page_size_bytes: u64,
        page_capacity_tokens: u32,
    ) -> Self {
        Self {
            pages,
            page_size_bytes,
            page_capacity_tokens,
        }
    }

    /// Look up a page by its page_id.
    pub fn get(&self, page_id: u32) -> Option<&PageDescriptor> {
        self.pages.get(page_id as usize)
    }

    /// Return the number of pages.
    pub fn len(&self) -> usize {
        self.pages.len()
    }

    /// True when there are no pages.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }
}
