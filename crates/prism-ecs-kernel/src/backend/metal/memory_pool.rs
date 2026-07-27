//! Memory pool (constitutional home, Metal FFI).
//!
//! Per the inventory v2.1 row 25, this replaces the engine's
//! `memory_pool.rs` (195 LOC). Placeholder.

pub struct MemoryPoolAllocator {
    _placeholder: (),
}

impl MemoryPoolAllocator {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for MemoryPoolAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_pool_constructs() {
        let _ = MemoryPoolAllocator::new();
    }
}
