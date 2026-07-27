//! Activation binding (constitutional home, Metal FFI).
//!
//! Per the inventory v2.1, this is the FFI half of activation
//! binding. The state half is in the runtime; the FFI half is here.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaBinding {
    pub offset: u64,
    pub size: u64,
}

impl ArenaBinding {
    pub fn new(offset: u64, size: u64) -> Self {
        Self { offset, size }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_carries_offset_and_size() {
        let b = ArenaBinding::new(16, 1024);
        assert_eq!(b.offset, 16);
        assert_eq!(b.size, 1024);
    }
}
