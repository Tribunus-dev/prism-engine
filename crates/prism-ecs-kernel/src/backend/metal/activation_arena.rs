//! Activation arena (constitutional home, Metal FFI).
//!
//! Per the inventory v2.1, this is the FFI half of activation
//! allocation. The state half (ActivationTransaction) is in
//! `state::activation_transaction`; the FFI half is here.

pub struct ActivationArena {
    capacity: u64,
    allocated: u64,
}

impl ActivationArena {
    pub fn new(capacity: u64) -> Self {
        Self { capacity, allocated: 0 }
    }

    pub fn allocated(&self) -> u64 {
        self.allocated
    }

    pub fn set_allocated(&mut self, value: u64) {
        self.allocated = value.min(self.capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_arena_has_zero_allocated() {
        let a = ActivationArena::new(1024);
        assert_eq!(a.allocated(), 0);
    }

    #[test]
    fn set_allocated_caps_at_capacity() {
        let mut a = ActivationArena::new(100);
        a.set_allocated(200);
        assert_eq!(a.allocated(), 100);
    }
}
