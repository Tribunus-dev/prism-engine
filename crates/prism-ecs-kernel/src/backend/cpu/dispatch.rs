//! CPU dispatch (constitutional home).
//!
//! Replaces the legacy `cpu_backend.rs` (1,894 LOC) and the
//! `metal_dispatch.rs` (1,821 LOC). The full implementation
//! arrives with the engine migration.

pub struct CpuDispatch {
    _placeholder: (),
}

impl CpuDispatch {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for CpuDispatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_dispatch_constructs() {
        let _ = CpuDispatch::new();
    }
}
