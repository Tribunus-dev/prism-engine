//! Legacy adapter (constitutional home).
//!
//! Per the inventory v2.1 row 24, this replaces the engine's
//! `legacy_adapter.rs` (241 LOC). Placeholder.

pub struct LegacyAdapter {
    _placeholder: (),
}

impl LegacyAdapter {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for LegacyAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_adapter_constructs() {
        let _ = LegacyAdapter::new();
    }
}
