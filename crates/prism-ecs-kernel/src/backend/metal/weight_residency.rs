//! Weight residency (constitutional home).
//!
//! Per the inventory v2.1 row 54, this replaces the engine's
//! `weight_residency.rs` (130 LOC). Placeholder.

pub struct WeightResidency {
    _placeholder: (),
}

impl WeightResidency {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for WeightResidency {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_residency_constructs() {
        let _ = WeightResidency::new();
    }
}
