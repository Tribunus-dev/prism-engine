//! Metal decoder (constitutional home).
//!
//! Per the inventory v2.1 row 26, this replaces the engine's
//! `metal_decoder.rs` (157 LOC). Placeholder.

pub struct MetalDecoder {
    _placeholder: (),
}

impl MetalDecoder {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for MetalDecoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_decoder_constructs() {
        let _ = MetalDecoder::new();
    }
}
