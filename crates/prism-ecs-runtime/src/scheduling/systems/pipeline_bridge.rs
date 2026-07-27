//! Pipeline bridge system (constitutional home).
//!
//! Placeholder for the engine's pipeline_bridge.rs. The engine file
//! is the legacy duplicate and is deleted in step 58. The full
//! implementation is added when the engine's pipeline types migrate.

/// Constitutional-side pipeline bridge.
pub struct PipelineBridge {
    _placeholder: (),
}

impl PipelineBridge {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for PipelineBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_constructs() {
        let _ = PipelineBridge::new();
    }
}
