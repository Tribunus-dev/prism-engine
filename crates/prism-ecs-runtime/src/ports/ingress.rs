//! Ingress port (constitutional home).
//!
//! Per the inventory v2.1 step 35, the engine's `ingress_bridge.rs`
//! is moved to `prism_ecs_runtime::ports::ingress`. Ingress is NOT
//! a hardware backend; it is the runtime's port for "submissions
//! from outside the system" (HTTP requests, message queue
//! items, agent task submissions, etc.).
//!
//! Placeholder: the engine's full IngressBridge (with the
//! ingress queue and routing logic) migrates when its
//! runtime-side callers are updated.

/// Placeholder for the engine's `IngressBridge` struct.
#[derive(Debug, Default)]
pub struct IngressBridge {
    _placeholder: (),
}

impl IngressBridge {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingress_bridge_constructs() {
        let _ = IngressBridge::new();
    }
}
