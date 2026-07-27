//! Agent bridge system (constitutional home, runtime half).
//!
//! The agent bridge in the engine is split during absorption:
//! - Runtime half (this file): the bridge logic that produces
//!   scheduling decisions for agent dispatch.
//! - Kernel half (ANE FFI): moves to `prism-ecs-kernel::backend::ane::agent_bridge`
//!   in step 50.
//!
//! The engine's `agent_bridge.rs` is the legacy duplicate; step 50
//! deletes it.

/// Constitutional-side agent bridge (runtime half).
pub struct AgentBridge {
    _placeholder: (),
}

impl AgentBridge {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for AgentBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_constructs() {
        let _ = AgentBridge::new();
    }
}
