//! ANE agent bridge (constitutional home, kernel half).
//!
//! Per the inventory v2.1, the engine's `agent_bridge.rs` is
//! split: runtime half → `state::systems::agent_bridge` (already
//! moved); ANE FFI half → this file.

pub struct AneAgentBridge {
    _placeholder: (),
}

impl AneAgentBridge {
    pub fn new() -> Self {
        Self { _placeholder: () }
    }
}

impl Default for AneAgentBridge {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ane_agent_bridge_constructs() {
        let _ = AneAgentBridge::new();
    }
}
