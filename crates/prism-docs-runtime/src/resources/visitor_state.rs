//! `VisitorState` resource — runtime-only state about the visitor.
//!
//! This is a transient resource: the SSG never has a visitor, so it
//! inserts a default. The WASM hydration reads it from `localStorage`
//! and updates it on interaction.

use prism_docs_content::{OpticalState, ObserverMode};
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisitorState {
    pub observer_mode: ObserverMode,
    pub optical_state: OpticalState,
    pub visited_routes: Vec<String>,
    pub selected_claim: Option<String>,
}

impl Default for VisitorState {
    fn default() -> Self {
        Self {
            observer_mode: ObserverMode::Observer,
            optical_state: OpticalState::Observation,
            visited_routes: Vec::new(),
            selected_claim: None,
        }
    }
}

impl Component for VisitorState {}
