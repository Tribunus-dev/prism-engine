//! Observer / optical state components — visitor-side state.

use prism_docs_content::{OpticalState, ObserverMode};
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObserverModeC(pub ObserverMode);
impl Component for ObserverModeC {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpticalStateC(pub OpticalState);
impl Component for OpticalStateC {}
