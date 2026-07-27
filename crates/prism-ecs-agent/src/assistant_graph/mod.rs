//! This module is the canonical authority for the assistant graph manifest
//! surface — regions, bridges, route graphs, authority policies, shared
//! state schema, and the structural validator that admits a manifest as
//! a valid graph identity.

pub mod authority;
pub mod bridge;
pub mod graph;
pub mod manifest;
pub mod receipts;
pub mod state;
pub mod validate;

#[cfg(test)]
mod tests;

pub use authority::{AuthorityRule, AuthorityRuleKind, RegionAuthorityPolicy};
pub use bridge::{
    AssistantResponseState, BridgeDecl, BridgeValueType, EmotionalRegister, EmphasisSpan,
    PacingPlan, PronunciationHint, SpeakingStyle, SpeechPlan, SpeechUtterance, TurnIntent,
};
pub use graph::{AssistantRouteGraph, RouteEdge, RouteKind};
pub use manifest::{
    AssistantContract, AssistantGraphManifest, AssistantRegionDecl, AssistantRegionKind,
    RegionOutputAuthority,
};
pub use receipts::{AssistantGraphValidationReceipt, AssistantGraphValidationStatus};
pub use state::{
    RegionStateAccess, SharedStateSchema, StatePersistence, StateStoreDecl, StateStoreKind,
};
pub use validate::AssistantGraphValidator;
