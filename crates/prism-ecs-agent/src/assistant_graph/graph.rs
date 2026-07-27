//! This module owns the canonical data structure for the route graph that
//! connects regions through bridges, with sequential, parallel, and
//! conditional kinds.

use serde::{Deserialize, Serialize};

use super::bridge::BridgeValueType;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantRouteGraph {
    pub edges: Vec<RouteEdge>,
    pub route_kind: RouteKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteEdge {
    pub from_region: String,
    pub to_region: String,
    pub allowed_types: Vec<BridgeValueType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteKind {
    Sequential,
    Parallel,
    Conditional,
}
