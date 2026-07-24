use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphNode {
    pub id: String,
    pub label: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GraphEvidence {
    pub source: String,
    pub detail: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphOperation {
    Expand,
    Explain,
    Trace,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum GraphAuthority {
    Local,
    Federated,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GraphProjection {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TraversalResult {
    pub projection: GraphProjection,
    pub evidence: Vec<GraphEvidence>,
}
pub trait FederatedGraphQuery: Send + Sync {
    fn query(&self, _operation: GraphOperation, _root: &str) -> Result<TraversalResult>;
}
