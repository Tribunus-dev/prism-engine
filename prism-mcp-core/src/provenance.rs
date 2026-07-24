use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProvenanceDomain {
    Model,
    Compilation,
    Evidence,
    Runtime,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProvenanceKind {
    Authoritative,
    Derived,
    Ambiguous,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceNode {
    pub id: String,
    pub domain: ProvenanceDomain,
    pub kind: ProvenanceKind,
    pub namespace: String,
    pub external_id: String,
    pub label: Option<String>,
    pub attributes: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProvenanceQuery {
    pub root: String,
    pub depth: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProvenanceSubgraph {
    pub nodes: Vec<ProvenanceNode>,
    pub edges: Vec<ProvenanceEdge>,
}
pub trait ProvenanceGraphStore: Send + Sync {
    fn upsert_node(&self, node: &ProvenanceNode) -> Result<()>;
    fn add_edge(&self, edge: &ProvenanceEdge) -> Result<()> {
        let _ = edge;
        Ok(())
    }
    fn query(&self, _query: &ProvenanceQuery) -> Result<ProvenanceSubgraph> {
        Ok(ProvenanceSubgraph::default())
    }
}
