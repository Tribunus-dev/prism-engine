//! Constitutional agent surface — assistant graph manifest types and
//! a small in-memory `AgentStore` for tracking agent runs.
//!
//! The `assistant_graph` module is the canonical home for the manifest
//! types (regions, bridges, route graphs, authority policies, shared
//! state schema) and the structural validator that admits a manifest
//! as a valid graph identity. The engine's `compute-core/src/ecs/
//! assistant_graph/` parallel surface has been absorbed here.

pub mod assistant_graph;

use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentState {
    Pending,
    Running,
    Succeeded,
    Failed,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentRecord {
    pub id: String,
    pub task: String,
    pub state: AgentState,
    pub error: Option<String>,
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AgentStore {
    pub records: Vec<AgentRecord>,
}
impl AgentStore {
    pub fn upsert(&mut self, r: AgentRecord) {
        if let Some(x) = self.records.iter_mut().find(|x| x.id == r.id) {
            *x = r
        } else {
            self.records.push(r)
        }
    }
    pub fn get(&self, id: &str) -> Option<&AgentRecord> {
        self.records.iter().find(|x| x.id == id)
    }
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
