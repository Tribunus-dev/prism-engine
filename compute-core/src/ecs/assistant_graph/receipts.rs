use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantGraphValidationStatus {
    Valid,
    ValidWithWarnings,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantGraphValidationReceipt {
    pub graph_id: String,
    pub contract_valid: bool,
    pub region_count: u32,
    pub bridge_count: u32,
    pub route_edges: u32,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub validation_status: AssistantGraphValidationStatus,
}
