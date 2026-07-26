use serde::{Deserialize, Serialize};

use crate::ecs::assistant_graph::authority::RegionAuthorityPolicy;
use crate::ecs::assistant_graph::bridge::BridgeDecl;
use crate::ecs::assistant_graph::bridge::BridgeValueType;
use crate::ecs::assistant_graph::graph::AssistantRouteGraph;
use crate::ecs::assistant_graph::state::{RegionStateAccess, SharedStateSchema};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantGraphManifest {
    pub graph_id: String,
    pub schema_version: u32,
    pub assistant_contract: AssistantContract,
    pub regions: Vec<AssistantRegionDecl>,
    pub bridges: Vec<BridgeDecl>,
    pub shared_state_schema: SharedStateSchema,
    pub route_graph: AssistantRouteGraph,
    pub authority_policy: RegionAuthorityPolicy,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantContract {
    pub contract_id: String,
    pub max_active_regions: u32,
    pub requires_bridge_types: bool,
    pub requires_authority: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantRegionDecl {
    pub region_id: String,
    pub partition_id: Option<String>,
    pub region_kind: AssistantRegionKind,
    pub input_types: Vec<BridgeValueType>,
    pub output_types: Vec<BridgeValueType>,
    pub state_access: RegionStateAccess,
    pub authority: Vec<RegionOutputAuthority>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantRegionKind {
    ReasoningDecode,
    VisionPerception,
    EmbeddingRetrieval,
    SpeechSynthesis,
    ToolDecision,
    DecoderLayerProof,
    SyntheticStub,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionOutputAuthority {
    PerceptionFacts,
    EmbeddingVectors,
    RetrievalCandidates,
    DraftText,
    ToolCallProposal,
    ToolCallDecision,
    CommittedAssistantResponse,
    SpeechPlan,
    AudioFrames,
    KvCacheUpdate,
}
