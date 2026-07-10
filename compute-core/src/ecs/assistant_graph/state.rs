use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionStateAccess {
    pub read_stores: Vec<String>,
    pub write_stores: Vec<String>,
    pub requires_epoch_check: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedStateSchema {
    pub stores: Vec<StateStoreDecl>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateStoreDecl {
    pub store_id: String,
    pub store_kind: StateStoreKind,
    pub owner_region: String,
    pub dtype: String,
    pub max_bytes: u64,
    pub persistence: StatePersistence,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateStoreKind {
    TextKvCache,
    VisionEmbeddingCache,
    SpeechSemanticCache,
    SpeechAcousticCache,
    RetrievalWorkingSet,
    AssistantResponseState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatePersistence {
    EpochOnly,
    Session,
    Permanent,
}
