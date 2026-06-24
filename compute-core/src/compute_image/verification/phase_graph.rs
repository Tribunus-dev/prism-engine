use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraphVerificationReceipt {
    pub artifact_identity: String,
    pub phase_count: u32,
    pub edge_count: u32,
    pub graph_valid: bool,
    pub graph_hash: ContentHash,
}
