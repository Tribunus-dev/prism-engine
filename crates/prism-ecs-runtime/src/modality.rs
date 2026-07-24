use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

/// Provider-neutral modality work carried by the canonical ECS world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModalityKind {
    Image,
    Audio,
    Video,
    Metal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityWork {
    pub kind: ModalityKind,
    pub model_path: String,
    pub prompt: String,
    pub output_path: String,
}

impl Component for ModalityWork {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityExecution {
    pub output_digest: String,
    pub output_bytes: u64,
}

impl Component for ModalityExecution {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalityFailure {
    pub error: String,
}

impl Component for ModalityFailure {}
