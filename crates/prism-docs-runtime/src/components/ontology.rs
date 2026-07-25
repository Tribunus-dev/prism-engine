//! Re-export the content ontology enums as components. We keep the
//! content crate's enums as the canonical authority and wrap them in
//! thin newtypes so they can act as components.

use prism_docs_content::{ClaimClass, ExistenceState, KnowledgeState};
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimClassC(pub ClaimClass);
impl Component for ClaimClassC {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KnowledgeStateC(pub KnowledgeState);
impl Component for KnowledgeStateC {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExistenceStateC(pub ExistenceState);
impl Component for ExistenceStateC {}
