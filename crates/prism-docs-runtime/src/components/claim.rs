//! Claim components.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimText(pub String);
impl Component for ClaimText {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimClassComponent(pub String);
impl Component for ClaimClassComponent {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KnowledgeStateComponent(pub String);
impl Component for KnowledgeStateComponent {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExistenceStateComponent(pub String);
impl Component for ExistenceStateComponent {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimSourceRefs(pub Vec<String>);
impl Component for ClaimSourceRefs {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClaimFramedBy(pub String);
impl Component for ClaimFramedBy {}
