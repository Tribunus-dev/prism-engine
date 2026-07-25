//! ADR components.

use prism_docs_content::AdrStatus;
use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrTitle(pub String);
impl Component for AdrTitle {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrSlug(pub String);
impl Component for AdrSlug {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrNumber(pub u32);
impl Component for AdrNumber {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrStatusComponent(pub AdrStatus);
impl Component for AdrStatusComponent {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrContext(pub String);
impl Component for AdrContext {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrDecision(pub String);
impl Component for AdrDecision {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrConsequences(pub String);
impl Component for AdrConsequences {}

/// Optional pointer to the ADR this one supersedes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrSupersedes(pub String);
impl Component for AdrSupersedes {}

/// Path to the ADR's markdown body, relative to the content root.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AdrBodyPath(pub String);
impl Component for AdrBodyPath {}
