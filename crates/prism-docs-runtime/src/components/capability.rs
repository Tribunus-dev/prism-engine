//! Capability components — the filterable grid on the
//! capabilities page.
//!
//! Each capability is a typed card. The class and state come
//! from the typed components; the filter UI reads them.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityId(pub String);
impl Component for CapabilityId {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityTitle(pub String);
impl Component for CapabilityTitle {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityDomain(pub String);
impl Component for CapabilityDomain {}

/// State: one of `Measured`, `Verified`, `Compile-verified`,
/// `Planned`, `Illustrative`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityState(pub String);
impl Component for CapabilityState {}

/// Class: one of `Measured`, `Architectural`, `Repository`,
/// `Compile-verified`, `Illustrative`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityClass(pub String);
impl Component for CapabilityClass {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityBody(pub String);
impl Component for CapabilityBody {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilitySourcePath(pub String);
impl Component for CapabilitySourcePath {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityLimitation(pub String);
impl Component for CapabilityLimitation {}
