//! Demo components — the apple silicon demo workflow.
//!
//! Gates are typed entities. Bands are typed entities. The
//! hydration JS reads their typed data.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoGateId(pub String);
impl Component for DemoGateId {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoGateNum(pub String);
impl Component for DemoGateNum {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoGateTitle(pub String);
impl Component for DemoGateTitle {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoGateBody(pub String);
impl Component for DemoGateBody {}

/// Order of the gate (1..4). Used to determine active state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoGateOrder(pub u32);
impl Component for DemoGateOrder {}

/// A milestone band. Status: `ready`, `active`, `gated`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoBandTitle(pub String);
impl Component for DemoBandTitle {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoBandBody(pub String);
impl Component for DemoBandBody {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DemoBandStatus(pub String);
impl Component for DemoBandStatus {}
