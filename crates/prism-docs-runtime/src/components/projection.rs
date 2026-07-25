//! Projection-repro components — the canonical subject's
//! 3D rebuilder surface.
//!
//! The subject is a typed component. The stages are typed
//! components. The hydration JS reads them and renders a
//! deterministic SVG projection.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionSubjectId(pub String);
impl Component for ProjectionSubjectId {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionSubjectName(pub String);
impl Component for ProjectionSubjectName {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionSubjectKind(pub String);
impl Component for ProjectionSubjectKind {}

/// The layers of the canonical subject: each layer is one
/// physical or logical stratum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionLayer {
    pub id: String,
    pub name: String,
    pub depth: u8,
    pub color: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionLayers(pub Vec<ProjectionLayer>);
impl Component for ProjectionLayers {}

/// A stage in the rebuild sequence. Stages: identity, search,
/// admission, seal, run, replay.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionStageId(pub String);
impl Component for ProjectionStageId {}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionStageLabel(pub String);
impl Component for ProjectionStageLabel {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionStageOrder(pub u32);
impl Component for ProjectionStageOrder {}
