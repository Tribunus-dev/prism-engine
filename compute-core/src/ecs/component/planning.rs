use crate::ecs::Component;
use serde::{Deserialize, Serialize};

// ── ProfileRunResult ───────────────────────────────────────────────────────

/// ECS component wrapping a profile run result from execution_profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRunResult(pub crate::ecs::execution_profile::ProfileRunResult);
impl Component for ProfileRunResult {}

// ── AcceptanceGates ────────────────────────────────────────────────────────

/// ECS component wrapping model equivalence acceptance gates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptanceGates(pub crate::ecs::plan::vectors::AcceptanceGates);
impl Component for AcceptanceGates {}

// ── ModelReferenceVector ───────────────────────────────────────────────────

/// ECS component wrapping a model reference vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelReferenceVector(pub crate::ecs::plan::vectors::ModelReferenceVector);
impl Component for ModelReferenceVector {}

// ── DriftStatus ────────────────────────────────────────────────────────────

/// ECS component wrapping drift status from model equivalence checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftStatus(pub crate::ecs::plan::vectors::DriftStatus);
impl Component for DriftStatus {}

// ── ModelExecutionPlanComp ──────────────────────────────────────────────────

/// ECS component wrapping a ModelExecutionPlan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelExecutionPlanComp(pub crate::ecs::plan::ModelExecutionPlan);
impl Component for ModelExecutionPlanComp {}

// ── ProfileRunConfigComp ────────────────────────────────────────────────────

/// ECS component wrapping a ProfileRunConfig.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRunConfigComp(pub crate::ecs::execution_profile::ProfileRunConfig);
impl Component for ProfileRunConfigComp {}
