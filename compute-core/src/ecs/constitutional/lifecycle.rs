use crate::ecs::constitutional::types::*;
use serde::{Deserialize, Serialize};

// ── Session Lifecycle (ownership & resource lifecycle) ────────────────────
//
// Orthogonal to InferencePhase. A session can be Active while in ToolWait.

/// Session lifecycle state — governs session ownership and resource allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SessionLifecycle {
    Created,
    Admitted,
    Active,
    Quiescing,
    Saving,
    Completed,
    Failed,
    Releasing,
    Released,
}

impl SessionLifecycle {
    /// Returns true if this state is terminal (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Released)
    }

    /// Returns true if cleanup effects are in progress.
    pub fn is_releasing(&self) -> bool {
        matches!(self, Self::Releasing | Self::Quiescing)
    }
}

// ── Inference Phase (current computational activity) ───────────────────────
//
// Orthogonal to SessionLifecycle. A session transitions through these
// within its current lifecycle state.

/// Current inference phase — the computational activity a session is performing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferencePhase {
    AwaitingInput,
    Prefill,
    Decode,
    ToolWait,
    Compaction,
    OutputFinalization,
}

impl InferencePhase {
    /// Returns true if this phase produces output tokens.
    pub fn is_generating(&self) -> bool {
        matches!(self, Self::Prefill | Self::Decode)
    }
}

// ── Two-Phase Teardown (generic entity teardown) ───────────────────────────
//
// Active → Quiescing (prevent new work) → Releasing (emit cleanup effects)
// → Released (all cleanup resolved). Tombstoning and slot reclamation are
// separate operations after Released.

/// Generic two-phase teardown state for any entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TeardownState {
    Active,
    Quiescing,
    Releasing,
    Released,
}

impl TeardownState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Released)
    }
}

// ── Entity Lifecycle States ────────────────────────────────────────────────

/// Lifecycle for an artifact entity (model weights, compiled graphs, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArtifactLifecycle {
    Discovered,
    Validated,
    Loaded,
    Invalid,
}

impl ArtifactLifecycle {
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Validated | Self::Loaded)
    }
}

/// Lifecycle for a device entity (Metal GPU, ANE, remote worker, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceLifecycle {
    Discovered,
    Initializing,
    Ready,
    Degraded,
    Unavailable,
    Removed,
}

impl DeviceLifecycle {
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Ready)
    }
}

/// Lifecycle for a residency relationship (model segment on a device).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidencyLifecycle {
    Desired,
    Binding,
    Resident,
    Evicting,
    Evicted,
}

impl ResidencyLifecycle {
    pub fn is_resident(&self) -> bool {
        matches!(self, Self::Resident)
    }
}

// ── Typed Relationship Components ──────────────────────────────────────────
//
// Replace Vec<CompEntity> stored on parent entities with typed components.
// A maintained reverse-index gives the parent's children without mutable vectors.

/// Declares that a session entity uses a model entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionUsesModel {
    pub session_id: u64,
    pub model_id: u64,
}

/// Declares that a residency entity targets a device entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ResidencyTargets {
    pub residency_id: u64,
    pub device_id: u64,
}

/// Declares that an entity has a parent entity (for two-phase teardown).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Parent {
    pub parent_id: u64,
}

// ── SessionCheckpoint ─────────────────────────────────────────────────────
//
// Metadata component: pointer + digest to blob-stored checkpoint data.
// The actual KV bytes belong in a checkpoint arena, mapped file, or object store.

/// Opaque handle to a storage location (arena, mapped file, blob store).
/// Actual storage backend wired in Stage 6 (persistence).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StorageHandle(pub String);

/// Checkpoint metadata attached to a session entity.
/// Payload bytes live in blob store, referenced by storage_handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    pub model_digest: [u8; 32],
    pub context_digest: [u8; 32],
    pub token_position: u32,
    pub world_epoch: WorldEpoch,
    pub kv_layout_version: u32,
    pub compatibility_digest: [u8; 32],
    pub payload_digest: [u8; 32],
    pub storage_handle: StorageHandle,
    pub created_at: Timestamp,
}

// ── Transition Guard Errors ───────────────────────────────────────────────

/// Errors for invalid lifecycle transitions.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LifecycleError {
    #[error("invalid pipeline lifecycle transition: from {from:?} to {to:?}")]
    InvalidPipelineTransition {
        from: super::multimodal::PipelineLifecycle,
        to: super::multimodal::PipelineLifecycle,
    },
    #[error("invalid session lifecycle transition: from {from:?} to {to:?}")]
    InvalidSessionTransition {
        from: SessionLifecycle,
        to: SessionLifecycle,
    },
    #[error("invalid inference phase transition: from {from:?} to {to:?}")]
    InvalidPhaseTransition {
        from: InferencePhase,
        to: InferencePhase,
    },
    #[error("invalid teardown transition: from {from:?} to {to:?}")]
    InvalidTeardownTransition {
        from: TeardownState,
        to: TeardownState,
    },
}

impl SessionLifecycle {
    /// Validate a lifecycle transition. Returns Ok(()) if allowed.
    pub fn can_transition_to(&self, target: Self) -> Result<(), LifecycleError> {
        let allowed = match (*self, target) {
            (Self::Created, Self::Admitted)
            | (Self::Admitted, Self::Active)
            | (Self::Active, Self::Quiescing)
            | (Self::Active, Self::Saving)
            | (Self::Quiescing, Self::Saving)
            | (Self::Quiescing, Self::Releasing)
            | (Self::Saving, Self::Completed)
            | (Self::Saving, Self::Active)
            | (Self::Completed, Self::Releasing)
            | (Self::Releasing, Self::Released) => true,
            // Allow failure from any non-terminal and non-completing state
            (Self::Failed, Self::Releasing) => true,
            // Failure allowed from any non-terminal, non-completing, and non-releasing state
            _ if target == Self::Failed
                && !self.is_terminal()
                && *self != Self::Completed
                && *self != Self::Releasing =>
            {
                true
            }
            _ => false,
        };
        if allowed {
            Ok(())
        } else {
            Err(LifecycleError::InvalidSessionTransition {
                from: *self,
                to: target,
            })
        }
    }
}

impl InferencePhase {
    /// Validate an inference phase transition. Returns Ok(()) if allowed.
    pub fn can_transition_to(&self, target: Self) -> Result<(), LifecycleError> {
        let allowed = match (*self, target) {
            (Self::AwaitingInput, Self::Prefill)
            | (Self::Prefill, Self::Decode)
            | (Self::Decode, Self::Decode)
            | (Self::Decode, Self::ToolWait)
            | (Self::Decode, Self::OutputFinalization)
            | (Self::ToolWait, Self::AwaitingInput)
            | (Self::Decode, Self::Compaction)
            | (Self::Compaction, Self::Decode)
            | (Self::OutputFinalization, Self::AwaitingInput) => true,
            _ => false,
        };
        if allowed {
            Ok(())
        } else {
            Err(LifecycleError::InvalidPhaseTransition {
                from: *self,
                to: target,
            })
        }
    }
}

impl TeardownState {
    /// Validate a teardown transition.
    pub fn can_transition_to(&self, target: Self) -> Result<(), LifecycleError> {
        let allowed = match (*self, target) {
            (Self::Active, Self::Quiescing)
            | (Self::Quiescing, Self::Releasing)
            | (Self::Releasing, Self::Released) => true,
            _ => false,
        };
        if allowed {
            Ok(())
        } else {
            Err(LifecycleError::InvalidTeardownTransition {
                from: *self,
                to: target,
            })
        }
    }
}
