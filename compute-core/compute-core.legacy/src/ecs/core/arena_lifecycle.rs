//! Arena lifecycle state machine for SharedTensorArena v1.
//!
//! Every arena moves through explicit states. Illegal transitions fail.
//! This is the runtime authority — not a debug assertion.

use std::sync::atomic::{AtomicU64, Ordering};
use uuid::Uuid;

/// Unique arena identity at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaId {
    pub id: Uuid,
    pub generation: u64,
}

impl ArenaId {
    pub fn new() -> Self {
        static NEXT_GEN: AtomicU64 = AtomicU64::new(0);
        ArenaId {
            id: Uuid::new_v4(),
            generation: NEXT_GEN.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Default for ArenaId {
    fn default() -> Self {
        Self::new()
    }
}

/// Arena lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    Free,
    MlxWriteLeased,
    MlxWritePending,
    CoreAiWriteLeased,
    CoreAiWritePending,
    Produced,
    MlxReadLeased,
    CoreAiReadLeased,
    Retiring,
    Released,
}

/// The lease describes who holds the arena and for what purpose.
#[derive(Debug, Clone)]
pub struct ArenaLease {
    pub arena_id: ArenaId,
    pub state: LifecycleState,
    pub owner_session: Uuid,
    pub job_id: Uuid,
    pub backend: LeasedBackend,
    pub access: AccessMode,
    pub acquired_at: std::time::Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LeasedBackend {
    Mlx,
    CoreAi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    Read,
    Write,
}

impl ArenaLease {
    pub fn new(
        arena_id: ArenaId,
        state: LifecycleState,
        owner_session: Uuid,
        job_id: Uuid,
        backend: LeasedBackend,
        access: AccessMode,
    ) -> Self {
        ArenaLease {
            arena_id,
            state,
            owner_session,
            job_id,
            backend,
            access,
            acquired_at: std::time::Instant::now(),
        }
    }
}

/// Validate a lifecycle transition. Returns the new state or an error.
pub fn validate_transition(
    current: LifecycleState,
    desired: LifecycleState,
) -> Result<LifecycleState, String> {
    use LifecycleState::*;
    match (current, desired) {
        // Allocation
        (Free, MlxWriteLeased) | (Free, CoreAiWriteLeased) => Ok(desired),
        // Write submission
        (MlxWriteLeased, MlxWritePending) | (CoreAiWriteLeased, CoreAiWritePending) => Ok(desired),
        // Write completion -> Produced
        (MlxWritePending, Produced) | (CoreAiWritePending, Produced) => Ok(desired),
        // Reader acquisition from Produced
        (Produced, MlxReadLeased) | (Produced, CoreAiReadLeased) => Ok(desired),
        // Reader returns to Produced
        (MlxReadLeased, Produced) | (CoreAiReadLeased, Produced) => Ok(desired),
        // Return to Free for reuse
        (Produced, Free) => Ok(Free),
        // Retiring path
        (Free, Retiring) | (Produced, Retiring) => Ok(Retiring),
        (Retiring, Released) => Ok(Released),
        _ => Err(format!(
            "illegal transition: {:?} -> {:?}",
            current, desired
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_transitions() {
        // Free -> WriteLeased
        assert_eq!(
            validate_transition(LifecycleState::Free, LifecycleState::MlxWriteLeased),
            Ok(LifecycleState::MlxWriteLeased)
        );
        assert_eq!(
            validate_transition(LifecycleState::Free, LifecycleState::CoreAiWriteLeased),
            Ok(LifecycleState::CoreAiWriteLeased)
        );

        // WriteLeased -> WritePending
        assert_eq!(
            validate_transition(
                LifecycleState::MlxWriteLeased,
                LifecycleState::MlxWritePending
            ),
            Ok(LifecycleState::MlxWritePending)
        );
        assert_eq!(
            validate_transition(
                LifecycleState::CoreAiWriteLeased,
                LifecycleState::CoreAiWritePending
            ),
            Ok(LifecycleState::CoreAiWritePending)
        );

        // WritePending -> Produced
        assert_eq!(
            validate_transition(LifecycleState::MlxWritePending, LifecycleState::Produced),
            Ok(LifecycleState::Produced)
        );
        assert_eq!(
            validate_transition(LifecycleState::CoreAiWritePending, LifecycleState::Produced),
            Ok(LifecycleState::Produced)
        );

        // Produced -> ReadLeased
        assert_eq!(
            validate_transition(LifecycleState::Produced, LifecycleState::MlxReadLeased),
            Ok(LifecycleState::MlxReadLeased)
        );
        assert_eq!(
            validate_transition(LifecycleState::Produced, LifecycleState::CoreAiReadLeased),
            Ok(LifecycleState::CoreAiReadLeased)
        );

        // ReadLeased -> Produced (reader release)
        assert_eq!(
            validate_transition(LifecycleState::MlxReadLeased, LifecycleState::Produced),
            Ok(LifecycleState::Produced)
        );
        assert_eq!(
            validate_transition(LifecycleState::CoreAiReadLeased, LifecycleState::Produced),
            Ok(LifecycleState::Produced)
        );

        // Produced -> Free (reuse)
        assert_eq!(
            validate_transition(LifecycleState::Produced, LifecycleState::Free),
            Ok(LifecycleState::Free)
        );

        // Retiring path
        assert_eq!(
            validate_transition(LifecycleState::Free, LifecycleState::Retiring),
            Ok(LifecycleState::Retiring)
        );
        assert_eq!(
            validate_transition(LifecycleState::Produced, LifecycleState::Retiring),
            Ok(LifecycleState::Retiring)
        );
        assert_eq!(
            validate_transition(LifecycleState::Retiring, LifecycleState::Released),
            Ok(LifecycleState::Released)
        );
    }

    #[test]
    fn test_illegal_transitions() {
        // Free cannot go directly to Produced
        assert!(validate_transition(LifecycleState::Free, LifecycleState::Produced).is_err());

        // Free cannot go directly to Released
        assert!(validate_transition(LifecycleState::Free, LifecycleState::Released).is_err());

        // Cross-backend transition is illegal
        assert!(validate_transition(
            LifecycleState::MlxWriteLeased,
            LifecycleState::CoreAiWritePending
        )
        .is_err());

        // Produced cannot directly to WriteLeased (must go through Free first)
        assert!(
            validate_transition(LifecycleState::Produced, LifecycleState::MlxWriteLeased).is_err()
        );

        // Cannot ReadLeased before Produced
        assert!(validate_transition(LifecycleState::Free, LifecycleState::MlxReadLeased).is_err());

        // Released is terminal
        assert!(validate_transition(LifecycleState::Released, LifecycleState::Free).is_err());
    }

    #[test]
    fn test_arena_id_unique() {
        let a = ArenaId::new();
        let b = ArenaId::new();
        assert_ne!(a, b);
        assert!(b.generation > a.generation);
    }
}
