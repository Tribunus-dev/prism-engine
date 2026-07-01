//! ComponentTypeRegistry — maps TypeId to decoder functions for semantic
//! receipt projection.
//!
//! Each typed ECS component that is inserted via a [`Command::Insert`] can be
//! projected into a stable [`SemanticCommandPayload`] variant by registering
//! a decoder with this registry.

use std::any::TypeId;
use std::collections::HashMap;

use serde::de::DeserializeOwned;

use crate::runtime::components::worker_assignment::WorkerAssignment;
use crate::runtime::components::worker_health::WorkerHeartbeat;
use crate::runtime::components::worker_lifecycle::{WorkerLifecycle, WorkerRequestPhase};
use crate::runtime::ledger::entry::SemanticCommandPayload;
use crate::runtime::ledger::error::LedgerProjectionError;

/// Boxed decoder: given raw component bytes, produce a semantic payload or
/// return a projection error.
type ComponentDecoder = Box<dyn Fn(&[u8]) -> Result<SemanticCommandPayload, LedgerProjectionError>>;

/// A static registry mapping [`TypeId`] to a decoder function.
///
/// Decoders are registered per-component-type and are used by
/// [`SemanticReceipt::semantic_receipt`] to project type-erased
/// [`Command::Insert`] payloads into stable semantic variants.
pub struct ComponentTypeRegistry {
    decoders: HashMap<TypeId, ComponentDecoder>,
}

impl ComponentTypeRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            decoders: HashMap::new(),
        }
    }

    /// Register a decoder for component type `T`.
    ///
    /// `name` is a human-readable label (currently unused; reserved for
    /// diagnostics and error messages).
    ///
    /// The `decoder` closure maps a fully deserialized `&T` to a
    /// [`SemanticCommandPayload`] and is expected to be infallible (all
    /// fallible work — deserialization — is handled by the registry).
    pub fn register<T: DeserializeOwned + 'static>(
        &mut self,
        _name: &'static str,
        decoder: fn(&T) -> SemanticCommandPayload,
    ) {
        let type_id = TypeId::of::<T>();
        self.decoders.insert(
            type_id,
            Box::new(move |bytes: &[u8]| {
                let data: T =
                    serde_json::from_slice(bytes).map_err(|_| LedgerProjectionError::InvalidSemanticPayload)?;
                Ok(decoder(&data))
            }),
        );
    }

    /// Project bytes carried by a [`Command::Insert`] into a semantic payload.
    ///
    /// Returns [`LedgerProjectionError::InvalidSemanticPayload`] if `type_id`
    /// has no registered decoder, or if the raw bytes cannot be deserialized
    /// as the registered component type.
    pub fn project(
        &self,
        type_id: &TypeId,
        bytes: &[u8],
    ) -> Result<SemanticCommandPayload, LedgerProjectionError> {
        let decoder = self
            .decoders
            .get(type_id)
            .ok_or(LedgerProjectionError::InvalidSemanticPayload)?;
        decoder(bytes)
    }

    /// Register the three built-in core component decoders.
    ///
    /// This is called automatically by [`new_core`] and may also be called on
    /// any existing registry to add the standard set on top of user-supplied
    /// decoders.
    pub fn register_core(&mut self) {
        // WorkerAssignment payload -> WorkerAssigned variant
        self.register::<WorkerAssignment>("WorkerAssignment", |data| {
            SemanticCommandPayload::WorkerAssigned {
                worker_id: data.worker_id.clone(),
                assignment_generation: data.generation as u64,
                request_class: format!("{:?}", data.dispatched),
            }
        });

        // WorkerLifecycle payload -> WorkerRequestPhaseTransitioned variant
        self.register::<WorkerLifecycle>("WorkerLifecycle", |data| {
            SemanticCommandPayload::WorkerRequestPhaseTransitioned {
                from: WorkerRequestPhase::Queued,
                to: data.phase,
                cause: "registry projection".to_string(),
            }
        });

        // WorkerHeartbeat payload -> WorkerHeartbeatObserved variant
        self.register::<WorkerHeartbeat>("WorkerHeartbeat", |data| {
            SemanticCommandPayload::WorkerHeartbeatObserved {
                worker_id: data.worker_id.clone(),
                assignment_generation: data.assignment_generation as u64,
                sequence: data.consecutive_misses as u64,
            }
        });
    }

    /// Create a new registry pre-populated with the three core component
    /// decoders.
    pub fn new_core() -> Self {
        let mut reg = Self::new();
        reg.register_core();
        reg
    }
}

impl Default for ComponentTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_returns_error() {
        let reg = ComponentTypeRegistry::new();
        let err = reg.project(&TypeId::of::<WorkerAssignment>(), &[]).unwrap_err();
        assert!(matches!(err, LedgerProjectionError::InvalidSemanticPayload));
    }

    #[test]
    fn registered_worker_assignment_projects() {
        let mut reg = ComponentTypeRegistry::new();
        let assignment = WorkerAssignment {
            worker_id: "w-1".into(),
            generation: 3,
            dispatched: true,
            assigned_at: Instant::now(),
        };
        let bytes = serde_json::to_vec(&assignment).unwrap();
        reg.register::<WorkerAssignment>("WorkerAssignment", |data| {
            SemanticCommandPayload::WorkerAssigned {
                worker_id: data.worker_id.clone(),
                assignment_generation: data.generation as u64,
                request_class: format!("{:?}", data.dispatched),
            }
        });

        let payload = reg.project(&TypeId::of::<WorkerAssignment>(), &bytes).unwrap();
        match payload {
            SemanticCommandPayload::WorkerAssigned {
                worker_id,
                assignment_generation,
                ..
            } => {
                assert_eq!(worker_id, "w-1");
                assert_eq!(assignment_generation, 3);
            }
            other => panic!("expected WorkerAssigned, got {other:?}"),
        }
    }

    #[test]
    fn core_registry_projects_all_three() {
        let reg = ComponentTypeRegistry::new_core();

        // WorkerAssignment
        let wa = WorkerAssignment {
            worker_id: "w-2".into(),
            generation: 1,
            dispatched: false,
            assigned_at: Instant::now(),
        };
        let bytes = serde_json::to_vec(&wa).unwrap();
        let p = reg.project(&TypeId::of::<WorkerAssignment>(), &bytes).unwrap();
        assert!(matches!(p, SemanticCommandPayload::WorkerAssigned { .. }));

        // WorkerLifecycle
        let wl = WorkerLifecycle {
            phase: WorkerRequestPhase::Streaming,
            retry_count: 0,
            last_transition_at: Instant::now(),
        };
        let bytes = serde_json::to_vec(&wl).unwrap();
        let p = reg.project(&TypeId::of::<WorkerLifecycle>(), &bytes).unwrap();
        assert!(matches!(p, SemanticCommandPayload::WorkerRequestPhaseTransitioned { .. }));

        // WorkerHeartbeat
        let wh = WorkerHeartbeat {
            worker_id: "w-2".into(),
            assignment_generation: 1,
            consecutive_misses: 0,
            last_heartbeat_at: Instant::now(),
        };
        let bytes = serde_json::to_vec(&wh).unwrap();
        let p = reg.project(&TypeId::of::<WorkerHeartbeat>(), &bytes).unwrap();
        assert!(matches!(p, SemanticCommandPayload::WorkerHeartbeatObserved { .. }));
    }

    #[test]
    fn invalid_bytes_returns_invalid_semantic_payload() {
        let mut reg = ComponentTypeRegistry::new();
        reg.register::<WorkerAssignment>("WorkerAssignment", |_| {
            SemanticCommandPayload::EntitySpawned {
                entity_kind: "test".into(),
            }
        });

        let err = reg.project(&TypeId::of::<WorkerAssignment>(), b"not json").unwrap_err();
        assert!(matches!(err, LedgerProjectionError::InvalidSemanticPayload));
    }

    /// Instant is only available with std; the tests above use it.
    /// Pull it from std::time (available in test configuration).
    use std::time::Instant;
}
