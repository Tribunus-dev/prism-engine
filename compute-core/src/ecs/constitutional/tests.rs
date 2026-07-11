#[cfg(test)]
mod tests {
    use crate::ecs::constitutional::persistence::{
        EventLogEntry, EventStore, InMemoryEventStore, ProjectionCheckpoint, ReplayEngine, Snapshot,
    };
    use crate::ecs::constitutional::*;
    use crate::ecs::receipt_bus::*;
    use crate::ecs::{CompWorld, EntityKind};

    // ── Helpers ────────────────────────────────────────────────────────────

    fn make_test_envelope() -> Envelope<String> {
        Envelope {
            id: MessageId::compute(b"test"),
            correlation_id: CorrelationId(uuid::Uuid::nil()),
            causation_id: None,
            target: DomainId(uuid::Uuid::nil()),
            originating_epoch: WorldEpoch(0),
            idempotency_key: IdempotencyKey(uuid::Uuid::nil()),
            timestamp: Timestamp::now(),
            aggregate_sequence: AggregateSequence(1),
            payload: "hello".to_string(),
        }
    }

    // ── WorldEpoch ─────────────────────────────────────────────────────────

    #[test]
    fn test_world_epoch_ordering() {
        let a = WorldEpoch(1);
        let b = WorldEpoch(2);
        assert!(a < b);
    }

    // ── MessageId ──────────────────────────────────────────────────────────

    #[test]
    fn test_message_id_determinism() {
        let id1 = MessageId::compute(b"hello");
        let id2 = MessageId::compute(b"hello");
        assert_eq!(id1, id2);
        let id3 = MessageId::compute(b"world");
        assert_ne!(id1, id3);
    }

    // ── Envelope ───────────────────────────────────────────────────────────

    #[test]
    fn test_envelope_roundtrip() {
        let envelope = Envelope {
            id: MessageId::compute(b"test"),
            correlation_id: CorrelationId(uuid::Uuid::nil()),
            causation_id: None,
            target: DomainId(uuid::Uuid::nil()),
            originating_epoch: WorldEpoch(0),
            idempotency_key: IdempotencyKey(uuid::Uuid::nil()),
            timestamp: Timestamp::now(),
            aggregate_sequence: AggregateSequence(1),
            payload: "hello".to_string(),
        };
        let json = serde_json::to_string(&envelope).unwrap();
        let deserialized: Envelope<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(envelope.id, deserialized.id);
        assert_eq!(envelope.payload, deserialized.payload);
    }

    #[test]
    fn test_envelope_compute_id_deterministic() {
        let e1 = make_test_envelope();
        let e2 = make_test_envelope();
        assert_eq!(e1.compute_id(), e2.compute_id());
    }

    #[test]
    fn test_correlation_and_causation() {
        let cmd_corr = CorrelationId(uuid::Uuid::new_v4());
        let cmd_id = MessageId::compute(b"command-1");
        let effect_causation = CausationId(format!("{}", cmd_id));

        let cmd_envelope = Envelope {
            id: cmd_id,
            correlation_id: cmd_corr,
            causation_id: None,
            target: DomainId(uuid::Uuid::nil()),
            originating_epoch: WorldEpoch(0),
            idempotency_key: IdempotencyKey(uuid::Uuid::nil()),
            timestamp: Timestamp::now(),
            aggregate_sequence: AggregateSequence(1),
            payload: Command {
                id: cmd_id,
                target_domain: DomainId(uuid::Uuid::nil()),
                payload: serde_json::json!({"action": "compute"}),
            },
        };

        let effect_id = MessageId::compute(b"effect-1");
        let effect_envelope = Envelope {
            id: effect_id,
            correlation_id: cmd_corr,
            causation_id: Some(effect_causation),
            target: DomainId(uuid::Uuid::nil()),
            originating_epoch: WorldEpoch(0),
            idempotency_key: IdempotencyKey(uuid::Uuid::nil()),
            timestamp: Timestamp::now(),
            aggregate_sequence: AggregateSequence(1),
            payload: EffectRequest {
                id: effect_id,
                kind: EffectKind::RunInference,
                params: serde_json::json!({"model": "test"}),
            },
        };

        // Same correlation links command and effect
        assert_eq!(cmd_envelope.correlation_id, effect_envelope.correlation_id);
        // Effect has a causation chain back to the command
        assert!(effect_envelope.causation_id.is_some());
        assert_eq!(
            effect_envelope.causation_id.as_ref().unwrap().0,
            format!("{}", cmd_envelope.id)
        );
    }

    // ── SystemDescriptor ───────────────────────────────────────────────────

    #[test]
    fn test_system_descriptor_construction() {
        let desc = SystemDescriptor {
            name: "test_sys".into(),
            read_schemas: vec![ComponentSchemaId(1), ComponentSchemaId(2)],
            write_schemas: vec![ComponentSchemaId(1)],
        };
        assert_eq!(desc.name, "test_sys");
        assert_eq!(desc.read_schemas.len(), 2);
        assert_eq!(desc.write_schemas.len(), 1);
    }

    // ── SchemaRegistry ─────────────────────────────────────────────────────

    #[test]
    fn test_schema_registry_basic() {
        let mut reg = SchemaRegistry::new();
        let sid = ComponentSchemaId(42);
        let entry = SchemaEntry {
            schema_id: sid,
            version: SchemaVersion(1),
            type_name: "TestComponent".into(),
            description: "A test component".into(),
        };
        reg.register(entry.clone());
        let retrieved = reg.get(&sid);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), &entry);
        assert_eq!(reg.get(&ComponentSchemaId(99)), None);
    }

    // ── EffectKind ─────────────────────────────────────────────────────────

    #[test]
    fn test_effect_kind_variants() {
        let kinds = [
            EffectKind::LoadFile,
            EffectKind::MapMemory,
            EffectKind::CreateDevice,
            EffectKind::EnumerateDrivers,
            EffectKind::CompileModel,
            EffectKind::RunInference,
            EffectKind::AcquireLease,
            EffectKind::ReleaseHandle,
            EffectKind::Custom(7),
        ];
        assert_eq!(kinds.len(), 9);
        assert_ne!(kinds[0], kinds[1]);
    }

    // ── EffectOutcome ──────────────────────────────────────────────────────

    #[test]
    fn test_effect_outcome_construction() {
        let outcome = EffectOutcome {
            id: MessageId::compute(b"outcome-1"),
            request_id: MessageId::compute(b"request-1"),
            success: true,
            output: serde_json::json!({"result": 42}),
        };
        assert!(outcome.success);
        assert_eq!(outcome.output["result"], 42);
    }

    // ── DomainEvent ────────────────────────────────────────────────────────

    #[test]
    fn test_domain_event_construction() {
        let event = DomainEvent {
            id: MessageId::compute(b"event-1"),
            kind: "EntityCreated".into(),
            entity_id: Some(EntityKindId(1)),
            payload: serde_json::json!({"entity": "foo"}),
        };
        assert_eq!(event.kind, "EntityCreated");
        assert_eq!(event.entity_id, Some(EntityKindId(1)));
    }

    // ── ReceiptCandidate ───────────────────────────────────────────────────

    #[test]
    fn test_receipt_candidate_construction() {
        let candidate = ReceiptCandidate {
            id: MessageId::compute(b"receipt-1"),
            kind: "InferenceResult".into(),
            payload: serde_json::json!({"tokens": 128}),
            payload_hash: [0u8; 32],
        };
        assert_eq!(candidate.kind, "InferenceResult");
        assert_eq!(candidate.payload_hash, [0u8; 32]);
    }

    // ── ReadDependency ─────────────────────────────────────────────────────

    #[test]
    fn test_read_dependency_construction() {
        let dep = ReadDependency {
            entity: 7,
            schema_id: ComponentSchemaId(1),
            observed_version: 3,
        };
        assert_eq!(dep.entity, 7);
        assert_eq!(dep.observed_version, 3);
    }

    // ── Map method ─────────────────────────────────────────────────────────

    #[test]
    fn test_envelope_map() {
        let e = make_test_envelope();
        let mapped = e.map(|s| s.len());
        assert_eq!(mapped.payload, 5); // "hello".len()
                                       // Metadata preserved
        assert_eq!(mapped.correlation_id, CorrelationId(uuid::Uuid::nil()));
    }

    // ── Display / Display impl ─────────────────────────────────────────────

    #[test]
    fn test_message_id_display() {
        let id = MessageId::compute(b"display-test");
        let s = format!("{}", id);
        assert_eq!(s.len(), 64); // hex-encoded blake3 = 64 chars
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  WorldTxn Tests  (Stage 2 — constitutional ECS migration)
    // ══════════════════════════════════════════════════════════════════════

    /// Create a minimal world with one spawned entity for txn tests.
    fn make_world() -> CompWorld {
        let mut world = CompWorld::new();
        world.spawn(EntityKind::Model, Some("test_entity".into()));
        world
    }

    /// Create a transaction against a world, staging an insert on entity 1.
    fn make_txn_with_insert(world: &CompWorld) -> WorldTxn {
        let mut txn = WorldTxn::new(world);
        txn.add_component(1, ComponentSchemaId(10), SchemaVersion(1), 42u64);
        txn
    }

    // ── test_atomic_commit ─────────────────────────────────────────────────

    #[test]
    fn test_atomic_commit() {
        let mut world = make_world();
        let start_epoch = world.current_epoch();

        let txn = make_txn_with_insert(&world);
        let committed = world.transit(txn).expect("commit should succeed");

        // Epoch advanced
        assert!(world.current_epoch() > start_epoch);
        assert_eq!(committed.0, world.current_epoch());

        // Journal contains the component change
        let journal = world.last_journal();
        assert_eq!(journal.len(), 1, "journal should have one change entry");

        let change = &journal[0];
        assert_eq!(change.entity, 1);
        assert_eq!(change.schema_id, ComponentSchemaId(10));
        assert_eq!(change.schema_version, SchemaVersion(1));
        assert_eq!(change.change_type, ChangeType::Insert);
        assert_eq!(change.world_epoch, committed.0);
    }

    // ── test_stale_epoch_rejects ────────────────────────────────────────────

    #[test]
    fn test_stale_epoch_rejects() {
        let mut world = make_world();

        // Start a transaction (records current epoch)
        let txn_stale = WorldTxn::new(&world);

        // Advance the world epoch by committing a different transaction
        let txn_advance = WorldTxn::new(&world);
        world
            .transit(txn_advance)
            .expect("advancing commit should succeed");

        // The stale transaction should be rejected
        let result = world.transit(txn_stale);
        assert!(
            matches!(result, Err(WorldTxnError::StaleEpoch { .. })),
            "expected StaleEpoch error, got {:?}",
            result
        );
    }

    // ── test_multiple_commits_advance_epoch ─────────────────────────────────

    #[test]
    fn test_multiple_commits_advance_epoch() {
        let mut world = make_world();

        // Starting epoch should be 1
        assert_eq!(world.current_epoch(), WorldEpoch(1));

        // Commit 3 transactions
        let commit1 = world
            .transit(WorldTxn::new(&world))
            .expect("commit 1 should succeed");
        assert_eq!(commit1.0, WorldEpoch(2));

        let commit2 = world
            .transit(WorldTxn::new(&world))
            .expect("commit 2 should succeed");
        assert_eq!(commit2.0, WorldEpoch(3));

        let commit3 = world
            .transit(WorldTxn::new(&world))
            .expect("commit 3 should succeed");
        assert_eq!(commit3.0, WorldEpoch(4));

        // Final epoch matches
        assert_eq!(world.current_epoch(), WorldEpoch(4));
    }

    // ── test_mutation_journal_records_changes ───────────────────────────────

    #[test]
    fn test_mutation_journal_records_changes() {
        let mut world = make_world();

        // Transaction with one insert
        let mut txn = WorldTxn::new(&world);
        txn.add_component(
            1,
            ComponentSchemaId(42),
            SchemaVersion(2),
            "hello".to_string(),
        );

        let committed = world.transit(txn).expect("commit should succeed");

        let journal = world.last_journal();
        assert_eq!(journal.len(), 1, "expected exactly one journal entry");

        let entry = &journal[0];
        assert_eq!(entry.entity, 1);
        assert_eq!(entry.schema_id, ComponentSchemaId(42));
        assert_eq!(entry.schema_version, SchemaVersion(2));
        assert_eq!(entry.change_type, ChangeType::Insert);
        assert_eq!(entry.world_epoch, committed.0);

        // before_hash is None for a fresh insert (no prior value); hashing comes in a future stage
        assert!(
            entry.before_hash.is_none(),
            "before_hash must be None for insert"
        );

        // after_hash is None until content hashing is wired in a future stage
        assert!(
            entry.after_hash.is_none(),
            "after_hash must be None for insert"
        );
    }

    // ── test_access_declaration_construction ────────────────────────────────

    #[test]
    fn test_access_declaration_construction() {
        // AccessKind variants
        let read = AccessKind::Read;
        let write = AccessKind::Write;
        assert_ne!(read, write);
        assert_eq!(format!("{:?}", read), "Read");
        assert_eq!(format!("{:?}", write), "Write");

        // ChangeType variants
        let insert = ChangeType::Insert;
        let update = ChangeType::Update;
        let remove = ChangeType::Remove;
        assert_ne!(insert, update);
        assert_ne!(update, remove);
        assert_ne!(insert, remove);
        assert_eq!(format!("{:?}", insert), "Insert");
        assert_eq!(format!("{:?}", update), "Update");
        assert_eq!(format!("{:?}", remove), "Remove");

        // AccessDeclaration construction
        let decl = AccessDeclaration {
            schema_id: ComponentSchemaId(7),
            entity: Some(42),
            access: AccessKind::Read,
        };
        assert_eq!(decl.schema_id, ComponentSchemaId(7));
        assert_eq!(decl.entity, Some(42));
        assert_eq!(decl.access, AccessKind::Read);

        // AccessDeclaration serialization roundtrip
        let json = serde_json::to_string(&decl).unwrap();
        let deserialized: AccessDeclaration = serde_json::from_str(&json).unwrap();
        assert_eq!(decl.schema_id, deserialized.schema_id);
        assert_eq!(decl.entity, deserialized.entity);
        assert_eq!(decl.access, deserialized.access);

        // ChangeType serialization roundtrip
        for ct in [ChangeType::Insert, ChangeType::Update, ChangeType::Remove] {
            let json = serde_json::to_string(&ct).unwrap();
            let deserialized: ChangeType = serde_json::from_str(&json).unwrap();
            assert_eq!(ct, deserialized);
        }

        // AccessKind serialization roundtrip
        for ak in [AccessKind::Read, AccessKind::Write] {
            let json = serde_json::to_string(&ak).unwrap();
            let deserialized: AccessKind = serde_json::from_str(&json).unwrap();
            assert_eq!(ak, deserialized);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Lifecycle Tests  (Stage 3 — entity lifecycle types)
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Pure type-level tests: no runtime behavior, no CompWorld dependency.
    //  Tests exercise transition guards, terminal/status predicates,
    //  typed relationship construction, and serde roundtrips.

    // ── test_session_lifecycle_valid_transitions ─────────────────────────────

    #[test]
    fn test_session_lifecycle_valid_transitions() {
        // Drive through the happy path: Created → Admitted → Active → Quiescing → Releasing → Released
        assert!(SessionLifecycle::Created
            .can_transition_to(SessionLifecycle::Admitted)
            .is_ok());
        assert!(SessionLifecycle::Admitted
            .can_transition_to(SessionLifecycle::Active)
            .is_ok());
        assert!(SessionLifecycle::Active
            .can_transition_to(SessionLifecycle::Quiescing)
            .is_ok());
        assert!(SessionLifecycle::Quiescing
            .can_transition_to(SessionLifecycle::Releasing)
            .is_ok());
        assert!(SessionLifecycle::Releasing
            .can_transition_to(SessionLifecycle::Released)
            .is_ok());
    }

    // ── test_session_lifecycle_rejects_invalid ───────────────────────────────

    #[test]
    fn test_session_lifecycle_rejects_invalid() {
        // Impossible forward jumps
        assert!(SessionLifecycle::Created
            .can_transition_to(SessionLifecycle::Completed)
            .is_err());
        assert!(SessionLifecycle::Admitted
            .can_transition_to(SessionLifecycle::Released)
            .is_err());
        // Backwards transition
        assert!(SessionLifecycle::Active
            .can_transition_to(SessionLifecycle::Created)
            .is_err());
        // Skipping states
        assert!(SessionLifecycle::Created
            .can_transition_to(SessionLifecycle::Released)
            .is_err());
        // Terminal → anything (Released is the only terminal)
        assert!(SessionLifecycle::Released
            .can_transition_to(SessionLifecycle::Admitted)
            .is_err());
    }

    // ── test_session_failure_from_any_non_terminal ───────────────────────────

    #[test]
    fn test_session_failure_from_any_non_terminal() {
        let non_terminal = [
            SessionLifecycle::Created,
            SessionLifecycle::Admitted,
            SessionLifecycle::Active,
            SessionLifecycle::Quiescing,
            SessionLifecycle::Saving,
            SessionLifecycle::Completed,
            SessionLifecycle::Releasing,
        ];
        for state in &non_terminal {
            assert!(
                state.can_transition_to(SessionLifecycle::Failed).is_ok(),
                "Failed should be reachable from {:?}",
                state
            );
        }
        // Terminal state (Released) cannot transition to Failed
        assert!(SessionLifecycle::Released
            .can_transition_to(SessionLifecycle::Failed)
            .is_err());
    }

    // ── test_session_terminal ────────────────────────────────────────────────

    #[test]
    fn test_session_terminal() {
        // Only Released is terminal
        for state in &[
            SessionLifecycle::Created,
            SessionLifecycle::Admitted,
            SessionLifecycle::Active,
            SessionLifecycle::Quiescing,
            SessionLifecycle::Saving,
            SessionLifecycle::Completed,
            SessionLifecycle::Failed,
            SessionLifecycle::Releasing,
        ] {
            assert!(!state.is_terminal(), "{:?} should not be terminal", state);
        }
        assert!(SessionLifecycle::Released.is_terminal());

        // is_releasing is true only for Releasing and Quiescing
        for state in &[
            SessionLifecycle::Created,
            SessionLifecycle::Admitted,
            SessionLifecycle::Active,
            SessionLifecycle::Saving,
            SessionLifecycle::Completed,
            SessionLifecycle::Failed,
            SessionLifecycle::Released,
        ] {
            assert!(!state.is_releasing(), "{:?} should not be releasing", state);
        }
        assert!(SessionLifecycle::Quiescing.is_releasing());
        assert!(SessionLifecycle::Releasing.is_releasing());
    }

    // ── test_inference_phase_valid_transitions ───────────────────────────────

    #[test]
    fn test_inference_phase_valid_transitions() {
        // Happy generate path
        assert!(InferencePhase::AwaitingInput
            .can_transition_to(InferencePhase::Prefill)
            .is_ok());
        assert!(InferencePhase::Prefill
            .can_transition_to(InferencePhase::Decode)
            .is_ok());
        // Self-transition (keep decoding)
        assert!(InferencePhase::Decode
            .can_transition_to(InferencePhase::Decode)
            .is_ok());
        // Decode → ToolWait → AwaitingInput
        assert!(InferencePhase::Decode
            .can_transition_to(InferencePhase::ToolWait)
            .is_ok());
        assert!(InferencePhase::ToolWait
            .can_transition_to(InferencePhase::AwaitingInput)
            .is_ok());
        // Decode → OutputFinalization → AwaitingInput
        assert!(InferencePhase::Decode
            .can_transition_to(InferencePhase::OutputFinalization)
            .is_ok());
        assert!(InferencePhase::OutputFinalization
            .can_transition_to(InferencePhase::AwaitingInput)
            .is_ok());
        // Decode → Compaction → Decode (CPU memory compaction)
        assert!(InferencePhase::Decode
            .can_transition_to(InferencePhase::Compaction)
            .is_ok());
        assert!(InferencePhase::Compaction
            .can_transition_to(InferencePhase::Decode)
            .is_ok());
    }

    // ── test_inference_phase_rejects_invalid ─────────────────────────────────

    #[test]
    fn test_inference_phase_rejects_invalid() {
        // Phase skipping
        assert!(InferencePhase::AwaitingInput
            .can_transition_to(InferencePhase::ToolWait)
            .is_err());
        assert!(InferencePhase::Prefill
            .can_transition_to(InferencePhase::AwaitingInput)
            .is_err());
        // Wrong direction
        assert!(InferencePhase::Decode
            .can_transition_to(InferencePhase::Prefill)
            .is_err());
        assert!(InferencePhase::OutputFinalization
            .can_transition_to(InferencePhase::Decode)
            .is_err());
    }

    // ── test_is_generating ───────────────────────────────────────────────────

    #[test]
    fn test_is_generating() {
        assert!(InferencePhase::Prefill.is_generating());
        assert!(InferencePhase::Decode.is_generating());
        for phase in &[
            InferencePhase::AwaitingInput,
            InferencePhase::ToolWait,
            InferencePhase::Compaction,
            InferencePhase::OutputFinalization,
        ] {
            assert!(
                !phase.is_generating(),
                "{:?} should not be generating",
                phase
            );
        }
    }

    // ── test_teardown_state_machine ──────────────────────────────────────────

    #[test]
    fn test_teardown_state_machine() {
        // Forward path
        assert!(TeardownState::Active
            .can_transition_to(TeardownState::Quiescing)
            .is_ok());
        assert!(TeardownState::Quiescing
            .can_transition_to(TeardownState::Releasing)
            .is_ok());
        assert!(TeardownState::Releasing
            .can_transition_to(TeardownState::Released)
            .is_ok());
        // Backwards transitions fail
        assert!(TeardownState::Released
            .can_transition_to(TeardownState::Releasing)
            .is_err());
        assert!(TeardownState::Releasing
            .can_transition_to(TeardownState::Quiescing)
            .is_err());
        assert!(TeardownState::Quiescing
            .can_transition_to(TeardownState::Active)
            .is_err());
        // Self-transition (identity) is not in the allowed list
        assert!(TeardownState::Active
            .can_transition_to(TeardownState::Active)
            .is_err());
        // Terminal
        assert!(!TeardownState::Active.is_terminal());
        assert!(!TeardownState::Quiescing.is_terminal());
        assert!(!TeardownState::Releasing.is_terminal());
        assert!(TeardownState::Released.is_terminal());
    }

    // ── test_artifact_lifecycle ──────────────────────────────────────────────

    #[test]
    fn test_artifact_lifecycle() {
        assert!(ArtifactLifecycle::Validated.is_usable());
        assert!(ArtifactLifecycle::Loaded.is_usable());
        assert!(!ArtifactLifecycle::Discovered.is_usable());
        assert!(!ArtifactLifecycle::Invalid.is_usable());

        // Serde roundtrip
        for variant in &[
            ArtifactLifecycle::Discovered,
            ArtifactLifecycle::Validated,
            ArtifactLifecycle::Loaded,
            ArtifactLifecycle::Invalid,
        ] {
            let json = serde_json::to_string(variant).unwrap();
            let deserialized: ArtifactLifecycle = serde_json::from_str(&json).unwrap();
            assert_eq!(*variant, deserialized);
        }
    }

    // ── test_device_lifecycle ────────────────────────────────────────────────

    #[test]
    fn test_device_lifecycle() {
        assert!(DeviceLifecycle::Ready.is_available());
        for state in &[
            DeviceLifecycle::Discovered,
            DeviceLifecycle::Initializing,
            DeviceLifecycle::Degraded,
            DeviceLifecycle::Unavailable,
            DeviceLifecycle::Removed,
        ] {
            assert!(!state.is_available(), "{:?} should not be available", state);
        }
    }

    // ── test_residency_lifecycle ─────────────────────────────────────────────

    #[test]
    fn test_residency_lifecycle() {
        assert!(ResidencyLifecycle::Resident.is_resident());
        for state in &[
            ResidencyLifecycle::Desired,
            ResidencyLifecycle::Binding,
            ResidencyLifecycle::Evicting,
            ResidencyLifecycle::Evicted,
        ] {
            assert!(!state.is_resident(), "{:?} should not be resident", state);
        }
    }

    // ── test_typed_relationships ─────────────────────────────────────────────

    #[test]
    fn test_typed_relationships() {
        let rel = SessionUsesModel {
            session_id: 1,
            model_id: 2,
        };
        assert_eq!(rel.session_id, 1);
        assert_eq!(rel.model_id, 2);

        let rt = ResidencyTargets {
            residency_id: 3,
            device_id: 4,
        };
        assert_eq!(rt.residency_id, 3);
        assert_eq!(rt.device_id, 4);

        let p = Parent { parent_id: 5 };
        assert_eq!(p.parent_id, 5);

        // Serde roundtrips
        let json = serde_json::to_string(&rel).unwrap();
        let deser: SessionUsesModel = serde_json::from_str(&json).unwrap();
        assert_eq!(rel, deser);

        let json = serde_json::to_string(&rt).unwrap();
        let deser: ResidencyTargets = serde_json::from_str(&json).unwrap();
        assert_eq!(rt, deser);

        let json = serde_json::to_string(&p).unwrap();
        let deser: Parent = serde_json::from_str(&json).unwrap();
        assert_eq!(p, deser);
    }

    // ── test_storage_handle ──────────────────────────────────────────────────

    #[test]
    fn test_storage_handle() {
        let handle = StorageHandle("test-handle-42".to_string());
        // Debug output contains the inner string
        let debug = format!("{:?}", handle);
        assert!(
            debug.contains("test-handle-42"),
            "Debug should contain inner value: {}",
            debug
        );

        // Serde roundtrip
        let json = serde_json::to_string(&handle).unwrap();
        let deserialized: StorageHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(handle, deserialized);
    }

    // ── test_session_checkpoint_serde ────────────────────────────────────────

    #[test]
    fn test_session_checkpoint_serde() {
        let cp = SessionCheckpoint {
            model_digest: [1u8; 32],
            context_digest: [2u8; 32],
            token_position: 128,
            world_epoch: WorldEpoch(42),
            kv_layout_version: 3,
            compatibility_digest: [4u8; 32],
            payload_digest: [5u8; 32],
            storage_handle: StorageHandle("arena-7/slot-3".to_string()),
            created_at: Timestamp(1_700_000_000_000_000_000),
        };

        let json = serde_json::to_string(&cp).unwrap();
        let deserialized: SessionCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(cp.model_digest, deserialized.model_digest);
        assert_eq!(cp.context_digest, deserialized.context_digest);
        assert_eq!(cp.token_position, deserialized.token_position);
        assert_eq!(cp.world_epoch, deserialized.world_epoch);
        assert_eq!(cp.kv_layout_version, deserialized.kv_layout_version);
        assert_eq!(cp.compatibility_digest, deserialized.compatibility_digest);
        assert_eq!(cp.payload_digest, deserialized.payload_digest);
        assert_eq!(cp.storage_handle, deserialized.storage_handle);
        assert_eq!(cp.created_at, deserialized.created_at);
    }

    // ── test_lifecycle_error_display ─────────────────────────────────────────

    #[test]
    fn test_lifecycle_error_display() {
        let err = LifecycleError::InvalidSessionTransition {
            from: SessionLifecycle::Created,
            to: SessionLifecycle::Completed,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid session"), "msg: {}", msg);
        assert!(msg.contains("Created"), "msg: {}", msg);
        assert!(msg.contains("Completed"), "msg: {}", msg);

        let err = LifecycleError::InvalidPhaseTransition {
            from: InferencePhase::AwaitingInput,
            to: InferencePhase::ToolWait,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid inference"), "msg: {}", msg);
        assert!(msg.contains("AwaitingInput"), "msg: {}", msg);
        assert!(msg.contains("ToolWait"), "msg: {}", msg);

        let err = LifecycleError::InvalidTeardownTransition {
            from: TeardownState::Released,
            to: TeardownState::Releasing,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("invalid teardown"), "msg: {}", msg);
        assert!(msg.contains("Released"), "msg: {}", msg);
        assert!(msg.contains("Releasing"), "msg: {}", msg);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Driver Registry Tests  (Stage 4)
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Pure type-level tests: construction, serde roundtrips, factory registration.
    //  No real backend hardware is involved.

    use crate::ecs::constitutional::driver::*;
    use std::sync::Arc;

    // ── Mock Factory ─────────────────────────────────────────────────────

    struct MockFactory {
        name: String,
        available: bool,
    }

    impl DriverFactory for MockFactory {
        fn enumerate(&self) -> Vec<DriverInfo> {
            vec![DriverInfo {
                name: self.name.clone(),
                version_major: 1,
                version_minor: 0,
                available: self.available,
                description: String::new(),
            }]
        }

        fn try_create(&self, info: &DriverInfo) -> Option<DriverCreateOutcome> {
            if self.available && info.available {
                Some(DriverCreateOutcome {
                    handle: "mock-handle".into(),
                    capabilities: vec![BackendCapability::MatMulF32],
                    device_metadata: DeviceMetadata {
                        name: "mock-device".into(),
                        device_id: DomainId(uuid::Uuid::nil()),
                        memory_bytes: 1024,
                        compute_units: 1,
                        max_alloc_bytes: 512,
                    },
                    validation_digest: [0u8; 32],
                })
            } else {
                None
            }
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    // ── test_backend_capability_variants ─────────────────────────────────

    #[test]
    fn test_backend_capability_variants() {
        let variants: Vec<BackendCapability> = vec![
            BackendCapability::MatMulF32,
            BackendCapability::MatMulF16,
            BackendCapability::MatMulInt8,
            BackendCapability::UnifiedMemory,
            BackendCapability::DedicatedMemory { size_bytes: 8192 },
            BackendCapability::MoeDispatch,
            BackendCapability::Attention,
            BackendCapability::FusedMlp,
            BackendCapability::RequiresHostCopy,
            BackendCapability::SupportsPeerAccess,
            BackendCapability::Other("custom".into()),
        ];

        // All are distinct
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }

        // Debug output
        assert_eq!(format!("{:?}", BackendCapability::MatMulF32), "MatMulF32");
        assert_eq!(
            format!("{:?}", BackendCapability::Other("custom".into())),
            "Other(\"custom\")"
        );

        // Clone
        let cloned = variants[0].clone();
        assert_eq!(variants[0], cloned);

        // Serde roundtrip — unit variants
        let json = serde_json::to_string(&BackendCapability::MatMulF32).unwrap();
        assert_eq!(json, "\"MatMulF32\"");
        let deser: BackendCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, BackendCapability::MatMulF32);

        // Serde roundtrip — Other with data
        let other = BackendCapability::Other("roundtrip-test".into());
        let json = serde_json::to_string(&other).unwrap();
        let deser: BackendCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, other);
    }

    // ── test_driver_info_construction ────────────────────────────────────

    #[test]
    fn test_driver_info_construction() {
        let info = DriverInfo {
            name: "test-backend".into(),
            version_major: 2,
            version_minor: 1,
            available: true,
            description: "A test backend".into(),
        };
        assert_eq!(info.name, "test-backend");
        assert_eq!(info.version_major, 2);
        assert_eq!(info.version_minor, 1);
        assert!(info.available);
        assert_eq!(info.description, "A test backend");

        // Serde roundtrip
        let json = serde_json::to_string(&info).unwrap();
        let deser: DriverInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info.name, deser.name);
        assert_eq!(info.available, deser.available);
        assert_eq!(info.description, deser.description);
        assert_eq!(info, deser);
    }

    // ── test_driver_registry_register_and_enumerate ──────────────────────

    #[test]
    fn test_driver_registry_register_and_enumerate() {
        let mut reg = DriverRegistry::new();
        assert_eq!(reg.factory_count(), 0);

        let factory = Arc::new(MockFactory {
            name: "mock-factory".into(),
            available: true,
        });
        reg.register_factory(factory);

        assert_eq!(reg.factory_count(), 1);

        let all = reg.enumerate_all();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].0, "mock-factory");
        assert_eq!(all[0].1.len(), 1);
        assert_eq!(all[0].1[0].name, "mock-factory");
        assert!(all[0].1[0].available);
    }

    // ── test_driver_registry_enumerate_preserves_order ───────────────────

    #[test]
    fn test_driver_registry_enumerate_preserves_order() {
        let mut reg = DriverRegistry::new();

        reg.register_factory(Arc::new(MockFactory {
            name: "first".into(),
            available: true,
        }));
        reg.register_factory(Arc::new(MockFactory {
            name: "second".into(),
            available: true,
        }));

        let all = reg.enumerate_all();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].0, "first");
        assert_eq!(all[1].0, "second");
    }

    // ── test_driver_factory_create_validated ─────────────────────────────

    #[test]
    fn test_driver_factory_create_validated() {
        let mut reg = DriverRegistry::new();
        reg.register_factory(Arc::new(MockFactory {
            name: "valid-factory".into(),
            available: true,
        }));

        // Valid info — matches factory name and is available
        let valid_info = DriverInfo {
            name: "valid-factory".into(),
            version_major: 1,
            version_minor: 0,
            available: true,
            description: String::new(),
        };
        let outcome = reg.try_create_from_info(&valid_info);
        assert!(outcome.is_some());
        assert_eq!(outcome.as_ref().unwrap().handle, "mock-handle");

        // Invalid info — matches factory name but not available
        let invalid_info = DriverInfo {
            name: "valid-factory".into(),
            version_major: 1,
            version_minor: 0,
            available: false,
            description: String::new(),
        };
        assert!(reg.try_create_from_info(&invalid_info).is_none());
    }

    // ── test_backend_capability_serde ────────────────────────────────────

    #[test]
    fn test_backend_capability_serde() {
        // Unit variants
        let json = serde_json::to_string(&BackendCapability::MatMulF32).unwrap();
        let deser: BackendCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, BackendCapability::MatMulF32);

        // Other
        let other = BackendCapability::Other("custom-backend".into());
        let json = serde_json::to_string(&other).unwrap();
        assert!(json.contains("\"Other\""));
        assert!(json.contains("custom-backend"));
        let deser: BackendCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, other);

        // DedicatedMemory with data field
        let mem = BackendCapability::DedicatedMemory { size_bytes: 4096 };
        let json = serde_json::to_string(&mem).unwrap();
        let deser: BackendCapability = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, mem);
    }

    // ── test_device_metadata_serde ───────────────────────────────────────

    #[test]
    fn test_device_metadata_serde() {
        let meta = DeviceMetadata {
            name: "test-device".into(),
            device_id: DomainId(uuid::Uuid::nil()),
            memory_bytes: 8_589_934_592,
            compute_units: 16,
            max_alloc_bytes: 4_294_967_296,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let deser: DeviceMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, deser);
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Scheduler Tests  (Stage 5)
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Pure type-level tests: construction, state predicates, drain behavior.

    // ── test_work_item_construction ──────────────────────────────────────

    #[test]
    fn test_work_item_construction() {
        let item = WorkItem::new(WorkKind::RunInference, 42);
        assert_eq!(item.kind, WorkKind::RunInference);
        assert_eq!(item.target_entity, 42);
        assert_eq!(item.state, WorkState::Pending);
        assert!(item.prerequisites.is_empty());
        assert!(item.is_ready());
    }

    // ── test_work_item_with_prereqs ──────────────────────────────────────

    #[test]
    fn test_work_item_with_prereqs() {
        let mut item = WorkItem::new(WorkKind::LoadModel, 7);
        item.prerequisites.push(Prerequisite {
            entity: 1,
            kind: PrereqKind::ComponentPresent,
            generation: 0,
        });
        assert_eq!(item.prerequisites.len(), 1);
        assert!(!item.is_ready());
    }

    // ── test_work_state_is_terminal ──────────────────────────────────────

    #[test]
    fn test_work_state_is_terminal() {
        assert!(WorkState::Completed.is_terminal());
        assert!(WorkState::Failed.is_terminal());
        assert!(WorkState::Cancelled.is_terminal());

        assert!(!WorkState::Pending.is_terminal());
        assert!(!WorkState::Ready.is_terminal());
        assert!(!WorkState::Leased.is_terminal());
    }

    // ── test_work_lease_construction ─────────────────────────────────────

    #[test]
    fn test_work_lease_construction() {
        let lease = WorkLease {
            work_entity: 42,
            kind: WorkKind::RunInference,
            lease_generation: 1,
            attempt: 0,
            cancellation_epoch: WorldEpoch(0),
            expiry: Timestamp(1_700_000_000_000_000_000),
            resource_claim: ResourceClaim {
                memory_bytes: 1024,
                compute_units: 2,
                priority: Priority::Normal,
            },
        };
        assert_eq!(lease.work_entity, 42);
        assert_eq!(lease.kind, WorkKind::RunInference);
        assert_eq!(lease.lease_generation, 1);
    }

    // ── test_scheduler_drain ─────────────────────────────────────────────

    #[test]
    fn test_scheduler_drain() {
        let mut sched = Scheduler::new();
        sched.mark_ready(1, WorkKind::RunInference);
        sched.mark_ready(2, WorkKind::RunInference);

        assert_eq!(sched.ready_count(), 2);

        let batch1 = sched.drain(1);
        assert_eq!(batch1.len(), 1);
        assert_eq!(sched.ready_count(), 1);

        let batch2 = sched.drain(1);
        assert_eq!(batch2.len(), 1);
        assert_eq!(sched.ready_count(), 0);

        let batch3 = sched.drain(1);
        assert_eq!(batch3.len(), 0);
    }

    // ── test_scheduler_drain_respects_max ────────────────────────────────

    #[test]
    fn test_scheduler_drain_respects_max() {
        let mut sched = Scheduler::new();
        for i in 0..5 {
            sched.mark_ready(i as u64, WorkKind::RunInference);
        }

        assert_eq!(sched.ready_count(), 5);

        let batch1 = sched.drain(3);
        assert_eq!(batch1.len(), 3);
        assert_eq!(sched.ready_count(), 2);

        let batch2 = sched.drain(3);
        assert_eq!(batch2.len(), 2);
        assert_eq!(sched.ready_count(), 0);

        let batch3 = sched.drain(3);
        assert_eq!(batch3.len(), 0);
    }

    // ── test_priority_ordering ───────────────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Low.as_u8() < Priority::Normal.as_u8());
        assert!(Priority::Normal.as_u8() < Priority::High.as_u8());
        assert!(Priority::High.as_u8() < Priority::Critical.as_u8());

        assert_eq!(Priority::Low.as_u8(), 0);
        assert_eq!(Priority::Normal.as_u8(), 1);
        assert_eq!(Priority::High.as_u8(), 2);
        assert_eq!(Priority::Critical.as_u8(), 3);
    }

    // ── test_prerequisite_construction ───────────────────────────────────

    #[test]
    fn test_prerequisite_construction() {
        let prereq = Prerequisite {
            entity: 7,
            kind: PrereqKind::EventReceived,
            generation: 3,
        };
        assert_eq!(prereq.entity, 7);
        assert_eq!(prereq.kind, PrereqKind::EventReceived);
        assert_eq!(prereq.generation, 3);

        let json = serde_json::to_string(&prereq).unwrap();
        let deser: Prerequisite = serde_json::from_str(&json).unwrap();
        assert_eq!(prereq, deser);
    }

    // ── test_resource_claim_serde ────────────────────────────────────────

    #[test]
    fn test_resource_claim_serde() {
        let claim = ResourceClaim {
            memory_bytes: 4_294_967_296,
            compute_units: 8,
            priority: Priority::High,
        };

        let json = serde_json::to_string(&claim).unwrap();
        let deser: ResourceClaim = serde_json::from_str(&json).unwrap();
        assert_eq!(claim, deser);
    }

    // ── test_scheduler_register_pending ──────────────────────────────────

    #[test]
    fn test_scheduler_register_pending() {
        let mut sched = Scheduler::new();
        assert_eq!(sched.ready_count(), 0);

        // register_pending does not change ready_count
        sched.register_pending(10);
        assert_eq!(sched.ready_count(), 0);

        // Only mark_ready increases ready_count
        sched.mark_ready(10, WorkKind::Validate);
        assert_eq!(sched.ready_count(), 1);
    }

    // ── test_multiple_kinds_drain ────────────────────────────────────────

    #[test]
    fn test_multiple_kinds_drain() {
        let mut sched = Scheduler::new();
        sched.mark_ready(1, WorkKind::LoadModel);
        sched.mark_ready(2, WorkKind::CompileGraph);
        sched.mark_ready(3, WorkKind::RunInference);

        let leases = sched.drain(10);
        assert_eq!(leases.len(), 3);

        let kinds: std::collections::HashSet<WorkKind> = leases.iter().map(|l| l.kind).collect();
        assert!(kinds.contains(&WorkKind::LoadModel));
        assert!(kinds.contains(&WorkKind::CompileGraph));
        assert!(kinds.contains(&WorkKind::RunInference));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Persistence & Projection Tests  (Stage 6)
    // ══════════════════════════════════════════════════════════════════════
    //
    //  Sync tests for InMemoryEventStore, ReplayEngine, ProjectionCheckpoint,
    //  and ReceiptBus subscriber accounting.

    // ── test_in_memory_event_store ─────────────────────────────────────────

    #[test]
    fn test_in_memory_event_store() {
        let mut store = InMemoryEventStore::new();
        assert_eq!(store.event_count(), 0);
        assert_eq!(store.latest_epoch(), None);

        let entry = EventLogEntry {
            epoch: WorldEpoch(1),
            sequence: 1,
            event: DomainEvent {
                id: MessageId::compute(b"ev-1"),
                kind: "Created".into(),
                entity_id: Some(EntityKindId(1)),
                payload: serde_json::json!({"x": 1}),
            },
            world_digest: [0u8; 32],
        };
        let entry2 = EventLogEntry {
            epoch: WorldEpoch(1),
            sequence: 2,
            event: DomainEvent {
                id: MessageId::compute(b"ev-2"),
                kind: "Updated".into(),
                entity_id: Some(EntityKindId(1)),
                payload: serde_json::json!({"x": 2}),
            },
            world_digest: [0u8; 32],
        };

        store
            .append_events(WorldEpoch(1), &[entry, entry2])
            .unwrap();
        assert_eq!(store.event_count(), 2);
        assert_eq!(store.latest_epoch(), Some(WorldEpoch(1)));
    }

    // ── test_event_store_get_events_from ────────────────────────────────────

    #[test]
    fn test_event_store_get_events_from() {
        let mut store = InMemoryEventStore::new();
        let base = DomainEvent {
            id: MessageId::compute(b"base"),
            kind: "Created".into(),
            entity_id: None,
            payload: serde_json::json!({}),
        };

        store
            .append_events(
                WorldEpoch(1),
                &[EventLogEntry {
                    epoch: WorldEpoch(1),
                    sequence: 1,
                    event: base.clone(),
                    world_digest: [0u8; 32],
                }],
            )
            .unwrap();
        store
            .append_events(
                WorldEpoch(2),
                &[EventLogEntry {
                    epoch: WorldEpoch(2),
                    sequence: 2,
                    event: base.clone(),
                    world_digest: [1u8; 32],
                }],
            )
            .unwrap();
        store
            .append_events(
                WorldEpoch(3),
                &[EventLogEntry {
                    epoch: WorldEpoch(3),
                    sequence: 3,
                    event: base.clone(),
                    world_digest: [2u8; 32],
                }],
            )
            .unwrap();

        let from_epoch_2 = store.get_events_from(WorldEpoch(2));
        assert_eq!(from_epoch_2.len(), 2);
        assert_eq!(from_epoch_2[0].epoch, WorldEpoch(2));
        assert_eq!(from_epoch_2[1].epoch, WorldEpoch(3));
    }

    // ── test_event_store_snapshot ───────────────────────────────────────────

    #[test]
    fn test_event_store_snapshot() {
        let mut store = InMemoryEventStore::new();
        assert_eq!(store.latest_snapshot(), None);

        let snap = Snapshot {
            epoch: WorldEpoch(5),
            world_digest: [0xab; 32],
            entity_count: 10,
            component_count: 42,
            created_at: Timestamp(1_700_000_000_000_000_000),
        };
        store.store_snapshot(snap.clone()).unwrap();

        let latest = store.latest_snapshot().expect("should have snapshot");
        assert_eq!(latest.epoch, WorldEpoch(5));
        assert_eq!(latest.world_digest, [0xab; 32]);
        assert_eq!(latest.entity_count, 10);
        assert_eq!(latest.component_count, 42);
    }

    // ── test_event_store_epoch_mismatch_rejected ────────────────────────────

    #[test]
    fn test_event_store_epoch_mismatch_rejected() {
        let mut store = InMemoryEventStore::new();

        let entry = EventLogEntry {
            epoch: WorldEpoch(1),
            sequence: 1,
            event: DomainEvent {
                id: MessageId::compute(b"ev"),
                kind: "Test".into(),
                entity_id: None,
                payload: serde_json::json!({}),
            },
            world_digest: [0u8; 32],
        };

        // Try appending an entry with epoch 1 under batch epoch 2
        let result = store.append_events(WorldEpoch(2), &[entry]);
        assert!(result.is_err(), "epoch mismatch should be rejected");
        assert!(
            result.unwrap_err().contains("epoch mismatch"),
            "error should contain 'epoch mismatch'"
        );
    }

    // ── test_replay_engine ──────────────────────────────────────────────────

    #[test]
    fn test_replay_engine() {
        let mut store = InMemoryEventStore::new();
        let base = DomainEvent {
            id: MessageId::compute(b"r"),
            kind: "Event".into(),
            entity_id: None,
            payload: serde_json::json!({}),
        };

        for epoch in 1..=3 {
            store
                .append_events(
                    WorldEpoch(epoch),
                    &[EventLogEntry {
                        epoch: WorldEpoch(epoch),
                        sequence: epoch,
                        event: base.clone(),
                        world_digest: [epoch as u8; 32],
                    }],
                )
                .unwrap();
        }

        let result = ReplayEngine::replay(&store, WorldEpoch(2));
        assert_eq!(result.events_replayed, 2);
        assert_eq!(result.last_epoch, WorldEpoch(3));
    }

    // ── test_projection_checkpoint_serde ────────────────────────────────────

    #[test]
    fn test_projection_checkpoint_serde() {
        let cp = ProjectionCheckpoint {
            last_epoch: WorldEpoch(42),
            last_sequence: 7,
        };

        let json = serde_json::to_string(&cp).unwrap();
        let deserialized: ProjectionCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(cp, deserialized);
        assert_eq!(deserialized.last_epoch, WorldEpoch(42));
        assert_eq!(deserialized.last_sequence, 7);
    }

    // ── test_async_subscriber_construction ──────────────────────────────────

    #[test]
    fn test_async_subscriber_construction() {
        let bus = ReceiptBus::new();
        assert_eq!(bus.subscriber_count(), 0);

        struct NoopSub;
        impl ReceiptSubscriber for NoopSub {}

        let _rx = bus.subscribe(Box::new(NoopSub));
        assert_eq!(bus.subscriber_count(), 1);
    }
}
