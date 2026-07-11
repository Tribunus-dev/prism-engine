#[cfg(test)]
mod tests {
    use crate::ecs::constitutional::command::*;
    use crate::ecs::constitutional::*;

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
}
