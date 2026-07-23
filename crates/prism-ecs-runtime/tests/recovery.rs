use prism_ecs_runtime::*;

/// Helper to create a kernel with in-memory adapters.
fn test_kernel() -> RuntimeKernel {
    RuntimeKernel::with_ports(
        Box::new(test_adapters::InMemoryCommandStore::new()),
        Box::new(test_adapters::InMemorySnapshotStore::new()),
        Box::new(test_adapters::InMemoryTickReceiptStore::new()),
        Box::new(test_adapters::InMemoryLeaseCoordinator::new()),
        Box::new(test_adapters::DeterministicClock::new(1000)),
    )
}

#[test]
fn test_recovery_from_snapshot_and_replay() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Submit a spawn command and verify it completes
    let result = handle
        .submit(CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 1,
            task: "recovery test".into(),
            max_steps: 5,
        }))
        .expect("spawn")
        .result;

    let entity_id = match result {
        CommandResult::Spawned { entity_id } => entity_id,
        _ => panic!("expected Spawned"),
    };

    // Verify entity is visible in query
    let agents = handle.query_agents();
    assert!(
        agents.iter().any(|a| a.entity_id == entity_id),
        "entity should exist before recovery"
    );

    // Capture and persist snapshot
    kernel.save_snapshot().expect("save snapshot");

    // Recover — restores allocator from snapshot and replays commands
    let report = kernel.recover().expect("recover");
    assert_eq!(
        report.recovery_state, "recovered",
        "should report recovered state"
    );
    assert!(
        report.replayed_commands >= 1,
        "should replay at least 1 command"
    );

    // After recovery, entity should still be visible
    let agents_after = handle.query_agents();
    assert!(
        agents_after.iter().any(|a| a.entity_id == entity_id),
        "entity should still exist after recovery"
    );
}

#[test]
fn test_recovery_with_no_snapshot() {
    let kernel = test_kernel();
    let report = kernel
        .recover()
        .expect("recovery should succeed with no snapshot");
    assert_eq!(report.recovery_state, "fresh");
    assert_eq!(report.replayed_commands, 0);
    assert_eq!(report.snapshot_epoch, 0);
    assert_eq!(report.unresolved_commands, 0);
}

#[test]
fn test_replay_preserves_entity_ids() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Spawn two entities
    let r1 = handle
        .submit(CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 1,
            task: "first".into(),
            max_steps: 3,
        }))
        .expect("spawn 1")
        .result;
    let id1 = match r1 {
        CommandResult::Spawned { entity_id } => entity_id,
        _ => panic!("expected Spawned"),
    };

    let r2 = handle
        .submit(CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 1,
            task: "second".into(),
            max_steps: 3,
        }))
        .expect("spawn 2")
        .result;
    let id2 = match r2 {
        CommandResult::Spawned { entity_id } => entity_id,
        _ => panic!("expected Spawned"),
    };

    assert!(id2 > id1, "entity IDs should be monotonic");

    // Capture snapshot and recover
    kernel.save_snapshot().expect("save snapshot");
    let report = kernel.recover().expect("recover");
    assert!(
        report.replayed_commands >= 2,
        "should replay at least 2 commands"
    );

    // After recovery, agents should be visible
    let agents = handle.query_agents();
    assert!(
        agents.iter().any(|a| a.phase == "Planning"),
        "recovered agents should have Planning phase"
    );
    assert!(
        agents.iter().any(|a| a.entity_id == id1),
        "first entity should exist after recovery"
    );
    assert!(
        agents.iter().any(|a| a.entity_id == id2),
        "second entity should exist after recovery"
    );
}

#[test]
fn test_recovery_with_cancel_before_replay() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Spawn and cancel an agent
    let r = handle
        .submit(CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 1,
            task: "to-cancel".into(),
            max_steps: 3,
        }))
        .expect("spawn")
        .result;
    let id = match r {
        CommandResult::Spawned { entity_id } => entity_id,
        _ => panic!("expected Spawned"),
    };

    handle
        .submit(CommandEnvelope::new(Command::CancelAgent { agent_id: id }))
        .expect("cancel");

    // Save snapshot and recover
    kernel.save_snapshot().expect("save snapshot");
    let report = kernel.recover().expect("recover");
    assert!(
        report.replayed_commands >= 2,
        "should replay spawn + cancel"
    );

    // After recovery, agent should have Completed lifecycle
    let agents = handle.query_agents();
    let agent = agents.iter().find(|a| a.entity_id == id);
    match agent {
        Some(a) => assert_eq!(
            a.lifecycle, "Completed",
            "cancelled agent should have Completed lifecycle after recovery"
        ),
        None => panic!("cancelled agent should exist after recovery"),
    }
}

#[test]
fn test_recover_rejects_corrupt_snapshot() {
    // Create a store pre-loaded with a deliberately corrupted snapshot
    let store = test_adapters::InMemorySnapshotStore::new();
    let bad_payload = SnapshotPayload {
        schema_version: 1,
        world_epoch: 99,
        next_entity_id: 0,
        last_command_sequence: 1,
        allocator_data: vec![],
        schedule_hash: [0u8; 32],
        created_at_ms: 0,
    };
    let bad_snapshot = WorldSnapshot {
        checksum: [0xAAu8; 32], // intentionally wrong — does not match payload
        payload: bad_payload,
    };
    store.save(&bad_snapshot).expect("save corrupt snapshot");

    // Build a kernel that inherits the corrupted snapshot store
    let kernel = RuntimeKernel::with_ports(
        Box::new(test_adapters::InMemoryCommandStore::new()),
        Box::new(store),
        Box::new(test_adapters::InMemoryTickReceiptStore::new()),
        Box::new(test_adapters::InMemoryLeaseCoordinator::new()),
        Box::new(test_adapters::DeterministicClock::new(1000)),
    );

    // Recovery must reject the corrupt snapshot
    let err = kernel.recover().unwrap_err();
    assert!(
        err.to_string().contains("checksum"),
        "error should mention checksum, got: {err}"
    );
}

#[test]
fn test_recovery_preserves_multiple_agents() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Spawn several agents with different parents
    let ids: Vec<u64> = (0..5)
        .map(|i| {
            let result = handle
                .submit(CommandEnvelope::new(Command::SpawnAgent {
                    parent_id: i,
                    task: format!("agent-{i}"),
                    max_steps: 10,
                }))
                .expect("spawn")
                .result;
            match result {
                CommandResult::Spawned { entity_id } => entity_id,
                _ => panic!("expected Spawned"),
            }
        })
        .collect();

    assert_eq!(ids.len(), 5, "all five agents should have unique IDs");

    kernel.save_snapshot().expect("save snapshot");
    let report = kernel.recover().expect("recover");
    assert!(
        report.replayed_commands >= 5,
        "should replay all 5 commands"
    );

    let agents = handle.query_agents();
    assert_eq!(
        agents.len(),
        5,
        "all 5 agents should be present after recovery"
    );

    for id in &ids {
        assert!(
            agents.iter().any(|a| a.entity_id == *id),
            "agent {id} should exist after recovery"
        );
    }
}

#[test]
fn test_crash_after_mutation_before_completion() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Submit a spawn and verify it produces an entity
    let result = handle
        .submit(CommandEnvelope::new(Command::SpawnAgent {
            parent_id: 1,
            task: "crash-test".into(),
            max_steps: 5,
        }))
        .expect("spawn")
        .result;
    let entity_id = match result {
        CommandResult::Spawned { entity_id } => entity_id,
        _ => panic!("expected Spawned"),
    };

    // Verify entity exists before recovery
    let agents_before = handle.query_agents();
    assert!(
        agents_before.iter().any(|a| a.entity_id == entity_id),
        "entity should exist before recovery"
    );

    // Recover without a snapshot — replays from command history
    let report = kernel.recover().expect("recover");
    assert!(
        report.replayed_commands >= 1,
        "should replay at least 1 command, got {}",
        report.replayed_commands
    );

    // Entity should exist after recovery
    let agents_after = handle.query_agents();
    assert!(
        agents_after.iter().any(|a| a.entity_id == entity_id),
        "entity should still exist after crash-recovery replay"
    );
}

#[test]
fn test_recovery_with_no_snapshot_nonempty_history() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Submit commands without ever saving a snapshot
    for i in 0..3 {
        let _ = handle
            .submit(CommandEnvelope::new(Command::SpawnAgent {
                parent_id: i,
                task: format!("no-snap-{i}"),
                max_steps: 5,
            }))
            .expect("spawn")
            .result;
    }

    // Recover — should replay from 0
    let report = kernel
        .recover()
        .expect("recover without snapshot, with nonempty history");
    assert!(
        report.replayed_commands >= 3,
        "should replay all 3 commands from history, got {}",
        report.replayed_commands
    );
    assert_eq!(
        report.recovery_state, "recovered",
        "should be recovered, not fresh"
    );

    // Verify all agents exist after recovery
    let agents = handle.query_agents();
    assert_eq!(
        agents.len(),
        3,
        "all 3 agents should be present after replay-from-zero"
    );
}

#[test]
fn test_crash_after_completion_before_response() {
    let kernel = test_kernel();
    let handle = kernel.handle();

    // Submit a spawn, capturing the idempotency key for later re-use
    let env = CommandEnvelope::new(Command::SpawnAgent {
        parent_id: 1,
        task: "done-before-reply".into(),
        max_steps: 5,
    });
    let original_key = env.idempotency_key;

    let r1 = handle.submit(env).expect("first submit");
    let _entity_id = match r1.result {
        CommandResult::Spawned { entity_id } => entity_id,
        _ => panic!("expected Spawned"),
    };

    // Simulate crash: recover resets the world and replays history.
    // The command store still holds the completed entry.
    let report = kernel.recover().expect("recover");
    assert!(
        report.replayed_commands >= 1,
        "should replay at least 1 command"
    );

    // Re-submit with the same idempotency key — should return the original outcome
    let mut env2 = CommandEnvelope::new(Command::SpawnAgent {
        parent_id: 1,
        task: "duplicate".into(),
        max_steps: 5,
    });
    env2.idempotency_key = original_key;

    let r2 = handle.submit(env2).expect("second submit with same key");
    assert_eq!(
        r1.sequence, r2.sequence,
        "duplicate should return same sequence: r1={} r2={}",
        r1.sequence, r2.sequence
    );
    // CommandResult does not implement PartialEq, so compare by variant
    match (&r1.result, &r2.result) {
        (CommandResult::Spawned { entity_id: e1 }, CommandResult::Spawned { entity_id: e2 }) => {
            assert_eq!(e1, e2, "duplicate spawn should return same entity_id");
        }
        _ => panic!("expected both results to be Spawned, got {r1:?} and {r2:?}",),
    }
}
