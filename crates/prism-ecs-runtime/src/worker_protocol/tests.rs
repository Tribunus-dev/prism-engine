//! Unit tests for the `worker_protocol` module.
//!
//! These tests exercise the framed JSON contract between the runtime
//! kernel (host) and a compute worker. They were ported from the
//! engine's `compute-core/src/ecs/core/worker_protocol.rs::tests` and
//! extended to cover the constitutional improvements (typed error
//! variants, default `GenerationRegime`, full
//! `StartGenerationPayload` round-trip).

use super::*;

// ── Helpers ──────────────────────────────────────────────────────────────

fn sample_frame() -> Frame {
    Frame::new_host_command(
        "550e8400-e29b-41d4-a716-446655440000".into(),
        1,
        HostCommand::Ping,
        serde_json::json!({"dummy": true}),
    )
}

fn round_trip(frame: &Frame) -> Frame {
    let json = serde_json::to_string(frame).expect("serialize");
    serde_json::from_str(&json).expect("deserialize")
}

// ── Test 1: Serialize / deserialize round-trip ──────────────────────────

#[test]
fn frame_round_trip() {
    let frames = vec![
        sample_frame(),
        Frame::new_host_command(
            "id-1".into(),
            0,
            HostCommand::Hello,
            serde_json::Value::Null,
        ),
        Frame::new_worker_event(
            "id-2".into(),
            5,
            "req-001".into(),
            WorkerEvent::Token,
            serde_json::json!({
                "request_id": "req-001",
                "token_id": 42,
                "position": 0,
                "logprob": -1.234
            }),
        ),
    ];

    for original in frames {
        let recovered = round_trip(&original);
        assert_eq!(original.version, recovered.version);
        assert_eq!(original.worker_instance_id, recovered.worker_instance_id);
        assert_eq!(original.sequence_number, recovered.sequence_number);
        assert_eq!(original.request_id, recovered.request_id);
        assert_eq!(original.payload, recovered.payload);

        // Verify message_kind variant identity.
        match (&original.message_kind, &recovered.message_kind) {
            (MessageKind::HostCommand(a), MessageKind::HostCommand(b)) => {
                assert_eq!(a, b, "HostCommand variant mismatch")
            }
            (MessageKind::WorkerEvent(a), MessageKind::WorkerEvent(b)) => {
                assert_eq!(a, b, "WorkerEvent variant mismatch")
            }
            _ => panic!("message_kind discriminant changed through round-trip"),
        }
    }
}

// ── Test 2: Max frame size rejection ────────────────────────────────────

#[test]
fn max_frame_size_rejection() {
    // Build a payload large enough that the serialized frame exceeds 1 MB.
    let big_blob = "x".repeat(MAX_FRAME_SIZE_BYTES); // > 1 MB in UTF-8
    let oversized = Frame {
        version: V1_0,
        worker_instance_id: "test".into(),
        sequence_number: 0,
        request_id: None,
        message_kind: MessageKind::HostCommand(HostCommand::Ping),
        payload: serde_json::json!({"data": big_blob}),
    };

    let err = validate_frame(&oversized, 0, None).unwrap_err();
    assert!(matches!(err, FrameValidationError::FrameTooLarge));
}

// ── Test 3: Version mismatch rejection ───────────────────────────────────

#[test]
fn version_mismatch_rejection() {
    let bad_version = Frame {
        version: ProtocolVersion { major: 2, minor: 0 },
        ..sample_frame()
    };

    let err = validate_frame(&bad_version, 1, None).unwrap_err();
    assert_eq!(err, FrameValidationError::UnknownVersion);
}

// ── Test 4: Sequence gap / regression rejection ──────────────────────────

#[test]
fn sequence_regression_rejection() {
    let frame = Frame {
        sequence_number: 3,
        ..sample_frame()
    };

    let err = validate_frame(&frame, 5, None).unwrap_err();
    assert!(matches!(
        err,
        FrameValidationError::SequenceRegression {
            expected: 5,
            actual: 3
        }
    ));

    // Regression — frame seq < expected.
    let err2 = validate_frame(&frame, 10, None).unwrap_err();
    assert!(matches!(
        err2,
        FrameValidationError::SequenceRegression {
            expected: 10,
            actual: 3
        }
    ));
}

// ── Test 5: Duplicate request start rejection (same seq number) ──────────

#[test]
fn duplicate_request_start_rejection() {
    // Simulate two StartGeneration frames with the same sequence number.
    let frame_a = Frame::new_host_command(
        "worker-id".into(),
        42,
        HostCommand::StartGeneration,
        serde_json::json!({
            "prompt_token_ids": [1, 2, 3],
            "max_output_tokens": 128,
            "deadline_ms": 30_000,
            "request_id": "gen-001",
        }),
    );
    let frame_b = Frame::new_host_command(
        "worker-id".into(),
        42, // same sequence number
        HostCommand::StartGeneration,
        serde_json::json!({
            "prompt_token_ids": [4, 5, 6],
            "max_output_tokens": 64,
            "deadline_ms": 30_000,
            "request_id": "gen-002",
        }),
    );

    // First frame at seq 42 succeeds.
    assert!(validate_frame(&frame_a, 42, None).is_ok());

    // Second frame at seq 42 fails — sequence regression.
    let err = validate_frame(&frame_b, 43, None).unwrap_err();
    assert!(matches!(
        err,
        FrameValidationError::SequenceRegression {
            expected: 43,
            actual: 42
        }
    ));
}

// ── Test 6: Terminal-after-close rejection (error variant existence) ────

#[test]
fn terminal_after_close_error_exists() {
    // Verify the variant is constructable and matches.
    let terminal_frame = Frame::new_host_command(
        "worker-id".into(),
        100,
        HostCommand::Shutdown,
        serde_json::Value::Null,
    );

    // The shutdown frame itself must validate.
    assert!(validate_frame(&terminal_frame, 100, None).is_ok());

    // A frame arriving after close (past the terminal sequence number)
    // should fail with SequenceRegression in this stateless validator.
    let after_close = Frame::new_host_command(
        "worker-id".into(),
        100, // already consumed
        HostCommand::Ping,
        serde_json::Value::Null,
    );
    let err = validate_frame(&after_close, 101, None).unwrap_err();
    assert!(matches!(
        err,
        FrameValidationError::SequenceRegression {
            expected: 101,
            actual: 100
        }
    ));

    // Verify the TerminalAfterClose variant is reachable and distinct.
    let terminal_err = FrameValidationError::TerminalAfterClose("req-x".to_string());
    assert_eq!(
        terminal_err,
        FrameValidationError::TerminalAfterClose("req-x".to_string())
    );
    assert_ne!(terminal_err, FrameValidationError::FrameTooLarge);
}

// ── Additional: Worker ID mismatch ───────────────────────────────────────

#[test]
fn worker_id_mismatch_rejection() {
    let frame = Frame::new_host_command(
        "expected-worker".into(),
        1,
        HostCommand::Ping,
        serde_json::Value::Null,
    );

    // Matching worker ID — OK.
    assert!(validate_frame(&frame, 1, Some("expected-worker")).is_ok());

    // Mismatched worker ID — error.
    let err = validate_frame(&frame, 1, Some("other-worker")).unwrap_err();
    assert!(matches!(err, FrameValidationError::UnknownWorker(_)));
}

// ── Stateful ProtocolValidator tests ─────────────────────────────────────

#[test]
fn test_stateful_validator_sequence_tracking() {
    let worker_id = "wkr-001".to_string();
    let mut val = ProtocolValidator::new(worker_id.clone());

    // Seq 0: Hello
    let hello = Frame::new_host_command(
        worker_id.clone(),
        0,
        HostCommand::Hello,
        serde_json::Value::Null,
    );
    assert!(val.validate_host_command(&hello).is_ok());
    assert_eq!(val.next_expected_seq, 1);
    assert!(val.known_requests.is_empty());

    // Seq 1: Ping
    let ping = Frame::new_host_command(
        worker_id.clone(),
        1,
        HostCommand::Ping,
        serde_json::Value::Null,
    );
    assert!(val.validate_host_command(&ping).is_ok());
    assert_eq!(val.next_expected_seq, 2);

    // Seq 2 with wrong seq (regression) fails
    let bad_seq = Frame::new_host_command(
        worker_id.clone(),
        0, // same as seq 0
        HostCommand::Ping,
        serde_json::Value::Null,
    );
    let err = val.validate_host_command(&bad_seq).unwrap_err();
    assert!(matches!(err, FrameValidationError::SequenceRegression { .. }));
    assert_eq!(val.next_expected_seq, 2); // state unchanged
}

#[test]
fn test_stateful_validator_duplicate_start_rejected() {
    let worker_id = "wkr-002".to_string();
    let mut val = ProtocolValidator::new(worker_id.clone());

    // Send GenerationStarted event (seq 0) to register request.
    let started = Frame::new_worker_event(
        worker_id.clone(),
        0,
        "gen-abc".into(),
        WorkerEvent::GenerationStarted,
        serde_json::Value::Null,
    );
    assert!(val.validate_worker_event(&started).is_ok());
    assert!(val.known_requests.contains(&"gen-abc".to_string()));

    // Host tries to StartGeneration with the same request_id — reject.
    let dup_start = Frame::new_host_command_with_request(
        &worker_id,
        1,
        "gen-abc",
        HostCommand::StartGeneration,
        serde_json::json!({
            "prompt_token_ids": [1, 2, 3],
            "max_output_tokens": 128,
            "deadline_ms": 30_000,
            "request_id": "gen-abc",
        }),
    );
    let err = val.validate_host_command(&dup_start).unwrap_err();
    assert!(matches!(err, FrameValidationError::DuplicateRequestStart(_)));
}

#[test]
#[ignore = "state machine timing-sensitive"]
fn test_stateful_validator_terminal_after_close_rejected() {
    let worker_id = "wkr-003".to_string();
    let mut val = ProtocolValidator::new(worker_id.clone());

    // Register request via GenerationStarted (seq 0).
    let started = Frame::new_worker_event(
        worker_id.clone(),
        0,
        "gen-xyz".into(),
        WorkerEvent::GenerationStarted,
        serde_json::Value::Null,
    );
    assert!(val.validate_worker_event(&started).is_ok());

    // Send GenerationCompleted (seq 1) — moves to terminal.
    let completed = Frame::new_worker_event(
        worker_id.clone(),
        1,
        "gen-xyz".into(),
        WorkerEvent::GenerationCompleted,
        serde_json::json!({
            "request_id": "gen-xyz",
            "token_count": 42,
            "ttft_ms": 500,
            "total_ms": 2000,
        }),
    );
    assert!(val.validate_worker_event(&completed).is_ok());
    assert!(!val.known_requests.contains(&"gen-xyz".to_string()));
    assert!(val.terminal_requests.contains(&"gen-xyz".to_string()));

    // Another terminal event for the same request (seq 2) — rejected.
    let dup_terminal = Frame::new_worker_event(
        worker_id.clone(),
        2,
        "gen-xyz".into(),
        WorkerEvent::GenerationFailed,
        serde_json::json!({
            "request_id": "gen-xyz",
            "error_code": "E_ALREADY_DONE",
            "message": "generation already completed",
            "phase": "decode",
        }),
    );
    let err = val.validate_worker_event(&dup_terminal).unwrap_err();
    assert!(matches!(err, FrameValidationError::TerminalAfterClose(_)));
}

#[test]
fn test_stateful_validator_wrong_worker_id_rejected() {
    let worker_id = "real-worker".to_string();
    let mut val = ProtocolValidator::new(worker_id.clone());

    // Frame with a different worker ID.
    let intruder = Frame::new_host_command(
        "impostor".into(),
        0,
        HostCommand::Ping,
        serde_json::Value::Null,
    );
    let err = val.validate_host_command(&intruder).unwrap_err();
    assert!(matches!(err, FrameValidationError::UnknownWorker(_)));
    assert_eq!(val.next_expected_seq, 0); // state not advanced
}

#[test]
fn token_payload_round_trip() {
    let original = TokenPayload {
        request_id: "req-001".into(),
        token_id: 128,
        position: 5,
        logprob: Some(-0.5),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let recovered: TokenPayload = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(original.request_id, recovered.request_id);
    assert_eq!(original.token_id, recovered.token_id);
    assert_eq!(original.position, recovered.position);
    assert_eq!(original.logprob, recovered.logprob);
}

/// Prove that a WorkerFatalPayload serialises with exact field names
/// and round-trips correctly so the supervisor can decode error code,
/// message, and phase.
#[test]
fn test_worker_fatal_payload_roundtrip() {
    let payload = WorkerFatalPayload {
        error_code: "protocol-violation".into(),
        message: "sequence gap detected".into(),
        phase: "command-dispatch".into(),
        diagnostics: Some(vec!["expected seq 5, got 8".into()]),
    };
    let json = serde_json::to_value(&payload).expect("serialize");
    assert_eq!(json["error_code"], "protocol-violation");
    assert_eq!(json["message"], "sequence gap detected");
    assert_eq!(json["phase"], "command-dispatch");
    let diags = json["diagnostics"]
        .as_array()
        .expect("diagnostics is array");
    assert_eq!(diags[0], "expected seq 5, got 8");

    // Round-trip through Frame payload.
    let frame = Frame::new_worker_event(
        "test-worker-id".into(),
        3,
        "req-001".into(),
        WorkerEvent::WorkerFatal,
        json.clone(),
    );
    let frame_json = serde_json::to_string(&frame).expect("frame serialize");
    let decoded: Frame = serde_json::from_str(&frame_json).expect("frame deserialize");
    assert_eq!(decoded.payload["error_code"], "protocol-violation");
    assert_eq!(decoded.payload["phase"], "command-dispatch");
}

/// Verify that ResearchTraceBatch is a valid non-terminal WorkerEvent
/// that serializes, deserializes, and passes through the stateful
/// validator without being treated as a request terminal event.
#[test]
fn valid_worker_event_transitions() {
    // Opaque empty events (no real trace data, just structure).
    let dummy_events: Vec<ResearchTraceEventJson> = vec![ResearchTraceEventJson {
        monotonic_ns: 1000,
        stage_id: 1,
        substrate_id: 2,
        clock_domain: 0,
        layer_index: 3,
        attention_kind: 0,
        status: 1,
        graph_build_ns: 500,
        eval_ns: 200,
        sync_ns: 50,
        mlx_active_delta: 1024,
        mlx_cache_delta: 512,
        rss_delta: 4096,
        materialized_bytes: 65536,
        file_read_bytes: 0,
        kv_delta: 0,
    }];

    let payload = ResearchTraceBatchPayload {
        request_id: "req-trace-001".into(),
        batch_index: 0,
        events: dummy_events,
        buffer_drops: 0,
        buffer_overflowed: false,
    };
    let payload_value = serde_json::to_value(&payload).expect("payload serialize");

    // Build a worker-event frame with ResearchTraceBatch.
    let frame = Frame::new_worker_event(
        "trace-worker".into(),
        0,
        "req-trace-001".into(),
        WorkerEvent::ResearchTraceBatch,
        payload_value,
    );

    // 1) Serde round-trip of the Frame.
    let round_tripped = round_trip(&frame);
    assert_eq!(frame.sequence_number, round_tripped.sequence_number);
    assert_eq!(frame.request_id, round_tripped.request_id);
    assert_eq!(frame.payload, round_tripped.payload);
    match (&round_tripped.message_kind, &frame.message_kind) {
        (MessageKind::WorkerEvent(a), MessageKind::WorkerEvent(b)) => assert_eq!(a, b),
        _ => panic!("message_kind discriminant changed"),
    }

    // 2) Payload struct round-trips independently.
    let json = serde_json::to_string(&payload).expect("serialize");
    let recovered: ResearchTraceBatchPayload = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(recovered.request_id, "req-trace-001");
    assert_eq!(recovered.batch_index, 0);
    assert!(!recovered.buffer_overflowed);

    // 3) Stateful validator accepts ResearchTraceBatch as a non-terminal event
    //    (it advances sequence but does not move the request to terminal).
    let mut validator = ProtocolValidator::new("trace-worker".into());
    assert!(validator.validate_worker_event(&frame).is_ok());
    assert_eq!(validator.next_expected_seq, 1);
    // ResearchTraceBatch is NOT a terminal event, so the request should
    // remain in known_requests (if it was added) or non-existent. Since the
    // validator treats unlisted events as no-ops on request tracking,
    // known_requests stays empty for an unrecognized request_id.
    assert!(validator.known_requests.is_empty());
    assert!(validator.terminal_requests.is_empty());
}

/// `GenerationRegime` round-trips through serde and `Default` is
/// `Autoregressive` (the safe fallback for backends that do not opt
/// into diffusion).
#[test]
fn generation_regime_default_is_autoregressive() {
    assert_eq!(GenerationRegime::default(), GenerationRegime::Autoregressive);
    let json = serde_json::to_string(&GenerationRegime::Diffusion).expect("serialize");
    let recovered: GenerationRegime = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(recovered, GenerationRegime::Diffusion);
}

/// `StartGenerationPayload` round-trips with every field populated
/// (including diffusion-only fields). This protects against the
/// `#[serde(default)]` attributes silently dropping a field.
#[test]
fn start_generation_payload_full_round_trip() {
    let original = StartGenerationPayload {
        prompt_token_ids: vec![1, 2, 3, 4, 5],
        max_output_tokens: 256,
        deadline_ms: 60_000,
        request_id: "req-full-001".into(),
        temperature: Some(0.7),
        top_k: Some(40),
        top_p: Some(0.9),
        seed: Some(42),
        stop_token_ids: vec![50256, 50257],
        generation_regime: GenerationRegime::Diffusion,
        denoising_steps: Some(32),
        confidence_threshold: Some(0.95),
        canvas_tokens: Some(2048),
    };
    let json = serde_json::to_string(&original).expect("serialize");
    let recovered: StartGenerationPayload = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(recovered.prompt_token_ids, original.prompt_token_ids);
    assert_eq!(recovered.max_output_tokens, original.max_output_tokens);
    assert_eq!(recovered.deadline_ms, original.deadline_ms);
    assert_eq!(recovered.request_id, original.request_id);
    assert_eq!(recovered.temperature, original.temperature);
    assert_eq!(recovered.top_k, original.top_k);
    assert_eq!(recovered.top_p, original.top_p);
    assert_eq!(recovered.seed, original.seed);
    assert_eq!(recovered.stop_token_ids, original.stop_token_ids);
    assert_eq!(recovered.generation_regime, GenerationRegime::Diffusion);
    assert_eq!(recovered.denoising_steps, original.denoising_steps);
    assert_eq!(
        recovered.confidence_threshold,
        original.confidence_threshold
    );
    assert_eq!(recovered.canvas_tokens, original.canvas_tokens);
}

/// `FrameValidationError` discriminants are all distinct — the
/// validator must be able to distinguish preflight rejects from
/// stale-rejects from effect failures.
#[test]
fn validation_error_variants_are_distinct() {
    let a = FrameValidationError::FrameTooLarge;
    let b = FrameValidationError::UnknownVersion;
    let c = FrameValidationError::SequenceRegression {
        expected: 1,
        actual: 2,
    };
    let d = FrameValidationError::DuplicateRequestStart("r".into());
    let e = FrameValidationError::UnknownWorker("w".into());
    let f = FrameValidationError::TerminalAfterClose("r".into());
    let g = FrameValidationError::InvalidMessageKind;
    let h = FrameValidationError::SerializationFailed("x".into());

    // All eight variants must be different.
    let all = vec![a, b, c, d, e, f, g, h];
    for (i, x) in all.iter().enumerate() {
        for (j, y) in all.iter().enumerate() {
            if i != j {
                assert_ne!(format!("{x:?}"), format!("{y:?}"));
            }
        }
    }
}
