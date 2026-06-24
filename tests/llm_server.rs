#![cfg(feature = "generation-image")]

// ── Prism LLM Inference — Integration Tests ─────────────────────────────
//
// Comprehensive hermetic tests for the LLM inference server runtime.
// All tests are self-contained: no hardware, network, or CImage files needed.

use prism_engine::llm::manifest;
use prism_engine::llm::runtime;
use prism_engine::llm::server;

// ── Helpers ──────────────────────────────────────────────────────────────

/// A minimal LlmCapabilityManifest that can be used in tests.
fn test_manifest() -> manifest::LlmCapabilityManifest {
    use manifest::{
        ComponentAvailability, ContextProfile, KvCacheContract, KvDtype,
        LlmCapabilityManifest, LlmModelFamily, LlmProviderArtifact,
        LlmQualificationRecord, QualificationStatus, ResidencyRequirements, RopeMode,
    };

    LlmCapabilityManifest {
        schema_version: 1,
        model_family: LlmModelFamily::Custom,
        tokenizer: ComponentAvailability::PresentQualified,
        embedding: ComponentAvailability::PresentQualified,
        transformer_blocks: ComponentAvailability::PresentQualified,
        lm_head: ComponentAvailability::PresentQualified,
        kv_cache_contract: KvCacheContract {
            layer_count: 32,
            attention_head_count: 32,
            kv_head_count: 8,
            head_dimension: 128,
            dtype: KvDtype::Fp16,
            rope_mode: RopeMode::Standard,
            supports_sparse_retention: true,
            supports_context_refresh: true,
            supports_position_renumbering: true,
            max_declared_context_tokens: 32768,
            page_token_capacity: 64,
        },
        supported_context_profiles: vec![ContextProfile {
            id: "default".into(),
            max_prompt_tokens: 4096,
            max_new_tokens: 1024,
            kv_page_capacity_tokens: 64,
            compression_threshold_tokens: None,
            refresh_threshold_tokens: None,
            memory_reservation_bytes: 1024 * 1024 * 1024,
        }],
        provider_artifacts: vec![LlmProviderArtifact {
            provider: "test".into(),
            artifact_id: "test-artifact".into(),
            compiler_id: "test-compiler".into(),
            abi_version: 1,
            required_hardware: vec![],
            tensor_layout: "default".into(),
        }],
        auxiliary_islands: vec![],
        residency_requirements: ResidencyRequirements {
            min_unified_memory_bytes: 1024 * 1024 * 1024,
            persistent_weight_bytes: 0,
            scratch_bytes: 0,
            kv_reservation_per_token: 0,
        },
        qualification: LlmQualificationRecord {
            status: QualificationStatus::Accepted,
            fixture_id: "unit-test".into(),
            verified_at: "2025-01-01T00:00:00Z".into(),
            failure_reason: None,
        },
    }
}

/// A minimal valid CreateSessionRequest.
fn create_request() -> server::CreateSessionRequest {
    use server::{AuxiliaryLanePolicy, CImageId, ContextProfileId, CreateSessionRequest, InferenceExecutionPolicy};
    CreateSessionRequest {
        cimage_id: CImageId("test-cimage".into()),
        context_profile: ContextProfileId("default".into()),
        execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
        auxiliary_lane_policy: AuxiliaryLanePolicy::Disabled,
    }
}

/// A test ScopedTempDir that cleans up when dropped.
struct ScopedTempDir(tempfile::TempDir);

impl ScopedTempDir {
    fn new() -> Self {
        Self(tempfile::tempdir().expect("failed to create temp dir"))
    }

    fn path(&self) -> &std::path::Path {
        self.0.path()
    }
}

// ── Test 1: Session Creation Lifecycle ──────────────────────────────────

#[test]
fn session_creation_lifecycle() {
    let mgr = runtime::SessionManager::new();

    let session_id = mgr
        .create_session(create_request())
        .expect("create_session should succeed");

    let state = mgr
        .get_state(&session_id)
        .expect("session should exist after creation");
    assert_eq!(
        state,
        server::InferenceSessionState::Ready,
        "new session should be in Ready state"
    );
}

// ── Test 2: Session State Transitions ───────────────────────────────────

#[test]
fn session_state_transitions() {
    let mgr = runtime::SessionManager::new();
    use server::InferenceSessionState;

    let session_id = mgr
        .create_session(create_request())
        .expect("create_session should succeed");

    // Starts at Ready after creation.
    assert_eq!(mgr.get_state(&session_id), Some(InferenceSessionState::Ready));

    // Ready -> Decoding
    mgr.transition(&session_id, InferenceSessionState::Decoding)
        .expect("Ready -> Decoding should succeed");
    assert_eq!(mgr.get_state(&session_id), Some(InferenceSessionState::Decoding));

    // Decoding -> Completed
    mgr.transition(&session_id, InferenceSessionState::Completed)
        .expect("Decoding -> Completed should succeed");
    assert_eq!(mgr.get_state(&session_id), Some(InferenceSessionState::Completed));

    // Completed -> Closed
    mgr.transition(&session_id, InferenceSessionState::Closed)
        .expect("Completed -> Closed should succeed");
    assert_eq!(mgr.get_state(&session_id), Some(InferenceSessionState::Closed));

    // Transition on non-existent session should fail.
    let bogus_id = manifest::SessionId(uuid::Uuid::new_v4());
    let result = mgr.transition(&bogus_id, InferenceSessionState::Ready);
    assert!(result.is_err(), "transition on non-existent session should fail");
}

// ── Test 3: Weight Residency Cache Hit ──────────────────────────────────

#[test]
fn weight_residency_cache_hit() {
    let mgr = runtime::WeightResidencyManager::new();
    let manifest = test_manifest();

    // Load a CImage.
    let key = mgr
        .load_cimage("/test/weights.safetensors", &manifest)
        .expect("load_cimage should succeed");

    // Pin once for session A.
    let session_a = manifest::SessionId(uuid::Uuid::new_v4());
    mgr.pin_for_session(&key, &session_a)
        .expect("first pin should succeed");

    // Pin again for session B.
    let session_b = manifest::SessionId(uuid::Uuid::new_v4());
    mgr.pin_for_session(&key, &session_b)
        .expect("second pin should succeed");

    // Verify cache hit.
    let receipt = mgr.get_residency_receipt(&key);
    assert!(receipt.cache_hit, "receipt should report cache_hit=true");
    assert_eq!(receipt.active_weight_leases, 2, "two sessions should hold leases");
    assert!(receipt.metal_visible, "weights should be metal visible");
}

// ── Test 4: KV Epoch Create, Seal, Dispatch ────────────────────────────

#[test]
fn kv_epoch_create_seal_dispatch() {
    use server::{KvEpochState, KvPage, KvPageId, KvPageState};

    let kv = runtime::KvManager::new(64, 32768);

    // Create an epoch.
    let epoch_id = kv.create_epoch(None).expect("create_epoch should succeed");

    // Add a page.
    let page = KvPage {
        page_id: KvPageId(0),
        layer_range: (0, 32),
        token_range: (0, 64),
        original_position_range: (0, 64),
        allocation_id: manifest::IslandAllocationId(0),
        residency: "default".into(),
        state: KvPageState::Allocated,
    };
    let _page_id = kv.add_page(&epoch_id, page).expect("add_page should succeed");

    // Seal the epoch.
    let epoch_receipt = kv.seal_epoch(&epoch_id).expect("seal_epoch should succeed");
    assert_eq!(epoch_receipt.epoch_id, epoch_id);
    assert_eq!(epoch_receipt.state, KvEpochState::Active);

    // Create a dispatch view.
    let view = kv
        .create_dispatch_view(&epoch_id, 0)
        .expect("create_dispatch_view should succeed");

    assert_eq!(view.epoch_id, epoch_id, "dispatch view should reference the correct epoch");
}

// ── Test 5: KV Sparse Retention Preserves Positions ────────────────────

#[test]
fn kv_sparse_retention_preserves_positions() {
    use server::{KvPage, KvPageId, KvPageState};

    let kv = runtime::KvManager::new(64, 32768);

    // Create and seal an epoch with multiple pages.
    let epoch_id = kv.create_epoch(None).expect("create_epoch");

    let page1 = KvPage {
        page_id: KvPageId(0),
        layer_range: (0, 32),
        token_range: (0, 64),
        original_position_range: (0, 64),
        allocation_id: manifest::IslandAllocationId(1),
        residency: "default".into(),
        state: KvPageState::Allocated,
    };
    let pid1 = kv.add_page(&epoch_id, page1).expect("add_page 1");

    let page2 = KvPage {
        page_id: KvPageId(0),
        layer_range: (0, 32),
        token_range: (64, 128),
        original_position_range: (64, 128),
        allocation_id: manifest::IslandAllocationId(2),
        residency: "default".into(),
        state: KvPageState::Allocated,
    };
    let pid2 = kv.add_page(&epoch_id, page2).expect("add_page 2");

    let _epoch_receipt = kv.seal_epoch(&epoch_id).expect("seal_epoch");

    // Build a sparse retention plan retaining both pages.
    let plan = kv
        .build_sparse_retention_plan(&epoch_id, vec![pid1, pid2])
        .expect("build_sparse_retention_plan");

    assert!(
        plan.preserves_absolute_positions,
        "sparse retention plan must preserve absolute positions"
    );
    assert_eq!(plan.source_epoch, epoch_id);
    assert_eq!(plan.retained_pages.len(), 2);
    assert!(plan.removed_pages.is_empty());
}

// ── Test 6: Cancellation Returns Receipt ───────────────────────────────

#[test]
fn cancellation_returns_receipt() {
    use server::InferenceSessionState;

    let mgr = runtime::CancellationManager::new();
    let session_id = manifest::SessionId(uuid::Uuid::new_v4());

    let handle = mgr.register_handle(session_id);
    assert_eq!(handle.session_id, session_id);

    // Cancel and verify receipt.
    let receipt = mgr.cancel(&handle).expect("cancel should succeed");
    assert_eq!(receipt.session_id, session_id, "receipt should have correct session_id");
    assert_eq!(receipt.state_at_cancellation, InferenceSessionState::Cancelling);

    // Verify the session is now marked cancelled.
    assert!(mgr.is_cancelled(&session_id), "session should be cancelled");
}

// ── Test 7: Memory Pressure Levels ─────────────────────────────────────

#[test]
fn memory_pressure_levels() {
    use server::{AllocationOwner, MemoryPressureLevel};

    let monitor = runtime::MemoryPressureMonitor::new(1000, 2000);

    // Level 1: Normal (below elevated threshold of 1000)
    monitor
        .record_allocation(500, AllocationOwner::WeightResidency)
        .expect("record_allocation 500 should succeed");
    assert_eq!(
        monitor.current_level(),
        MemoryPressureLevel::Normal,
        "500 bytes allocated should be Normal (threshold < 1000)"
    );

    // Level 2: Elevated (between 1000 and 2000)
    monitor
        .record_allocation(600, AllocationOwner::KvCache)
        .expect("record_allocation 600 should succeed");
    // Total: 1100
    assert_eq!(
        monitor.current_level(),
        MemoryPressureLevel::Elevated,
        "1100 bytes allocated should be Elevated"
    );

    // Level 3: Critical (above 2000)
    monitor
        .record_allocation(1000, AllocationOwner::TokenBuffer)
        .expect("record_allocation 1000 should succeed");
    // Total: 2100
    assert_eq!(
        monitor.current_level(),
        MemoryPressureLevel::Critical,
        "2100 bytes allocated should be Critical"
    );

    // Verify history recorded level transitions.
    let history = monitor.get_history();
    assert!(!history.is_empty(), "memory transitions should be recorded");
}

// ── Test 8: Receipt Store Full Lifecycle ────────────────────────────────

#[test]
fn receipt_store_full_lifecycle() {
    use server::{
        CImageId, ContextProfileId, CoreMlVisibilityState, DispatchId,
        InferenceAdmissionReceipt, InferenceExecutionPolicy, InferenceOutputReceipt,
        InferenceTerminalState, LaneExecutionReceipt, MetalExecutionReceipt, RequestId,
        WeightEvictionStatus, WeightResidencyReceipt,
    };

    let tmp = ScopedTempDir::new();
    let store = runtime::ReceiptStore::new(tmp.path().to_string_lossy().to_string());
    let session_id = manifest::SessionId(uuid::Uuid::new_v4());
    let request_id = RequestId(uuid::Uuid::new_v4());

    // Step 1: Create base receipt.
    let base = store.create_base_receipt(&session_id, &request_id);
    assert_eq!(base.session_id, session_id);
    assert_eq!(base.request_id, request_id);

    // Step 2: Record admission.
    let admission = InferenceAdmissionReceipt {
        cimage_id: CImageId("test".into()),
        context_profile: ContextProfileId("default".into()),
        execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
        admitted: true,
        refusal_reason: None,
    };
    store.record_admission(&session_id, admission.clone());

    // Step 3: Record residency.
    let residency = WeightResidencyReceipt {
        cimage_digest: prism_engine::image::types::ArtifactDigest("digest".into()),
        cache_hit: true,
        initial_load_bytes: 1024,
        decode_step_reload_count: 0,
        active_weight_leases: 1,
        metal_visible: true,
        accelerate_visible: false,
        coreml_auxiliary_visibility: CoreMlVisibilityState::NotVisible,
        materialization_events: vec![],
        eviction_status: WeightEvictionStatus::Retained,
    };
    store.record_residency(&session_id, residency);

    // Step 4: Record a lane execution receipt.
    let lane = LaneExecutionReceipt {
        lane: manifest::ExecutionLane::Metal,
        metal: Some(MetalExecutionReceipt {
            dispatch_id: DispatchId(1),
            phase: manifest::InferencePhase::Decode,
            kv_epoch: None,
            command_submission_time: "2025-01-01T00:00:00Z".into(),
            completion_time: "2025-01-01T00:00:01Z".into(),
            input_allocation_ids: vec![],
            output_allocation_ids: vec![],
            authoritative_result_committed: true,
        }),
        accelerate: None,
        coreml: None,
    };
    store.record_lane(&session_id, lane);

    // Step 5: Record output.
    let output = InferenceOutputReceipt {
        total_tokens: 42,
        tokens_per_second: 10.5,
        total_latency_ms: 4000.0,
        metal_decode_latency_ms: 4000.0,
    };
    store.record_output(&session_id, output);

    // Step 6: Finalize.
    let finalized = store
        .finalize(&session_id, InferenceTerminalState::Succeeded)
        .expect("finalize should succeed");
    assert_eq!(finalized.terminal_state, InferenceTerminalState::Succeeded);
    assert!(!finalized.completed_at.is_empty(), "completed_at should be set");

    // Step 7: Retrieve.
    let retrieved = store
        .get_receipt(&session_id)
        .expect("receipt should be retrievable");
    assert_eq!(retrieved.admission.admitted, true);
    assert_eq!(retrieved.output.as_ref().map(|o| o.total_tokens), Some(42));
}

// ── Test 9: Multi-Session Independence ──────────────────────────────────

#[test]
fn multi_session_independence() {
    use server::{
        AuxiliaryLanePolicy, CImageId, ContextProfileId, CreateSessionRequest,
        InferenceExecutionPolicy, InferenceSessionState,
    };

    let mgr = runtime::SessionManager::new();

    // Create 3 sessions with different context profiles.
    let profiles = ["profile-a", "profile-b", "profile-c"];
    let mut session_ids = Vec::new();

    for (i, profile) in profiles.iter().enumerate() {
        let request = CreateSessionRequest {
            cimage_id: CImageId(format!("cimage-{i}")),
            context_profile: ContextProfileId(profile.to_string()),
            execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
            auxiliary_lane_policy: AuxiliaryLanePolicy::Optional,
        };
        let sid = mgr.create_session(request).expect("create_session");
        session_ids.push(sid);
    }

    // All sessions should be independent — each in Ready state.
    for (i, sid) in session_ids.iter().enumerate() {
        let state = mgr
            .get_state(sid)
            .unwrap_or_else(|| panic!("session {i} should exist"));
        assert_eq!(
            state,
            InferenceSessionState::Ready,
            "session {i} should be Ready"
        );
    }

    // Transition session 0 to Decoding — session 1 and 2 must stay Ready.
    mgr.transition(&session_ids[0], InferenceSessionState::Decoding)
        .expect("session 0 -> Decoding");
    assert_eq!(
        mgr.get_state(&session_ids[0]),
        Some(InferenceSessionState::Decoding)
    );
    for (i, sid) in session_ids[1..].iter().enumerate() {
        assert_eq!(
            mgr.get_state(sid),
            Some(InferenceSessionState::Ready),
            "session {} should still be Ready when session 0 transitions",
            i + 1
        );
    }

    // Close session 0 — session 1 and 2 must remain Ready.
    mgr.close_session(&session_ids[0]).expect("close session 0");
    assert_eq!(
        mgr.get_state(&session_ids[0]),
        Some(InferenceSessionState::Closed)
    );
    for (i, sid) in session_ids[1..].iter().enumerate() {
        assert_eq!(
            mgr.get_state(sid),
            Some(InferenceSessionState::Ready),
            "session {} should remain Ready after session 0 closed",
            i + 1
        );
    }
}

// ── Test 10: PrismInferenceServer Create and Close ──────────────────────

#[test]
fn prism_inference_server_create_and_close() {
    use server::{
        AuxiliaryLanePolicy, CImageId, ContextProfileId, CreateSessionRequest,
        InferenceExecutionPolicy, InferenceSessionState,
    };

    let tmp = ScopedTempDir::new();
    let receipt_path = tmp.path().join("receipts");
    let config = runtime::ServerConfig {
        cimage_path: tmp.path().to_string_lossy().to_string(),
        context_profiles: vec![manifest::ContextProfile {
            id: "default".into(),
            max_prompt_tokens: 4096,
            max_new_tokens: 1024,
            kv_page_capacity_tokens: 64,
            compression_threshold_tokens: None,
            refresh_threshold_tokens: None,
            memory_reservation_bytes: 1024 * 1024 * 1024,
        }],
        execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
        max_concurrent_sessions: 10,
        http_listen: None,
        receipt_store_path: receipt_path.to_string_lossy().to_string(),
        memory_elevated_threshold_bytes: 1_000_000_000,
        memory_critical_threshold_bytes: 2_000_000_000,
    };

    let server_inst = runtime::PrismInferenceServer::new(config);

    // Create a session via the server.
    let request = CreateSessionRequest {
        cimage_id: CImageId("test-cimage".into()),
        context_profile: ContextProfileId("default".into()),
        execution_policy: InferenceExecutionPolicy::RequireMetalDecode,
        auxiliary_lane_policy: AuxiliaryLanePolicy::Disabled,
    };

    let session_id = server_inst
        .create_session(request)
        .expect("server create_session should succeed");

    // Verify session exists and is Ready.
    let state = server_inst
        .session_manager
        .get_state(&session_id)
        .expect("session should exist");
    assert_eq!(state, InferenceSessionState::Ready);

    // Close the session via the server.
    server_inst
        .close_session(session_id)
        .expect("close_session should succeed");

    // Verify session is now Closed.
    let closed_state = server_inst
        .session_manager
        .get_state(&session_id)
        .expect("session should still exist after close");
    assert_eq!(closed_state, InferenceSessionState::Closed);
}
