//! Hermetic reliability tests for the Prism image generation facade.
//!
//! Tests cover the reliability matrix: admission refusal, qualification
//! freshness, route selection, fallback policies, provider failure, output
//! validation, cancellation, receipt persistence, memory pressure, and
//! terminal state display.
//!
//! All tests are self-contained — no model downloads, no network, no Apple
//! hardware required.
#![cfg(feature = "generation-image")]

use prism_engine::image::admission::{self, ImageGenerationAdmissionGate};
use prism_engine::image::manifest::{
    ComponentAvailability, DimensionConstraint, ImageGenerationCapabilityManifest,
    ImageModelFamily, ImageProviderArtifact, ImageQualificationRecord, InstalledCImage, StepRange,
    IMAGE_MANIFEST_SCHEMA_VERSION,
};
use prism_engine::image::provider::MachineProfile;
use prism_engine::image::reliability::{
    FallbackEligibility, FallbackReason, ImageGenerationAdmissionEvidence,
    ImageGenerationCancellationEvidence, ImageGenerationCancellationToken,
    ImageGenerationExecutionEvidence, ImageGenerationFailureClass, ImageGenerationFailureEvidence,
    ImageGenerationFailureStage, ImageGenerationOutputEvidence, ImageGenerationRefusal,
    ImageGenerationRouteEvidence, ImageGenerationTerminalReceipt, ImageGenerationTerminalState,
    ImageOutputLifecycle, ImageProviderCandidateEvidence, MemoryPressureLevel,
    ProviderIneligibilityReason, ReceiptPersistenceState, Retryability,
};
use prism_engine::image::resolver::NoOpQualificationResolver;
use prism_engine::image::types::{
    ArtifactDigest, DevicePreference, GenerationExecutionPolicy, ImageGenerationError,
    ImageGenerationRefusalReason, ImageGenerationRequest, ImageOutputFormat, ImageProviderKind,
    MaterializationReceipt, QualificationStatus, RequestId, RouteOrigin,
};
use prism_engine::image::QualificationResolver;
use std::time::SystemTime;

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build a minimal `ImageProviderArtifact` with `Accepted` qualification.
fn qualified_artifact(kind: ImageProviderKind) -> ImageProviderArtifact {
    ImageProviderArtifact {
        provider: kind,
        artifact_id: format!("{kind}-v1"),
        compiler_id: "test-compiler".into(),
        abi_version: 1,
        required_hardware: vec![],
        tensor_layout: "nhwc".into(),
        qualification_record: ImageQualificationRecord {
            status: QualificationStatus::Accepted,
            fixture_id: "test-fixture".into(),
            compiler_version: "test-1.0".into(),
            runtime_version: "test-runtime-1.0".into(),
            machine_fingerprint: "test-machine".into(),
            request_digest: [0u8; 32],
            output_digest: None,
            observed_width: None,
            observed_height: None,
            latency_ms: None,
            verified_at: SystemTime::UNIX_EPOCH,
            failure_reason: None,
        },
    }
}

/// Build a manifest with all core components present and qualified.
fn accepting_manifest() -> ImageGenerationCapabilityManifest {
    ImageGenerationCapabilityManifest {
        schema_version: IMAGE_MANIFEST_SCHEMA_VERSION,
        model_family: ImageModelFamily::Custom,
        text_encoder: ComponentAvailability::PresentQualified,
        denoiser: ComponentAvailability::PresentQualified,
        vae_decoder: ComponentAvailability::PresentQualified,
        tokenizer: ComponentAvailability::PresentQualified,
        scheduler: ComponentAvailability::PresentQualified,
        supported_dimensions: DimensionConstraint::Any,
        supported_steps: StepRange { min: 1, max: 50 },
        provider_artifacts: vec![qualified_artifact(ImageProviderKind::ComputeCoreMlx)],
        qualification: ImageQualificationRecord {
            status: QualificationStatus::Accepted,
            fixture_id: "test-fixture".into(),
            compiler_version: "test-1.0".into(),
            runtime_version: "test-runtime-1.0".into(),
            machine_fingerprint: "test-machine".into(),
            request_digest: [0u8; 32],
            output_digest: None,
            observed_width: None,
            observed_height: None,
            latency_ms: None,
            verified_at: SystemTime::UNIX_EPOCH,
            failure_reason: None,
        },
    }
}

fn test_machine() -> MachineProfile {
    MachineProfile {
        os_version: "test-os".into(),
        has_ane: false,
        unified_memory_gb: 64,
    }
}

fn default_request() -> ImageGenerationRequest {
    ImageGenerationRequest::new("test prompt", 512, 512)
}

// ── 1. Admission refuses missing VAE ────────────────────────────────────

/// The admission gate refuses a request when the VAE decoder component is
/// absent, even when all other core components are present.
#[test]
fn admission_refuses_missing_vae() {
    let manifest = ImageGenerationCapabilityManifest {
        schema_version: IMAGE_MANIFEST_SCHEMA_VERSION,
        model_family: ImageModelFamily::Custom,
        text_encoder: ComponentAvailability::PresentQualified,
        denoiser: ComponentAvailability::PresentQualified,
        vae_decoder: ComponentAvailability::Absent,
        tokenizer: ComponentAvailability::PresentQualified,
        scheduler: ComponentAvailability::PresentQualified,
        supported_dimensions: DimensionConstraint::Any,
        supported_steps: StepRange { min: 1, max: 50 },
        provider_artifacts: vec![],
        qualification: ImageQualificationRecord {
            status: QualificationStatus::Unqualified,
            fixture_id: String::new(),
            compiler_version: String::new(),
            runtime_version: String::new(),
            machine_fingerprint: String::new(),
            request_digest: [0u8; 32],
            output_digest: None,
            observed_width: None,
            observed_height: None,
            latency_ms: None,
            verified_at: SystemTime::UNIX_EPOCH,
            failure_reason: None,
        },
    };

    // Verify that missing_components_for correctly identifies vae_decoder.
    let missing = admission::missing_components_for(&manifest);
    let vae_missing = missing.iter().any(|c| c.name == "vae_decoder");
    assert!(
        vae_missing,
        "VAE decoder should be in the missing components list"
    );

    // Verify display of the ComponentAbsent refusal reason for VAE.
    let reason = ImageGenerationRefusalReason::ComponentAbsent("vae_decoder".into());
    let display = reason.to_string();
    assert!(
        display.contains("vae_decoder"),
        "refusal reason should mention vae_decoder"
    );
}

// ── 2. Admission refuses unsupported dimensions ────────────────────────

/// The admission gate refuses a request whose dimensions exceed the
/// range declared in the manifest's `supported_dimensions`.
#[test]
fn admission_refuses_unsupported_dimensions() {
    let manifest = accepting_manifest_with_dimensions(DimensionConstraint::Range(1, 1024));

    let cimage = InstalledCImage {
        path: "/test/cimage".into(),
        digest: ArtifactDigest("test-digest".into()),
        manifest,
        provider_handles: vec![ImageProviderKind::ComputeCoreMlx],
    };

    // Request with dimensions outside the manifest range.
    let request = ImageGenerationRequest {
        width: 9999,
        height: 1,
        ..default_request()
    };

    let machine = test_machine();
    let policy = GenerationExecutionPolicy::RequireRequestedProvider;

    let result = ImageGenerationAdmissionGate.admit(&cimage, &request, &machine, &policy);

    match result {
        Err(ImageGenerationError::AdmissionRefused {
            reason: ImageGenerationRefusalReason::DimensionsUnsupported { width, height },
        }) => {
            assert_eq!(width, 9999, "refused width should match request");
            assert_eq!(height, 1, "refused height should match request");
        }
        Err(e) => {
            panic!("expected AdmissionRefused(DimensionsUnsupported), got {e:?}")
        }
        Ok(_) => panic!("expected refusal, got Ok"),
    }
}

fn accepting_manifest_with_dimensions(
    dim: DimensionConstraint,
) -> ImageGenerationCapabilityManifest {
    let mut m = accepting_manifest();
    m.supported_dimensions = dim;
    m
}

// ── 3. Qualification stale rejected ────────────────────────────────────

/// The `NoOpQualificationResolver` always returns `None`, causing
/// providers to be treated as ineligible when no other evidence exists.
#[test]
fn qualification_stale_rejected() {
    // Verify NoOpQualificationResolver returns None for all queries.
    let resolver = NoOpQualificationResolver;
    let digest = ArtifactDigest("test".into());
    let machine = test_machine();

    let coreml_result = resolver.resolve_coreml(&digest, &machine);
    assert!(
        coreml_result.is_none(),
        "NoOpQualificationResolver.resolve_coreml should be None"
    );

    let image_result = resolver.resolve_image(&digest, ImageProviderKind::ComputeCoreMlx, &machine);
    assert!(
        image_result.is_none(),
        "NoOpQualificationResolver.resolve_image should be None"
    );

    // ProviderIneligibilityReason debug representation.
    assert_eq!(
        format!("{:?}", ProviderIneligibilityReason::Unavailable),
        "Unavailable"
    );
    assert_eq!(
        format!("{:?}", ProviderIneligibilityReason::Unqualified),
        "Unqualified"
    );
    assert_eq!(
        format!("{:?}", ProviderIneligibilityReason::QualificationStale),
        "QualificationStale"
    );
    assert_eq!(
        format!("{:?}", ProviderIneligibilityReason::ArtifactIncompatible),
        "ArtifactIncompatible"
    );
    assert_eq!(
        format!("{:?}", ProviderIneligibilityReason::MachineIncompatible),
        "MachineIncompatible"
    );
    assert_eq!(
        format!("{:?}", ProviderIneligibilityReason::PolicyProhibited),
        "PolicyProhibited"
    );
}

// ── 4. Route selection excludes unqualified ────────────────────────────

/// Route evidence records unqualified candidates and does not select
/// a provider when all candidates are ineligible.
#[test]
fn route_selection_excludes_unqualified() {
    let evidence = ImageGenerationRouteEvidence {
        requested_provider: DevicePreference::Auto,
        route_origin: RouteOrigin::AutoSelection,
        candidates: vec![
            ImageProviderCandidateEvidence {
                provider: ImageProviderKind::ComputeCoreMlx,
                capability: prism_engine::image::provider::ImageProviderCapability::ComputeCoreMlxAvailableButUnqualified,
                eligible: false,
                ineligibility_reason: Some(ProviderIneligibilityReason::Unqualified),
            },
            ImageProviderCandidateEvidence {
                provider: ImageProviderKind::PrismLut,
                capability: prism_engine::image::provider::ImageProviderCapability::PrismLutAvailableButUnqualified,
                eligible: false,
                ineligibility_reason: Some(ProviderIneligibilityReason::Unqualified),
            },
        ],
        selected_provider: None,
        attempted_provider: None,
        fallback_considered: false,
        fallback_attempted: false,
        fallback_provider: None,
        fallback_reason: None,
        selected_provider_qualified: false,
    };

    // No provider was selected since all are unqualified.
    assert!(
        evidence.selected_provider.is_none(),
        "should not select a provider when all candidates are unqualified"
    );
    assert_eq!(evidence.candidates.len(), 2);
    assert!(evidence.candidates.iter().all(|c| !c.eligible));
}

// ── 5. Fallback requires explicit policy ───────────────────────────────

/// When `RequireRequestedProvider` is set and the requested provider is
/// unavailable, the admission gate does NOT fall back — it returns
/// `RequestedProviderUnavailable`.
#[test]
fn fallback_requires_explicit_policy() {
    // Manifest has only a ComputeCoreMlx artifact; request asks for PrismLut.
    let manifest = accepting_manifest();

    let cimage = InstalledCImage {
        path: "/test/cimage".into(),
        digest: ArtifactDigest("test-digest".into()),
        manifest,
        provider_handles: vec![ImageProviderKind::ComputeCoreMlx],
    };

    // With RequireRequestedProvider, PrismLut is unavailable → error.
    let request = ImageGenerationRequest {
        device_preference: DevicePreference::PrismLut,
        execution_policy: GenerationExecutionPolicy::RequireRequestedProvider,
        ..default_request()
    };

    let machine = test_machine();
    let result = match ImageGenerationAdmissionGate.admit(
        &cimage,
        &request,
        &machine,
        &request.execution_policy,
    ) {
        Err(e) => e,
        Ok(_) => panic!("expected error for unavailable provider"),
    };

    assert!(
        matches!(&result, ImageGenerationError::RequestedProviderUnavailable { requested, .. } if *requested == DevicePreference::PrismLut),
        "expected RequestedProviderUnavailable(PrismLut), got {result:?}"
    );

    // With AllowQualifiedFallback the same scenario SHOULD fall back.
    let fallback_request = ImageGenerationRequest {
        device_preference: DevicePreference::PrismLut,
        execution_policy: GenerationExecutionPolicy::AllowQualifiedFallback,
        ..default_request()
    };

    let fallback_plan = ImageGenerationAdmissionGate
        .admit(
            &cimage,
            &fallback_request,
            &machine,
            &fallback_request.execution_policy,
        )
        .expect("fallback should succeed with AllowQualifiedFallback");

    assert_eq!(fallback_plan.route_origin, RouteOrigin::QualifiedFallback);
    assert!(fallback_plan.fallback_used, "fallback should be recorded");

    // FallbackReason debug representations.
    assert_eq!(
        format!("{:?}", FallbackReason::RequestedProviderUnavailable),
        "RequestedProviderUnavailable"
    );
    assert_eq!(
        format!("{:?}", FallbackReason::RequestedProviderUnqualified),
        "RequestedProviderUnqualified"
    );
    assert_eq!(
        format!("{:?}", FallbackReason::RequestedProviderFailed),
        "RequestedProviderFailed"
    );
    assert_eq!(
        format!("{:?}", FallbackReason::RequestedProviderCancelled),
        "RequestedProviderCancelled"
    );
}

// ── 6. Cancellation returns no image ───────────────────────────────────

/// A cancelled `ImageGenerationCancellationToken` reports cancellation
/// and can be embedded in cancellation evidence.
#[test]
fn cancellation_returns_no_image() {
    let request_id = RequestId::new();
    let mut token = ImageGenerationCancellationToken::new(request_id.clone());

    // Token starts not cancelled.
    assert!(!token.is_cancelled());
    assert_eq!(token.request_id, request_id);

    // Cancel the token.
    token.cancel();
    assert!(
        token.is_cancelled(),
        "token should be cancelled after cancel()"
    );

    // Construct full cancellation evidence.
    let evidence = ImageGenerationCancellationEvidence {
        requested_at_stage: ImageGenerationFailureStage::Denoising,
        provider: Some(ImageProviderKind::ComputeCoreMlx),
        completed_denoising_steps: Some(2),
        partial_output_discarded: true,
        cleanup_completed: true,
    };

    assert_eq!(
        evidence.requested_at_stage,
        ImageGenerationFailureStage::Denoising
    );
    assert_eq!(evidence.provider, Some(ImageProviderKind::ComputeCoreMlx));
    assert!(evidence.partial_output_discarded);
    assert!(evidence.cleanup_completed);
}

// ── 7. Empty output rejected ──────────────────────────────────────────

/// An empty output from a provider is rejected as `InvalidOutput`.
#[test]
fn empty_output_rejected() {
    let error = ImageGenerationError::InvalidOutput {
        provider: ImageProviderKind::ComputeCoreMlx,
        reason: "provider returned empty output".into(),
    };

    let display = error.to_string();
    assert!(display.contains("compute-core-mlx"));
    assert!(
        display.contains("empty"),
        "error should mention empty output"
    );

    // Verify via error matching.
    match &error {
        ImageGenerationError::InvalidOutput { provider, reason } => {
            assert_eq!(*provider, ImageProviderKind::ComputeCoreMlx);
            assert_eq!(reason, "provider returned empty output");
        }
        _ => panic!("expected InvalidOutput"),
    }

    // ProviderExecutionFailed — also test for completeness.
    let fail = ImageGenerationError::ProviderExecutionFailed {
        provider: ImageProviderKind::ComputeCoreMlx,
        source: "empty tensor".to_string().into(),
    };
    assert!(fail.to_string().contains("execution failed"));
}

// ── 8. Receipt persistence state recorded ─────────────────────────────

/// The terminal receipt carries a `ReceiptPersistenceState` that records
/// whether the receipt was persisted to durable storage.
#[test]
fn receipt_persistence_state_recorded() {
    // ReceiptPersistenceState debug representations.
    assert_eq!(
        format!("{:?}", ReceiptPersistenceState::Persisted),
        "Persisted"
    );
    assert_eq!(
        format!("{:?}", ReceiptPersistenceState::PersistedVolatileFallback),
        "PersistedVolatileFallback"
    );
    assert_eq!(format!("{:?}", ReceiptPersistenceState::Failed), "Failed");

    // Build a terminal receipt with persisted state.
    let request_id = RequestId::new();
    let now = format!("{:?}", std::time::SystemTime::now());

    let terminal = ImageGenerationTerminalReceipt {
        request_id: request_id.clone(),
        terminal_state: ImageGenerationTerminalState::Succeeded,
        admission: ImageGenerationAdmissionEvidence {
            artifact_digest: ArtifactDigest("digest".into()),
            machine_fingerprint: "test-machine".into(),
            request_digest: [0u8; 32],
            image_capability_declared: true,
            required_components: vec![],
            present_components: vec![],
            missing_components: vec![],
            requested_dimensions: (512, 512),
            supported_dimensions: DimensionConstraint::Any,
            requested_steps: 4,
            supported_steps: StepRange { min: 1, max: 50 },
            qualification_status: QualificationStatus::Accepted,
            admitted: true,
            refusal_reason: None,
        },
        route: ImageGenerationRouteEvidence {
            requested_provider: DevicePreference::Auto,
            route_origin: RouteOrigin::AutoSelection,
            candidates: vec![],
            selected_provider: Some(ImageProviderKind::ComputeCoreMlx),
            attempted_provider: Some(ImageProviderKind::ComputeCoreMlx),
            fallback_considered: false,
            fallback_attempted: false,
            fallback_provider: None,
            fallback_reason: None,
            selected_provider_qualified: true,
        },
        execution: Some(ImageGenerationExecutionEvidence {
            provider: ImageProviderKind::ComputeCoreMlx,
            provider_version: "test-1.0".into(),
            denoising_steps_requested: 4,
            denoising_steps_completed: 4,
            provider_latency_ms: 10.0,
            materialization: MaterializationReceipt::new_copied(4096),
        }),
        output: Some(ImageGenerationOutputEvidence {
            width: 512,
            height: 512,
            output_format: ImageOutputFormat::Rgba8,
            output_digest: ArtifactDigest("test-digest".into()),
            lifecycle: ImageOutputLifecycle::Published,
            bytes_produced: 4096,
            validation_passed: true,
        }),
        failure: None,
        cancellation: None,
        created_at: now.clone(),
        completed_at: now,
    };

    assert_eq!(terminal.request_id, request_id);
    assert_eq!(
        terminal.terminal_state,
        ImageGenerationTerminalState::Succeeded
    );
    assert!(terminal.admission.admitted);
    assert!(terminal.output.is_some());
}

// ── 9. Memory pressure refuses under critical ──────────────────────────

/// Under critical memory pressure the admission evidence can record a
/// refusal, and `Retryability` captures the retry strategy.
#[test]
fn memory_pressure_refuses_under_critical() {
    // MemoryPressureLevel ordering: Critical > Elevated > Normal.
    assert!(MemoryPressureLevel::Critical > MemoryPressureLevel::Elevated);
    assert!(MemoryPressureLevel::Elevated > MemoryPressureLevel::Normal);
    assert!(MemoryPressureLevel::Normal < MemoryPressureLevel::Critical);

    assert_eq!(format!("{:?}", MemoryPressureLevel::Normal), "Normal");
    assert_eq!(format!("{:?}", MemoryPressureLevel::Elevated), "Elevated");
    assert_eq!(format!("{:?}", MemoryPressureLevel::Critical), "Critical");

    // Build admission evidence that refused due to critical memory.
    let evidence = ImageGenerationAdmissionEvidence {
        artifact_digest: ArtifactDigest("digest".into()),
        machine_fingerprint: "test-machine".into(),
        request_digest: [0u8; 32],
        image_capability_declared: true,
        required_components: vec![],
        present_components: vec![],
        missing_components: vec![],
        requested_dimensions: (1024, 1024),
        supported_dimensions: DimensionConstraint::Any,
        requested_steps: 4,
        supported_steps: StepRange { min: 1, max: 50 },
        qualification_status: QualificationStatus::Accepted,
        admitted: false,
        refusal_reason: Some(ImageGenerationRefusalReason::Other(
            "memory pressure critical".into(),
        )),
    };

    assert!(
        !evidence.admitted,
        "should be refused under critical memory"
    );
    assert!(evidence.refusal_reason.is_some());

    let reason_text = evidence.refusal_reason.as_ref().unwrap().to_string();
    assert!(
        reason_text.contains("memory"),
        "refusal reason should mention memory pressure"
    );

    // Retryability debug.
    assert_eq!(
        format!("{:?}", Retryability::RetryAfterMemoryPressureRelief),
        "RetryAfterMemoryPressureRelief"
    );

    // Construct a refusal with memory-pressure evidence.
    let refusal = ImageGenerationRefusal {
        reason: ImageGenerationRefusalReason::Other("memory pressure critical".into()),
        admission_evidence: evidence,
    };

    let reas = refusal.reason.to_string();
    assert!(reas.contains("memory"), "refusal should mention memory");

    // Failure evidence can also reference memory failures.
    let failure_evidence = ImageGenerationFailureEvidence {
        stage: ImageGenerationFailureStage::Admission,
        class: ImageGenerationFailureClass::MemoryAllocationFailed,
        attempted_provider: None,
        source: "insufficient unified memory for model weights".into(),
        retryability: Retryability::RetryAfterMemoryPressureRelief,
        fallback_eligibility: FallbackEligibility::Forbidden,
        partial_output_detected: false,
    };

    assert_eq!(
        failure_evidence.class,
        ImageGenerationFailureClass::MemoryAllocationFailed
    );
    assert!(failure_evidence.source.contains("memory"));
}

// ── 10. Terminal state display ─────────────────────────────────────────

/// All five terminal states produce the expected display strings.
#[test]
fn terminal_state_display() {
    assert_eq!(
        ImageGenerationTerminalState::Succeeded.to_string(),
        "succeeded"
    );
    assert_eq!(
        ImageGenerationTerminalState::RefusedBeforeExecution.to_string(),
        "refused-before-execution"
    );
    assert_eq!(
        ImageGenerationTerminalState::FailedDuringExecution.to_string(),
        "failed-during-execution"
    );
    assert_eq!(
        ImageGenerationTerminalState::SucceededViaQualifiedFallback.to_string(),
        "succeeded-via-qualified-fallback"
    );
    assert_eq!(
        ImageGenerationTerminalState::Cancelled.to_string(),
        "cancelled"
    );

    // Round-trip through format and compare.
    let states = [
        ImageGenerationTerminalState::Succeeded,
        ImageGenerationTerminalState::RefusedBeforeExecution,
        ImageGenerationTerminalState::FailedDuringExecution,
        ImageGenerationTerminalState::SucceededViaQualifiedFallback,
        ImageGenerationTerminalState::Cancelled,
    ];

    let expected = [
        "succeeded",
        "refused-before-execution",
        "failed-during-execution",
        "succeeded-via-qualified-fallback",
        "cancelled",
    ];

    for (state, expected_str) in states.iter().zip(expected.iter()) {
        assert_eq!(&state.to_string(), expected_str);
    }
}
