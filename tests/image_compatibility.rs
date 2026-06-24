//! Hermetic compatibility matrix tests for the Prism image generation facade.
//!
//! Every test is self-contained — no model downloads, no network, no Apple
//! hardware required.  Tests exercise schema roundtrips, status transitions,
//! the DryRunCompatibilityRunner, machine fingerprint determinism,
//! performance tolerance defaults, repeatability evidence construction,
//! and ISO 8601 timestamp formatting.
//!
//! All types are exercised through the public API surface
//! (`use prism_engine::image::*`).
#![cfg(feature = "generation-image")]

use prism_engine::image::*;

// ── Helpers ──────────────────────────────────────────────────────────────

fn sample_artifact() -> ImageCompatibilityArtifact {
    ImageCompatibilityArtifact {
        artifact_id: "test-flux-schnell-v1".into(),
        artifact_digest: ArtifactDigest("aabbccdd".into()),
        model_family: ImageModelFamily::Flux,
        cimage_schema_version: 1,
        tokenizer_digest: ArtifactDigest("token123".into()),
        scheduler_kind: SchedulerKind::FlowMatch,
        tensor_dtype_profile: TensorDtypeProfile::MixedFp16,
        provider_requirements: vec![ProviderRequirement::MlxRuntime("0.20".into())],
        supported_request_profiles: vec!["smoke".into(), "nominal".into(), "boundary".into()],
    }
}

fn sample_machine() -> ImageQualificationMachineProfile {
    ImageQualificationMachineProfile {
        machine_fingerprint: MachineFingerprint("deadbeef".into()),
        product_name: "MacBook Pro".into(),
        chip_family: "Apple M1".into(),
        cpu_core_count: 8,
        gpu_core_count: 8,
        unified_memory_bytes: 16_000_000_000,
        macos_version: "15.5".into(),
        coreml_runtime_version: "4.0".into(),
        mlx_runtime_version: "0.20.0".into(),
        prism_version: "0.1.0".into(),
        compute_core_version: "0.1.0".into(),
    }
}

fn sample_profile(id: &str) -> ImageRequestProfile {
    ImageRequestProfile {
        id: id.into(),
        width: 256,
        height: 256,
        steps: 4,
        seed: 42,
        guidance_scale: None,
        prompt_fixture_id: "synthetic-prompt-v1".into(),
        output_format: ImageOutputFormat::Rgba8,
    }
}

fn sample_receipt(
    artifact: &ImageCompatibilityArtifact,
    machine: &ImageQualificationMachineProfile,
    profile: &ImageRequestProfile,
    status: CompatibilityStatus,
    provider: ImageProviderKind,
) -> ImageCompatibilityReceipt {
    ImageCompatibilityReceipt {
        receipt_id: ReceiptId::new(),
        artifact: artifact.clone(),
        machine: machine.clone(),
        request_profile: profile.clone(),
        provider,
        qualification_status: status,
        admission_receipts: vec![],
        terminal_receipts: vec![],
        repeatability: None,
        performance: vec![],
        performance_tolerance: None,
        failure_summary: None,
        generated_at: iso_now(),
    }
}

// ── 1. Schema & Serialization Roundtrips ─────────────────────────────────

#[test]
fn artifact_serde_roundtrip() {
    let a = sample_artifact();
    let json = serde_json::to_string(&a).unwrap();
    let back: ImageCompatibilityArtifact = serde_json::from_str(&json).unwrap();
    assert_eq!(a.artifact_id, back.artifact_id);
    assert_eq!(a.artifact_digest, back.artifact_digest);
    assert_eq!(a.model_family, back.model_family);
    assert_eq!(a.cimage_schema_version, back.cimage_schema_version);
    assert_eq!(a.scheduler_kind, back.scheduler_kind);
    assert_eq!(a.tensor_dtype_profile, back.tensor_dtype_profile);
    assert_eq!(a.provider_requirements, back.provider_requirements);
    assert_eq!(
        a.supported_request_profiles,
        back.supported_request_profiles
    );
}

#[test]
fn machine_profile_serde_roundtrip() {
    let m = sample_machine();
    let json = serde_json::to_string(&m).unwrap();
    let back: ImageQualificationMachineProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(m.machine_fingerprint, back.machine_fingerprint);
    assert_eq!(m.product_name, back.product_name);
    assert_eq!(m.chip_family, back.chip_family);
    assert_eq!(m.cpu_core_count, back.cpu_core_count);
    assert_eq!(m.gpu_core_count, back.gpu_core_count);
    assert_eq!(m.unified_memory_bytes, back.unified_memory_bytes);
    assert_eq!(m.macos_version, back.macos_version);
    assert_eq!(m.coreml_runtime_version, back.coreml_runtime_version);
    assert_eq!(m.mlx_runtime_version, back.mlx_runtime_version);
    assert_eq!(m.prism_version, back.prism_version);
    assert_eq!(m.compute_core_version, back.compute_core_version);
}

#[test]
fn request_profile_serde_roundtrip() {
    let p = sample_profile("smoke");
    let json = serde_json::to_string(&p).unwrap();
    let back: ImageRequestProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(p.id, back.id);
    assert_eq!(p.width, back.width);
    assert_eq!(p.height, back.height);
    assert_eq!(p.steps, back.steps);
    assert_eq!(p.seed, back.seed);
    assert_eq!(p.guidance_scale, back.guidance_scale);
    assert_eq!(p.prompt_fixture_id, back.prompt_fixture_id);
    assert_eq!(p.output_format, back.output_format);
}

#[test]
fn request_profile_with_guidance_serde() {
    let p = ImageRequestProfile {
        guidance_scale: Some(7.5),
        ..sample_profile("cfg-test")
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: ImageRequestProfile = serde_json::from_str(&json).unwrap();
    assert_eq!(back.guidance_scale, Some(7.5));
}

#[test]
fn receipt_serde_roundtrip() {
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("smoke");
    let receipt = sample_receipt(
        &artifact,
        &machine,
        &profile,
        CompatibilityStatus::FunctionallyQualified,
        ImageProviderKind::ComputeCoreMlx,
    );
    let json = serde_json::to_string(&receipt).unwrap();
    let back: ImageCompatibilityReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(receipt.receipt_id, back.receipt_id);
    assert_eq!(receipt.qualification_status, back.qualification_status);
    assert_eq!(receipt.provider, back.provider);
    assert_eq!(receipt.artifact.artifact_id, back.artifact.artifact_id);
    // Fields with #[serde(skip)] are not serialized — deserialized back as empty
    assert!(back.admission_receipts.is_empty());
    assert!(back.terminal_receipts.is_empty());
}

#[test]
fn receipt_with_tolerance_serde() {
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("perf");
    let tolerance = ImagePerformanceTolerance {
        max_median_latency_regression_pct: 0.10,
        max_p95_latency_regression_pct: 0.15,
        max_peak_memory_regression_pct: 0.10,
    };
    let receipt = ImageCompatibilityReceipt {
        performance_tolerance: Some(tolerance),
        ..sample_receipt(
            &artifact,
            &machine,
            &profile,
            CompatibilityStatus::PerformanceQualified,
            ImageProviderKind::ComputeCoreMlx,
        )
    };
    let json = serde_json::to_string(&receipt).unwrap();
    let back: ImageCompatibilityReceipt = serde_json::from_str(&json).unwrap();
    let tol = back.performance_tolerance.unwrap();
    assert!((tol.max_median_latency_regression_pct - 0.10).abs() < 1e-6);
}

#[test]
fn receipt_with_failure_summary_serde() {
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("fail");
    let receipt = ImageCompatibilityReceipt {
        qualification_status: CompatibilityStatus::Incompatible,
        failure_summary: Some("ANE hardware required but absent".into()),
        ..sample_receipt(
            &artifact,
            &machine,
            &profile,
            CompatibilityStatus::Incompatible,
            ImageProviderKind::ComputeCoreMlx,
        )
    };
    let json = serde_json::to_string(&receipt).unwrap();
    let back: ImageCompatibilityReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(
        back.failure_summary.as_deref(),
        Some("ANE hardware required but absent")
    );
}

#[test]
fn manifest_serde_roundtrip() {
    let manifest = PrismImageCompatibilityManifest {
        schema_version: 1,
        generated_at: iso_now(),
        prism_version: "0.1.0".into(),
        compute_core_version: "0.1.0".into(),
        cells: vec![],
    };
    let json = serde_json::to_string_pretty(&manifest).unwrap();
    assert!(json.contains("prism_version"));
    assert!(json.contains("compute_core_version"));
    assert!(json.contains("schema_version"));
    let back: PrismImageCompatibilityManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.schema_version, 1);
    assert_eq!(back.prism_version, "0.1.0");
    assert_eq!(back.compute_core_version, "0.1.0");
    assert!(back.cells.is_empty());
}

#[test]
fn manifest_with_cells_serde() {
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("nominal");
    let receipt = sample_receipt(
        &artifact,
        &machine,
        &profile,
        CompatibilityStatus::PerformanceQualified,
        ImageProviderKind::ComputeCoreMlx,
    );
    let manifest = PrismImageCompatibilityManifest {
        schema_version: 1,
        generated_at: iso_now(),
        prism_version: "0.1.0".into(),
        compute_core_version: "0.2.0".into(),
        cells: vec![receipt],
    };
    let json = serde_json::to_string(&manifest).unwrap();
    let back: PrismImageCompatibilityManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(back.cells.len(), 1);
    assert_eq!(
        back.cells[0].qualification_status,
        CompatibilityStatus::PerformanceQualified
    );
}

// ── 2. CompatibilityStatus transitions & properties ─────────────────────

#[test]
fn status_display_all_variants() {
    assert_eq!(CompatibilityStatus::Untried.to_string(), "untried");
    assert_eq!(
        CompatibilityStatus::FixtureUnavailable.to_string(),
        "fixture-unavailable"
    );
    assert_eq!(
        CompatibilityStatus::AdmissionRefused.to_string(),
        "admission-refused"
    );
    assert_eq!(
        CompatibilityStatus::ProviderUnavailable.to_string(),
        "provider-unavailable"
    );
    assert_eq!(
        CompatibilityStatus::ProviderUnqualified.to_string(),
        "provider-unqualified"
    );
    assert_eq!(
        CompatibilityStatus::FunctionallyQualified.to_string(),
        "functionally-qualified"
    );
    assert_eq!(
        CompatibilityStatus::RepeatabilityQualified.to_string(),
        "repeatability-qualified"
    );
    assert_eq!(
        CompatibilityStatus::PerformanceQualified.to_string(),
        "performance-qualified"
    );
    assert_eq!(
        CompatibilityStatus::PerformanceRegressed.to_string(),
        "performance-regressed"
    );
    assert_eq!(
        CompatibilityStatus::ReliabilityFailed.to_string(),
        "reliability-failed"
    );
    assert_eq!(
        CompatibilityStatus::Incompatible.to_string(),
        "incompatible"
    );
}

#[test]
fn status_route_eligibility() {
    // Route-eligible
    assert!(CompatibilityStatus::PerformanceQualified.is_route_eligible());
    assert!(CompatibilityStatus::RepeatabilityQualified.is_route_eligible());
    // Not route-eligible
    assert!(!CompatibilityStatus::FunctionallyQualified.is_route_eligible());
    assert!(!CompatibilityStatus::Untried.is_route_eligible());
    assert!(!CompatibilityStatus::FixtureUnavailable.is_route_eligible());
    assert!(!CompatibilityStatus::AdmissionRefused.is_route_eligible());
    assert!(!CompatibilityStatus::ProviderUnavailable.is_route_eligible());
    assert!(!CompatibilityStatus::ProviderUnqualified.is_route_eligible());
    assert!(!CompatibilityStatus::PerformanceRegressed.is_route_eligible());
    assert!(!CompatibilityStatus::ReliabilityFailed.is_route_eligible());
    assert!(!CompatibilityStatus::Incompatible.is_route_eligible());
}

#[test]
fn status_development_eligibility() {
    // Development-eligible
    assert!(CompatibilityStatus::PerformanceQualified.is_development_eligible());
    assert!(CompatibilityStatus::RepeatabilityQualified.is_development_eligible());
    assert!(CompatibilityStatus::FunctionallyQualified.is_development_eligible());
    // Not development-eligible
    assert!(!CompatibilityStatus::Untried.is_development_eligible());
    assert!(!CompatibilityStatus::FixtureUnavailable.is_development_eligible());
    assert!(!CompatibilityStatus::AdmissionRefused.is_development_eligible());
    assert!(!CompatibilityStatus::ProviderUnavailable.is_development_eligible());
    assert!(!CompatibilityStatus::ProviderUnqualified.is_development_eligible());
    assert!(!CompatibilityStatus::PerformanceRegressed.is_development_eligible());
    assert!(!CompatibilityStatus::ReliabilityFailed.is_development_eligible());
    assert!(!CompatibilityStatus::Incompatible.is_development_eligible());
}

#[test]
fn status_failure_predicates() {
    // Failures
    assert!(CompatibilityStatus::AdmissionRefused.is_failure());
    assert!(CompatibilityStatus::ProviderUnavailable.is_failure());
    assert!(CompatibilityStatus::ProviderUnqualified.is_failure());
    assert!(CompatibilityStatus::ReliabilityFailed.is_failure());
    assert!(CompatibilityStatus::Incompatible.is_failure());
    // Not failures
    assert!(!CompatibilityStatus::Untried.is_failure());
    assert!(!CompatibilityStatus::FixtureUnavailable.is_failure());
    assert!(!CompatibilityStatus::FunctionallyQualified.is_failure());
    assert!(!CompatibilityStatus::RepeatabilityQualified.is_failure());
    assert!(!CompatibilityStatus::PerformanceQualified.is_failure());
    assert!(!CompatibilityStatus::PerformanceRegressed.is_failure());
}

#[test]
fn status_eq_and_hash() {
    use std::collections::HashSet;
    let variants = [
        CompatibilityStatus::Untried,
        CompatibilityStatus::FixtureUnavailable,
        CompatibilityStatus::AdmissionRefused,
        CompatibilityStatus::ProviderUnavailable,
        CompatibilityStatus::ProviderUnqualified,
        CompatibilityStatus::FunctionallyQualified,
        CompatibilityStatus::RepeatabilityQualified,
        CompatibilityStatus::PerformanceQualified,
        CompatibilityStatus::PerformanceRegressed,
        CompatibilityStatus::ReliabilityFailed,
        CompatibilityStatus::Incompatible,
    ];
    let set: HashSet<_> = variants.iter().collect();
    assert_eq!(
        set.len(),
        11,
        "all 11 variants must be distinct under Eq + Hash"
    );
}

#[test]
fn status_clone_and_copy() {
    let s = CompatibilityStatus::PerformanceQualified;
    let s2 = s; // Copy
    assert_eq!(s, s2);
    let s3 = s.clone(); // Clone
    assert_eq!(s, s3);
}

// ── 3. Dry-run runner tests ─────────────────────────────────────────────

#[test]
fn dry_run_produces_receipt() {
    let runner = DryRunCompatibilityRunner;
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("smoke");

    let receipt = runner
        .qualify(
            &artifact,
            &machine,
            &profile,
            ImageProviderKind::ComputeCoreMlx,
        )
        .unwrap();

    assert_eq!(receipt.artifact.artifact_id, "test-flux-schnell-v1");
    assert_eq!(
        receipt.qualification_status,
        CompatibilityStatus::FunctionallyQualified
    );
    assert_eq!(receipt.provider, ImageProviderKind::ComputeCoreMlx);
}

#[test]
fn dry_run_produces_correct_status() {
    let runner = DryRunCompatibilityRunner;
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("boundary");

    // DryRunCompatibilityRunner always produces FunctionallyQualified
    let receipt = runner
        .qualify(
            &artifact,
            &machine,
            &profile,
            ImageProviderKind::PrismLut,
        )
        .unwrap();

    assert_eq!(
        receipt.qualification_status,
        CompatibilityStatus::FunctionallyQualified
    );
    assert!(receipt.qualification_status.is_development_eligible());
    assert!(!receipt.qualification_status.is_route_eligible());
    assert!(!receipt.qualification_status.is_failure());
}

#[test]
fn dry_run_different_providers() {
    let runner = DryRunCompatibilityRunner;
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("multi-provider");

    for kind in &[
        ImageProviderKind::ComputeCoreMlx,
        ImageProviderKind::PrismLut,
    ] {
        let receipt = runner
            .qualify(&artifact, &machine, &profile, *kind)
            .unwrap();
        assert_eq!(receipt.provider, *kind);
    }
}

#[test]
fn dry_run_roundtrip_receipt_json() {
    let runner = DryRunCompatibilityRunner;
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("serde");

    let receipt = runner
        .qualify(
            &artifact,
            &machine,
            &profile,
            ImageProviderKind::ComputeCoreMlx,
        )
        .unwrap();

    // Serialize the dry-run receipt and deserialize back
    let json = serde_json::to_string(&receipt).unwrap();
    let back: ImageCompatibilityReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(receipt.receipt_id, back.receipt_id);
    assert_eq!(receipt.qualification_status, back.qualification_status);
    assert_eq!(receipt.provider, back.provider);
    assert!(back.admission_receipts.is_empty());
    assert!(back.terminal_receipts.is_empty());
}

#[test]
fn dry_run_generated_at_populated() {
    let runner = DryRunCompatibilityRunner;
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("timestamp");

    let receipt = runner
        .qualify(
            &artifact,
            &machine,
            &profile,
            ImageProviderKind::ComputeCoreMlx,
        )
        .unwrap();

    assert!(!receipt.generated_at.is_empty());
    assert_eq!(receipt.generated_at.len(), 27);
}

#[test]
fn dry_run_receipt_id_unique() {
    let runner = DryRunCompatibilityRunner;
    let artifact = sample_artifact();
    let machine = sample_machine();
    let profile = sample_profile("unique");

    let r1 = runner
        .qualify(
            &artifact,
            &machine,
            &profile,
            ImageProviderKind::ComputeCoreMlx,
        )
        .unwrap();
    let r2 = runner
        .qualify(
            &artifact,
            &machine,
            &profile,
            ImageProviderKind::ComputeCoreMlx,
        )
        .unwrap();

    assert_ne!(r1.receipt_id, r2.receipt_id);
}

// ── 4. Machine fingerprint tests ───────────────────────────────────────

#[test]
fn fingerprint_deterministic() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    assert_eq!(fp1, fp2);
}

#[test]
fn fingerprint_changes_on_chip() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M2", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    assert_ne!(fp1, fp2);
}

#[test]
fn fingerprint_changes_on_cpu_cores() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M1", 10, 8, 8_000_000_000, "15.5", "0.1.0");
    assert_ne!(fp1, fp2);
}

#[test]
fn fingerprint_changes_on_gpu_cores() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M1", 8, 16, 8_000_000_000, "15.5", "0.1.0");
    assert_ne!(fp1, fp2);
}

#[test]
fn fingerprint_changes_on_memory() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M1", 8, 8, 16_000_000_000, "15.5", "0.1.0");
    assert_ne!(fp1, fp2);
}

#[test]
fn fingerprint_changes_on_os_version() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.4", "0.1.0");
    assert_ne!(fp1, fp2);
}

#[test]
fn fingerprint_changes_on_prism_version() {
    let fp1 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let fp2 = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.2.0");
    assert_ne!(fp1, fp2);
}

#[test]
fn fingerprint_display() {
    let fp = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let s = format!("{}", fp.0);
    // Fingerprint is a 16-char hex string
    assert_eq!(s.len(), 16);
    assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
}

// ── 5. Performance tolerance defaults ───────────────────────────────────

#[test]
fn performance_tolerance_defaults() {
    let t = ImagePerformanceTolerance::default();
    assert!((t.max_median_latency_regression_pct - 0.20).abs() < 1e-6);
    assert!((t.max_p95_latency_regression_pct - 0.30).abs() < 1e-6);
    assert!((t.max_peak_memory_regression_pct - 0.20).abs() < 1e-6);
}

#[test]
fn performance_tolerance_construct_and_clone() {
    let t = ImagePerformanceTolerance {
        max_median_latency_regression_pct: 0.15,
        max_p95_latency_regression_pct: 0.25,
        max_peak_memory_regression_pct: 0.30,
    };
    let t2 = t.clone();
    assert_eq!(
        t.max_median_latency_regression_pct,
        t2.max_median_latency_regression_pct
    );
}

#[test]
fn performance_tolerance_serde_roundtrip() {
    let t = ImagePerformanceTolerance {
        max_median_latency_regression_pct: 0.10,
        max_p95_latency_regression_pct: 0.20,
        max_peak_memory_regression_pct: 0.15,
    };
    let json = serde_json::to_string(&t).unwrap();
    let back: ImagePerformanceTolerance = serde_json::from_str(&json).unwrap();
    assert!((back.max_median_latency_regression_pct - 0.10).abs() < 1e-6);
    assert!((back.max_p95_latency_regression_pct - 0.20).abs() < 1e-6);
    assert!((back.max_peak_memory_regression_pct - 0.15).abs() < 1e-6);
}

// ── 6. Repeatability evidence construction ──────────────────────────────

#[test]
fn repeatability_evidence_construct() {
    let evidence = ImageRepeatabilityEvidence {
        run_count: 3,
        output_digests: vec![
            ArtifactDigest("a1b2c3d4".into()),
            ArtifactDigest("a1b2c3d4".into()),
            ArtifactDigest("a1b2c3d4".into()),
        ],
        exact_matches: 3,
        perceptual_distances: vec![0.0, 0.0],
        policy: ImageRepeatabilityPolicy::ExactDigest,
        passed: true,
    };
    assert_eq!(evidence.run_count, 3);
    assert_eq!(evidence.exact_matches, 3);
    assert!(evidence.passed);
}

#[test]
fn repeatability_evidence_digest_allowlist() {
    let allowed = vec![
        ArtifactDigest("a1b2c3d4".into()),
        ArtifactDigest("e5f6g7h8".into()),
    ];
    let evidence = ImageRepeatabilityEvidence {
        run_count: 2,
        output_digests: vec![
            ArtifactDigest("a1b2c3d4".into()),
            ArtifactDigest("e5f6g7h8".into()),
        ],
        exact_matches: 1,
        perceptual_distances: vec![0.01],
        policy: ImageRepeatabilityPolicy::DigestAllowlist(allowed),
        passed: true,
    };
    assert_eq!(evidence.run_count, 2);
    assert!(evidence.passed);
}

#[test]
fn repeatability_evidence_structural_policy() {
    let evidence = ImageRepeatabilityEvidence {
        run_count: 2,
        output_digests: vec![ArtifactDigest("abc".into()), ArtifactDigest("def".into())],
        exact_matches: 0,
        perceptual_distances: vec![0.05],
        policy: ImageRepeatabilityPolicy::StructuralAndPerceptual,
        passed: false,
    };
    assert_eq!(evidence.exact_matches, 0);
    assert!(!evidence.passed);
    assert_eq!(evidence.perceptual_distances.len(), 1);
}

#[test]
fn repeatability_evidence_serde_roundtrip() {
    let evidence = ImageRepeatabilityEvidence {
        run_count: 3,
        output_digests: vec![
            ArtifactDigest("a1b2c3d4".into()),
            ArtifactDigest("a1b2c3d4".into()),
            ArtifactDigest("a1b2c3d4".into()),
        ],
        exact_matches: 3,
        perceptual_distances: vec![0.0, 0.0],
        policy: ImageRepeatabilityPolicy::ExactDigest,
        passed: true,
    };
    let json = serde_json::to_string(&evidence).unwrap();
    let back: ImageRepeatabilityEvidence = serde_json::from_str(&json).unwrap();
    assert_eq!(back.run_count, 3);
    assert_eq!(back.exact_matches, 3);
    assert!(back.passed);
    assert_eq!(back.output_digests.len(), 3);
    assert_eq!(back.perceptual_distances.len(), 2);
}

#[test]
fn repeatability_policy_eq_and_clone() {
    let p1 = ImageRepeatabilityPolicy::ExactDigest;
    let p2 = ImageRepeatabilityPolicy::ExactDigest;
    assert_eq!(p1, p2);

    let allow1 = ImageRepeatabilityPolicy::DigestAllowlist(vec![ArtifactDigest("abc".into())]);
    let allow2 = allow1.clone();
    assert_eq!(allow1, allow2);

    let struct1 = ImageRepeatabilityPolicy::StructuralAndPerceptual;
    let struct2 = ImageRepeatabilityPolicy::StructuralAndPerceptual;
    assert_eq!(struct1, struct2);
    assert_ne!(p1, allow1);
}

// ── 7. iso_now format validation ────────────────────────────────────────

#[test]
fn iso_now_format_length() {
    let ts = iso_now();
    // ISO 8601: YYYY-MM-DDTHH:MM:SS.ffffffZ = 27 chars
    assert_eq!(ts.len(), 27, "expected 27-char ISO 8601: got {ts}");
}

#[test]
fn iso_now_format_structure() {
    let ts = iso_now();
    // Verify fixed separator positions
    assert_eq!(
        &ts[4..5],
        "-",
        "position 4 should be '-' (year-month separator)"
    );
    assert_eq!(
        &ts[7..8],
        "-",
        "position 7 should be '-' (month-day separator)"
    );
    assert_eq!(
        &ts[10..11],
        "T",
        "position 10 should be 'T' (date-time separator)"
    );
    assert_eq!(
        &ts[13..14],
        ":",
        "position 13 should be ':' (hour-minute separator)"
    );
    assert_eq!(
        &ts[16..17],
        ":",
        "position 16 should be ':' (minute-second separator)"
    );
    assert_eq!(
        &ts[19..20],
        ".",
        "position 19 should be '.' (second-microsecond separator)"
    );
    assert_eq!(
        &ts[26..27],
        "Z",
        "position 26 should be 'Z' (UTC indicator)"
    );
}

#[test]
fn iso_now_year_range() {
    let ts = iso_now();
    let year: u32 = ts[0..4].parse().expect("year should be numeric");
    // Must be a reasonable recent year
    assert!(year >= 2024, "year should be >= 2024, got {year}");
    assert!(year <= 2030, "year should be <= 2030, got {year}");
}

#[test]
fn iso_now_month_range() {
    let ts = iso_now();
    let month: u32 = ts[5..7].parse().expect("month should be numeric");
    assert!(
        (1..=12).contains(&month),
        "month should be 01-12, got {month:02}"
    );
}

#[test]
fn iso_now_day_range() {
    let ts = iso_now();
    let day: u32 = ts[8..10].parse().expect("day should be numeric");
    assert!((1..=31).contains(&day), "day should be 01-31, got {day:02}");
}

#[test]
fn iso_now_time_fields() {
    let ts = iso_now();
    let hours: u32 = ts[11..13].parse().expect("hours should be numeric");
    let minutes: u32 = ts[14..16].parse().expect("minutes should be numeric");
    let seconds: u32 = ts[17..19].parse().expect("seconds should be numeric");
    assert!(hours < 24, "hours should be 00-23, got {hours:02}");
    assert!(minutes < 60, "minutes should be 00-59, got {minutes:02}");
    assert!(seconds < 60, "seconds should be 00-59, got {seconds:02}");
}

#[test]
fn iso_now_microseconds() {
    let ts = iso_now();
    let micros: u32 = ts[20..26].parse().expect("microseconds should be numeric");
    assert!(
        micros < 1_000_000,
        "microseconds should be 000000-999999, got {micros:06}"
    );
}

// ── 8. ReceiptId construction ──────────────────────────────────────────

#[test]
fn receipt_id_new_unique() {
    let id1 = ReceiptId::new();
    let id2 = ReceiptId::new();
    assert_ne!(id1, id2);
}

// ── 9. ImagePerformanceBaseline construction & serde ────────────────────

#[test]
fn performance_baseline_construct() {
    let fp = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let baseline = ImagePerformanceBaseline {
        artifact_id: "test-flux-schnell-v1".into(),
        machine_fingerprint: fp,
        request_profile_id: "smoke".into(),
        provider: ImageProviderKind::ComputeCoreMlx,
        total_latency_ms: 1234.5,
        provider_latency_ms: 1100.0,
        text_encoding_latency_ms: Some(50.2),
        denoising_latency_ms: Some(900.0),
        vae_decode_latency_ms: Some(149.8),
        peak_estimated_memory_bytes: Some(4_000_000_000),
        output_bytes: 524_288,
        completed_steps: 4,
        timestamp: iso_now(),
    };
    assert_eq!(baseline.artifact_id, "test-flux-schnell-v1");
    assert_eq!(baseline.completed_steps, 4);
    assert_eq!(baseline.peak_estimated_memory_bytes, Some(4_000_000_000));
}

#[test]
fn performance_baseline_serde_roundtrip() {
    let fp = build_machine_fingerprint("Apple M1", 8, 8, 8_000_000_000, "15.5", "0.1.0");
    let baseline = ImagePerformanceBaseline {
        artifact_id: "test-flux-schnell-v1".into(),
        machine_fingerprint: fp,
        request_profile_id: "smoke".into(),
        provider: ImageProviderKind::ComputeCoreMlx,
        total_latency_ms: 1234.5,
        provider_latency_ms: 1100.0,
        text_encoding_latency_ms: None,
        denoising_latency_ms: None,
        vae_decode_latency_ms: None,
        peak_estimated_memory_bytes: None,
        output_bytes: 524_288,
        completed_steps: 4,
        timestamp: iso_now(),
    };
    let json = serde_json::to_string(&baseline).unwrap();
    let back: ImagePerformanceBaseline = serde_json::from_str(&json).unwrap();
    assert_eq!(back.artifact_id, "test-flux-schnell-v1");
    assert!(back.text_encoding_latency_ms.is_none());
    assert!(back.peak_estimated_memory_bytes.is_none());
    assert!((back.total_latency_ms - 1234.5).abs() < 1e-9);
}

// ── 10. ImageModelFamily, SchedulerKind, TensorDtypeProfile display ─────

#[test]
fn image_model_family_display() {
    assert_eq!(ImageModelFamily::StableDiffusion3.to_string(), "sd3");
    assert_eq!(ImageModelFamily::Flux.to_string(), "flux");
    assert_eq!(ImageModelFamily::Sdxl.to_string(), "sdxl");
    assert_eq!(
        ImageModelFamily::DiffusionGemma.to_string(),
        "diffusion-gemma"
    );
    assert_eq!(ImageModelFamily::Custom.to_string(), "custom");
}

#[test]
fn scheduler_kind_display() {
    assert_eq!(SchedulerKind::FlowMatch.to_string(), "flow-match");
    assert_eq!(
        SchedulerKind::FlowMatchContinuous.to_string(),
        "flow-match-continuous"
    );
    assert_eq!(SchedulerKind::Ddpm.to_string(), "ddpm");
    assert_eq!(SchedulerKind::Pndm.to_string(), "pndm");
    assert_eq!(SchedulerKind::DpmSolverPP.to_string(), "dpm-solver++");
    assert_eq!(SchedulerKind::EulerAncestral.to_string(), "euler-ancestral");
    assert_eq!(SchedulerKind::Custom.to_string(), "custom");
}

#[test]
fn tensor_dtype_profile_display() {
    assert_eq!(TensorDtypeProfile::Fp32.to_string(), "fp32");
    assert_eq!(TensorDtypeProfile::Fp16.to_string(), "fp16");
    assert_eq!(TensorDtypeProfile::MixedFp16.to_string(), "mixed-fp16");
    assert_eq!(TensorDtypeProfile::Bf16.to_string(), "bf16");
    assert_eq!(
        TensorDtypeProfile::Nf4 { block_size: 64 }.to_string(),
        "nf4-b64"
    );
    assert_eq!(
        TensorDtypeProfile::Custom("my-dtype".into()).to_string(),
        "custom-my-dtype"
    );
}

#[test]
fn tensor_dtype_profile_serde_roundtrip() {
    let profiles = vec![
        TensorDtypeProfile::Fp32,
        TensorDtypeProfile::Fp16,
        TensorDtypeProfile::MixedFp16,
        TensorDtypeProfile::Bf16,
        TensorDtypeProfile::Nf4 { block_size: 128 },
        TensorDtypeProfile::Custom("test".into()),
    ];
    for p in &profiles {
        let json = serde_json::to_string(p).unwrap();
        let back: TensorDtypeProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(*p, back);
    }
}

#[test]
fn provider_requirement_serde_roundtrip() {
    let reqs = vec![
        ProviderRequirement::MlxRuntime("0.20".into()),
        ProviderRequirement::CoreMlRuntime("4.0".into()),
        ProviderRequirement::AneHardware,
        ProviderRequirement::MinimumMemory(8_000_000_000),
        ProviderRequirement::MinimumGpuCores(8),
        ProviderRequirement::Custom("need-foo".into()),
    ];
    for r in &reqs {
        let json = serde_json::to_string(r).unwrap();
        let back: ProviderRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(*r, back);
    }
}
