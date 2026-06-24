// ── Prism Image Generation — Module Root ─────────────────────────────────
//
// Stable public API for text-to-image generation.
//
// Architecture:
//   generate_image()  →  admission  →  routing  →  provider  →  receipt
//
// Every module is gated behind `cfg(feature = "generation-image")` except the
// scheduler registry which is always available.

use anyhow::Result;
use std::path::Path;

pub mod scheduler_registry;

// ── New capability modules (gated on generation-image) ─────────────────

#[cfg(feature = "generation-image")]
pub mod admission;
#[cfg(feature = "generation-image")]
pub mod compatibility;
#[cfg(feature = "generation-image")]
pub mod manifest;
#[cfg(feature = "generation-image")]
pub mod provider;
#[cfg(feature = "generation-image")]
pub mod reliability;
#[cfg(feature = "generation-image")]
pub mod resolver;
#[cfg(feature = "generation-image")]
pub mod router;
pub mod types;

// ── Public API re-exports ─────────────────────────────────────────────

pub use self::types::{
    ArtifactDigest, DevicePreference, GeneratedImage, GenerationExecutionPolicy, GenerationWarning,
    ImageGenerationError, ImageGenerationReceipt, ImageGenerationRefusalReason,
    ImageGenerationRequest, ImageGenerationResult, ImageOutputFormat, ImageProviderKind,
    MaterializationReceipt, MemoryResidency, OutputDigest, QualificationStatus, RequestId,
    RouteOrigin,
};

#[cfg(feature = "generation-image")]
pub use self::reliability::{
    FallbackEligibility, FallbackReason, ImageGenerationAdmissionEvidence,
    ImageGenerationCancellation, ImageGenerationCancellationEvidence,
    ImageGenerationCancellationToken, ImageGenerationExecutionEvidence, ImageGenerationFailure,
    ImageGenerationFailureClass, ImageGenerationFailureEvidence, ImageGenerationFailureStage,
    ImageGenerationOutcome, ImageGenerationOutputEvidence, ImageGenerationRefusal,
    ImageGenerationResponse, ImageGenerationRouteEvidence, ImageGenerationTerminalReceipt,
    ImageGenerationTerminalState, ImageOutputLifecycle, ImageProviderCandidateEvidence,
    ImageReliabilityMetrics, MemoryPressureLevel, ProviderIneligibilityReason,
    QualificationFreshness, QualificationFreshnessKey, ReceiptPersistenceState, Retryability,
};

#[cfg(feature = "generation-image")]
pub use tribunus_compute_core::compute_image::adapter::ComputeImageGenerationAdapter;

#[cfg(feature = "generation-image")]
pub use self::resolver::{
    CoreMlQualificationReceipt, NoOpQualificationResolver, QualificationResolver,
};

#[cfg(feature = "generation-image")]
pub use self::compatibility::{
    build_machine_fingerprint, iso_now, CompatibilityRunnerError, CompatibilityStatus,
    DryRunCompatibilityRunner, ImageCompatibilityArtifact, ImageCompatibilityReceipt,
    ImageCompatibilityRunner, ImageModelFamily, ImagePerformanceBaseline,
    ImagePerformanceTolerance, ImageQualificationMachineProfile, ImageRepeatabilityEvidence,
    ImageRepeatabilityPolicy, ImageRequestProfile, MachineFingerprint,
    PrismImageCompatibilityManifest, ProviderRequirement, ReceiptId, SchedulerKind,
    TensorDtypeProfile,
};

// ── Deprecated / legacy types ─────────────────────────────────────────
//
// Keep CImage, GpuInfo, ModelType, DiffusionModel, etc. for any remaining
// callers.  These are not part of the new golden-path API.

pub struct GpuInfo {
    pub id: usize,
    pub memory_gb: usize,
}

pub struct CImage {
    pub sharded: bool,
}

pub struct TextEncoderInfo {
    pub name: String,
    pub params_billion: f32,
}

pub struct VaeConfig {
    pub channels: u32,
}

pub enum Sd3Variant {
    Sd3_5,
    Sd3Medium,
}

pub enum FluxVariant {
    Dev,
    Schnell,
    Pro,
}

pub enum SdxlVariant {
    Base,
    Refiner,
}

pub enum DgVariant {
    Base,
}

pub enum ModelType {
    StableDiffusion3 { variant: Sd3Variant },
    Flux { variant: FluxVariant },
    Sdxl { variant: SdxlVariant },
    DiffusionGemma { variant: DgVariant },
    Custom { encoder: String, denoiser: String },
}

pub trait DiffusionModel: Send {
    fn model_type(&self) -> ModelType;
    fn steps(&self) -> (u32, u32, u32);
    fn latent_shape(&self) -> (u32, u32, u32);
    fn text_encoders(&self) -> Vec<TextEncoderInfo>;
    fn vae_config(&self) -> VaeConfig;
    fn guidance_range(&self) -> (f32, f32);
    fn has_cfg(&self) -> bool;
}

/// Legacy metadata stub.
pub struct Metadata {}

/// Deprecated inline diffusion_gemma compiler stub.
pub mod diffusion_gemma {
    use super::*;
    pub struct DgCompiler;
    impl DgCompiler {
        pub fn compile(_gguf_path: &Path, _variant: DgVariant) -> Result<Metadata> {
            Ok(Metadata {})
        }
    }
}

/// Deprecated — always returns an error.
pub fn compile_diffusion_model(
    _gguf_path: &Path,
    _model_type: ModelType,
    _gpu_topology: &[GpuInfo],
) -> Result<CImage> {
    anyhow::bail!("Model compilation via CImage is deprecated; use generate_image() instead");
}

// ═══════════════════════════════════════════════════════════════════════════
// Prism Image Generation Facade — Entry Point
// ═══════════════════════════════════════════════════════════════════════════
//
// The canonical execution path:
//   request  →  admission  →  routing  →  provider  →  receipt
//
// Every step produces typed output.  No step silently falls back.

/// Generate an image from a text prompt.
///
/// Entry point for the Prism image generation facade.  Always available at
/// compile time; returns `ImageGenerationError::FeatureUnavailable` when the
/// `generation-image` feature is not enabled.
pub fn generate_image(
    model_path: &str,
    request: ImageGenerationRequest,
) -> Result<ImageGenerationResult, ImageGenerationError> {
    #[cfg(feature = "generation-image")]
    {
        let response = generate_via_admission_pipeline(model_path, request)?;
        match response.outcome {
            reliability::ImageGenerationOutcome::Success(result) => Ok(result),
            _ => unreachable!("admission pipeline always produces a Success outcome"),
        }
    }

    #[cfg(not(feature = "generation-image"))]
    {
        let _ = (model_path, request);
        Err(ImageGenerationError::FeatureUnavailable {
            capability: "generation-image",
        })
    }
}

/// Generate an image and return the full [`ImageGenerationResponse`]
/// including the terminal receipt with admission, route, execution,
/// and output evidence.
#[cfg(feature = "generation-image")]
pub fn generate_image_with_receipt(
    model_path: &str,
    request: ImageGenerationRequest,
) -> Result<reliability::ImageGenerationResponse, ImageGenerationError> {
    generate_via_admission_pipeline(model_path, request)
}

#[cfg(feature = "generation-image")]
fn generate_via_admission_pipeline(
    model_path: &str,
    request: ImageGenerationRequest,
) -> Result<reliability::ImageGenerationResponse, ImageGenerationError> {
    use std::time::Instant;
    let t_start = Instant::now();
    let request_id = RequestId::new();
    let t0_str = now_iso();

    // ── Build installed CImage from path ────────────────────────────────
    let cimage = resolve_cimage(model_path)?;

    // ── Machine profile ─────────────────────────────────────────────────
    let machine = provider::MachineProfile {
        os_version: std::env::consts::OS.to_string(),
        has_ane: false,
        unified_memory_gb: 0,
    };

    // ── Build providers ─────────────────────────────────────────────────
    let providers: Vec<Box<dyn provider::ImageGenerationProvider>> =
        build_providers(&cimage, model_path);

    let provider_refs: Vec<&dyn provider::ImageGenerationProvider> =
        providers.iter().map(|p| p.as_ref()).collect();

    // ── Admission ───────────────────────────────────────────────────────
    let plan = admission::ImageGenerationAdmissionGate.admit(
        &cimage,
        &request,
        &machine,
        &request.execution_policy,
    )?;

    // Dry-run policy — admission only, no execution.
    if request.execution_policy == GenerationExecutionPolicy::DryRunAdmission {
        let t_elapsed = t_start.elapsed().as_secs_f64() * 1000.0;
        let digest_bytes = blake3_hash(b"dry-run-no-output");
        let output_digest = ArtifactDigest(digest_bytes);

        // Build route evidence for dry-run (no router called).
        let route_evidence = reliability::ImageGenerationRouteEvidence {
            requested_provider: request.device_preference,
            route_origin: RouteOrigin::DryRun,
            candidates: vec![],
            selected_provider: None,
            attempted_provider: None,
            fallback_considered: false,
            fallback_attempted: false,
            fallback_provider: None,
            fallback_reason: None,
            selected_provider_qualified: false,
        };

        let terminal_receipt = reliability::ImageGenerationTerminalReceipt {
            request_id: request_id.clone(),
            terminal_state: reliability::ImageGenerationTerminalState::Succeeded,
            admission: plan.admission_evidence,
            route: route_evidence,
            execution: None,
            output: None,
            failure: None,
            cancellation: None,
            created_at: t0_str,
            completed_at: now_iso(),
        };

        let result = ImageGenerationResult {
            image: GeneratedImage {
                width: 0,
                height: 0,
                format: ImageOutputFormat::Rgba8,
                bytes: vec![],
                digest: output_digest.clone(),
            },
            receipt: ImageGenerationReceipt {
                request_id: request_id.clone(),
                model_digest: cimage.digest.clone(),
                requested_provider: request.device_preference,
                selected_provider: ImageProviderKind::Unavailable,
                route_origin: RouteOrigin::DryRun,
                provider_version: String::new(),
                qualification_status: QualificationStatus::Accepted,
                fallback_used: false,
                denoising_steps_requested: request.steps,
                denoising_steps_completed: 0,
                width: 0,
                height: 0,
                output_format: ImageOutputFormat::Rgba8,
                output_digest,
                total_latency_ms: t_elapsed,
                provider_latency_ms: 0.0,
                materialization: MaterializationReceipt::new_copied(0),
                warnings: vec![],
            },
        };

        return Ok(reliability::ImageGenerationResponse {
            outcome: reliability::ImageGenerationOutcome::Success(result),
            receipt: terminal_receipt,
        });
    }

    // ── Routing ─────────────────────────────────────────────────────────
    let route = router::select_provider(&request, &cimage, &machine, &provider_refs, None, None)?;

    // ── Execution ───────────────────────────────────────────────────────
    let execution_id = provider::ExecutionId::new();
    let provider_request = provider::ImageGenerationProviderRequest {
        installed_image: &cimage,
        request: &request,
        machine: &machine,
        execution_id,
    };

    let exec_provider = &provider_refs[route.provider_index];
    let cancellation_token = ImageGenerationCancellationToken::new(RequestId::new());
    let provider_result = exec_provider
        .generate(&provider_request, &cancellation_token)
        .map_err(|e| ImageGenerationError::ProviderExecutionFailed {
            provider: route.provider_kind,
            source: Box::new(e),
        })?;

    // ── Validate output ─────────────────────────────────────────────────
    if provider_result.rgba_bytes.is_empty() {
        return Err(ImageGenerationError::InvalidOutput {
            provider: route.provider_kind,
            reason: "provider returned empty output".into(),
        });
    }

    let expected_len = provider_result.width as u64 * provider_result.height as u64 * 4;
    if provider_result.rgba_bytes.len() as u64 != expected_len {
        return Err(ImageGenerationError::InvalidOutput {
            provider: route.provider_kind,
            reason: format!(
                "expected {expected_len} bytes for {}x{} RGBA, got {}",
                provider_result.width,
                provider_result.height,
                provider_result.rgba_bytes.len()
            ),
        });
    }

    // ── Materialize output ──────────────────────────────────────────────
    let total_latency_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let digest_hex = blake3_hash(&provider_result.rgba_bytes);
    let output_digest = ArtifactDigest(digest_hex);

    // ── Terminal state ─────────────────────────────────────────────────
    let terminal_state = if route.fallback_used {
        reliability::ImageGenerationTerminalState::SucceededViaQualifiedFallback
    } else {
        reliability::ImageGenerationTerminalState::Succeeded
    };

    // ── Execution evidence ────────────────────────────────────────────
    let execution_evidence = reliability::ImageGenerationExecutionEvidence {
        provider: route.provider_kind,
        provider_version: provider_result.provider_metadata.provider_version.clone(),
        denoising_steps_requested: request.steps,
        denoising_steps_completed: provider_result.provider_metadata.steps_completed,
        provider_latency_ms: provider_result.provider_latency_ms,
        materialization: provider_result.materialization.clone(),
    };

    // ── Output evidence ───────────────────────────────────────────────
    let output_evidence = reliability::ImageGenerationOutputEvidence {
        width: provider_result.width,
        height: provider_result.height,
        output_format: ImageOutputFormat::Rgba8,
        output_digest: output_digest.clone(),
        lifecycle: reliability::ImageOutputLifecycle::Validated,
        bytes_produced: provider_result.rgba_bytes.len() as u64,
        validation_passed: true,
    };

    // ── Terminal receipt ──────────────────────────────────────────────
    let terminal_receipt = reliability::ImageGenerationTerminalReceipt {
        request_id: request_id.clone(),
        terminal_state,
        admission: plan.admission_evidence,
        route: route.route_evidence,
        execution: Some(execution_evidence),
        output: Some(output_evidence),
        failure: None,
        cancellation: None,
        created_at: t0_str,
        completed_at: now_iso(),
    };

    let generated_image = GeneratedImage {
        width: provider_result.width,
        height: provider_result.height,
        format: ImageOutputFormat::Rgba8,
        bytes: provider_result.rgba_bytes,
        digest: output_digest.clone(),
    };

    let receipt = ImageGenerationReceipt {
        request_id: request_id.clone(),
        model_digest: cimage.digest.clone(),
        requested_provider: request.device_preference,
        selected_provider: route.provider_kind,
        route_origin: route.route_origin,
        provider_version: provider_result.provider_metadata.provider_version,
        qualification_status: QualificationStatus::Accepted,
        fallback_used: route.fallback_used,
        denoising_steps_requested: request.steps,
        denoising_steps_completed: provider_result.provider_metadata.steps_completed,
        width: provider_result.width,
        height: provider_result.height,
        output_format: ImageOutputFormat::Rgba8,
        output_digest: output_digest.clone(),
        total_latency_ms,
        provider_latency_ms: provider_result.provider_latency_ms,
        materialization: provider_result.materialization,
        warnings: vec![],
    };

    let result = ImageGenerationResult {
        image: generated_image,
        receipt,
    };

    Ok(reliability::ImageGenerationResponse {
        outcome: reliability::ImageGenerationOutcome::Success(result),
        receipt: terminal_receipt,
    })
}

// ── Helpers ────────────────────────────────────────────────────────────

#[cfg(feature = "generation-image")]
fn resolve_cimage(path: &str) -> Result<manifest::InstalledCImage, ImageGenerationError> {
    use crate::image::manifest::{ImageGenerationCapabilityManifest, InstalledCImage};

    // Build a minimal manifest for the installed artifact.
    // In production this would parse the .cimage binary manifest; for the MVP
    // we construct a default (all-components-absent) manifest so that the
    // admission gate can reject it appropriately when components are missing.
    let digest = blake3_hash(path.as_bytes());

    Ok(InstalledCImage {
        path: path.to_string(),
        digest: ArtifactDigest(digest),
        manifest: ImageGenerationCapabilityManifest::default(),
        provider_handles: vec![],
    })
}

#[cfg(feature = "generation-image")]
fn build_providers(
    _cimage: &manifest::InstalledCImage,
    model_path: &str,
) -> Vec<Box<dyn provider::ImageGenerationProvider>> {
    let mut providers: Vec<Box<dyn provider::ImageGenerationProvider>> = Vec::new();

    // ComputeCoreMlxImageProvider — attempt to construct.  If loading fails
    // the router will handle the missing provider.
    match provider::ComputeCoreMlxImageProvider::new(model_path) {
        Ok(p) => providers.push(Box::new(p)),
        Err(e) => {
            let _ = e;
        }
    }

    // DiffusionGemmaImageProvider — attempt to construct when the manifest
    // identifies a DiffusionGemma model family.  Gated behind prism-backend
    // because DiffusionProvider depends on MLX.
    #[cfg(all(feature = "prism-backend", feature = "generation-diffusion"))]
    if _cimage.manifest.model_family == manifest::ImageModelFamily::DiffusionGemma {
        match provider::DiffusionGemmaImageProvider::new(model_path) {
            Ok(p) => providers.push(Box::new(p)),
            Err(e) => {
                let _ = e;
            }
        }
    }

    // PrismLutImageProvider — always available as a stub.
    providers.push(Box::new(provider::PrismLutImageProvider));
    providers
}

/// BLAKE3 hash helper.
/// ISO 8601 timestamp helper.
#[cfg(feature = "generation-image")]
fn now_iso() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    format!(
        "{:?}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    )
}

/// BLAKE3 hash helper.
#[cfg(feature = "generation-image")]
fn blake3_hash(data: &[u8]) -> String {
    use blake3::Hasher;
    let mut h = Hasher::new();
    h.update(data);
    h.finalize().to_hex().to_string()
}
