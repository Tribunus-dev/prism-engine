// ── Prism Image Generation — Admission Gate ──────────────────────────
//
// Validates ImageGenerationRequest parameters against an InstalledCImage
// manifest before provider dispatch. All validation logic is pure — no
// I/O, no network, no model loading.

use super::manifest::*;
use super::provider::MachineProfile;
use super::reliability::ImageGenerationAdmissionEvidence;
use super::types::*;

// ── Admission gate ───────────────────────────────────────────────────

/// Unit struct whose [`admit`](Self::admit) method gates image generation.
pub struct ImageGenerationAdmissionGate;

impl ImageGenerationAdmissionGate {
    /// Validate `request` against the capability manifest of `image`.
    ///
    /// Returns an [`ImageGenerationExecutionPlan`] on success, or an
    /// [`ImageGenerationError`] describing why the request was refused.
    ///
    /// # Validation order
    ///
    /// 1. Dry-run policy → plan with `DryRun` origin, `Unavailable` provider.
    /// 2. Manifest schema version.
    /// 3. Core-component availability (`manifest.is_admittable()`).
    /// 4. Requested dimensions against manifest constraint.
    /// 5. Denoising step count against manifest step range.
    /// 6. Output format (Rgba8 always ok; Png requires a provider artifact).
    /// 7. Select a provider artifact, if one is present.
    pub fn admit(
        &self,
        image: &InstalledCImage,
        request: &ImageGenerationRequest,
        _machine: &MachineProfile,
        policy: &GenerationExecutionPolicy,
    ) -> Result<ImageGenerationExecutionPlan, ImageGenerationError> {
        let manifest = &image.manifest;

        // ── 1. Dry-run ───────────────────────────────────────────────
        if *policy == GenerationExecutionPolicy::DryRunAdmission {
            let req_comps = required_components_for(manifest);
            let present_components: Vec<ComponentRequirement> = req_comps
                .iter()
                .filter(|c| c.actual == ComponentAvailability::PresentQualified)
                .cloned()
                .collect();
            let missing_components: Vec<ComponentRequirement> = req_comps
                .iter()
                .filter(|c| c.actual != ComponentAvailability::PresentQualified)
                .cloned()
                .collect();
            let machine_fingerprint = format!(
                "{}|{}|{}",
                _machine.os_version, _machine.has_ane, _machine.unified_memory_gb
            );
            let request_bytes = format!("{:?}", request);
            let request_digest = *blake3::hash(request_bytes.as_bytes()).as_bytes();
            let image_capability_declared = manifest.is_admittable();
            let requested_dimensions = (request.width, request.height);
            let supported_dimensions = manifest.supported_dimensions.clone();
            let requested_steps = request.steps;
            let supported_steps = manifest.supported_steps;

            return Ok(ImageGenerationExecutionPlan {
                selected_provider: ImageProviderKind::Unavailable,
                route_origin: RouteOrigin::DryRun,
                fallback_used: false,
                provider_version: String::new(),
                qualification_status: QualificationStatus::Accepted,
                admission_evidence: ImageGenerationAdmissionEvidence {
                    artifact_digest: image.digest.clone(),
                    machine_fingerprint,
                    request_digest,
                    image_capability_declared,
                    required_components: req_comps,
                    present_components,
                    missing_components,
                    requested_dimensions,
                    supported_dimensions,
                    requested_steps,
                    supported_steps,
                    qualification_status: QualificationStatus::Accepted,
                    admitted: true,
                    refusal_reason: None,
                },
            });
        }

        // ── 2. Schema version ────────────────────────────────────────
        if manifest.schema_version != IMAGE_MANIFEST_SCHEMA_VERSION {
            return Err(ImageGenerationError::AdmissionRefused {
                reason: ImageGenerationRefusalReason::Other(format!(
                    "manifest schema version {} != expected {}",
                    manifest.schema_version, IMAGE_MANIFEST_SCHEMA_VERSION
                )),
            });
        }

        // ── 3. Image-capable (core components present & qualified) ───
        if !manifest.is_admittable() {
            let miss = missing_components_for(manifest);
            if let Some(mc) = miss.first() {
                return Err(ImageGenerationError::MissingComponent {
                    component: mc.name.clone(),
                });
            }
            return Err(ImageGenerationError::ArtifactNotImageCapable {
                artifact: image.digest.clone(),
            });
        }

        // ── 4. Dimensions ────────────────────────────────────────────
        if !manifest.supported_dimensions.accepts(request.width)
            || !manifest.supported_dimensions.accepts(request.height)
        {
            return Err(ImageGenerationError::AdmissionRefused {
                reason: ImageGenerationRefusalReason::DimensionsUnsupported {
                    width: request.width,
                    height: request.height,
                },
            });
        }

        // ── 5. Step range ────────────────────────────────────────────
        if !manifest.supported_steps.contains(request.steps) {
            return Err(ImageGenerationError::AdmissionRefused {
                reason: ImageGenerationRefusalReason::StepsOutOfRange {
                    steps: request.steps,
                    min: manifest.supported_steps.min,
                    max: manifest.supported_steps.max,
                },
            });
        }

        // ── 6. Output format ─────────────────────────────────────────
        if request.output_format == ImageOutputFormat::Png && manifest.provider_artifacts.is_empty()
        {
            return Err(ImageGenerationError::AdmissionRefused {
                reason: ImageGenerationRefusalReason::FormatUnsupported(ImageOutputFormat::Png),
            });
        }

        // ── 7. Select provider ───────────────────────────────────────
        let selected_provider =
            select_provider(manifest).ok_or_else(|| ImageGenerationError::AdmissionRefused {
                reason: ImageGenerationRefusalReason::Other(
                    "no provider artifact available".into(),
                ),
            })?;

        // Determine route origin and fallback flag.
        let requested_kind = match request.device_preference {
            DevicePreference::Auto => None,
            DevicePreference::ComputeCoreMlx => Some(ImageProviderKind::ComputeCoreMlx),
            DevicePreference::PrismLut => Some(ImageProviderKind::PrismLut),
        };

        let (route_origin, fallback_used) = match (requested_kind, *policy) {
            (Some(requested), _) if requested == selected_provider => {
                (RouteOrigin::ExplicitRequest, false)
            }
            (Some(_), GenerationExecutionPolicy::AllowQualifiedFallback) => {
                (RouteOrigin::QualifiedFallback, true)
            }
            (Some(_), GenerationExecutionPolicy::RequireRequestedProvider) => {
                return Err(ImageGenerationError::RequestedProviderUnavailable {
                    requested: request.device_preference,
                    available: vec![selected_provider],
                });
            }
            _ => (RouteOrigin::AutoSelection, false),
        };

        // Provider version from the selected artifact's qualification record.
        let pa = manifest
            .find_provider_artifact(selected_provider)
            .expect("select_provider returned a kind present in provider_artifacts ");

        // Build admission evidence.
        let req_comps = required_components_for(manifest);
        let present_components: Vec<ComponentRequirement> = req_comps
            .iter()
            .filter(|c| c.actual == ComponentAvailability::PresentQualified)
            .cloned()
            .collect();
        let missing_components: Vec<ComponentRequirement> = req_comps
            .iter()
            .filter(|c| c.actual != ComponentAvailability::PresentQualified)
            .cloned()
            .collect();
        let machine_fingerprint = format!(
            "{}|{}|{}",
            _machine.os_version, _machine.has_ane, _machine.unified_memory_gb
        );
        let request_bytes = format!("{:?}", request);
        let request_digest = *blake3::hash(request_bytes.as_bytes()).as_bytes();
        let image_capability_declared = manifest.is_admittable();
        let requested_dimensions = (request.width, request.height);
        let supported_dimensions = manifest.supported_dimensions.clone();
        let requested_steps = request.steps;
        let supported_steps = manifest.supported_steps;

        let admission_evidence = ImageGenerationAdmissionEvidence {
            artifact_digest: image.digest.clone(),
            machine_fingerprint,
            request_digest,
            image_capability_declared,
            required_components: req_comps.clone(),
            present_components: present_components.clone(),
            missing_components: missing_components.clone(),
            requested_dimensions,
            supported_dimensions,
            requested_steps,
            supported_steps,
            qualification_status: pa.qualification_record.status.clone(),
            admitted: true,
            refusal_reason: None,
        };

        Ok(ImageGenerationExecutionPlan {
            selected_provider,
            route_origin,
            fallback_used,
            provider_version: pa.qualification_record.compiler_version.clone(),
            qualification_status: pa.qualification_record.status.clone(),
            admission_evidence,
        })
    }
}

// ── Plan type ────────────────────────────────────────────────────────

/// Result of a successful admission gate run.
///
/// Contains everything the router and executor need to dispatch generation
/// to the correct provider.
pub struct ImageGenerationExecutionPlan {
    /// The provider selected to serve this request.
    pub selected_provider: ImageProviderKind,
    /// How the provider was chosen (explicit, auto, fallback, or dry-run).
    pub route_origin: RouteOrigin,
    /// Whether a fallback route was used.
    pub fallback_used: bool,
    /// Version of the selected provider artifact.
    pub provider_version: String,
    /// Qualification status of the selected artifact.
    pub qualification_status: QualificationStatus,
    /// Admission evidence captured during validation.
    pub admission_evidence: ImageGenerationAdmissionEvidence,
}

/// A named component requirement describing the expected vs actual
/// availability of an artifact component.
#[derive(Debug, Clone)]
pub struct ComponentRequirement {
    /// Human-readable component name (e.g. `"text_encoder"`).
    pub name: String,
    /// The availability level required for admission.
    pub required: ComponentAvailability,
    /// The actual availability declared in the manifest.
    pub actual: ComponentAvailability,
}

// ── Helper functions ─────────────────────────────────────────────────

/// All components that **should** be `PresentQualified` for image
/// generation to proceed.
pub fn required_components_for(
    manifest: &ImageGenerationCapabilityManifest,
) -> Vec<ComponentRequirement> {
    let mut comps = Vec::with_capacity(5);

    // Core components — all must be PresentQualified for admission.
    comps.push(ComponentRequirement {
        name: "text_encoder".into(),
        required: ComponentAvailability::PresentQualified,
        actual: manifest.text_encoder,
    });
    comps.push(ComponentRequirement {
        name: "denoiser".into(),
        required: ComponentAvailability::PresentQualified,
        actual: manifest.denoiser,
    });
    comps.push(ComponentRequirement {
        name: "vae_decoder".into(),
        required: ComponentAvailability::PresentQualified,
        actual: manifest.vae_decoder,
    });
    comps.push(ComponentRequirement {
        name: "tokenizer".into(),
        required: ComponentAvailability::PresentQualified,
        actual: manifest.tokenizer,
    });
    comps.push(ComponentRequirement {
        name: "scheduler".into(),
        required: ComponentAvailability::PresentQualified,
        actual: manifest.scheduler,
    });

    comps
}

/// Components that are **not** `PresentQualified`.
pub fn missing_components_for(
    manifest: &ImageGenerationCapabilityManifest,
) -> Vec<ComponentRequirement> {
    required_components_for(manifest)
        .into_iter()
        .filter(|c| c.actual != ComponentAvailability::PresentQualified)
        .collect()
}

/// Select the best available provider from the manifest.
///
/// Preference order:
/// 1. First qualified provider (qualification `Accepted`).
/// 2. First provider artifact with any other qualification status.
/// 3. `None` if no provider artifacts exist.
fn select_provider(manifest: &ImageGenerationCapabilityManifest) -> Option<ImageProviderKind> {
    // Prefer a qualified provider.
    if let Some(pa) = manifest
        .provider_artifacts
        .iter()
        .find(|pa| pa.qualification_record.status == QualificationStatus::Accepted)
    {
        return Some(pa.provider);
    }

    // Fall back to the first artifact.
    manifest.provider_artifacts.first().map(|pa| pa.provider)
}
