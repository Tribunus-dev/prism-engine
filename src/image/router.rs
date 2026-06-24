// ── Prism Image Generation — Route Selector ───────────────────────────
//
// Deterministic provider selection for text-to-image generation.
// Implements the full route matrix: auto-selection, explicit request,
// and qualified fallback.  Every decision is runtime-grounded —
// no compile-time feature gating, no silent degradation.

use super::compatibility::{CompatibilityStatus, ImageCompatibilityReceipt};
use super::manifest::*;
use super::provider::*;
use super::reliability::*;
use super::resolver::QualificationResolver;
use super::types::*;

/// Outcome of provider selection.
#[derive(Debug)]
pub struct SelectedRoute {
    /// The provider kind that will serve the request.
    pub provider_kind: ImageProviderKind,
    /// How this provider was chosen.
    pub route_origin: RouteOrigin,
    /// Whether a fallback occurred from the originally requested provider.
    pub fallback_used: bool,
    /// Index into the `providers` slice passed to `select_provider`.
    pub provider_index: usize,
    /// Whether this route bypassed qualification (used under experimental policy).
    pub qualification_override: bool,
    /// Non-fatal warning about this route (e.g. experimental override).
    pub warning: Option<String>,
    /// Route evidence captured during provider selection.
    pub route_evidence: ImageGenerationRouteEvidence,
}

/// Evaluate whether a single provider is qualified for a given CImage + machine.
#[cfg(test)]
fn is_provider_qualified(
    provider: &dyn ImageGenerationProvider,
    cimage: &InstalledCImage,
    machine: &MachineProfile,
) -> ImageProviderCapability {
    provider.capability_report(cimage, machine)
}

/// Find the index of the first qualified provider matching `kind`.
///
/// This is a simpler variant of `find_eligible_provider` that does not
/// consult the qualification resolver.
#[allow(dead_code)]
fn find_qualified_provider(
    kind: ImageProviderKind,
    providers: &[&dyn ImageGenerationProvider],
    cimage: &InstalledCImage,
    machine: &MachineProfile,
) -> Option<usize> {
    providers
        .iter()
        .position(|p| p.kind() == kind && p.capability_report(cimage, machine).is_qualified())
}

/// Look up the best (most advanced) compatibility status for a given provider
/// kind from a slice of compatibility receipts.
///
/// Returns `None` when no receipts match the provider kind.
fn compatibility_status_for_provider(
    kind: ImageProviderKind,
    receipts: &[ImageCompatibilityReceipt],
) -> Option<CompatibilityStatus> {
    // Rank statuses by qualification pipeline progress:
    // PerformanceQualified (highest) > RepeatabilityQualified > FunctionallyQualified > everything else
    fn rank(s: CompatibilityStatus) -> u8 {
        match s {
            CompatibilityStatus::PerformanceQualified => 5,
            CompatibilityStatus::RepeatabilityQualified => 4,
            CompatibilityStatus::FunctionallyQualified => 3,
            // Untried, FixtureUnavailable, failures, and regressions are all
            // below the eligibility threshold.
            _ => 0,
        }
    }

    receipts
        .iter()
        .filter(|r| r.provider == kind)
        .map(|r| r.qualification_status)
        .max_by_key(|s| rank(*s))
}

/// Extended provider lookup that also checks the qualification resolver.
/// Compatibility receipts further gate eligibility under production policy.
/// Returns `(index, qualification_override, warning)` where:
/// - `index`: provider position in the slice
/// - `qualification_override`: whether an unqualified provider was used under experimental policy
/// - `warning`: set when experimental override is active or compatibility is below threshold
fn find_eligible_provider(
    kind: ImageProviderKind,
    providers: &[&dyn ImageGenerationProvider],
    cimage: &InstalledCImage,
    machine: &MachineProfile,
    qualification: Option<&dyn QualificationResolver>,
    compatibility: Option<&[ImageCompatibilityReceipt]>,
) -> Option<(usize, bool, Option<String>)> {
    for (idx, provider) in providers.iter().enumerate() {
        if provider.kind() != kind {
            continue;
        }
        let cap = provider.capability_report(cimage, machine);
        if !cap.is_qualified() {
            continue;
        }
        // Compatibility-based eligibility check
        if let Some(receipts) = compatibility {
            let compat_status = compatibility_status_for_provider(kind, receipts);
            match compat_status {
                // PerformanceQualified or RepeatabilityQualified → eligible, proceed normally
                Some(s) if s.is_route_eligible() => {}
                // FunctionallyQualified (development-eligible but not route-eligible)
                Some(s) if s.is_development_eligible() => {
                    if qualification.is_some() {
                        // Production policy: FunctionallyQualified is not sufficient
                        continue;
                    }
                    // Development policy: allow with experimental override
                    return Some((
                        idx,
                        true,
                        Some(
                            "Using functionally-qualified provider under experimental policy"
                                .to_string(),
                        ),
                    ));
                }
                // Any other status (Untried, FixtureUnavailable, failures, etc.) → ineligible
                _ => continue,
            }
        }

        // Under default policy, treat AvailableButUnqualified as ineligible
        // unless the resolver can provide qualification evidence.
        match cap {
            ImageProviderCapability::ComputeCoreMlxAvailableButUnqualified
            | ImageProviderCapability::PrismLutAvailableButUnqualified
            | ImageProviderCapability::CoreMlAneAvailableButUnqualified => {
                if let Some(resolver) = qualification {
                    // Production mode: check for qualification evidence
                    if resolver
                        .resolve_image(&cimage.digest, kind, machine)
                        .is_some()
                    {
                        return Some((idx, false, None));
                    }
                } else {
                    // Development mode: allow with experimental override
                    return Some((
                        idx,
                        true,
                        Some("Using unqualified provider under experimental policy".to_string()),
                    ));
                }
            }
            _ => return Some((idx, false, None)),
        }
    }
    None
}

/// Select a provider for `request` from the available `providers`.
///
/// The route matrix (highest priority first):
///
/// | Request preference | Conditions | Result |
/// |---|---|---|
/// | Auto | Qualified Compute MLX available | ComputeCoreMlx (AutoSelection) |
/// | Auto | Compute MLX unavailable, qualified PrismLut available | PrismLut (AutoSelection) |
/// | Auto | No qualified provider | `RequestedProviderUnavailable` |
/// | ComputeCoreMlx | Qualified | ComputeCoreMlx (ExplicitRequest) |
/// | ComputeCoreMlx | Unavailable | `RequestedProviderUnavailable` |
/// | PrismLut | Qualified | PrismLut (ExplicitRequest) |
/// | PrismLut | Unavailable, `AllowQualifiedFallback`, qualified Compute MLX | ComputeCoreMlx (QualifiedFallback, `fallback_used = true`) |
/// | PrismLut | Unavailable, no fallback path | `RequestedProviderUnavailable` |
///
/// Compatibility receipts (`compatibility`) gate production routing: only
/// providers with `RepeatabilityQualified` or `PerformanceQualified` status
/// are eligible under production policy.  `FunctionallyQualified` providers
/// are allowed only under development (experimental) policy.
pub fn select_provider(
    request: &ImageGenerationRequest,
    cimage: &InstalledCImage,
    machine: &MachineProfile,
    providers: &[&dyn ImageGenerationProvider],
    qualification: Option<&dyn QualificationResolver>,
    compatibility: Option<&[ImageCompatibilityReceipt]>,
) -> Result<SelectedRoute, ImageGenerationError> {
    match request.device_preference {
        DevicePreference::Auto => {
            // 1. Auto + qualified Compute MLX → ComputeCoreMlx (AutoSelection)
            if let Some((idx, qualification_override, warning)) = find_eligible_provider(
                ImageProviderKind::ComputeCoreMlx,
                providers,
                cimage,
                machine,
                qualification,
                compatibility,
            ) {
                let route = SelectedRoute {
                    provider_kind: ImageProviderKind::ComputeCoreMlx,
                    route_origin: RouteOrigin::AutoSelection,
                    fallback_used: false,
                    provider_index: idx,
                    qualification_override,
                    warning,
                    route_evidence: build_route_evidence(
                        request,
                        ImageProviderKind::ComputeCoreMlx,
                        RouteOrigin::AutoSelection,
                        false,
                        qualification_override,
                        providers,
                        cimage,
                        machine,
                    ),
                };
                return Ok(route);
            }

            // 2. Auto + unavailable Compute MLX + qualified PrismLut → PrismLut (AutoSelection)
            if let Some((idx, qualification_override, warning)) = find_eligible_provider(
                ImageProviderKind::PrismLut,
                providers,
                cimage,
                machine,
                qualification,
                compatibility,
            ) {
                let route = SelectedRoute {
                    provider_kind: ImageProviderKind::PrismLut,
                    route_origin: RouteOrigin::AutoSelection,
                    fallback_used: false,
                    provider_index: idx,
                    qualification_override,
                    warning,
                    route_evidence: build_route_evidence(
                        request,
                        ImageProviderKind::PrismLut,
                        RouteOrigin::AutoSelection,
                        false,
                        qualification_override,
                        providers,
                        cimage,
                        machine,
                    ),
                };
                return Ok(route);
            }

            // 3. Auto + no qualified provider → error
            let available: Vec<ImageProviderKind> = providers.iter().map(|p| p.kind()).collect();
            Err(ImageGenerationError::RequestedProviderUnavailable {
                requested: DevicePreference::Auto,
                available,
            })
        }

        DevicePreference::ComputeCoreMlx => {
            // 4. ComputeCoreMlx + qualified → ComputeCoreMlx (ExplicitRequest)
            if let Some((idx, qualification_override, warning)) = find_eligible_provider(
                ImageProviderKind::ComputeCoreMlx,
                providers,
                cimage,
                machine,
                qualification,
                compatibility,
            ) {
                let route = SelectedRoute {
                    provider_kind: ImageProviderKind::ComputeCoreMlx,
                    route_origin: RouteOrigin::ExplicitRequest,
                    fallback_used: false,
                    provider_index: idx,
                    qualification_override,
                    warning,
                    route_evidence: build_route_evidence(
                        request,
                        ImageProviderKind::ComputeCoreMlx,
                        RouteOrigin::ExplicitRequest,
                        false,
                        qualification_override,
                        providers,
                        cimage,
                        machine,
                    ),
                };
                return Ok(route);
            }

            // 5. ComputeCoreMlx + unavailable → error
            let available: Vec<ImageProviderKind> = providers.iter().map(|p| p.kind()).collect();
            Err(ImageGenerationError::RequestedProviderUnavailable {
                requested: DevicePreference::ComputeCoreMlx,
                available,
            })
        }

        DevicePreference::PrismLut => {
            // 6. PrismLut + qualified → PrismLut (ExplicitRequest)
            if let Some((idx, qualification_override, warning)) = find_eligible_provider(
                ImageProviderKind::PrismLut,
                providers,
                cimage,
                machine,
                qualification,
                compatibility,
            ) {
                let route = SelectedRoute {
                    provider_kind: ImageProviderKind::PrismLut,
                    route_origin: RouteOrigin::ExplicitRequest,
                    fallback_used: false,
                    provider_index: idx,
                    qualification_override,
                    warning,
                    route_evidence: build_route_evidence(
                        request,
                        ImageProviderKind::PrismLut,
                        RouteOrigin::ExplicitRequest,
                        false,
                        qualification_override,
                        providers,
                        cimage,
                        machine,
                    ),
                };
                return Ok(route);
            }

            // 7. PrismLut + unavailable + AllowQualifiedFallback + qualified Compute MLX
            //    → ComputeCoreMlx (QualifiedFallback, fallback_used = true)
            if request.execution_policy == GenerationExecutionPolicy::AllowQualifiedFallback {
                if let Some((idx, qualification_override, warning)) = find_eligible_provider(
                    ImageProviderKind::ComputeCoreMlx,
                    providers,
                    cimage,
                    machine,
                    qualification,
                    compatibility,
                ) {
                    let route = SelectedRoute {
                        provider_kind: ImageProviderKind::ComputeCoreMlx,
                        route_origin: RouteOrigin::QualifiedFallback,
                        fallback_used: true,
                        provider_index: idx,
                        qualification_override,
                        warning,
                        route_evidence: build_route_evidence(
                            request,
                            ImageProviderKind::ComputeCoreMlx,
                            RouteOrigin::QualifiedFallback,
                            true,
                            qualification_override,
                            providers,
                            cimage,
                            machine,
                        ),
                    };
                    return Ok(route);
                }
            }

            // PrismLut + unavailable + no fallback path → error
            let available: Vec<ImageProviderKind> = providers.iter().map(|p| p.kind()).collect();
            Err(ImageGenerationError::RequestedProviderUnavailable {
                requested: DevicePreference::PrismLut,
                available,
            })
        }
    }
}

// ── Route evidence builder ───────────────────────────────────────────

/// Build candidate evidence for every available provider.
pub fn build_candidate_evidence(
    providers: &[&dyn ImageGenerationProvider],
    cimage: &InstalledCImage,
    machine: &MachineProfile,
) -> Vec<ImageProviderCandidateEvidence> {
    providers
        .iter()
        .map(|p| {
            let cap = p.capability_report(cimage, machine);
            let eligible = cap.is_qualified();
            let ineligibility_reason = if eligible {
                None
            } else {
                match cap {
                    ImageProviderCapability::ComputeCoreMlxAvailableButUnqualified
                    | ImageProviderCapability::PrismLutAvailableButUnqualified
                    | ImageProviderCapability::CoreMlAneAvailableButUnqualified => {
                        Some(ProviderIneligibilityReason::Unqualified)
                    }
                    ImageProviderCapability::CoreMlAneUnavailable
                    | ImageProviderCapability::ProviderUnavailable => {
                        Some(ProviderIneligibilityReason::Unavailable)
                    }
                    ImageProviderCapability::CoreMlAneRefusedByArtifactPolicy => {
                        Some(ProviderIneligibilityReason::PolicyProhibited)
                    }
                    _ => Some(ProviderIneligibilityReason::Unavailable),
                }
            };
            ImageProviderCandidateEvidence {
                provider: p.kind(),
                capability: cap,
                eligible,
                ineligibility_reason,
            }
        })
        .collect()
}

/// Build route evidence from a selected route and the full provider list.
pub fn build_route_evidence(
    request: &ImageGenerationRequest,
    provider_kind: ImageProviderKind,
    route_origin: RouteOrigin,
    fallback_used: bool,
    qualification_override: bool,
    providers: &[&dyn ImageGenerationProvider],
    cimage: &InstalledCImage,
    machine: &MachineProfile,
) -> ImageGenerationRouteEvidence {
    let candidates = build_candidate_evidence(providers, cimage, machine);
    let fallback_provider = if fallback_used {
        Some(provider_kind)
    } else {
        None
    };
    let fallback_reason = if fallback_used {
        Some(FallbackReason::RequestedProviderUnavailable)
    } else {
        None
    };

    let fallback_considered =
        fallback_used || candidates.iter().any(|c| c.provider != provider_kind);

    ImageGenerationRouteEvidence {
        requested_provider: request.device_preference,
        route_origin,
        candidates,
        selected_provider: Some(provider_kind),
        attempted_provider: Some(provider_kind),
        fallback_considered,
        fallback_attempted: fallback_used,
        fallback_provider,
        fallback_reason,
        selected_provider_qualified: !qualification_override,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test providers ─────────────────────────────────────────────────

    struct MockProvider {
        kind: ImageProviderKind,
        qualified: bool,
    }

    impl MockProvider {
        fn qualified(kind: ImageProviderKind) -> Self {
            Self {
                kind,
                qualified: true,
            }
        }

        fn unqualified(kind: ImageProviderKind) -> Self {
            Self {
                kind,
                qualified: false,
            }
        }
    }

    impl ImageGenerationProvider for MockProvider {
        fn kind(&self) -> ImageProviderKind {
            self.kind
        }

        fn capability_report(
            &self,
            _cimage: &InstalledCImage,
            _machine: &MachineProfile,
        ) -> ImageProviderCapability {
            if self.qualified {
                ImageProviderCapability::ComputeCoreMlxQualified
            } else {
                ImageProviderCapability::ProviderUnavailable
            }
        }

        fn generate(
            &self,
            _request: &ImageGenerationProviderRequest,
            _cancellation: &ImageGenerationCancellationToken,
        ) -> std::result::Result<ImageGenerationProviderResult, ImageProviderError> {
            unimplemented!("mock provider does not generate")
        }
    }

    fn auto_request() -> ImageGenerationRequest {
        ImageGenerationRequest::new("test", 512, 512)
    }

    fn compute_mlx_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            device_preference: DevicePreference::ComputeCoreMlx,
            ..ImageGenerationRequest::new("test", 512, 512)
        }
    }

    fn prism_lut_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            device_preference: DevicePreference::PrismLut,
            ..ImageGenerationRequest::new("test", 512, 512)
        }
    }

    fn prism_lut_fallback_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            device_preference: DevicePreference::PrismLut,
            execution_policy: GenerationExecutionPolicy::AllowQualifiedFallback,
            ..ImageGenerationRequest::new("test", 512, 512)
        }
    }

    fn prism_lut_strict_request() -> ImageGenerationRequest {
        ImageGenerationRequest {
            device_preference: DevicePreference::PrismLut,
            execution_policy: GenerationExecutionPolicy::RequireRequestedProvider,
            ..ImageGenerationRequest::new("test", 512, 512)
        }
    }

    fn default_cimage() -> InstalledCImage {
        // Minimal CImage — the mock provider ignores it anyway.
        InstalledCImage {
            path: String::new(),
            digest: ArtifactDigest(String::new()),
            manifest: ImageGenerationCapabilityManifest::default(),
            provider_handles: vec![],
        }
    }

    fn default_machine() -> MachineProfile {
        MachineProfile {
            os_version: String::new(),
            has_ane: false,
            unified_memory_gb: 0,
        }
    }

    // ── Helper: build a provider list from mock providers ──────────────

    fn build_providers(
        compute_mlx_qualified: bool,
        prism_lut_qualified: bool,
    ) -> Vec<MockProvider> {
        vec![
            MockProvider::qualified(ImageProviderKind::ComputeCoreMlx),
            MockProvider::qualified(ImageProviderKind::PrismLut),
        ]
        .into_iter()
        .filter(|p| match p.kind {
            ImageProviderKind::ComputeCoreMlx => compute_mlx_qualified,
            ImageProviderKind::PrismLut => prism_lut_qualified,
            _ => true,
        })
        .map(|p| {
            if p.qualified {
                p
            } else {
                MockProvider::unqualified(p.kind)
            }
        })
        .collect()
    }

    fn to_refs(providers: &[MockProvider]) -> Vec<&dyn ImageGenerationProvider> {
        providers
            .iter()
            .map(|p| p as &dyn ImageGenerationProvider)
            .collect()
    }

    // ── Route matrix tests ────────────────────────────────────────────

    #[test]
    fn auto_selects_compute_mlx_when_qualified() {
        let providers = build_providers(true, false);
        let refs = to_refs(&providers);

        let route = select_provider(
            &auto_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("auto should pick compute mlx");

        assert_eq!(route.provider_kind, ImageProviderKind::ComputeCoreMlx);
        assert_eq!(route.route_origin, RouteOrigin::AutoSelection);
        assert!(!route.fallback_used);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn auto_selects_prism_lut_when_compute_mlx_unavailable() {
        let providers = build_providers(false, true);
        let refs = to_refs(&providers);

        let route = select_provider(
            &auto_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("auto should pick prism lut");

        assert_eq!(route.provider_kind, ImageProviderKind::PrismLut);
        assert_eq!(route.route_origin, RouteOrigin::AutoSelection);
        assert!(!route.fallback_used);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn auto_errors_when_no_qualified_provider() {
        let providers = build_providers(false, false);
        let refs = to_refs(&providers);

        let err = select_provider(
            &auto_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect_err("auto with no qualified provider should error");

        assert!(
            matches!(
                err,
                ImageGenerationError::RequestedProviderUnavailable { .. }
            ),
            "expected RequestedProviderUnavailable, got {err:?}"
        );
    }

    #[test]
    fn compute_mlx_explicit_when_qualified() {
        let providers = build_providers(true, false);
        let refs = to_refs(&providers);

        let route = select_provider(
            &compute_mlx_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("explicit compute mlx should succeed when qualified");

        assert_eq!(route.provider_kind, ImageProviderKind::ComputeCoreMlx);
        assert_eq!(route.route_origin, RouteOrigin::ExplicitRequest);
        assert!(!route.fallback_used);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn compute_mlx_explicit_errors_when_unavailable() {
        let providers = build_providers(false, true);
        let refs = to_refs(&providers);

        let err = select_provider(
            &compute_mlx_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect_err("explicit compute mlx with no qualified provider should error");

        assert!(
            matches!(
                err,
                ImageGenerationError::RequestedProviderUnavailable {
                    requested: DevicePreference::ComputeCoreMlx,
                    ..
                }
            ),
            "expected RequestedProviderUnavailable(ComputeCoreMlx), got {err:?}"
        );
    }

    #[test]
    fn prism_lut_explicit_when_qualified() {
        let providers = build_providers(false, true);
        let refs = to_refs(&providers);

        let route = select_provider(
            &prism_lut_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("explicit prism lut should succeed when qualified");

        assert_eq!(route.provider_kind, ImageProviderKind::PrismLut);
        assert_eq!(route.route_origin, RouteOrigin::ExplicitRequest);
        assert!(!route.fallback_used);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn prism_lut_unqualified_falls_back_to_compute_mlx() {
        let providers = build_providers(true, false);
        let refs = to_refs(&providers);

        let route = select_provider(
            &prism_lut_fallback_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("prism lut with fallback should pick compute mlx");

        assert_eq!(route.provider_kind, ImageProviderKind::ComputeCoreMlx);
        assert_eq!(route.route_origin, RouteOrigin::QualifiedFallback);
        assert!(route.fallback_used);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn prism_lut_unqualified_no_fallback_errors() {
        let providers = build_providers(true, false);
        let refs = to_refs(&providers);

        let err = select_provider(
            &prism_lut_strict_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect_err("prism lut without fallback should error");

        assert!(
            matches!(
                err,
                ImageGenerationError::RequestedProviderUnavailable {
                    requested: DevicePreference::PrismLut,
                    ..
                }
            ),
            "expected RequestedProviderUnavailable(PrismLut), got {err:?}"
        );
    }

    #[test]
    fn prism_lut_fallback_no_compute_mlx_errors() {
        let providers = build_providers(false, false);
        let refs = to_refs(&providers);

        let err = select_provider(
            &prism_lut_fallback_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect_err("prism lut with fallback but no compute mlx should error");

        assert!(
            matches!(
                err,
                ImageGenerationError::RequestedProviderUnavailable {
                    requested: DevicePreference::PrismLut,
                    ..
                }
            ),
            "expected RequestedProviderUnavailable(PrismLut), got {err:?}"
        );
    }

    #[test]
    fn auto_prefers_compute_mlx_over_prism_lut() {
        let providers = build_providers(true, true);
        let refs = to_refs(&providers);

        let route = select_provider(
            &auto_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("auto should pick compute mlx over prism lut");

        assert_eq!(route.provider_kind, ImageProviderKind::ComputeCoreMlx);
        assert_eq!(route.route_origin, RouteOrigin::AutoSelection);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn provider_index_matches_position() {
        let providers = build_providers(true, true);
        let refs = to_refs(&providers);

        let route = select_provider(
            &compute_mlx_request(),
            &default_cimage(),
            &default_machine(),
            &refs,
            None,
            None,
        )
        .expect("explicit compute mlx should succeed");

        // ComputeCoreMlx should be at index 0 (first in build_providers(true, true))
        assert_eq!(route.provider_index, 0);
        assert!(!route.qualification_override);
        assert!(route.warning.is_none());
    }

    #[test]
    fn is_provider_qualified_wraps_capability_report() {
        let provider = MockProvider::qualified(ImageProviderKind::ComputeCoreMlx);
        let cap = is_provider_qualified(&provider, &default_cimage(), &default_machine());
        assert_eq!(cap, ImageProviderCapability::ComputeCoreMlxQualified);

        let provider = MockProvider::unqualified(ImageProviderKind::ComputeCoreMlx);
        let cap = is_provider_qualified(&provider, &default_cimage(), &default_machine());
        assert_eq!(cap, ImageProviderCapability::ProviderUnavailable);
    }

    #[test]
    fn find_qualified_provider_returns_correct_index() {
        let providers = build_providers(true, false);
        let refs = to_refs(&providers);

        let idx = find_qualified_provider(
            ImageProviderKind::ComputeCoreMlx,
            &refs,
            &default_cimage(),
            &default_machine(),
        );
        assert_eq!(idx, Some(0));

        let idx = find_qualified_provider(
            ImageProviderKind::PrismLut,
            &refs,
            &default_cimage(),
            &default_machine(),
        );
        assert_eq!(idx, None);
    }
}
