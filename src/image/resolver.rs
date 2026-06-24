// ── Prism Image Generation — Qualification Resolver ──────────────────────
//
// Defines the trait used by the router to look up qualification evidence for
// a provider + artifact + machine combination.

use super::manifest::ImageQualificationRecord;
use super::provider::MachineProfile;
use super::types::*;

/// Minimal Core ML qualification receipt.
///
/// This is a Prism-owned stub that mirrors the shape of compute-core's
/// CoreMlQualificationReceipt without depending on that crate directly.
#[derive(Debug, Clone)]
pub struct CoreMlQualificationReceipt {
    pub fixture_id: String,
    pub qualification_status: QualificationStatus,
    pub machine_fingerprint: String,
}

/// Provides qualification evidence for a provider + artifact + machine.
pub trait QualificationResolver: Send + Sync {
    /// Look up a Core ML qualification receipt for the given artifact and machine.
    fn resolve_coreml(
        &self,
        artifact: &ArtifactDigest,
        machine: &MachineProfile,
    ) -> Option<CoreMlQualificationReceipt>;

    /// Look up an image qualification record for the given artifact, provider, and machine.
    fn resolve_image(
        &self,
        artifact: &ArtifactDigest,
        provider: ImageProviderKind,
        machine: &MachineProfile,
    ) -> Option<ImageQualificationRecord>;
}

/// A no-qualification resolver that always returns None (unqualified).
/// Used as default when no qualification store is configured.
pub struct NoOpQualificationResolver;

impl QualificationResolver for NoOpQualificationResolver {
    fn resolve_coreml(
        &self,
        _: &ArtifactDigest,
        _: &MachineProfile,
    ) -> Option<CoreMlQualificationReceipt> {
        None
    }

    fn resolve_image(
        &self,
        _: &ArtifactDigest,
        _: ImageProviderKind,
        _: &MachineProfile,
    ) -> Option<ImageQualificationRecord> {
        None
    }
}
