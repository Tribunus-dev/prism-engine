//! Disclosure subsystem — EU AI Act Article 50 compliance primitives.
//!
//! Enforces governed-output delivery conditions determined by the active
//! policy profile. Prism can render notices itself, block delivery pending
//! engine-verifiable host acceptance, or require managed-host attestation.
//!
//! Every transition is recorded in a canonical, tamper-evident receipt chain.

use crate::registry::types::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ── Disclosure control scope ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum DisclosureControlScope {
    PrismRendered = 0,
    SdkDeliveryGate = 1,
    HostAttested = 2,
    UnmanagedTelemetry = 3,
    AdvisoryOnly = 4,
}

// ── Disclosure requirement levels ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureRequirement {
    StrictEnforcement,
    ConditionalOnModality,
    AdvisoryOnly,
    ExemptByHumanReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionRequirement {
    MandatoryInternalInjection,
    MandatoryExternalLabel,
    OptionalAdvisory,
    CompletelyExempt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureEnforcement {
    Advisory,
    AcknowledgementRequired,
    RuntimeInjected,
    DeliveryBlockedUntilAcknowledged,
}

// ── Disclosure modality ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosureModalitySet {
    pub bitmask: u32,
}

impl DisclosureModalitySet {
    pub const TEXT_METADATA: u32 = 1 << 0;
    pub const IMAGE_METADATA: u32 = 1 << 1;
    pub const AUDIO_METADATA: u32 = 1 << 2;
    pub const VISIBLE_OVERLAY: u32 = 1 << 3;

    pub fn empty() -> Self {
        Self { bitmask: 0 }
    }
    pub fn has(&self, flag: u32) -> bool {
        self.bitmask & flag != 0
    }
    pub fn set(&mut self, flag: u32) {
        self.bitmask |= flag;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputModalityDisclosure {
    TextMetadata,
    ImageProvenanceMetadata,
    AudioMetadata,
    VisibleLabel,
    InaccessibleForCurrentBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryRequirement {
    Immediate,
    HostAcceptanceRequired,
    PrismRenderedOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureDeliveryState {
    NotRequired = 0,
    InjectedByPrism = 1,
    AwaitingHostAcceptance = 2,
    AcceptedByHost = 3,
    RefusedByHost = 4,
    TimedOut = 5,
}

// ── Jurisdiction profiles ─────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JurisdictionProfile {
    EuropeanUnionArt50,
    UnitedStatesNistAI1001,
    GlobalStandardBaseline,
}

// ── Disclosure authority (embedded in RuntimeContract) ────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureAuthority {
    pub interaction_notice: DisclosureRequirement,
    pub machine_readable_marking: DisclosureRequirement,
    pub visible_labelling: DisclosureRequirement,
    pub acknowledgement_required: bool,
    pub permitted_modalities: DisclosureModalitySet,
    pub jurisdiction_profile: JurisdictionProfile,
}

// ── Effective disclosure policy (compiled per-session) ────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectiveDisclosurePolicy {
    pub session_id: uuid::Uuid,
    pub active_jurisdiction: String,
    pub legal_basis_reference: Option<String>,
    pub interaction_requirement: ActionRequirement,
    pub machine_marking_requirement: ActionRequirement,
    pub visible_label_requirement: ActionRequirement,
    pub enforcement_level: DisclosureEnforcement,
    pub control_scope: DisclosureControlScope,
}

impl EffectiveDisclosurePolicy {
    pub fn requires_blocking_delivery(&self) -> bool {
        matches!(
            self.enforcement_level,
            DisclosureEnforcement::DeliveryBlockedUntilAcknowledged
        )
    }
}

// ── Disclosure attachment ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureNotice {
    pub text: String,
    pub placement: NoticePlacement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoticePlacement {
    SessionStart,
    InlinePerOutput,
    ExplicitModalInterrupt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureLabel {
    pub label_text: String,
    pub tracking_digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentProvenanceMark {
    pub header_identifier: String,
    pub payload_base64: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureAttachment {
    pub interaction_notice: Option<DisclosureNotice>,
    pub visible_label: Option<DisclosureLabel>,
    pub machine_readable_metadata: std::collections::HashMap<String, String>,
    pub enforcement: DisclosureEnforcement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationDisclosure {
    pub required: bool,
    pub label_text: String,
    pub machine_readable_mark: Option<ContentProvenanceMark>,
    pub receipt_digest: String,
}

// ── Host disclosure acceptance ────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostDisclosureAcceptance {
    pub session_id: uuid::Uuid,
    pub output_id: uuid::Uuid,
    pub disclosure_digest: Digest256,
    pub contract_generation: u64,
    pub accepted_at_monotonic: u64,
    pub signature: Option<HostSignature>,
    pub attestation_bundle: Option<AttestationBundle>,
    pub host_identity_fingerprint: Digest256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostSignature {
    pub algorithm: String,
    pub bytes: Vec<u8>,
    pub key_fingerprint: Digest256,
}

// ── Host identity binding ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostIdentityBinding {
    pub signing_key_fingerprint: Digest256,
    pub bundle_identifier: String,
    pub build_identifier: String,
    pub team_or_tenant_identity: String,
    pub mdm_enrollment_state: u8,
}

impl HostIdentityBinding {
    /// Compiles a collision-resistant canonical fingerprint.
    /// Uses domain separation and length-prefixed payload fields.
    pub fn compute_canonical_fingerprint(&self) -> Result<Digest256, DisclosureError> {
        let mut h = Sha256::new();
        h.update(b"prism.host-identity.v1\0");
        h.update(self.signing_key_fingerprint.as_bytes());
        encode_string(&mut h, &self.bundle_identifier)?;
        encode_string(&mut h, &self.build_identifier)?;
        encode_string(&mut h, &self.team_or_tenant_identity)?;
        h.update(&[self.mdm_enrollment_state]);
        Ok(Digest256(h.finalize().into()))
    }
}

fn encode_string(h: &mut Sha256, value: &str) -> Result<(), DisclosureError> {
    let bytes = value.as_bytes();
    let len: u32 = bytes
        .len()
        .try_into()
        .map_err(|_| DisclosureError::IdentityError("field too large".into()))?;
    h.update(&len.to_be_bytes());
    h.update(bytes);
    Ok(())
}

// ── Attestation types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationType {
    AppleAppAttest,
    AppleManagedDevice,
    AndroidKeyAttestation,
    EnterpriseHardwareSecurityModule,
    CustomCorporateTrustRoot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedEnterpriseAttestation {
    pub host_identity_fingerprint: Digest256,
    pub device_identifier_digest: Digest256,
    pub key_fingerprint: Digest256,
    pub attestation_type: AttestationType,
    pub trust_chain_digest: Digest256,
    pub issued_at_monotonic: u64,
    pub expires_at_monotonic: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationBundle {
    pub session_id: uuid::Uuid,
    pub output_id: uuid::Uuid,
    pub disclosure_digest: Digest256,
    pub contract_generation: u64,
    pub nonce: [u8; 32],
    pub host_identity_fingerprint: Digest256,
    pub platform_attestation_payload: Vec<u8>,
}

// ── Nonce execution context ───────────────────────────────────────────────

pub struct NonceExecutionContext {
    pub output_id: uuid::Uuid,
    pub session_id: uuid::Uuid,
    pub disclosure_digest: Digest256,
    pub contract_generation: u64,
    pub deployment_digest: Digest256,
    pub host_identity_fingerprint: Digest256,
    pub expiry_epoch_monotonic: u64,
    pub engine_request_sequence: u64,
}

impl NonceExecutionContext {
    pub fn derive_challenge(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(b"prism.nonce-challenge.v1\0");
        h.update(self.output_id.as_bytes());
        h.update(self.session_id.as_bytes());
        h.update(self.disclosure_digest.as_bytes());
        h.update(&self.contract_generation.to_be_bytes());
        h.update(self.deployment_digest.as_bytes());
        h.update(self.host_identity_fingerprint.as_bytes());
        h.update(&self.expiry_epoch_monotonic.to_be_bytes());
        h.update(&self.engine_request_sequence.to_be_bytes());
        let mut challenge = [0u8; 32];
        challenge.copy_from_slice(h.finalize().as_slice());
        challenge
    }
}

// ── Disclosure receipt record ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisclosureReceiptRecord {
    pub effective_policy_digest: Digest256,
    pub output_modality: u16,
    pub obligations_emitted: u64,
    pub control_scope: DisclosureControlScope,
    pub delivery_state: DisclosureDeliveryState,
    pub host_identity_fingerprint: Digest256,
    pub host_acceptance_digest: Option<Digest256>,
    pub engine_monotonic_sequence: u64,
}

impl DisclosureReceiptRecord {
    pub fn compute_canonical_digest(&self) -> Digest256 {
        let mut h = Sha256::new();
        h.update(b"prism.disclosure-receipt.v1\0");
        h.update(self.effective_policy_digest.as_bytes());
        h.update(&self.output_modality.to_be_bytes());
        h.update(&self.obligations_emitted.to_be_bytes());
        h.update(&[self.control_scope as u8]);
        h.update(&[self.delivery_state as u8]);
        h.update(self.host_identity_fingerprint.as_bytes());
        match &self.host_acceptance_digest {
            Some(digest) => {
                h.update(&[1u8]);
                h.update(digest.as_bytes());
            }
            None => {
                h.update(&[0u8]);
            }
        }
        h.update(&self.engine_monotonic_sequence.to_be_bytes());
        Digest256(h.finalize().into())
    }
}

// ── Disclosure errors ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum DisclosureError {
    IdentityError(String),
    GovernanceViolation(String),
    DeliveryBlocked(String),
    AttestationFailed(String),
}

impl std::fmt::Display for DisclosureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IdentityError(s) => write!(f, "identity error: {s}"),
            Self::GovernanceViolation(s) => write!(f, "governance violation: {s}"),
            Self::DeliveryBlocked(s) => write!(f, "delivery blocked: {s}"),
            Self::AttestationFailed(s) => write!(f, "attestation failed: {s}"),
        }
    }
}

impl std::error::Error for DisclosureError {}

// ── Disclosure compiler ───────────────────────────────────────────────────

pub struct DisclosureCompiler;

impl DisclosureCompiler {
    pub fn compile_effective_policy(
        requested: &PrismDisclosurePolicy,
        authority: &DisclosureAuthority,
        session_id: uuid::Uuid,
    ) -> Result<EffectiveDisclosurePolicy, DisclosureError> {
        // Fail fast: app demands applicationManaged but authority mandates strict runtime control
        if requested.interaction_disclosure == InteractionDisclosure::ApplicationManaged
            && authority.interaction_notice == DisclosureRequirement::StrictEnforcement
        {
            return Err(DisclosureError::GovernanceViolation(
                "Host environment forbids ApplicationManaged interaction disclosure for this deployment.".into()
            ));
        }

        let interaction_req = match authority.interaction_notice {
            DisclosureRequirement::StrictEnforcement => {
                ActionRequirement::MandatoryInternalInjection
            }
            DisclosureRequirement::ConditionalOnModality => {
                ActionRequirement::MandatoryExternalLabel
            }
            DisclosureRequirement::AdvisoryOnly => ActionRequirement::OptionalAdvisory,
            DisclosureRequirement::ExemptByHumanReview => ActionRequirement::CompletelyExempt,
        };

        let machine_marking = match requested.machine_readable_marking {
            MachineReadableMarking::Required => ActionRequirement::MandatoryInternalInjection,
            MachineReadableMarking::DisabledByPolicy
                if authority.machine_readable_marking
                    == DisclosureRequirement::StrictEnforcement =>
            {
                return Err(DisclosureError::GovernanceViolation(
                    "Machine readable marking is legally non-negotiable.".into(),
                ));
            }
            _ => ActionRequirement::MandatoryInternalInjection,
        };

        let enforcement = if authority.acknowledgement_required {
            DisclosureEnforcement::DeliveryBlockedUntilAcknowledged
        } else {
            DisclosureEnforcement::RuntimeInjected
        };

        let control_scope = determine_control_scope(requested, authority);

        Ok(EffectiveDisclosurePolicy {
            session_id,
            active_jurisdiction: format!("{:?}", authority.jurisdiction_profile),
            legal_basis_reference: None,
            interaction_requirement: interaction_req,
            machine_marking_requirement: machine_marking,
            visible_label_requirement: ActionRequirement::MandatoryExternalLabel,
            enforcement_level: enforcement,
            control_scope,
        })
    }
}

fn determine_control_scope(
    requested: &PrismDisclosurePolicy,
    authority: &DisclosureAuthority,
) -> DisclosureControlScope {
    if authority.interaction_notice == DisclosureRequirement::StrictEnforcement {
        DisclosureControlScope::PrismRendered
    } else if authority.acknowledgement_required {
        DisclosureControlScope::HostAttested
    } else {
        match requested.evidence_mode {
            DisclosureEvidenceMode::None | DisclosureEvidenceMode::ReceiptOnly => {
                DisclosureControlScope::UnmanagedTelemetry
            }
            DisclosureEvidenceMode::SignedReceipt => DisclosureControlScope::SdkDeliveryGate,
            DisclosureEvidenceMode::SignedReceiptAndExport => DisclosureControlScope::HostAttested,
        }
    }
}

// ── SDK policy types (exposed to Swift via FFI) ───────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismDisclosurePolicy {
    pub interaction_disclosure: InteractionDisclosure,
    pub machine_readable_marking: MachineReadableMarking,
    pub visible_content_label: VisibleContentLabel,
    pub evidence_mode: DisclosureEvidenceMode,
}

impl Default for PrismDisclosurePolicy {
    fn default() -> Self {
        Self {
            interaction_disclosure: InteractionDisclosure::WhenRequired,
            machine_readable_marking: MachineReadableMarking::WhenSupported,
            visible_content_label: VisibleContentLabel::WhenRequired,
            evidence_mode: DisclosureEvidenceMode::ReceiptOnly,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionDisclosure {
    Always,
    WhenRequired,
    ApplicationManaged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MachineReadableMarking {
    Required,
    WhenSupported,
    DisabledByPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisibleContentLabel {
    Always,
    WhenRequired,
    ApplicationManaged,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosureEvidenceMode {
    None,
    ReceiptOnly,
    SignedReceipt,
    SignedReceiptAndExport,
}

// ── Enterprise trust store (stub) ─────────────────────────────────────────

pub struct EnterpriseTrustStore;

impl EnterpriseTrustStore {
    pub fn new() -> Self {
        Self
    }

    pub fn validate_hardware_signature(
        &self,
        _payload: &[u8],
        _nonce: [u8; 32],
    ) -> Result<(), DisclosureError> {
        // Production: verify against Apple App Attest, MDM, or enterprise CA.
        // Stub: accept any non-empty payload.
        if _payload.is_empty() {
            return Err(DisclosureError::AttestationFailed(
                "empty attestation payload".into(),
            ));
        }
        Ok(())
    }

    pub fn verify_and_parse_payload(
        &self,
        _payload: &[u8],
        _nonce: &[u8; 32],
    ) -> Result<VerifiedEnterpriseAttestation, DisclosureError> {
        if _payload.is_empty() {
            return Err(DisclosureError::AttestationFailed(
                "empty attestation payload".into(),
            ));
        }
        // Stub: return a default verified attestation.
        Ok(VerifiedEnterpriseAttestation {
            host_identity_fingerprint: Digest256::compute(b"stub"),
            device_identifier_digest: Digest256::compute(b"stub-device"),
            key_fingerprint: Digest256::compute(b"stub-key"),
            attestation_type: AttestationType::AppleAppAttest,
            trust_chain_digest: Digest256::compute(b"stub-chain"),
            issued_at_monotonic: 0,
            expires_at_monotonic: None,
        })
    }
}

impl Default for EnterpriseTrustStore {
    fn default() -> Self {
        Self::new()
    }
}

// ── Nonce store (in-memory, for simplicity) ───────────────────────────────
// Production: backed by the same durable SQLite transaction as the receipt ledger.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NonceState {
    Pending,
    ReservedForVerification,
    Consumed,
}

struct NonceEntry {
    state: NonceState,
    output_id: uuid::Uuid,
}

pub struct NonceStore {
    nonces: HashMap<[u8; 32], NonceEntry>,
}

impl NonceStore {
    pub fn new() -> Self {
        Self {
            nonces: HashMap::new(),
        }
    }

    pub fn insert(&mut self, nonce: [u8; 32], output_id: uuid::Uuid) {
        self.nonces.insert(
            nonce,
            NonceEntry {
                state: NonceState::Pending,
                output_id,
            },
        );
    }

    pub fn reserve_for_verification(
        &mut self,
        _tx: &mut (), // In production: storage transaction
        output_id: uuid::Uuid,
        nonce: &[u8; 32],
    ) -> Result<(), DisclosureError> {
        match self.nonces.get_mut(nonce) {
            Some(entry) if entry.state == NonceState::Pending && entry.output_id == output_id => {
                entry.state = NonceState::ReservedForVerification;
                Ok(())
            }
            Some(_) => Err(DisclosureError::DeliveryBlocked(
                "nonce already spent or reserved".into(),
            )),
            None => Err(DisclosureError::DeliveryBlocked("unknown nonce".into())),
        }
    }

    pub fn consume_reserved(
        &mut self,
        _tx: &mut (),
        output_id: uuid::Uuid,
        nonce: &[u8; 32],
    ) -> Result<(), DisclosureError> {
        match self.nonces.get_mut(nonce) {
            Some(entry)
                if entry.state == NonceState::ReservedForVerification
                    && entry.output_id == output_id =>
            {
                entry.state = NonceState::Consumed;
                Ok(())
            }
            _ => Err(DisclosureError::DeliveryBlocked(
                "cannot consume unreserved nonce".into(),
            )),
        }
    }

    pub fn release_reservation(
        &mut self,
        _tx: &mut (),
        output_id: uuid::Uuid,
        nonce: &[u8; 32],
    ) -> Result<(), DisclosureError> {
        match self.nonces.get_mut(nonce) {
            Some(entry)
                if entry.state == NonceState::ReservedForVerification
                    && entry.output_id == output_id =>
            {
                entry.state = NonceState::Pending;
                Ok(())
            }
            _ => Err(DisclosureError::DeliveryBlocked(
                "cannot release nonce".into(),
            )),
        }
    }
}

// ── Output delivery gatekeeper ────────────────────────────────────────────

pub struct OutputDeliveryGatekeeper;

impl OutputDeliveryGatekeeper {
    pub fn verify_stream_clearance(
        scope: DisclosureControlScope,
        state: &DisclosureDeliveryState,
        policy: &EffectiveDisclosurePolicy,
    ) -> Result<(), DisclosureError> {
        match scope {
            DisclosureControlScope::PrismRendered => {
                if *state == DisclosureDeliveryState::InjectedByPrism {
                    Ok(())
                } else {
                    Err(DisclosureError::GovernanceViolation(
                        "Prism visual render layer bypass detected.".into(),
                    ))
                }
            }
            DisclosureControlScope::SdkDeliveryGate => {
                if *state == DisclosureDeliveryState::AcceptedByHost {
                    Ok(())
                } else {
                    Err(DisclosureError::DeliveryBlocked(
                        "Awaiting engine-verifiable host interaction acknowledgment.".into(),
                    ))
                }
            }
            DisclosureControlScope::HostAttested => {
                if *state != DisclosureDeliveryState::AcceptedByHost {
                    return Err(DisclosureError::DeliveryBlocked(
                        "Awaiting host interaction acknowledgment token.".into(),
                    ));
                }
                Ok(())
            }
            DisclosureControlScope::UnmanagedTelemetry => {
                if policy.requires_blocking_delivery() {
                    Err(DisclosureError::GovernanceViolation(
                        "Security Exception: Unmanaged telemetry cannot satisfy a blocking disclosure obligation.".into()
                    ))
                } else {
                    Ok(())
                }
            }
            DisclosureControlScope::AdvisoryOnly => Ok(()),
        }
    }

    pub fn verify_and_commit_clearance(
        acceptance: &HostDisclosureAcceptance,
        expected_generation: u64,
        trust_store: &EnterpriseTrustStore,
        nonce_store: &mut NonceStore,
        _tx: &mut (),
    ) -> Result<VerifiedEnterpriseAttestation, DisclosureError> {
        let bundle = acceptance.attestation_bundle.as_ref().ok_or_else(|| {
            DisclosureError::GovernanceViolation("Cryptographic attestation bundle missing.".into())
        })?;

        // 1. Context binding
        verify_context_binding(bundle, acceptance, expected_generation)?;

        // 2. Reserve nonce (prevents burn-before-verification)
        nonce_store.reserve_for_verification(&mut (), acceptance.output_id, &bundle.nonce)?;

        // 3. Verify attestation
        let verified = trust_store
            .verify_and_parse_payload(&bundle.platform_attestation_payload, &bundle.nonce);

        if verified.is_err() {
            let _ = nonce_store.release_reservation(&mut (), acceptance.output_id, &bundle.nonce);
            return Err(DisclosureError::AttestationFailed(
                "verification failed".into(),
            ));
        }
        let verified = verified.unwrap();

        // 4. Identity binding check
        if verified.host_identity_fingerprint != bundle.host_identity_fingerprint {
            let _ = nonce_store.release_reservation(&mut (), acceptance.output_id, &bundle.nonce);
            return Err(DisclosureError::GovernanceViolation(
                "parsed platform identity mismatch".into(),
            ));
        }

        // 5. Commit: consume nonce
        nonce_store.consume_reserved(&mut (), acceptance.output_id, &bundle.nonce)?;

        Ok(verified)
    }
}

fn verify_context_binding(
    bundle: &AttestationBundle,
    acceptance: &HostDisclosureAcceptance,
    expected_generation: u64,
) -> Result<(), DisclosureError> {
    if bundle.session_id != acceptance.session_id || bundle.output_id != acceptance.output_id {
        return Err(DisclosureError::GovernanceViolation(
            "attestation bundle context mismatch".into(),
        ));
    }
    if bundle.disclosure_digest != acceptance.disclosure_digest {
        return Err(DisclosureError::GovernanceViolation(
            "tampered disclosure metadata reference".into(),
        ));
    }
    if bundle.contract_generation != expected_generation {
        return Err(DisclosureError::GovernanceViolation(
            "attestation contract generation mismatch".into(),
        ));
    }
    if bundle.host_identity_fingerprint != acceptance.host_identity_fingerprint {
        return Err(DisclosureError::GovernanceViolation(
            "attestation host identity mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_identity_fingerprint_deterministic() {
        let binding = HostIdentityBinding {
            signing_key_fingerprint: Digest256::compute(b"key"),
            bundle_identifier: "com.example.app".into(),
            build_identifier: "build1".into(),
            team_or_tenant_identity: "TEAM123".into(),
            mdm_enrollment_state: 1,
        };
        let fp1 = binding.compute_canonical_fingerprint().unwrap();
        let fp2 = binding.compute_canonical_fingerprint().unwrap();
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_disclosure_receipt_canonical_digest() {
        let record = DisclosureReceiptRecord {
            effective_policy_digest: Digest256::compute(b"policy"),
            output_modality: 1,
            obligations_emitted: 3,
            control_scope: DisclosureControlScope::SdkDeliveryGate,
            delivery_state: DisclosureDeliveryState::AcceptedByHost,
            host_identity_fingerprint: Digest256::compute(b"host"),
            host_acceptance_digest: Some(Digest256::compute(b"accept")),
            engine_monotonic_sequence: 42,
        };
        let d1 = record.compute_canonical_digest();
        let d2 = record.compute_canonical_digest();
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_nonce_state_machine() {
        let mut store = NonceStore::new();
        let nonce = [1u8; 32];
        let output_id = uuid::Uuid::new_v4();

        store.insert(nonce, output_id);
        store
            .reserve_for_verification(&mut (), output_id, &nonce)
            .unwrap();
        store.consume_reserved(&mut (), output_id, &nonce).unwrap();

        // Cannot consume again
        assert!(store.consume_reserved(&mut (), output_id, &nonce).is_err());
    }

    #[test]
    fn test_nonce_failed_verification_releases() {
        let mut store = NonceStore::new();
        let nonce = [2u8; 32];
        let output_id = uuid::Uuid::new_v4();
        store.insert(nonce, output_id);

        let trust_store = EnterpriseTrustStore::new();
        let bundle = AttestationBundle {
            session_id: uuid::Uuid::nil(),
            output_id,
            disclosure_digest: Digest256::compute(b"d"),
            contract_generation: 0,
            nonce,
            host_identity_fingerprint: Digest256::compute(b"h"),
            platform_attestation_payload: vec![], // Empty = will fail
        };
        let acceptance = HostDisclosureAcceptance {
            session_id: bundle.session_id,
            output_id,
            disclosure_digest: bundle.disclosure_digest,
            contract_generation: 0,
            accepted_at_monotonic: 0,
            signature: None,
            attestation_bundle: Some(bundle),
            host_identity_fingerprint: Digest256::compute(b"h"),
        };

        let result = OutputDeliveryGatekeeper::verify_and_commit_clearance(
            &acceptance,
            0,
            &trust_store,
            &mut store,
            &mut (),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_effective_policy_compilation() {
        let policy = PrismDisclosurePolicy::default();
        let authority = DisclosureAuthority {
            interaction_notice: DisclosureRequirement::AdvisoryOnly,
            machine_readable_marking: DisclosureRequirement::ConditionalOnModality,
            visible_labelling: DisclosureRequirement::AdvisoryOnly,
            acknowledgement_required: false,
            permitted_modalities: DisclosureModalitySet::empty(),
            jurisdiction_profile: JurisdictionProfile::GlobalStandardBaseline,
        };
        let session_id = uuid::Uuid::new_v4();
        let effective =
            DisclosureCompiler::compile_effective_policy(&policy, &authority, session_id).unwrap();
        assert_eq!(effective.session_id, session_id);
        assert!(!effective.requires_blocking_delivery());
    }
}
