//! PRISM-MODEL-TO-CIMAGE-0002 — Campaign receipt types for installation,
//! epoch execution, and session lifecycle.
//!
//! These types capture the observable outcome of each phase of the
//! Prism runtime: artifact installation, per-epoch execution, and
//! end-to-end session.  [`DiagnosticsBundle`] aggregates all receipts
//! into a single evidence package for offline analysis or automated
//! qualification.

use serde::{Deserialize, Serialize};

// ── Installation receipts ────────────────────────────────────────────────

/// Outcome of loading one compiled artifact (Core ML .mlmodelc, Metal .metallib,
/// or CPU fallback binary) during installation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactLoadResult {
    /// Path to the artifact on disk or within the compute image bundle.
    pub artifact_path: String,
    /// Whether the artifact was loaded without error.
    pub loaded: bool,
    /// Human-readable error message when `loaded` is `false`.
    pub error: Option<String>,
}

/// IOSurface allocation attestation for one arena slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSurfaceAttestationEntry {
    /// Index of the arena slot this IOSurface was allocated for.
    pub slot_index: u32,
    /// Mach IOSurface ID returned by the kernel.
    pub iosurface_id: u64,
    /// OSType pixel format (e.g. `bgra8Unorm` → `1111970369`).
    pub pixel_format: u32,
    /// Width of the IOSurface in pixels.
    pub width: u64,
    /// Height of the IOSurface in pixels.
    pub height: u64,
    /// Whether the surface passed post-allocation validation.
    pub attested: bool,
}

/// Metal resource binding record for one arena slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalBindingEntry {
    /// Index of the arena slot this binding corresponds to.
    pub slot_index: u32,
    /// Optional MTLTexture label or resource ID for diagnostics.
    pub texture_id: Option<String>,
    /// Whether the Metal resource was successfully bound.
    pub bound: bool,
}

/// Outcome of one Core ML executable warmup invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResultEntry {
    /// Name of the Core ML executable that was warmed up.
    pub executable_name: String,
    /// Whether the warmup prediction completed without error.
    pub success: bool,
    /// Whether the executable was loaded into the model store.
    pub load_success: bool,
    /// Whether the warmup invocation produced valid output.
    pub output_present: bool,
    /// Observed latency of the warmup invocation in microseconds.
    pub latency_us: u64,
}

/// Overall status of a compute image installation attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstallationStatus {
    /// Every artifact was installed and validated.
    Succeeded,
    /// Installation failed with a description of the error.
    Failed(String),
    /// Installation was rolled back after a partial or failed install.
    RolledBack(String),
}

/// Complete receipt for a compute image installation attempt.
///
/// Aggregates artifact load outcomes, IOSurface attestations, Metal
/// resource bindings, and Core ML warmup results into a single record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismInstallationReceipt {
    /// Digest of the compute image that was being installed.
    pub image_digest: String,
    /// Digest of the capability signature used for this installation.
    pub capability_signature_digest: String,
    /// Outcomes of loading each compiled artifact.
    pub artifact_load_results: Vec<ArtifactLoadResult>,
    /// IOSurface allocation attestations for each arena slot.
    pub iosurface_attestations: Vec<IOSurfaceAttestationEntry>,
    /// Metal resource binding records for each arena slot.
    pub metal_resource_bindings: Vec<MetalBindingEntry>,
    /// Core ML executable warmup results.
    pub coreml_warmup_results: Vec<WarmupResultEntry>,
    /// Overall installation status.
    pub installation_status: InstallationStatus,
    /// ISO-8601 timestamp when the installation completed.
    pub installed_at: String,
}

// ── Epoch execution receipts ─────────────────────────────────────────────

/// Per-epoch resource utilisation counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochResourceCounters {
    /// Number of IOSurface allocations performed during the epoch.
    pub iosurface_allocations: u64,
    /// Number of Metal texture creations performed during the epoch.
    pub metal_texture_creations: u64,
    /// Number of Metal command queue creations performed during the epoch.
    pub command_queue_creations: u64,
    /// Number of Metal compute pipeline state creations.
    pub command_pipeline_creations: u64,
    /// Number of Core ML model loads performed during the epoch.
    pub coreml_model_loads: u64,
    /// Number of CPU readbacks performed during the epoch.
    pub cpu_readbacks: u64,
}

/// Receipt for one epoch of execution within a session.
///
/// Tracks which route was selected, whether fallback was triggered,
/// the timing of Core ML and Metal phases, and resource counters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismEpochReceipt {
    /// Identifier of the session this epoch belongs to.
    pub session_id: String,
    /// Epoch number within the session (0-based).
    pub epoch: u64,
    /// Token emitted by this epoch, if applicable (e.g. for autoregressive models).
    pub emitted_token: Option<u32>,
    /// Which route origin was selected for this epoch.
    pub route_origin: String,
    /// Whether a fallback path was taken during this epoch.
    pub fallback_used: bool,
    /// Whether the Core ML prediction phase completed.
    pub coreml_prediction_completed: bool,
    /// Whether the Metal command buffer phase completed.
    pub metal_command_buffer_completed: bool,
    /// Wall-clock time for this epoch in nanoseconds.
    pub wall_time_ns: u64,
    /// Resource utilisation counters accumulated during this epoch.
    pub resource_counters: EpochResourceCounters,
}

// ── Session receipts ─────────────────────────────────────────────────────

/// Aggregate receipt for an entire inference session.
///
/// Summarises the start/end times, total generated tokens, fallback
/// count, and links back to the compute image and its epoch-level
/// receipts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrismSessionReceipt {
    /// Identifier of the session.
    pub session_id: String,
    /// Digest of the compute image used for the session.
    pub image_digest: String,
    /// ISO-8601 timestamp when the session started.
    pub start_time: String,
    /// ISO-8601 timestamp when the session ended.
    pub end_time: String,
    /// Total number of tokens generated during the session.
    pub generated_tokens: u32,
    /// Total number of epochs where fallback was triggered.
    pub fallback_count: u32,
    /// Terminal status string (e.g. "completed", "cancelled", "error").
    pub terminal_status: String,
}

// ── Diagnostics ──────────────────────────────────────────────────────────

/// Platform metadata captured at diagnostics time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformMetadata {
    /// Operating system version string (e.g. "macOS 15.2").
    pub os_version: String,
    /// System-on-chip identifier (e.g. "Apple M1", "Apple M3 Max").
    pub soc: String,
    /// Core ML framework version, if available.
    pub coreml_version: Option<String>,
    /// Metal feature-set string (e.g. "macOS_GPUFamily2_v1"), if available.
    pub metal_feature_set: Option<String>,
    /// Total system memory in bytes.
    pub memory_bytes: u64,
}

/// Aggregated diagnostics bundle combining receipts and metadata.
///
/// Used for offline analysis, automated qualification, and error
/// reporting across all phases of a Prism runtime session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticsBundle {
    /// Digest of the compute image used.
    pub image_digest: String,
    /// Digest of the capability signature used.
    pub capability_signature_digest: String,
    /// Digests of every compiled artifact loaded.
    pub artifact_digests: Vec<String>,
    /// Installation receipt, if installation has occurred.
    pub installation_receipt: Option<PrismInstallationReceipt>,
    /// Epoch-level execution receipts.
    pub epoch_receipts: Vec<PrismEpochReceipt>,
    /// Chronological history of fallback reasons.
    pub fallback_history: Vec<String>,
    /// Aggregated resource counters across all epochs.
    pub resource_counters: EpochResourceCounters,
    /// Platform metadata at diagnostics time.
    pub platform_metadata: PlatformMetadata,
    /// Chronological error chain for diagnostics.
    pub error_chain: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that every variant of InstallationStatus roundtrips through
    /// JSON serialization without data loss.
    #[test]
    fn installation_status_serde_roundtrip() {
        for status in [
            InstallationStatus::Succeeded,
            InstallationStatus::Failed("disk full".into()),
            InstallationStatus::RolledBack("checksum mismatch".into()),
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: InstallationStatus = serde_json::from_str(&json).unwrap();
            match (&status, &back) {
                (InstallationStatus::Succeeded, InstallationStatus::Succeeded) => {}
                (InstallationStatus::Failed(a), InstallationStatus::Failed(b)) => {
                    assert_eq!(a, b);
                }
                (InstallationStatus::RolledBack(a), InstallationStatus::RolledBack(b)) => {
                    assert_eq!(a, b);
                }
                _ => panic!("variant mismatch: {status:?} vs {back:?}"),
            }
        }
    }

    /// Verify that a full PrismInstallationReceipt roundtrips through JSON.
    #[test]
    fn installation_receipt_serde_roundtrip() {
        let receipt = PrismInstallationReceipt {
            image_digest: "abc123".into(),
            capability_signature_digest: "def456".into(),
            artifact_load_results: vec![ArtifactLoadResult {
                artifact_path: "model.mlmodelc".into(),
                loaded: true,
                error: None,
            }],
            iosurface_attestations: vec![IOSurfaceAttestationEntry {
                slot_index: 0,
                iosurface_id: 42,
                pixel_format: 1111970369,
                width: 4096,
                height: 4096,
                attested: true,
            }],
            metal_resource_bindings: vec![MetalBindingEntry {
                slot_index: 0,
                texture_id: Some("slot0_tex".into()),
                bound: true,
            }],
            coreml_warmup_results: vec![WarmupResultEntry {
                executable_name: "encoder".into(),
                success: true,
                load_success: true,
                output_present: true,
                latency_us: 1500,
            }],
            installation_status: InstallationStatus::Succeeded,
            installed_at: "2026-06-25T12:00:00Z".into(),
        };

        let json = serde_json::to_string(&receipt).unwrap();
        let back: PrismInstallationReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.image_digest, back.image_digest);
        assert_eq!(receipt.artifact_load_results.len(), back.artifact_load_results.len());
    assert!(matches!(back.installation_status, InstallationStatus::Succeeded));
    }

    /// Verify that a full PrismEpochReceipt roundtrips through JSON.
    #[test]
    fn epoch_receipt_serde_roundtrip() {
        let receipt = PrismEpochReceipt {
            session_id: "sess-001".into(),
            epoch: 0,
            emitted_token: Some(42),
            route_origin: "metal".into(),
            fallback_used: false,
            coreml_prediction_completed: true,
            metal_command_buffer_completed: true,
            wall_time_ns: 1_234_567,
            resource_counters: EpochResourceCounters {
                iosurface_allocations: 2,
                metal_texture_creations: 1,
                command_queue_creations: 1,
                command_pipeline_creations: 2,
                coreml_model_loads: 0,
                cpu_readbacks: 1,
            },
        };

        let json = serde_json::to_string(&receipt).unwrap();
        let back: PrismEpochReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.session_id, back.session_id);
        assert_eq!(receipt.epoch, back.epoch);
        assert_eq!(receipt.emitted_token, back.emitted_token);
    }

    /// Verify that a full PrismSessionReceipt roundtrips through JSON.
    #[test]
    fn session_receipt_serde_roundtrip() {
        let receipt = PrismSessionReceipt {
            session_id: "sess-001".into(),
            image_digest: "abc123".into(),
            start_time: "2026-06-25T12:00:00Z".into(),
            end_time: "2026-06-25T12:00:05Z".into(),
            generated_tokens: 128,
            fallback_count: 1,
            terminal_status: "completed".into(),
        };

        let json = serde_json::to_string(&receipt).unwrap();
        let back: PrismSessionReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt.generated_tokens, back.generated_tokens);
        assert_eq!(receipt.terminal_status, back.terminal_status);
    }

    /// Verify that a full DiagnosticsBundle roundtrips through JSON.
    #[test]
    fn diagnostics_bundle_serde_roundtrip() {
        let bundle = DiagnosticsBundle {
            image_digest: "abc123".into(),
            capability_signature_digest: "def456".into(),
            artifact_digests: vec!["hash1".into(), "hash2".into()],
            installation_receipt: None,
            epoch_receipts: vec![],
            fallback_history: vec!["route_metal:fallback_to_cpu".into()],
            resource_counters: EpochResourceCounters {
                iosurface_allocations: 0,
                metal_texture_creations: 0,
                command_queue_creations: 0,
                command_pipeline_creations: 0,
                coreml_model_loads: 0,
                cpu_readbacks: 0,
            },
            platform_metadata: PlatformMetadata {
                os_version: "macOS 15.2".into(),
                soc: "Apple M1".into(),
                coreml_version: Some("9.0".into()),
                metal_feature_set: Some("macOS_GPUFamily2_v1".into()),
                memory_bytes: 8_589_934_592,
            },
            error_chain: vec![],
        };

        let json = serde_json::to_string(&bundle).unwrap();
        let back: DiagnosticsBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(bundle.image_digest, back.image_digest);
        assert_eq!(bundle.platform_metadata.soc, back.platform_metadata.soc);
    }
}
