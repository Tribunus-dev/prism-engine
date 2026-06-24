//! Apple tri-lane artifact manifest — sealed CImage extension.
//!
//! Contains compile-time metadata for three-lane (ANE/GPU/CPU) heterogeneous
//! execution on Apple Silicon.  The manifest is embedded in the CImage and
//! consumed by the runtime scheduler, admission gate, and IOSurface arena
//! installer.
//!
//! All types are `Serialize + Deserialize` for embedding in the CImage
//! `manifest.json` as a dedicated section alongside the base Manifest.

use serde::{Deserialize, Serialize};

pub use crate::backend::placement::ExecutionLane;
pub use crate::compilation::tri_lane::{ExecutionEpoch, LaneDependency, NumericalPolicy, ShapeClass};

// ── Content digest ───────────────────────────────────────────────────────

/// Content digest (hex-encoded SHA-256 or similar).
pub type Digest = String;

// ── Hardware compatibility ───────────────────────────────────────────────

/// Compatibility binding for hardware targets.
///
/// Describes the minimum SoC, OS, Core ML runtime, and Metal feature set
/// that this tri-lane plan requires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleHardwareCompatibility {
    /// Minimum supported SoC family, e.g. "M1"
    pub min_soc_family: String,
    /// Minimum macOS version, e.g. "14.0"
    pub min_macos_version: String,
    /// Minimum Core ML runtime version, e.g. "7.2.0"
    pub min_coreml_version: String,
    /// Required ANE presence
    pub require_ane: bool,
    /// Required Metal feature set
    pub required_metal_features: Vec<String>,
    /// Supported compute policies
    pub supported_compute_policies: Vec<String>,
    /// Page/alignment constraints in bytes
    pub alignment_bytes: u64,
}

// ── IOSurface slot manifestation ─────────────────────────────────────────

/// IOSurface slot manifestation (compile-time, embedded in manifest).
///
/// Describes the compile-time binding of a tensor to an IOSurface slot
/// within the shared arena.  Consumers use this manifest to install and
/// map IOSurfaces at runtime — no live pointers are stored here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSurfaceSlotManifest {
    pub slot_id: u32,
    pub tensor_id: String,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub dtype: String,
    pub logical_shape: Vec<u32>,
    pub physical_shape: Vec<u32>,
    pub strides_bytes: Vec<u64>,
    pub layout: String,
    pub producer: ExecutionLane,
    pub consumer: ExecutionLane,
    pub reuse_class: String, // "exclusive", "shared_readonly", "ring_reuse"
    pub required_alignment: u64,
}

// ── Shared arena manifest ────────────────────────────────────────────────

/// Shared arena manifest (immutable, no live pointers).
///
/// Describes the total IOSurface arena layout — allocation size, alignment,
/// ring depth, and every slot that participates in tri-lane execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleSharedArenaManifest {
    pub arena_layout_digest: String,
    pub allocation_bytes: u64,
    pub alignment_bytes: u64,
    pub ring_depth: u8,
    pub slots: Vec<IOSurfaceSlotManifest>,
}

// ── Artifact manifests ───────────────────────────────────────────────────

/// Core ML artifact manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMlArtifactManifest {
    pub artifact_id: String,
    pub mlmodelc_name: String,
    pub package_digest: String,
    pub compiled_model_digest: String,
    pub compute_policy: String, // "cpuAndNeuralEngine", etc.
    pub input_slots: Vec<String>,
    pub output_slots: Vec<String>,
}

/// Metal artifact manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalArtifactManifest {
    pub artifact_id: String,
    pub function_name: String,
    pub pipeline_digest: String,
    pub input_slots: Vec<String>,
    pub output_slots: Vec<String>,
}

/// CPU artifact manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuArtifactManifest {
    pub artifact_id: String,
    pub function: String,
    pub description: String,
}

// ── Numerical policy ─────────────────────────────────────────────────────

/// Numerical policy for Apple tri-lane execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleNumericalPolicy {
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub validation_mode: String, // "full", "sampled", "none"
    pub sample_period_epochs: Option<u64>,
    pub failure_action: String, // "warn", "fallback", "abort"
}

// ── Admission manifest ───────────────────────────────────────────────────

/// Admission manifest for Apple tri-lane.
///
/// Documents which model regions were admitted to the ANE lane at compile
/// time and which were rejected (with reasons).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleTriLaneAdmissionManifest {
    pub region_count: u32,
    pub admitted_regions: Vec<String>,
    pub rejected_regions: Vec<String>,
    pub fallback_available: bool,
}

// ── Fallback manifest ────────────────────────────────────────────────────

/// Fallback manifest — describes the lane and artifact to use when the
/// primary ANE lane is unhealthy or unavailable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleFallbackManifest {
    pub replacement_lane: String,
    pub replacement_artifact: String,
    pub input_slots: Vec<u32>,
    pub output_slots: Vec<u32>,
    pub epoch_boundary: u64,
}

// ── Validation ───────────────────────────────────────────────────────────

/// Validate an `AppleTriLaneArtifactManifest` for structural consistency.
///
/// Returns `Ok(())` if the manifest passes all checks, or `Err(reasons)`
/// with every violation collected.
pub fn validate_manifest(manifest: &AppleTriLaneArtifactManifest) -> Result<(), Vec<String>> {
    let mut errors: Vec<String> = Vec::new();

    if manifest.manifest_version == 0 {
        errors.push("manifest_version must be >= 1".to_string());
    }

    if manifest.arena.allocation_bytes == 0 {
        errors.push("arena.allocation_bytes must be > 0".to_string());
    }

    if manifest.arena.slots.is_empty() {
        errors.push("arena.slots must not be empty".to_string());
    }

    if manifest.hardware_compatibility.alignment_bytes == 0 {
        errors.push("hardware_compatibility.alignment_bytes must be > 0".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Check whether the given hardware compatibility descriptor is satisfied
/// by the runtime environment described by the provided metadata.
///
/// Returns `Ok(())` if compatible, or `Err(reason)` on the first
/// incompatibility detected.
pub fn check_hardware_compatibility(
    compatibility: &AppleHardwareCompatibility,
    _soc_family: &str,
    macos_version: &str,
    coreml_version: &str,
    has_ane: bool,
) -> Result<(), String> {
    if compatibility.require_ane && !has_ane {
        return Err("ANE required but not available".to_string());
    }

    // Simple lexical version comparison (semver prefix slice — sufficient
    // for "14.0" >= "13.0" sorting without pulling in a semver crate).
    if macos_version < compatibility.min_macos_version.as_str() {
        return Err(format!(
            "macOS version {} is below minimum {}",
            macos_version, compatibility.min_macos_version
        ));
    }

    if coreml_version < compatibility.min_coreml_version.as_str() {
        return Err(format!(
            "Core ML version {} is below minimum {}",
            coreml_version, compatibility.min_coreml_version
        ));
    }

    Ok(())
}

// ── Top-level manifest ───────────────────────────────────────────────────

/// Top-level Apple tri-lane artifact manifest, embedded in CImage.
///
/// Contains all compile-time metadata for the three-lane execution plan:
/// hardware compatibility, IOSurface arena layout, per-lane artifact
/// bindings, epoch schedule, dependency graph, numerical policy, admission
/// decisions, and fallback topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppleTriLaneArtifactManifest {
    pub manifest_version: u32,
    pub hardware_compatibility: AppleHardwareCompatibility,
    pub plan_digest: String,
    pub arena: AppleSharedArenaManifest,
    pub coreml_artifacts: Vec<CoreMlArtifactManifest>,
    pub metal_artifacts: Vec<MetalArtifactManifest>,
    pub cpu_artifacts: Vec<CpuArtifactManifest>,
    pub epochs: Vec<crate::compilation::tri_lane::ExecutionEpoch>,
    pub dependencies: Vec<crate::compilation::tri_lane::LaneDependency>,
    pub fallback: AppleFallbackManifest,
    pub numerical_policy: AppleNumericalPolicy,
    pub admission: AppleTriLaneAdmissionManifest,
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::placement::ExecutionLane;

    fn minimal_hardware_compatibility() -> AppleHardwareCompatibility {
        AppleHardwareCompatibility {
            min_soc_family: "M1".into(),
            min_macos_version: "14.0".into(),
            min_coreml_version: "7.2.0".into(),
            require_ane: true,
            required_metal_features: vec!["apple_m1".into()],
            supported_compute_policies: vec!["cpuAndNeuralEngine".into()],
            alignment_bytes: 16384,
        }
    }

    fn minimal_arena() -> AppleSharedArenaManifest {
        AppleSharedArenaManifest {
            arena_layout_digest: "abc123".into(),
            allocation_bytes: 1_048_576,
            alignment_bytes: 16384,
            ring_depth: 2,
            slots: vec![IOSurfaceSlotManifest {
                slot_id: 0,
                tensor_id: "input_0".into(),
                byte_offset: 0,
                byte_length: 262_144,
                dtype: "f16".into(),
                logical_shape: vec![1, 64],
                physical_shape: vec![1, 64],
                strides_bytes: vec![128, 2],
                layout: "NHWC".into(),
                producer: ExecutionLane::CoreMlAne,
                consumer: ExecutionLane::MlxGpu,
                reuse_class: "exclusive".into(),
                required_alignment: 16384,
            }],
        }
    }

    fn minimal_manifest() -> AppleTriLaneArtifactManifest {
        AppleTriLaneArtifactManifest {
            manifest_version: 1,
            hardware_compatibility: minimal_hardware_compatibility(),
            plan_digest: "deadbeef".into(),
            arena: minimal_arena(),
            coreml_artifacts: vec![],
            metal_artifacts: vec![],
            cpu_artifacts: vec![],
            epochs: vec![],
            dependencies: vec![],
            fallback: AppleFallbackManifest {
                replacement_lane: "MlxGpu".into(),
                replacement_artifact: "fallback_projection".into(),
                input_slots: vec![0],
                output_slots: vec![1],
                epoch_boundary: 0,
            },
            numerical_policy: AppleNumericalPolicy {
                absolute_tolerance: 1e-3,
                relative_tolerance: 1e-2,
                validation_mode: "sampled".into(),
                sample_period_epochs: Some(100),
                failure_action: "fallback".into(),
            },
            admission: AppleTriLaneAdmissionManifest {
                region_count: 4,
                admitted_regions: vec!["attention_proj".into(), "ffn_gate".into()],
                rejected_regions: vec!["rms_norm".into()],
                fallback_available: true,
            },
        }
    }

    // ── test_manifest_roundtrip ─────────────────────────────────────────

    #[test]
    fn test_manifest_roundtrip() {
        let manifest = minimal_manifest();

        let json = serde_json::to_string_pretty(&manifest)
            .expect("serialize manifest");
        let deserialized: AppleTriLaneArtifactManifest = serde_json::from_str(&json)
            .expect("deserialize manifest");

        // Verify top-level fields survived the round trip.
        assert_eq!(deserialized.manifest_version, 1);
        assert_eq!(deserialized.plan_digest, "deadbeef");
        assert_eq!(deserialized.admission.region_count, 4);

        // Verify hardware compatibility.
        assert_eq!(deserialized.hardware_compatibility.min_soc_family, "M1");
        assert!(deserialized.hardware_compatibility.require_ane);
        assert_eq!(deserialized.hardware_compatibility.alignment_bytes, 16384);

        // Verify arena.
        assert_eq!(deserialized.arena.allocation_bytes, 1_048_576);
        assert_eq!(deserialized.arena.ring_depth, 2);
        assert_eq!(deserialized.arena.slots.len(), 1);
        assert_eq!(deserialized.arena.slots[0].tensor_id, "input_0");
        assert_eq!(
            deserialized.arena.slots[0].producer,
            ExecutionLane::CoreMlAne
        );
        assert_eq!(
            deserialized.arena.slots[0].consumer,
            ExecutionLane::MlxGpu
        );

        // Verify fallback.
        assert_eq!(deserialized.fallback.replacement_lane, "MlxGpu");
        assert_eq!(deserialized.fallback.replacement_artifact, "fallback_projection");

        // Verify numerical policy.
        assert_eq!(deserialized.numerical_policy.validation_mode, "sampled");
        assert_eq!(
            deserialized.numerical_policy.sample_period_epochs,
            Some(100)
        );
        assert_eq!(deserialized.numerical_policy.failure_action, "fallback");

        // Verify admission.
        assert_eq!(deserialized.admission.admitted_regions.len(), 2);
        assert!(deserialized.admission.fallback_available);
    }

    // ── test_manifest_rejects_zero_arena_bytes ──────────────────────────

    #[test]
    fn test_manifest_rejects_zero_arena_bytes() {
        let mut manifest = minimal_manifest();
        manifest.arena.allocation_bytes = 0;

        let result = validate_manifest(&manifest);
        assert!(result.is_err(), "validate should reject zero arena bytes");

        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("allocation_bytes")),
            "error message should mention allocation_bytes: {:?}",
            errors
        );
    }

    // ── test_hardware_compatibility_check ───────────────────────────────

    #[test]
    fn test_hardware_compatibility_check() {
        let compat = minimal_hardware_compatibility();

        // Compatible: M1 + macOS 14.5 + Core ML 7.3.0 + ANE present.
        let ok = check_hardware_compatibility(&compat, "M1", "14.5", "7.3.0", true);
        assert!(ok.is_ok(), "expected compatible: {:?}", ok);

        // Missing ANE when required.
        let err = check_hardware_compatibility(&compat, "M1", "14.5", "7.3.0", false);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("ANE required"));

        // macOS version too old.
        let err = check_hardware_compatibility(&compat, "M1", "13.6", "7.3.0", true);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("macOS"));

        // Core ML version too old.
        let err = check_hardware_compatibility(&compat, "M1", "14.5", "7.0.0", true);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("Core ML"));
    }
}
