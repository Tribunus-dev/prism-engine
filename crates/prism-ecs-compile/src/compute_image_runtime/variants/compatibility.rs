//! Variant compatibility — pure data types and pure algorithms.

use serde::{Deserialize, Serialize};

/// Reasons a variant may violate runtime compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityViolation {
    /// Hardware feature is missing.
    MissingHardwareFeature {
        /// Feature name.
        feature: String,
    },
    /// Required capability is missing.
    MissingCapability {
        /// Capability name.
        capability: String,
    },
    /// Hardware contract mismatch.
    HardwareContractMismatch,
}

/// Runtime capability snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeCapabilitySnapshot {
    /// Hardware features available.
    pub hardware_features: Vec<String>,
    /// Capabilities available.
    pub capabilities: Vec<String>,
    /// Hardware family identifier.
    pub hardware_family: String,
}

/// Variant compatibility report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCompatibilityReport {
    /// Variant identifier.
    pub variant_id: String,
    /// Whether the variant is compatible.
    pub compatible: bool,
    /// Violations found.
    pub violations: Vec<CompatibilityViolation>,
}
