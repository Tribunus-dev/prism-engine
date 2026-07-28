//! Bridge provider trait and capability types for Level 3 routing.
//!
//! The core scheduler does not care which provider is chosen; it consumes the
//! capability declaration and its measured cost. Level 3 adds a BridgeProvider
//! trait. One implementation is explicit materialization (which remains valid).
//! Others may be verified shared routes on specific OS/API pairings.
//!
//! Every bridge that fails capability probing, runtime execution, lifetime
//! validation, or memory admission must atomically revert to the explicit
//! Level 2 materialization route.

use serde::{Deserialize, Serialize};

use super::phase_types::TensorDescriptor;
pub use super::receipt::BridgeReceipt;

// ── Bridge capability ───────────────────────────────────────────────────────

/// Result of probing what a bridge provider can do for a given device/OS pair.
///
/// The probe must be cached by hardware identifier, OS version, Core ML
/// version, model representation, tensor dtype, and physical layout. Any
/// mismatch invalidates the cached result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeCapability {
    pub supports_borrowing: bool,
    pub supports_aliasing: bool,
    pub supports_exporting: bool,
    pub supports_importing: bool,
    pub supports_materialization: bool,
    pub supports_cpu_visible_staging: bool,
    pub layout_constraints: Vec<String>,
    pub alignment_constraints: Vec<usize>,
    pub allowed_element_types: Vec<String>,
    pub max_tensor_bytes: u64,
    pub synchronization_requirements: Vec<String>,
    /// Whether this route is stable enough to be enabled outside research mode.
    pub stable_for_production: bool,
}

// ── Bridge plan ─────────────────────────────────────────────────────────────

/// A prepared bridge plan — the output of `BridgeProvider::prepare`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePlan {
    pub source_slot: u64,
    pub destination_slot: u64,
    pub requested_route: String,
    pub allocation_class: String,
    pub estimated_bytes: u64,
    pub requires_sync: bool,
}

// ── Bridge verification ─────────────────────────────────────────────────────

/// Result of validating a bridge route with instrumentation.
///
/// For a route to be called `zero_copy_verified`, repeated measurements must
/// show no bridge-sized allocation or copy event attributable to the handoff,
/// AND the provider must establish the relevant physical-storage relationship.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeVerification {
    pub passed: bool,
    pub zero_copy_proved: bool,
    pub lifetime_safe: bool,
    pub digest_match: bool,
    pub failure_reason: Option<String>,
    pub verification_details: Vec<String>,
}

// ── Bridge provider trait ───────────────────────────────────────────────────

/// Trait for Level 3 bridge providers.
///
/// The core scheduler does not care which provider is chosen; it consumes the
/// capability declaration and its measured cost. The fallback invariant: any
/// bridge that fails must atomically revert to explicit materialization.
pub trait BridgeProvider: Send + Sync {
    /// Probe what this provider can do for a given device/OS/tensor pair.
    fn probe_capability(
        &self,
        device: &str,
        os: &str,
        source_layout: &TensorDescriptor,
        destination_layout: &TensorDescriptor,
    ) -> BridgeCapability;

    /// Prepare a bridge plan for the given source and destination.
    fn prepare(&self, source_slot: u64, destination: &TensorDescriptor) -> BridgePlan;

    /// Execute the bridge plan, returning a receipt.
    fn execute(&self, plan: &BridgePlan) -> BridgeReceipt;

    /// Validate the bridge route with instrumentation.
    fn validate(&self, plan: &BridgePlan, instrumentation: &str) -> BridgeVerification;
}
