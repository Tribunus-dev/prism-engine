//! ANE page-migration policy — config and tier-transition types.
//!
//! Authority: the canonical page-migration policy config and the
//! tier-transition types.
//!
//! The actual [`PageMigrationPolicy`] trait implementation lives in
//! the engine's `legacy_ane/page_migration_policy.rs` because it
//! depends on engine-internal `KVCacheTier` / `PageBacking` /
//! `TiersPage` / `PageMigrationPolicy` types and on the
//! `AneCompressor` engine-coupled adapter. The constitutional
//! surface provides the policy config and the tier-transition
//! types in an engine-neutral form.
//!
//! # Tier mapping (old ANE → platform-agnostic)
//!
//! - L1AneSram  → [`MigrationTier::L0Device`]
//! - L2Iosurface → [`MigrationTier::L1Shared`]
//! - L3DramHeap  → [`MigrationTier::L2System`]
//! - L4Disk      → [`MigrationTier::L3Disk`]
//!
//! # Data format per tier
//!
//! - L0Device: 3.5-bit packed (device-local).
//! - L1Shared: 3.5-bit packed (shared memory, e.g. IOSurface).
//! - L2System: 2-bit packed (host DRAM).
//! - L3Disk:   2-bit packed (disk, no resident data).

use std::time::Duration;

/// Platform-agnostic migration tier.
///
/// Mirrors the engine's `KVCacheTier` enum. The engine's
/// `legacy_ane/page_migration_policy.rs` maps each
/// `MigrationTier` variant to the engine-internal tier and backing
/// representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MigrationTier {
    /// Device-local memory (e.g. ANE SRAM, 3.5-bit packed).
    L0Device,
    /// Shared memory (e.g. IOSurface, 3.5-bit packed).
    L1Shared,
    /// System memory (e.g. host DRAM, 2-bit packed).
    L2System,
    /// Disk-backed (cold storage, 2-bit packed).
    L3Disk,
}

impl MigrationTier {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::L0Device => "L0Device",
            Self::L1Shared => "L1Shared",
            Self::L2System => "L2System",
            Self::L3Disk => "L3Disk",
        }
    }
}

/// Backend-agnostic config for [`AnePageMigrationPolicy`].
///
/// `head_dim` is the KV head dimension (e.g. 120 for Gemma4, 64
/// for Qwen). `n_kv_heads` is the number of KV heads. The
/// `cold_threshold` and `hot_threshold` durations drive
/// promotion/demotion decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnePageMigrationPolicyConfig {
    /// KV head dimension.
    pub head_dim: u32,
    /// Number of KV heads per layer.
    pub n_kv_heads: u32,
    /// Duration of inactivity before a page is considered cold
    /// (demotion candidate).
    pub cold_threshold: Duration,
    /// Duration since last access for a page to be considered hot
    /// (promotion candidate).
    pub hot_threshold: Duration,
}

impl AnePageMigrationPolicyConfig {
    /// Create a new config.
    pub fn new(
        head_dim: u32,
        n_kv_heads: u32,
        cold_threshold: Duration,
        hot_threshold: Duration,
    ) -> Self {
        Self {
            head_dim,
            n_kv_heads,
            cold_threshold,
            hot_threshold,
        }
    }
}

/// Migration policy name for log lines.
pub const ANE_MIGRATION_POLICY_NAME: &str = "ane_page_migration_policy";

/// ANE platform-specific page-migration policy.
///
/// The constitutional surface stores the config; the engine's
/// `legacy_ane/page_migration_policy.rs` adds the `compressor` field
/// (an `Arc<AneCompressor>`) and implements the engine's
/// `PageMigrationPolicy` trait.
#[derive(Debug, Clone)]
pub struct AnePageMigrationPolicy {
    /// Public config (read-only after construction).
    pub config: AnePageMigrationPolicyConfig,
}

impl AnePageMigrationPolicy {
    /// Create a new policy with the given config.
    pub fn new(config: AnePageMigrationPolicyConfig) -> Self {
        Self { config }
    }

    /// Name of this policy.
    pub fn name(&self) -> &'static str {
        ANE_MIGRATION_POLICY_NAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_names() {
        assert_eq!(MigrationTier::L0Device.name(), "L0Device");
        assert_eq!(MigrationTier::L1Shared.name(), "L1Shared");
        assert_eq!(MigrationTier::L2System.name(), "L2System");
        assert_eq!(MigrationTier::L3Disk.name(), "L3Disk");
    }

    #[test]
    fn policy_name() {
        let config = AnePageMigrationPolicyConfig::new(
            64,
            8,
            Duration::from_secs(60),
            Duration::from_secs(5),
        );
        let policy = AnePageMigrationPolicy::new(config);
        assert_eq!(policy.name(), ANE_MIGRATION_POLICY_NAME);
    }
}
