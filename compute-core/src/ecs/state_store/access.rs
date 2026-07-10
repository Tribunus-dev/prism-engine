use serde::{Deserialize, Serialize};

use crate::ecs::state_store::schema::AccessKind;
use crate::ecs::state_store::schema::StateAccessPolicy;

/// Result of an access check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateAccessCheck {
    pub store_id: String,
    pub region_id: String,
    pub access: String,
    pub epoch_id: Option<u64>,
    pub allowed: bool,
    pub reason: String,
}

/// Check whether `region_id` has `access`-level permission on `store_id`
/// according to the given set of policies.
pub fn check_access(
    policies: &[StateAccessPolicy],
    store_id: &str,
    region_id: &str,
    access: &str,
) -> StateAccessCheck {
    let access_lower = access.to_lowercase();
    let required = match access_lower.as_str() {
        "read" => AccessKind::Read,
        "write" => AccessKind::Write,
        _ => {
            return StateAccessCheck {
                store_id: store_id.to_string(),
                region_id: region_id.to_string(),
                access: access.to_string(),
                epoch_id: None,
                allowed: false,
                reason: format!("unknown access kind '{}'", access),
            };
        }
    };

    // Collect policies for this store.
    let relevant: Vec<&StateAccessPolicy> =
        policies.iter().filter(|p| p.store_id == store_id).collect();

    for policy in &relevant {
        if !policy.allowed_regions.iter().any(|r| r == region_id) {
            continue;
        }
        let granted = policy.access;
        let allowed = match (required, granted) {
            // ReadWrite grants everything.
            (_, AccessKind::ReadWrite) => true,
            // Exact match.
            (a, b) if a == b => true,
            // Read <= ReadWrite; Write not granted by Read-only.
            (AccessKind::Read, AccessKind::Read) => true,
            (AccessKind::Write, AccessKind::Write) => true,
            _ => false,
        };
        if allowed {
            return StateAccessCheck {
                store_id: store_id.to_string(),
                region_id: region_id.to_string(),
                access: access.to_string(),
                epoch_id: None,
                allowed: true,
                reason: "access granted by policy".to_string(),
            };
        }
    }

    // No matching policy — check if region is the owner (implicit full access).
    // Ownership is determined externally; if no policy matched, deny.
    StateAccessCheck {
        store_id: store_id.to_string(),
        region_id: region_id.to_string(),
        access: access.to_string(),
        epoch_id: None,
        allowed: false,
        reason: format!(
            "no policy grants '{}' access to region '{}' on store '{}'",
            access, region_id, store_id
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::state_store::schema::{AccessKind, StateAccessPolicy};

    fn make_policy(
        id: &str,
        store: &str,
        regions: &[&str],
        access: AccessKind,
    ) -> StateAccessPolicy {
        StateAccessPolicy {
            policy_id: id.to_string(),
            store_id: store.to_string(),
            allowed_regions: regions.iter().map(|r| r.to_string()).collect(),
            access,
        }
    }

    #[test]
    fn check_access_allows_explicit_read() {
        let policies = vec![make_policy("p1", "kv1", &["region_a"], AccessKind::Read)];
        let r = check_access(&policies, "kv1", "region_a", "read");
        assert!(r.allowed, "expected allowed: {}", r.reason);
    }

    #[test]
    fn check_access_allows_explicit_write() {
        let policies = vec![make_policy("p1", "kv1", &["region_a"], AccessKind::Write)];
        let r = check_access(&policies, "kv1", "region_a", "write");
        assert!(r.allowed, "expected allowed: {}", r.reason);
    }

    #[test]
    fn check_access_denies_write_from_readonly() {
        let policies = vec![make_policy("p1", "kv1", &["region_a"], AccessKind::Read)];
        let r = check_access(&policies, "kv1", "region_a", "write");
        assert!(!r.allowed, "expected denied");
    }

    #[test]
    fn check_access_readwrite_grants_both() {
        let policies = vec![make_policy(
            "p1",
            "kv1",
            &["region_a"],
            AccessKind::ReadWrite,
        )];
        let r1 = check_access(&policies, "kv1", "region_a", "read");
        let r2 = check_access(&policies, "kv1", "region_a", "write");
        assert!(r1.allowed, "read denied under ReadWrite");
        assert!(r2.allowed, "write denied under ReadWrite");
    }

    #[test]
    fn check_access_denies_unknown_region() {
        let policies = vec![make_policy("p1", "kv1", &["region_a"], AccessKind::Read)];
        let r = check_access(&policies, "kv1", "region_b", "read");
        assert!(!r.allowed, "expected denied for unknown region");
    }

    #[test]
    fn check_access_denies_unknown_access_kind() {
        let policies = vec![];
        let r = check_access(&policies, "kv1", "region_a", "delete");
        assert!(!r.allowed);
        assert!(r.reason.contains("unknown access kind"));
    }
}
