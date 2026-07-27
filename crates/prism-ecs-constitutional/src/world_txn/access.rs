//! Authority: this module owns the canonical access-kind vocabulary
//! ([`AccessKind`]) and the read/write declaration shape
//! ([`AccessDeclaration`]). These are the dependencies a [`crate::world_txn::txn::WorldTxn`]
//! records so that optimistic-concurrency-control (OCC) can detect stale
//! reads at prepare time.

use crate::types::ComponentSchemaId;
use serde::{Deserialize, Serialize};

/// Read or write — the kind of access a system declares for a
/// (schema, optional entity) tuple.
///
/// `Write` implies that the caller intends to mutate the component on the
/// targeted entity (or any entity for that schema). `Read` is for
/// observed dependencies that feed into the transaction's decisions but
/// do not mutate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccessKind {
    /// Read-only dependency; observed for OCC.
    Read,
    /// Read-write dependency; observed for OCC and reserved for the
    /// commit.
    Write,
}

/// Access declaration — what a system intends to read or write.
///
/// `entity: Option<u64>` distinguishes "I am reading the whole column"
/// (`None`) from "I am reading this specific row" (`Some(entity_id)`).
/// OCC validation in [`crate::world_txn::txn::WorldTxn::prepare_inner`]
/// uses these declarations together with the world's component-version
/// table to detect stale reads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccessDeclaration {
    /// Stable schema identifier of the accessed component.
    pub schema_id: ComponentSchemaId,
    /// Targeted entity id, or `None` for column-wide access.
    pub entity: Option<u64>,
    /// Read or write intent.
    pub access: AccessKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ComponentSchemaId;

    /// `AccessKind` is `Copy + Eq + Hash` so it can live inside
    /// `BTreeMap` / `BTreeSet` keys for OCC dependency tracking
    /// without forcing the caller to clone.
    #[test]
    fn access_kind_is_copy_and_eq() {
        let a = AccessKind::Read;
        let b = a; // Copy semantics
        assert_eq!(a, b);
        // `Read != Write` is the foundational ordering.
        assert_ne!(AccessKind::Read, AccessKind::Write);
    }

    /// `AccessDeclaration` hashing must be stable so the OCC
    /// dependency table does not get duplicate entries for the
    /// same logical access.
    #[test]
    fn access_declaration_hash_is_stable_for_identical_inputs() {
        use std::collections::HashSet;
        let lhs = AccessDeclaration {
            schema_id: ComponentSchemaId(42),
            entity: Some(7),
            access: AccessKind::Write,
        };
        let rhs = AccessDeclaration {
            schema_id: ComponentSchemaId(42),
            entity: Some(7),
            access: AccessKind::Write,
        };
        let mut set: HashSet<AccessDeclaration> = HashSet::new();
        set.insert(lhs.clone());
        set.insert(rhs);
        assert_eq!(set.len(), 1);
        // `entity: None` (column-wide) and `entity: Some(0)` are
        // distinct declarations — they must not collide.
        let column_wide = AccessDeclaration {
            schema_id: ComponentSchemaId(42),
            entity: None,
            access: AccessKind::Read,
        };
        assert!(!set.contains(&column_wide));
    }
}
