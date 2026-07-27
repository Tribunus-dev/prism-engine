//! This module owns the canonical authority-rule vocabulary that constrains
//! which regions may emit, mutate, or be required to consume specific
//! output types.

use serde::{Deserialize, Serialize};

use super::manifest::RegionOutputAuthority;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionAuthorityPolicy {
    pub policy_id: String,
    pub rules: Vec<AuthorityRule>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorityRule {
    pub rule_kind: AuthorityRuleKind,
    pub reject_message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuthorityRuleKind {
    OnlyOneRegionMayEmit(RegionOutputAuthority),
    RegionMayNotMutate {
        region: String,
        store_kind: String,
    },
    TtsMustNotConsumeDraftText,
    RouteRequiresAuthority {
        source: RegionOutputAuthority,
        target: RegionOutputAuthority,
    },
}
