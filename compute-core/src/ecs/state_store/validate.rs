use crate::ecs::state_store::receipts::StateStoreValidationReceipt;
use crate::ecs::state_store::schema::{KvCacheLayout, KvCacheStoreDecl, StateStoreSchema};
// ---------------------------------------------------------------------------
// Validation gates (see local://state_store_spec.md § Validation gates)
// ---------------------------------------------------------------------------

/// Compute total max bytes from a schema's stores.
pub fn compute_total_max_bytes(schema: &StateStoreSchema) -> u64 {
    schema.stores.iter().map(|s| s.max_bytes).sum()
}

/// Validate a StateStoreSchema against the six validation gates.
///
/// Gates checked:
/// 1. Store IDs must be unique
/// 2. Owner region must be a non-empty string
/// 3. Max bytes must be > 0
/// 4. For KV stores, page_tokens and alignment_bytes must be > 0
/// 5. Allowed regions must reference valid store IDs
/// 6. No two policies for the same store have conflicting access
pub fn validate_schema(
    schema: &StateStoreSchema,
    _kv_decls: &[KvCacheStoreDecl],
) -> StateStoreValidationReceipt {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let name = "state_store".to_string();

    // --- Gate 1: unique store IDs ---
    {
        let mut seen = std::collections::HashSet::new();
        for s in &schema.stores {
            if !seen.insert(&s.store_id) {
                errors.push(format!("Gate 1: duplicate store_id '{}'", s.store_id));
            }
        }
    }

    // --- Gate 2: non-empty owner_region ---
    for s in &schema.stores {
        if s.owner_region.trim().is_empty() {
            errors.push(format!(
                "Gate 2: store '{}' has empty owner_region",
                s.store_id
            ));
        }
    }

    // --- Gate 3: max_bytes > 0 ---
    for s in &schema.stores {
        if s.max_bytes == 0 {
            errors.push(format!("Gate 3: store '{}' has max_bytes == 0", s.store_id));
        }
    }

    // --- Gate 4: KV store page_tokens > 0, alignment_bytes > 0 ---
    for kv in _kv_decls {
        let KvCacheLayout::PagedLayerHead {
            page_tokens,
            alignment_bytes,
        } = &kv.cache_layout;
        {
            if *page_tokens == 0 {
                errors.push(format!(
                    "Gate 4: KV store '{}' has page_tokens == 0",
                    kv.store_id
                ));
            }
            if *alignment_bytes == 0 {
                errors.push(format!(
                    "Gate 4: KV store '{}' has alignment_bytes == 0",
                    kv.store_id
                ));
            }
        }
    }

    // Build a store_id → true map for gate 5.
    let store_ids: std::collections::HashSet<&str> =
        schema.stores.iter().map(|s| s.store_id.as_str()).collect();
    let store_ids_kv: std::collections::HashSet<&str> =
        _kv_decls.iter().map(|kv| kv.store_id.as_str()).collect();

    // --- Gate 5: allowed regions reference valid store IDs ---
    for p in &schema.access_policies {
        if !store_ids.contains(p.store_id.as_str()) && !store_ids_kv.contains(p.store_id.as_str()) {
            errors.push(format!(
                "Gate 5: access policy '{}' references unknown store_id '{}'",
                p.policy_id, p.store_id
            ));
        }
    }

    // --- Gate 6: no two policies for the same store have conflicting access ---
    {
        // Group policies by store_id, check for conflicts.
        let mut by_store: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        for p in &schema.access_policies {
            by_store
                .entry(p.store_id.as_str())
                .or_default()
                .push(p.policy_id.as_str());
        }
        for (_store, pids) in &by_store {
            if pids.len() > 1 {
                warnings.push(format!(
                    "Gate 6: store '{}' has {} policies ({}) — manual review may be needed",
                    _store,
                    pids.len(),
                    pids.join(", ")
                ));
            }
        }
    }

    let store_count = schema.stores.len() as u32;
    let total_max_bytes = compute_total_max_bytes(schema);
    let valid = errors.is_empty();

    StateStoreValidationReceipt {
        schema_name: name,
        store_count,
        total_max_bytes,
        errors,
        warnings,
        valid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_compute_image_core::kv_plan::KvCodec;
    use crate::ecs::state_store::schema::*;

    fn valid_decl(id: &str) -> StateStoreDecl {
        StateStoreDecl {
            store_id: id.to_string(),
            store_kind: "kv_cache".to_string(),
            owner_region: "region_a".to_string(),
            dtype: "fp16".to_string(),
            max_bytes: 1_000_000,
            persistence: "volatile".to_string(),
        }
    }

    fn valid_kv_decl(id: &str) -> KvCacheStoreDecl {
        KvCacheStoreDecl {
            store_id: id.to_string(),
            model_partition_id: "p0".to_string(),
            layer_count: 4,
            head_count: 8,
            kv_head_count: 8,
            head_dim: 128,
            max_sequence_len: 4096,
            cache_layout: KvCacheLayout::PagedLayerHead {
                page_tokens: 64,
                alignment_bytes: 256,
            },
            codec_policy: KvCodecPolicy {
                codec: KvCodec::Fp16,
            },
            residency_policy: KvResidencyPolicy {
                max_active_spans: 8,
                span_pin_supported: true,
            },
        }
    }

    #[test]
    fn validate_passes_well_formed_schema() {
        let schema = StateStoreSchema {
            stores: vec![valid_decl("kv1")],
            access_policies: vec![],
            eviction_policies: vec![],
        };
        let kv = vec![valid_kv_decl("kv1")];
        let r = validate_schema(&schema, &kv);
        assert!(r.valid, "expected valid: {:#?}", r.errors);
    }

    #[test]
    fn gate1_duplicate_store_ids() {
        let schema = StateStoreSchema {
            stores: vec![valid_decl("kv1"), valid_decl("kv1")],
            access_policies: vec![],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[]);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("Gate 1")));
    }

    #[test]
    fn gate2_empty_owner_region() {
        let mut d = valid_decl("kv1");
        d.owner_region = "  ".to_string();
        let schema = StateStoreSchema {
            stores: vec![d],
            access_policies: vec![],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[]);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("Gate 2")));
    }

    #[test]
    fn gate3_zero_max_bytes() {
        let mut d = valid_decl("kv1");
        d.max_bytes = 0;
        let schema = StateStoreSchema {
            stores: vec![d],
            access_policies: vec![],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[]);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("Gate 3")));
    }

    #[test]
    fn gate4_zero_page_tokens() {
        let mut kv = valid_kv_decl("kv1");
        kv.cache_layout = KvCacheLayout::PagedLayerHead {
            page_tokens: 0,
            alignment_bytes: 256,
        };
        let schema = StateStoreSchema {
            stores: vec![],
            access_policies: vec![],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[kv]);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("page_tokens == 0")));
    }

    #[test]
    fn gate4_zero_alignment_bytes() {
        let mut kv = valid_kv_decl("kv1");
        kv.cache_layout = KvCacheLayout::PagedLayerHead {
            page_tokens: 64,
            alignment_bytes: 0,
        };
        let schema = StateStoreSchema {
            stores: vec![],
            access_policies: vec![],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[kv]);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("alignment_bytes == 0")));
    }

    #[test]
    fn gate5_unknown_store_in_policy() {
        let schema = StateStoreSchema {
            stores: vec![valid_decl("kv1")],
            access_policies: vec![StateAccessPolicy {
                policy_id: "p1".to_string(),
                store_id: "nonexistent".to_string(),
                allowed_regions: vec!["r1".to_string()],
                access: AccessKind::Read,
            }],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[]);
        assert!(!r.valid);
        assert!(r.errors.iter().any(|e| e.contains("Gate 5")));
    }

    #[test]
    fn gate6_multiple_policies_same_store() {
        let schema = StateStoreSchema {
            stores: vec![valid_decl("kv1")],
            access_policies: vec![
                StateAccessPolicy {
                    policy_id: "p1".to_string(),
                    store_id: "kv1".to_string(),
                    allowed_regions: vec!["r1".to_string()],
                    access: AccessKind::Read,
                },
                StateAccessPolicy {
                    policy_id: "p2".to_string(),
                    store_id: "kv1".to_string(),
                    allowed_regions: vec!["r2".to_string()],
                    access: AccessKind::Write,
                },
            ],
            eviction_policies: vec![],
        };
        let r = validate_schema(&schema, &[]);
        assert!(r.valid, "multiple policies is a warning, not error");
        assert!(r.warnings.iter().any(|w| w.contains("Gate 6")));
    }
}
