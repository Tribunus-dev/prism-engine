use crate::state_store::access::check_access;
use crate::state_store::epochs::StateEpoch;
use crate::state_store::kv::KvCacheManager;
use crate::state_store::pages::PageTable;
use crate::state_store::receipts::{KvAppendReceipt, KvReadReceipt, StateStoreValidationReceipt};
use crate::state_store::schema::{
    AccessKind, EvictionKind, KvCacheLayout, KvCacheStoreDecl, KvPrecisionPolicy,
    KvResidencyPolicy, StateAccessPolicy, StateStoreDecl, StateStoreSchema,
};
use crate::state_store::validate::validate_schema;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn make_kv_config(store_id: &str) -> KvCacheStoreDecl {
    KvCacheStoreDecl {
        store_id: store_id.to_string(),
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
        precision_policy: KvPrecisionPolicy {
            key_dtype: "fp16".to_string(),
            value_dtype: "fp16".to_string(),
        },
        residency_policy: KvResidencyPolicy {
            max_active_spans: 8,
            span_pin_supported: true,
        },
    }
}

// ---------------------------------------------------------------------------
// kv_store_allocates_paged_layout
// ---------------------------------------------------------------------------

#[test]
fn kv_store_allocates_paged_layout() {
    let config = make_kv_config("kv1");
    let mut mgr = KvCacheManager::new("kv1", config, "region_a");

    // page_tokens=64, head_dim=128, key=2B+value=2B => 32768 per page
    const PAGE_ALIGNED: u64 = 32768;

    mgr.allocate_store().unwrap();

    assert!(!mgr.pages.is_empty(), "expected pages after allocation");
    // Expected page count:
    //   page_tokens=64, max_sequence_len=4096 => 64 pages per (layer, head)
    //   layer_count=4, head_count=8 => 4*8*64 = 2048
    assert_eq!(mgr.pages.len(), 2048);
    assert_eq!(mgr.max_bytes, 2048 * PAGE_ALIGNED);

    // Verify page layout invariants.
    for p in &mgr.pages {
        assert_eq!(p.byte_length, 32768);
        assert!(p.layer_index < 4);
        assert!(p.head_index < 8);
        assert_eq!(p.owner_region, "region_a");
    }

    let (used, max) = mgr.memory_usage();
    assert_eq!(used, 0);
    assert_eq!(max, 2048 * PAGE_ALIGNED);
}

// ---------------------------------------------------------------------------
// kv_append_produces_page_descriptors
// ---------------------------------------------------------------------------

#[test]
fn kv_append_produces_page_descriptors() {
    let config = make_kv_config("kv_append");
    let mut mgr = KvCacheManager::new("kv_append", config, "owner");
    mgr.allocate_store().unwrap();

    let receipt = mgr
        .append_tokens(0, 128, "span_0", 0, "owner")
        .expect("append should succeed");

    assert_eq!(receipt.store_id, "kv_append");
    assert!(receipt.pages_allocated > 0);
    assert_eq!(receipt.total_pages, 2048);
    assert!(receipt.memory_ok);

    // Should have created one span with pages assigned.
    assert_eq!(mgr.spans.len(), 1);
    assert_eq!(mgr.spans[0].span_id, "span_0");
    assert!(!mgr.spans[0].pages.is_empty());

    // Verify the claimed_pages count matches.
    let (used, _) = mgr.memory_usage();
    assert_eq!(used, receipt.pages_allocated as u64 * 32768);
}

// ---------------------------------------------------------------------------
// kv_read_window_resolves_pages
// ---------------------------------------------------------------------------

#[test]
fn kv_read_window_resolves_pages() {
    let config = make_kv_config("kv_read");
    let mut mgr = KvCacheManager::new("kv_read", config, "reader_region");
    mgr.allocate_store().unwrap();

    // Append tokens first.
    mgr.append_tokens(0, 128, "span_a", 0, "reader_region")
        .expect("append failed");

    // Read the same window.
    let receipt = mgr
        .read_window(0, 64, 0, "reader_region")
        .expect("read should succeed");

    assert!(receipt.access_granted);
    assert!(receipt.pages_resolved > 0, "expected pages to be resolved");
    assert!(receipt.byte_length > 0);
    assert_eq!(receipt.store_id, "kv_read");

    // Read a range that doesn't exist yet — should get 0 pages.
    let missing = mgr
        .read_window(10000, 64, 0, "reader_region")
        .expect("read should succeed");
    assert_eq!(missing.pages_resolved, 0);
}

// ---------------------------------------------------------------------------
// kv_rejects_write_by_non_owner
// ---------------------------------------------------------------------------

#[test]
fn kv_rejects_write_by_non_owner() {
    let config = make_kv_config("kv_write_chk");
    let mut mgr = KvCacheManager::new("kv_write_chk", config, "owner_region");
    mgr.allocate_store().unwrap();

    // Non-owner tries to append.
    let result = mgr.append_tokens(0, 64, "bad_span", 0, "intruder");
    assert!(result.is_err(), "non-owner append should be rejected");

    // Owner can still append.
    let result = mgr.append_tokens(0, 64, "good_span", 0, "owner_region");
    assert!(result.is_ok(), "owner append should succeed");
}

// ---------------------------------------------------------------------------
// kv_rejects_draft_epoch_read_by_unpermitted_region
// ---------------------------------------------------------------------------

#[test]
fn kv_rejects_draft_epoch_read_by_unpermitted_region() {
    let config = make_kv_config("kv_draft_epoch");
    let mut mgr = KvCacheManager::new("kv_draft_epoch", config, "owner");
    mgr.allocate_store().unwrap();

    // Populate some data as owner.
    mgr.append_tokens(0, 128, "span_draft", 1, "owner")
        .expect("owner append");

    // Another region tries to read epoch 1 (draft/uncommitted) without
    // a read policy granting it access.
    let receipt = mgr
        .read_window(0, 64, 1, "other_region")
        .expect("read should not error, but access denied");

    assert!(
        !receipt.access_granted,
        "unpermitted region should be denied read on draft epoch"
    );
    assert_eq!(receipt.pages_resolved, 0);

    // Owner can still read.
    let owner_receipt = mgr.read_window(0, 64, 1, "owner").expect("owner read");
    assert!(owner_receipt.access_granted);
}

// ---------------------------------------------------------------------------
// kv_memory_budget_accounting_is_deterministic
// ---------------------------------------------------------------------------

#[test]
fn kv_memory_budget_accounting_is_deterministic() {
    let config = make_kv_config("kv_mem_budget");
    let mut mgr = KvCacheManager::new("kv_mem_budget", config, "region_x");
    mgr.allocate_store().unwrap();

    const PAGE_ALIGNED: u64 = 32768;

    let (used1, max1) = mgr.memory_usage();
    assert_eq!(used1, 0);
    assert_eq!(max1, 2048 * PAGE_ALIGNED);

    // Append tokens twice; memory usage should increase deterministically.
    let r1 = mgr
        .append_tokens(0, 128, "s1", 0, "region_x")
        .expect("append s1");

    let (used2, _) = mgr.memory_usage();
    assert_eq!(used2, r1.pages_allocated as u64 * PAGE_ALIGNED);

    let r2 = mgr
        .append_tokens(128, 128, "s2", 0, "region_x")
        .expect("append s2");
    let (used3, _) = mgr.memory_usage();
    assert_eq!(
        used3,
        (r1.pages_allocated + r2.pages_allocated) as u64 * PAGE_ALIGNED
    );

    // Total should never exceed max.
    assert!(used3 <= max1);
}

// ---------------------------------------------------------------------------
// serde roundtrips for all major types
// ---------------------------------------------------------------------------

#[test]
fn serde_roundtrip_state_store_schema() {
    let schema = StateStoreSchema {
        stores: vec![StateStoreDecl {
            store_id: "s1".to_string(),
            store_kind: "kv_cache".to_string(),
            owner_region: "r1".to_string(),
            dtype: "fp16".to_string(),
            max_bytes: 1_000_000,
            persistence: "volatile".to_string(),
        }],
        access_policies: vec![StateAccessPolicy {
            policy_id: "p1".to_string(),
            store_id: "s1".to_string(),
            allowed_regions: vec!["r1".to_string()],
            access: AccessKind::ReadWrite,
        }],
        eviction_policies: vec![],
    };

    let json = serde_json::to_string(&schema).unwrap();
    let deserialized: StateStoreSchema = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.stores.len(), 1);
    assert_eq!(deserialized.stores[0].store_id, "s1");
    assert_eq!(deserialized.access_policies.len(), 1);
}

#[test]
fn serde_roundtrip_kv_cache_store_decl() {
    let decl = KvCacheStoreDecl {
        store_id: "k1".to_string(),
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
        precision_policy: KvPrecisionPolicy {
            key_dtype: "fp16".to_string(),
            value_dtype: "fp16".to_string(),
        },
        residency_policy: KvResidencyPolicy {
            max_active_spans: 8,
            span_pin_supported: true,
        },
    };
    let json = serde_json::to_string(&decl).unwrap();
    let d: KvCacheStoreDecl = serde_json::from_str(&json).unwrap();
    assert_eq!(d.store_id, "k1");
    assert_eq!(d.layer_count, 4);
}

#[test]
fn serde_roundtrip_state_epoch() {
    let e = StateEpoch::new(5, Some(3), "draft");
    let json = serde_json::to_string(&e).unwrap();
    let d: StateEpoch = serde_json::from_str(&json).unwrap();
    assert_eq!(d.epoch_id, 5);
    assert_eq!(d.parent_epoch_id, Some(3));
    assert!(!d.committed);
}

#[test]
fn serde_roundtrip_receipts() {
    let ar = KvAppendReceipt {
        store_id: "s".to_string(),
        span_id: "sp".to_string(),
        epoch_id: 1,
        pages_allocated: 10,
        total_pages: 100,
        total_bytes_after: 2560,
        memory_ok: true,
    };
    let json = serde_json::to_string(&ar).unwrap();
    let d: KvAppendReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(d.store_id, "s");
    assert_eq!(d.pages_allocated, 10);

    let rr = KvReadReceipt {
        store_id: "s".to_string(),
        token_start: 0,
        token_count: 64,
        epoch_id: 1,
        pages_resolved: 2,
        byte_offset: 0,
        byte_length: 512,
        access_granted: true,
    };
    let json = serde_json::to_string(&rr).unwrap();
    let d: KvReadReceipt = serde_json::from_str(&json).unwrap();
    assert!(d.access_granted);
    assert_eq!(d.pages_resolved, 2);
}

#[test]
fn serde_roundtrip_page_table() {
    let desc = crate::state_store::pages::PageDescriptor {
        page_id: 0,
        layer_index: 0,
        head_index: 0,
        token_offset: 0,
        token_count: 64,
        byte_offset: 0,
        byte_length: 256,
        epoch_id: 0,
        owner_region: "r1".to_string(),
    };
    let pt = PageTable::new(vec![desc], 256, 64);
    let json = serde_json::to_string(&pt).unwrap();
    let d: PageTable = serde_json::from_str(&json).unwrap();
    assert_eq!(d.pages.len(), 1);
    assert_eq!(d.page_size_bytes, 256);
}

#[test]
fn serde_roundtrip_access_check() {
    use crate::state_store::access::StateAccessCheck;
    let ac = StateAccessCheck {
        store_id: "s1".to_string(),
        region_id: "r1".to_string(),
        access: "read".to_string(),
        epoch_id: None,
        allowed: true,
        reason: "ok".to_string(),
    };
    let json = serde_json::to_string(&ac).unwrap();
    let d: StateAccessCheck = serde_json::from_str(&json).unwrap();
    assert!(d.allowed);
}

// ---------------------------------------------------------------------------
// Validation gate integration (validate.rs is tested inline, but we also
// verify the public API works end-to-end)
// ---------------------------------------------------------------------------

#[test]
fn validation_gate_end_to_end() {
    let schema = StateStoreSchema {
        stores: vec![
            StateStoreDecl {
                store_id: "v1".to_string(),
                store_kind: "kv_cache".to_string(),
                owner_region: "r1".to_string(),
                dtype: "fp16".to_string(),
                max_bytes: 1_000_000,
                persistence: "volatile".to_string(),
            },
            StateStoreDecl {
                store_id: "v2".to_string(),
                store_kind: "kv_cache".to_string(),
                owner_region: "r2".to_string(),
                dtype: "fp16".to_string(),
                max_bytes: 2_000_000,
                persistence: "volatile".to_string(),
            },
        ],
        access_policies: vec![StateAccessPolicy {
            policy_id: "p1".to_string(),
            store_id: "v1".to_string(),
            allowed_regions: vec!["r1".to_string()],
            access: AccessKind::ReadWrite,
        }],
        eviction_policies: vec![],
    };

    let kv_decls = vec![make_kv_config("v1"), make_kv_config("v2")];
    let receipt = validate_schema(&schema, &kv_decls);
    assert!(receipt.valid, "expected valid: {:?}", receipt.errors);
    assert_eq!(receipt.store_count, 2);
    assert_eq!(receipt.total_max_bytes, 3_000_000);
}

// ---------------------------------------------------------------------------
// KvCacheManager pin_span and edge cases
// ---------------------------------------------------------------------------

#[test]
fn pin_span_marks_pinned_and_prevents_duplicate_pin() {
    let config = make_kv_config("kv_pin");
    let mut mgr = KvCacheManager::new("kv_pin", config, "owner");
    mgr.allocate_store().unwrap();

    mgr.append_tokens(0, 64, "pin_me", 0, "owner")
        .expect("append");

    mgr.pin_span("pin_me").unwrap();
    assert!(mgr.spans[0].pinned);

    // Double pin should error.
    let err = mgr.pin_span("pin_me");
    assert!(err.is_err());
}

#[test]
fn pin_span_not_found() {
    let config = make_kv_config("kv_pin_nf");
    let mut mgr = KvCacheManager::new("kv_pin_nf", config, "owner");
    mgr.allocate_store().unwrap();

    let err = mgr.pin_span("nonexistent");
    assert!(err.is_err());
}
