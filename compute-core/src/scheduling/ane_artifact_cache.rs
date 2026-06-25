//! ANE artifact cache — Core ML packet cache for ANE execution.
//!
//! Manages residency state transitions (Cold → Compiling → Loaded → Warmed),
//! LRU eviction, weight-budget enforcement, and lease-gated eviction protection
//! for in-flight Warmed entries.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};


// ── Key ────────────────────────────────────────────────────────────────────

/// Key identifying a Core ML artifact for ANE execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactKey {
    pub model_family: String,
    pub packet_kind: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub shape_bucket: u32,
    pub precision: String,
}

// ── Residency state ────────────────────────────────────────────────────────

/// Residency lifecycle of an ANE artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactResidencyState {
    /// Not yet compiled or loaded.
    Cold,
    /// Core ML compilation in progress.
    Compiling,
    /// Compiled and resident in memory.
    Loaded,
    /// Warmed up and ready for execution.
    Warmed,
    /// Marked for eviction.
    Evictable,
    /// Compilation or warmup failed.
    Failed(String),
}

// ── Cache entry ────────────────────────────────────────────────────────────

/// A single entry in the ANE artifact cache.
fn serde_instant_now() -> Instant { Instant::now() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneArtifactCacheEntry {
    pub key: ArtifactKey,
    pub state: ArtifactResidencyState,
    /// Last access timestamp. Skipped during serialization since `Instant`
    /// does not implement `Serialize`/`Deserialize`.
    #[serde(skip_serializing, skip_deserializing, default = "serde_instant_now")]
    pub last_used: Instant,
    pub compile_latency: Duration,
    pub warmup_latency: Duration,
    pub steady_state_latency: Option<Duration>,
    pub memory_footprint_bytes: u64,
}

// ── Eviction policy ────────────────────────────────────────────────────────

/// Eviction policy selection for the ANE artifact cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AneEvictionPolicy {
    /// Least-recently-used eviction.
    Lru,
    /// LRU with a hard byte budget.
    WeightBudget { max_bytes: u64 },
    /// Prioritise by admission gate priority (placeholder).
    AdmissionPriority,
}

// ── Cache ──────────────────────────────────────────────────────────────────

/// ANE artifact cache managing Core ML packet residency and lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AneArtifactCache {
    pub max_slots: u32,
    pub max_weight_bytes: u64,
    pub entries: HashMap<ArtifactKey, AneArtifactCacheEntry>,
    pub eviction_policy: AneEvictionPolicy,
    /// Tracks leases for in-flight Warmed entries.
    #[serde(skip)]
    pub active_leases: HashSet<ArtifactKey>,
}

impl AneArtifactCache {
    /// Create a new empty cache.
    pub fn new(
        max_slots: u32,
        max_weight_bytes: u64,
        eviction_policy: AneEvictionPolicy,
    ) -> Self {
        Self {
            max_slots,
            max_weight_bytes,
            entries: HashMap::new(),
            eviction_policy,
            active_leases: HashSet::new(),
        }
    }

    /// Insert an entry, evicting LRU entries if the cache exceeds slot or
    /// weight capacity.  Entries in the `Warmed` state with an active lease
    /// are protected from eviction.
    pub fn insert(&mut self, key: ArtifactKey, entry: AneArtifactCacheEntry) {
        self.entries.insert(key.clone(), entry);
        self.evict_to_capacity();
    }

    /// Return the current residency state of `key`, if present.
    pub fn get_state(&self, key: &ArtifactKey) -> Option<ArtifactResidencyState> {
        self.entries.get(key).map(|e| e.state.clone())
    }

    /// Transition `key` to `new_state` with validation.
    ///
    /// Valid forward transitions:
    /// - `Cold` → `Compiling`
    /// - `Cold` → `Loaded` (bulk-warmup shortcut)
    /// - `Compiling` → `Loaded`
    /// - `Loaded` → `Warmed`
    /// - *any* → `Failed`
    /// - *any* → `Evictable` (except `Warmed` with an active lease)
    ///
    /// Terminal states (`Failed`, `Evictable`) cannot transition further.
    pub fn transition(
        &mut self,
        key: &ArtifactKey,
        new_state: ArtifactResidencyState,
    ) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(key)
            .ok_or_else(|| "key not found".to_string())?;

        // Terminal → anything is rejected.
        if matches!(
            entry.state,
            ArtifactResidencyState::Failed(_) | ArtifactResidencyState::Evictable
        ) {
            return Err(format!(
                "Cannot transition from terminal state {:?}",
                entry.state
            ));
        }

        // any → Failed is always valid.
        if matches!(&new_state, ArtifactResidencyState::Failed(_)) {
            entry.state = new_state;
            entry.last_used = Instant::now();
            return Ok(());
        }

        // any → Evictable, except Warmed with a lease.
        if new_state == ArtifactResidencyState::Evictable {
            if entry.state == ArtifactResidencyState::Warmed
                && self.active_leases.contains(key)
            {
                return Err(
                    "Cannot mark Warmed entry with active lease as Evictable".to_string(),
                );
            }
            entry.state = new_state;
            entry.last_used = Instant::now();
            return Ok(());
        }

        // Validate the forward chain.
        let valid = match (&entry.state, &new_state) {
            (ArtifactResidencyState::Cold, ArtifactResidencyState::Compiling) => true,
            (ArtifactResidencyState::Cold, ArtifactResidencyState::Loaded) => true,
            (ArtifactResidencyState::Compiling, ArtifactResidencyState::Loaded) => true,
            (ArtifactResidencyState::Loaded, ArtifactResidencyState::Warmed) => true,
            _ => false,
        };

        if !valid {
            return Err(format!(
                "Invalid transition from {:?} to {:?}",
                entry.state, new_state
            ));
        }

        entry.state = new_state;
        entry.last_used = Instant::now();
        Ok(())
    }

    /// Transition a `Loaded` entry to `Warmed` and record its warmup latency.
    pub fn mark_warmed(&mut self, key: &ArtifactKey, latency: Duration) -> Result<(), String> {
        let entry = self
            .entries
            .get_mut(key)
            .ok_or_else(|| "key not found".to_string())?;

        if entry.state != ArtifactResidencyState::Loaded {
            return Err(format!(
                "Cannot mark_warmed from state {:?}; expected Loaded",
                entry.state
            ));
        }

        entry.state = ArtifactResidencyState::Warmed;
        entry.last_used = Instant::now();
        entry.warmup_latency = latency;
        Ok(())
    }

    /// Mark an entry as failed with a reason.  No-op if already `Failed`.
    pub fn mark_failed(&mut self, key: &ArtifactKey, reason: String) {
        if let Some(entry) = self.entries.get_mut(key) {
            if matches!(entry.state, ArtifactResidencyState::Failed(_)) {
                return;
            }
            entry.state = ArtifactResidencyState::Failed(reason);
            entry.last_used = Instant::now();
        }
    }

    /// Remove an entry from the cache.
    ///
    /// Returns an error if the entry is `Warmed` and has an active lease.
    pub fn evict(&mut self, key: &ArtifactKey) -> Result<(), String> {
        let entry = self
            .entries
            .get(key)
            .ok_or_else(|| "key not found".to_string())?;

        if entry.state == ArtifactResidencyState::Warmed && self.active_leases.contains(key) {
            return Err("Cannot evict Warmed entry with active lease".to_string());
        }

        self.active_leases.remove(key);
        self.entries.remove(key);
        Ok(())
    }

    /// Bulk-warm artefacts by transitioning `Cold` → `Loaded` for each key.
    ///
    /// Entries in any other state are left untouched.
    pub fn warm_portfolio(&mut self, keys: &[ArtifactKey]) {
        for key in keys {
            if let Some(entry) = self.entries.get_mut(key) {
                if entry.state == ArtifactResidencyState::Cold {
                    entry.state = ArtifactResidencyState::Loaded;
                    entry.last_used = Instant::now();
                }
            }
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Evict LRU entries until both slot and weight budgets are satisfied.
    fn evict_to_capacity(&mut self) {
        loop {
            // Check slot capacity.
            if self.entries.len() > self.max_slots as usize {
                if let Some(key) = self.find_lru_evictable() {
                    self.entries.remove(&key);
                    self.active_leases.remove(&key);
                    continue;
                }
            }

            // Check weight capacity.
            if self.exceeds_weight_budget() {
                if let Some(key) = self.find_lru_evictable() {
                    self.entries.remove(&key);
                    self.active_leases.remove(&key);
                    continue;
                }
            }

            break;
        }
    }

    /// Total memory footprint of all entries.
    fn total_weight(&self) -> u64 {
        self.entries
            .values()
            .map(|e| e.memory_footprint_bytes)
            .sum()
    }

    /// Whether the total resident weight exceeds the budget.
    fn exceeds_weight_budget(&self) -> bool {
        if self.max_weight_bytes == 0 {
            return false;
        }
        self.total_weight() > self.max_weight_bytes
    }

    /// Find the least-recently-used key that is eligible for eviction.
    /// Entries that are `Warmed` with an active lease are skipped.
    fn find_lru_evictable(&self) -> Option<ArtifactKey> {
        self.entries
            .iter()
            .filter(|(key, entry)| {
                !(entry.state == ArtifactResidencyState::Warmed
                    && self.active_leases.contains(key))
            })
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_key(i: u32) -> ArtifactKey {
        ArtifactKey {
            model_family: "llama".into(),
            packet_kind: "self_attn".into(),
            layer_start: i,
            layer_end: i + 1,
            shape_bucket: 0,
            precision: "fp16".into(),
        }
    }

    fn entry(state: ArtifactResidencyState) -> AneArtifactCacheEntry {
        AneArtifactCacheEntry {
            key: sample_key(0),
            state,
            last_used: Instant::now(),
            compile_latency: Duration::from_millis(500),
            warmup_latency: Duration::from_millis(100),
            steady_state_latency: Some(Duration::from_millis(20)),
            memory_footprint_bytes: 4_000_000,
        }
    }

    fn filled_cache(slots: u32) -> AneArtifactCache {
        let mut cache = AneArtifactCache::new(
            slots,
            1_000_000_000,
            AneEvictionPolicy::Lru,
        );
        for i in 0..slots {
            let mut e = entry(ArtifactResidencyState::Loaded);
            e.key = sample_key(i);
            cache.entries.insert(sample_key(i), e);
        }
        cache
    }

    // ── new ───────────────────────────────────────────────────────────

    #[test]
    fn test_new_cache_is_empty() {
        let cache = AneArtifactCache::new(10, 1_000_000, AneEvictionPolicy::Lru);
        assert_eq!(cache.max_slots, 10);
        assert_eq!(cache.max_weight_bytes, 1_000_000);
        assert!(cache.entries.is_empty());
        assert!(cache.active_leases.is_empty());
    }

    // ── insert / eviction ─────────────────────────────────────────────

    #[test]
    fn test_insert_evicts_lru() {
        let mut cache = filled_cache(3);

        // The LRU entry is the one with the oldest last_used.
        // Simulate ageing by giving each a different Instant (via sleep is
        // impractical, so we set last_used manually).
        {
            let mut timestamps: Vec<_> = cache.entries.iter_mut().collect();
            timestamps.sort_by_key(|(k, _)| k.layer_start);
            for (i, (_, e)) in timestamps.iter_mut().enumerate() {
                e.last_used = Instant::now() - Duration::from_secs(10 - i as u64);
            }
        }
        // key0 (layer_start=0) is now oldest.

        let new_key = sample_key(99);
        let new_entry = AneArtifactCacheEntry {
            key: new_key.clone(),
            state: ArtifactResidencyState::Loaded,
            last_used: Instant::now(),
            compile_latency: Duration::default(),
            warmup_latency: Duration::default(),
            steady_state_latency: None,
            memory_footprint_bytes: 1,
        };
        cache.insert(new_key, new_entry);

        // LRU entry (key0) should be gone.
        assert!(cache.entries.contains_key(&sample_key(1)));
        assert!(cache.entries.contains_key(&sample_key(2)));
        assert!(cache.entries.contains_key(&sample_key(99)));
        assert!(!cache.entries.contains_key(&sample_key(0)));
        assert_eq!(cache.entries.len(), 3);
    }

    // ── transition ────────────────────────────────────────────────────

    #[test]
    fn test_transition_loaded_to_warmed() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache.entries.insert(
            key.clone(),
            entry(ArtifactResidencyState::Loaded),
        );

        let result = cache.transition(&key, ArtifactResidencyState::Warmed);
        assert!(result.is_ok());
        assert_eq!(
            cache.get_state(&key),
            Some(ArtifactResidencyState::Warmed)
        );
    }

    #[test]
    fn test_transition_cold_to_warmed_rejected() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache
            .entries
            .insert(key.clone(), entry(ArtifactResidencyState::Cold));

        let result = cache.transition(&key, ArtifactResidencyState::Warmed);
        assert!(result.is_err());
        assert_eq!(
            cache.get_state(&key),
            Some(ArtifactResidencyState::Cold)
        );
    }

    // ── failed ────────────────────────────────────────────────────────

    #[test]
    fn test_failed_entry_not_admitted() {
        // A failed entry cannot transition to any other state.
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache
            .entries
            .insert(key.clone(), entry(ArtifactResidencyState::Failed("OOM".into())));

        let r = cache.transition(&key, ArtifactResidencyState::Loaded);
        assert!(r.is_err());

        let r = cache.transition(&key, ArtifactResidencyState::Evictable);
        assert!(r.is_err());

        // Still Failed.
        assert_eq!(
            cache.get_state(&key),
            Some(ArtifactResidencyState::Failed("OOM".into()))
        );
    }

    // ── in-flight protection ──────────────────────────────────────────

    #[test]
    fn test_in_flight_not_evicted() {
        let mut cache = AneArtifactCache::new(2, 0, AneEvictionPolicy::Lru);
        let key0 = sample_key(0);
        let key1 = sample_key(1);

        cache
            .entries
            .insert(key0.clone(), entry(ArtifactResidencyState::Warmed));
        cache.active_leases.insert(key0.clone());
        cache
            .entries
            .insert(key1.clone(), entry(ArtifactResidencyState::Warmed));

        // Try to insert a third entry — should evict. The lease-protected
        // key0 must stay.  Mark key1 as older so it becomes LRU.
        cache.entries.get_mut(&key1).unwrap().last_used =
            Instant::now() - Duration::from_secs(60);
        cache.entries.get_mut(&key0).unwrap().last_used = Instant::now();

        let key2 = sample_key(2);
        cache.insert(
            key2.clone(),
            AneArtifactCacheEntry {
                key: key2,
                state: ArtifactResidencyState::Loaded,
                last_used: Instant::now(),
                compile_latency: Duration::default(),
                warmup_latency: Duration::default(),
                steady_state_latency: None,
                memory_footprint_bytes: 1,
            },
        );

        // key0 stays (lease-protected), key1 was evicted (LRU).
        assert!(cache.entries.contains_key(&key0));
        assert!(!cache.entries.contains_key(&key1));
        assert_eq!(cache.entries.len(), 2);

        // Explicit evict on key0 must also fail.
        let r = cache.evict(&key0);
        assert!(r.is_err());
    }

    // ── warm_portfolio ────────────────────────────────────────────────

    #[test]
    fn test_warm_portfolio() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let keys: Vec<_> = (0..5).map(sample_key).collect();

        for k in &keys {
            cache
                .entries
                .insert(k.clone(), entry(ArtifactResidencyState::Cold));
        }

        // One entry already Loaded — should not be touched.
        let loaded_key = sample_key(99);
        cache.entries.insert(
            loaded_key.clone(),
            entry(ArtifactResidencyState::Loaded),
        );

        cache.warm_portfolio(&keys);

        for k in &keys {
            assert_eq!(
                cache.get_state(k),
                Some(ArtifactResidencyState::Loaded),
                "key {:?} should be Loaded after warm_portfolio",
                k,
            );
        }

        // The already-Loaded entry stays Loaded.
        assert_eq!(
            cache.get_state(&loaded_key),
            Some(ArtifactResidencyState::Loaded),
        );
    }

    // ── serde ─────────────────────────────────────────────────────────

    #[test]
    fn test_serde_roundtrip() {
        let mut cache = AneArtifactCache::new(5, 2_000_000, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache.entries.insert(
            key.clone(),
            AneArtifactCacheEntry {
                key: key.clone(),
                state: ArtifactResidencyState::Loaded,
                last_used: Instant::now(),
                compile_latency: Duration::from_millis(500),
                warmup_latency: Duration::from_millis(100),
                steady_state_latency: Some(Duration::from_millis(20)),
                memory_footprint_bytes: 1_000_000,
            },
        );

        // Serialize individual entry (HashMap with ArtifactKey key can't use standard JSON)
        let entry = cache.entries.get(&key).unwrap();
        let entry_json = serde_json::to_string(entry).expect("serialize entry");
        let restored: AneArtifactCacheEntry =
            serde_json::from_str(&entry_json).expect("deserialize entry");
        assert_eq!(restored.key, entry.key);
        assert_eq!(restored.memory_footprint_bytes, entry.memory_footprint_bytes);
    }

    // ── Additional coverage ───────────────────────────────────────────

    #[test]
    fn test_mark_warmed_validates_state() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache
            .entries
            .insert(key.clone(), entry(ArtifactResidencyState::Cold));

        let r = cache.mark_warmed(&key, Duration::from_millis(50));
        assert!(r.is_err());
    }

    #[test]
    fn test_mark_failed_any_state() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        for state in [
            ArtifactResidencyState::Cold,
            ArtifactResidencyState::Compiling,
            ArtifactResidencyState::Loaded,
            ArtifactResidencyState::Warmed,
        ] {
            let key = sample_key(42);
            cache.entries.clear();
            cache.entries.insert(key.clone(), entry(state));
            cache.mark_failed(&key, "hardware_error".into());
            assert_eq!(
                cache.get_state(&key),
                Some(ArtifactResidencyState::Failed("hardware_error".into()))
            );
        }
    }

    #[test]
    fn test_mark_failed_idempotent() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache.entries.insert(
            key.clone(),
            entry(ArtifactResidencyState::Failed("OOM".into())),
        );
        // Second mark should be a no-op.
        cache.mark_failed(&key, "new_error".into());
        assert_eq!(
            cache.get_state(&key),
            Some(ArtifactResidencyState::Failed("OOM".into()))
        );
    }

    #[test]
    fn test_evict_absent_key() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let r = cache.evict(&sample_key(99));
        assert!(r.is_err());
    }

    #[test]
    fn test_warm_portfolio_skips_non_cold() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache
            .entries
            .insert(key.clone(), entry(ArtifactResidencyState::Compiling));
        cache.warm_portfolio(&[key.clone()]);
        // Should remain Compiling.
        assert_eq!(
            cache.get_state(&key),
            Some(ArtifactResidencyState::Compiling)
        );
    }

    #[test]
    fn test_full_forward_chain() {
        let mut cache = AneArtifactCache::new(10, 0, AneEvictionPolicy::Lru);
        let key = sample_key(0);
        cache
            .entries
            .insert(key.clone(), entry(ArtifactResidencyState::Cold));

        cache.transition(&key, ArtifactResidencyState::Compiling).unwrap();
        cache.transition(&key, ArtifactResidencyState::Loaded).unwrap();
        cache.mark_warmed(&key, Duration::from_millis(80)).unwrap();

        assert_eq!(
            cache.get_state(&key),
            Some(ArtifactResidencyState::Warmed)
        );
    }
}
