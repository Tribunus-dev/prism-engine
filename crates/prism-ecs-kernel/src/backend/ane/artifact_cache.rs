//! ANE artifact cache (constitutional home).
//!
//! Per the inventory v2.1 row 7, this replaces the engine's
//! `ane_artifact_cache.rs` (649 LOC). The full implementation
//! arrives with the engine migration.

use std::collections::BTreeMap;

#[derive(Debug, Default)]
pub struct AneArtifactCache {
    /// BTreeMap for stable iteration order. The receipt snapshot
    /// iterates the cache and the result must be deterministic.
    artifacts: BTreeMap<String, AneArtifactEntry>,
}

#[derive(Debug, Clone)]
pub struct AneArtifactEntry {
    pub artifact_digest: String,
    pub model_path: String,
}

impl AneArtifactCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, digest: String, model_path: String) {
        self.artifacts.insert(
            digest.clone(),
            AneArtifactEntry {
                artifact_digest: digest,
                model_path,
            },
        );
    }

    pub fn get(&self, digest: &str) -> Option<&AneArtifactEntry> {
        self.artifacts.get(digest)
    }

    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_cache_is_empty() {
        let c = AneArtifactCache::new();
        assert!(c.is_empty());
    }

    #[test]
    fn insert_and_get() {
        let mut c = AneArtifactCache::new();
        c.insert("digest1".into(), "/path/to/model.mlmodelc".into());
        let e = c.get("digest1").expect("entry present");
        assert_eq!(e.model_path, "/path/to/model.mlmodelc");
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn duplicate_digest_replaces() {
        // Architectural invariant: a second insert with the same
        // digest replaces the first (BTreeMap insert semantics).
        let mut c = AneArtifactCache::new();
        c.insert("d".into(), "old".into());
        c.insert("d".into(), "new".into());
        assert_eq!(c.len(), 1);
        assert_eq!(c.get("d").unwrap().model_path, "new");
    }
}
