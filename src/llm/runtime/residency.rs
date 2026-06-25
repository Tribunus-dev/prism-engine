// ── Prism LLM Inference — Weight Residency Manager ────────────────────────
//
// Manages loaded model weights in the unified memory pool. Tracks which
// CImage artifacts have been loaded, their visibility to Metal/Accelerate/
// CoreML execution lanes, and session-level lease counts for eviction
// eligibility.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use super::super::manifest::{LlmCapabilityManifest, LlmModelFamily, SessionId};
use super::super::server::{
    CoreMlVisibilityState, WeightEvictionStatus, WeightResidencyKey,
    WeightResidencyReceipt,
};

/// Handle tracking a loaded model's runtime state in unified memory.
///
/// Each entry corresponds to one WeightResidencyKey loaded into the
/// memory pool. The lease counter tracks how many active sessions depend
/// on these weights; eviction is permitted only when the count reaches zero.
// Fields below may only be read indirectly through WeightResidencyReceipt.
#[allow(dead_code)]
struct LoadedModelHandle {
    /// The full key for this residency entry (stored for receipt/eviction queries).
    key: WeightResidencyKey,
    /// Total bytes of weight data resident in unified memory.
    weight_bytes: u64,
    /// Whether weights are visible to Metal compute dispatches.
    metal_visible: bool,
    /// Whether weights are visible to Accelerate framework dispatches.
    accelerate_visible: bool,
    /// Visibility state of the weights from CoreML auxiliary islands.
    coreml_visibility: CoreMlVisibilityState,
    /// Number of active session leases on this weight residency.
    lease_count: u32,
    /// Tracks which sessions currently hold a lease (prevents double-count).
    session_leases: HashSet<SessionId>,
    /// True once load_cimage has completed.
    loaded: bool,
    /// Model family recorded from the manifest at load time.
    manifest_model_family: LlmModelFamily,
}

/// Manages weight residency for LLM inference.
///
/// Thread-safe: all interior state is Mutex-protected. Provides operations
/// to load CImage artifacts, pin/release weight allocations per session,
/// query residency receipts, and enumerate eviction-eligible entries.
///
/// Internally keyed by `cimage_digest` (which implements Hash + Eq) since
/// `WeightResidencyKey` does not derive those traits directly.
pub struct WeightResidencyManager {
    loaded: Mutex<HashMap<crate::image::types::ArtifactDigest, LoadedModelHandle>>,
}

impl WeightResidencyManager {
    /// Creates a new, empty WeightResidencyManager.
    pub fn new() -> Self {
        Self {
            loaded: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a reference to a handle by its cimage digest, or None.
    fn get_ref<'a>(
        map: &'a HashMap<crate::image::types::ArtifactDigest, LoadedModelHandle>,
        key: &WeightResidencyKey,
    ) -> Option<&'a LoadedModelHandle> {
        map.get(&key.cimage_digest)
    }

    /// Returns a mutable reference to a handle by its cimage digest, or None.
    fn get_mut_ref<'a>(
        map: &'a mut HashMap<crate::image::types::ArtifactDigest, LoadedModelHandle>,
        key: &WeightResidencyKey,
    ) -> Option<&'a mut LoadedModelHandle> {
        map.get_mut(&key.cimage_digest)
    }

    /// Simulates loading a CImage artifact's weights into unified memory.
    ///
    /// Records the artifact path and manifest metadata, then returns a
    /// synthetic `WeightResidencyKey` identifying the residency entry.
    pub fn load_cimage(
        &self,
        path: &str,
        manifest: &LlmCapabilityManifest,
    ) -> Result<WeightResidencyKey, String> {
        let key = WeightResidencyKey {
            cimage_digest: crate::image::types::ArtifactDigest(path.to_string()),
            tensor_manifest_digest: crate::image::types::ArtifactDigest(format!(
                "tensor:{path}"
            )),
            provider_kind: "runtime:llm".into(),
            dtype_profile: "default".into(),
        };

        let handle = LoadedModelHandle {
            key: key.clone(),
            weight_bytes: 0,
            metal_visible: true,
            accelerate_visible: true,
            coreml_visibility: CoreMlVisibilityState::Full,
            lease_count: 0,
            session_leases: HashSet::new(),
            loaded: true,
            manifest_model_family: manifest.model_family,
        };

        let digest = key.cimage_digest.clone();
        let mut map = self
            .loaded
            .lock()
            .map_err(|e| format!("weight residency lock poisoned: {e}"))?;

        map.insert(digest, handle);
        Ok(key)
    }

    /// Pins a weight residency for the given session, incrementing the
    /// lease counter.
    ///
    /// Idempotent: calling multiple times with the same session is a no-op.
    pub fn pin_for_session(
        &self,
        key: &WeightResidencyKey,
        session_id: &SessionId,
    ) -> Result<(), String> {
        let mut map = self
            .loaded
            .lock()
            .map_err(|e| format!("weight residency lock poisoned: {e}"))?;

        let handle = Self::get_mut_ref(&mut map, key)
            .ok_or_else(|| "weight residency key not found".to_string())?;

        if handle.session_leases.insert(*session_id) {
            handle.lease_count += 1;
        }
        Ok(())
    }

    /// Releases a session's lease on a weight residency, decrementing the
    /// lease counter.
    ///
    /// Returns an error if the session was not holding a lease on this key.
    pub fn release_session(
        &self,
        key: &WeightResidencyKey,
        session_id: &SessionId,
    ) -> Result<(), String> {
        let mut map = self
            .loaded
            .lock()
            .map_err(|e| format!("weight residency lock poisoned: {e}"))?;

        let handle = Self::get_mut_ref(&mut map, key)
            .ok_or_else(|| "weight residency key not found".to_string())?;

        if !handle.session_leases.remove(session_id) {
            return Err(
                "session not holding a lease on this weight residency".to_string(),
            );
        }
        handle.lease_count = handle.lease_count.saturating_sub(1);
        Ok(())
    }

    /// Returns a `WeightResidencyReceipt` for the given key.
    ///
    /// When the key is not found the receipt reports `cache_hit: false` and
    /// `eviction_status: Ineligible`; otherwise it reflects the current
    /// handle state with `cache_hit: true`.
    pub fn get_residency_receipt(&self, key: &WeightResidencyKey) -> WeightResidencyReceipt {
        let map = self.loaded.lock().unwrap_or_else(|e| e.into_inner());

        match Self::get_ref(&map, key) {
            Some(handle) => WeightResidencyReceipt {
                cimage_digest: key.cimage_digest.clone(),
                cache_hit: handle.loaded,
                initial_load_bytes: handle.weight_bytes,
                decode_step_reload_count: 0,
                active_weight_leases: handle.lease_count,
                metal_visible: handle.metal_visible,
                accelerate_visible: handle.accelerate_visible,
                coreml_auxiliary_visibility: handle.coreml_visibility,
                materialization_events: Vec::new(),
                eviction_status: WeightEvictionStatus::Retained,
            },
            None => WeightResidencyReceipt {
                cimage_digest: key.cimage_digest.clone(),
                cache_hit: false,
                initial_load_bytes: 0,
                decode_step_reload_count: 0,
                active_weight_leases: 0,
                metal_visible: false,
                accelerate_visible: false,
                coreml_auxiliary_visibility: CoreMlVisibilityState::NotVisible,
                materialization_events: Vec::new(),
                eviction_status: WeightEvictionStatus::Ineligible,
            },
        }
    }

    /// Returns all weight residency keys whose lease count is zero,
    /// making them eligible for eviction.
    pub fn eviction_eligible(&self) -> Vec<WeightResidencyKey> {
        let map = self.loaded.lock().unwrap_or_else(|e| e.into_inner());

        map.iter()
            .filter(|(_, handle)| handle.lease_count == 0)
            .map(|(_, handle)| handle.key.clone())
            .collect()
    }
}

impl Default for WeightResidencyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::types::ArtifactDigest;
    use crate::llm::manifest::{
        ComponentAvailability, KvCacheContract, KvDtype, LlmQualificationRecord,
        QualificationStatus, ResidencyRequirements, RopeMode,
    };

    fn test_manifest() -> LlmCapabilityManifest {
        LlmCapabilityManifest {
            schema_version: 1,
            model_family: LlmModelFamily::Qwen3,
            tokenizer: ComponentAvailability::PresentQualified,
            embedding: ComponentAvailability::PresentQualified,
            transformer_blocks: ComponentAvailability::PresentQualified,
            lm_head: ComponentAvailability::PresentQualified,
            kv_cache_contract: KvCacheContract {
                layer_count: 48,
                attention_head_count: 32,
                kv_head_count: 8,
                head_dimension: 128,
                dtype: KvDtype::Fp16,
                rope_mode: RopeMode::Standard,
                supports_sparse_retention: false,
                supports_context_refresh: false,
                supports_position_renumbering: false,
                max_declared_context_tokens: 131_072,
                page_token_capacity: 128,
            },
            supported_context_profiles: vec![],
            provider_artifacts: vec![],
            auxiliary_islands: vec![],
            residency_requirements: ResidencyRequirements {
                min_unified_memory_bytes: 8_589_934_592,
                persistent_weight_bytes: 4_294_967_296,
                scratch_bytes: 1_073_741_824,
                kv_reservation_per_token: 4096,
            },
            qualification: LlmQualificationRecord {
                status: QualificationStatus::Accepted,
                fixture_id: "test-fixture".into(),
                verified_at: "2026-06-24T00:00:00Z".into(),
                failure_reason: None,
            },
        }
    }

    #[test]
    fn test_new_is_empty() {
        let mgr = WeightResidencyManager::new();
        assert!(mgr.eviction_eligible().is_empty());
    }

    #[test]
    fn test_load_cimage_produces_valid_key() {
        let mgr = WeightResidencyManager::new();
        let manifest = test_manifest();
        let key = mgr.load_cimage("/models/test.cimage", &manifest).unwrap();

        // The key should have a digest derived from the path
        assert_eq!(key.cimage_digest.0, "/models/test.cimage");
        assert_eq!(key.provider_kind, "runtime:llm");

        // After loading, the receipt should report a cache hit
        let receipt = mgr.get_residency_receipt(&key);
        assert!(receipt.cache_hit);
        assert_eq!(receipt.active_weight_leases, 0);
    }

    #[test]
    fn test_pin_increments_lease() {
        let mgr = WeightResidencyManager::new();
        let manifest = test_manifest();
        let key = mgr.load_cimage("/models/test.cimage", &manifest).unwrap();
        let sid = SessionId(uuid::Uuid::new_v4());

        mgr.pin_for_session(&key, &sid).unwrap();

        let receipt = mgr.get_residency_receipt(&key);
        assert_eq!(receipt.active_weight_leases, 1);
    }

    #[test]
    fn test_release_decrements_lease() {
        let mgr = WeightResidencyManager::new();
        let manifest = test_manifest();
        let key = mgr.load_cimage("/models/test.cimage", &manifest).unwrap();
        let sid = SessionId(uuid::Uuid::new_v4());

        mgr.pin_for_session(&key, &sid).unwrap();
        mgr.pin_for_session(&key, &sid).unwrap(); // idempotent
        let receipt1 = mgr.get_residency_receipt(&key);
        assert_eq!(receipt1.active_weight_leases, 1);

        mgr.release_session(&key, &sid).unwrap();
        let receipt2 = mgr.get_residency_receipt(&key);
        assert_eq!(receipt2.active_weight_leases, 0);
    }

    #[test]
    fn test_release_without_pin_errors() {
        let mgr = WeightResidencyManager::new();
        let manifest = test_manifest();
        let key = mgr.load_cimage("/models/test.cimage", &manifest).unwrap();
        let sid = SessionId(uuid::Uuid::new_v4());

        let err = mgr.release_session(&key, &sid).unwrap_err();
        assert!(err.contains("not holding a lease"));
    }

    #[test]
    fn test_pin_unknown_key_errors() {
        let mgr = WeightResidencyManager::new();
        let key = WeightResidencyKey {
            cimage_digest: ArtifactDigest("unknown".into()),
            tensor_manifest_digest: ArtifactDigest("unknown".into()),
            provider_kind: "none".into(),
            dtype_profile: "none".into(),
        };
        let sid = SessionId(uuid::Uuid::new_v4());

        let err = mgr.pin_for_session(&key, &sid).unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_eviction_eligible_after_all_releases() {
        let mgr = WeightResidencyManager::new();
        let manifest = test_manifest();
        let key = mgr.load_cimage("/models/test.cimage", &manifest).unwrap();
        let sid = SessionId(uuid::Uuid::new_v4());

        // Loaded but not pinned -> eligible
        assert!(mgr.eviction_eligible().contains(&key));

        // Pin -> no longer eligible
        mgr.pin_for_session(&key, &sid).unwrap();
        assert!(!mgr.eviction_eligible().contains(&key));

        // Release -> eligible again
        mgr.release_session(&key, &sid).unwrap();
        assert!(mgr.eviction_eligible().contains(&key));
    }

    #[test]
    fn test_multiple_sessions_independent_leases() {
        let mgr = WeightResidencyManager::new();
        let manifest = test_manifest();
        let key = mgr.load_cimage("/models/test.cimage", &manifest).unwrap();
        let sid1 = SessionId(uuid::Uuid::new_v4());
        let sid2 = SessionId(uuid::Uuid::new_v4());

        mgr.pin_for_session(&key, &sid1).unwrap();
        mgr.pin_for_session(&key, &sid2).unwrap();

        let receipt = mgr.get_residency_receipt(&key);
        assert_eq!(receipt.active_weight_leases, 2);

        // Release one session
        mgr.release_session(&key, &sid1).unwrap();
        let receipt = mgr.get_residency_receipt(&key);
        assert_eq!(receipt.active_weight_leases, 1);
        assert!(!mgr.eviction_eligible().contains(&key)); // still pinned by sid2

        // Release the other
        mgr.release_session(&key, &sid2).unwrap();
        assert!(mgr.eviction_eligible().contains(&key));
    }

    #[test]
    fn test_get_receipt_for_unknown_key() {
        let mgr = WeightResidencyManager::new();
        let key = WeightResidencyKey {
            cimage_digest: ArtifactDigest("missing".into()),
            tensor_manifest_digest: ArtifactDigest("missing".into()),
            provider_kind: "none".into(),
            dtype_profile: "none".into(),
        };
        let receipt = mgr.get_residency_receipt(&key);
        assert!(!receipt.cache_hit);
        assert_eq!(receipt.active_weight_leases, 0);
        assert_eq!(receipt.eviction_status, WeightEvictionStatus::Ineligible);
    }

    #[test]
    fn test_default_creates_empty() {
        let mgr = WeightResidencyManager::default();
        assert!(mgr.eviction_eligible().is_empty());
    }
}

#[cfg(feature = "prism-backend")]
mod prism_backend {
    use std::collections::HashSet;
    use std::path::Path;
    use std::sync::Mutex;

    use tribunus_compute_core::kv_cache::KvCache;
    use tribunus_compute_core::profiled_executor::{
        LoadedProfiledModel, ProfiledInferenceSession,
     };
    use tribunus_compute_core::residency::ResidencyManager;
     
     use super::super::super::manifest::{LlmCapabilityManifest, SessionId};
     use super::super::super::server::{
        CoreMlVisibilityState, WeightEvictionStatus, WeightResidencyKey,
        WeightResidencyReceipt,
    };
    use crate::image::types::ArtifactDigest;

    /// Real weight-residency manager backed by compute-core types.
    ///
    /// Delegates model weight loading to [`ProfiledInferenceSession`] /
    /// [`LoadedProfiledModel`] and tracks segment lifecycle through
    /// [`ResidencyManager`].
    ///
    /// This type offers the same public API as the stub
    /// [`super::WeightResidencyManager`] but drives actual compute-core
    /// loading and memory-budget tracking instead of synthetic bookkeeping.
    ///
    /// Thread-safe: all mutable state is Mutex-protected.
    #[allow(dead_code)]
    pub struct ComputeWeightResidencyManager {
        /// The loaded model runtime (immutable, shared across sessions).
        model: Mutex<Option<LoadedProfiledModel>>,
        /// Inference session created during load — delegates weight loading
        /// to the underlying `ProfiledInferenceSession` lifecycle.
        session: Mutex<Option<ProfiledInferenceSession>>,
        /// Residency manager tracking memory budget and segment lifecycle.
        residency: Mutex<ResidencyManager>,
        /// The key returned by the most recent load_cimage call.
        loaded_key: Mutex<Option<WeightResidencyKey>>,
        /// Number of sessions currently holding a lease.
        lease_count: Mutex<u32>,
        /// Set of session IDs currently holding leases.
        session_leases: Mutex<HashSet<SessionId>>,
        /// Residency — memory budget bytes.
        memory_budget_bytes: u64,
        /// Residency — safety reserve bytes.
        safety_reserve_bytes: u64,
    }

    #[allow(dead_code)]
    impl ComputeWeightResidencyManager {
        /// Creates a new, empty `ComputeWeightResidencyManager`.
        ///
        /// `memory_budget_bytes` sets the total budget for weight residency;
        /// `safety_reserve_bytes` is kept free by the admission gate.
        pub fn new(memory_budget_bytes: u64, safety_reserve_bytes: u64) -> Self {
            Self {
                model: Mutex::new(None),
                session: Mutex::new(None),
                residency: Mutex::new(ResidencyManager::new(
                    memory_budget_bytes,
                    safety_reserve_bytes,
                )),
                loaded_key: Mutex::new(None),
                lease_count: Mutex::new(0),
                session_leases: Mutex::new(HashSet::new()),
                memory_budget_bytes,
                safety_reserve_bytes,
            }
        }

        /// Loads a CImage artifact's weights into unified memory.
        ///
        /// Delegates weight loading to compute-core's
        /// [`LoadedProfiledModel::new`] (used internally by
        /// [`ProfiledInferenceSession`]), then registers the segment with
        /// [`ResidencyManager`] for lifecycle tracking.
        pub fn load_cimage(
            &self,
            path: &str,
            manifest: &LlmCapabilityManifest,
        ) -> Result<WeightResidencyKey, String> {
            // 1. Delegate to LoadedProfiledModel (the weight-loading half of
            //    ProfiledInferenceSession).
            let loaded_model = LoadedProfiledModel::new(Path::new(path))
                .map_err(|e| {
                    format!("ComputeWeightResidencyManager::load_cimage: {e}")
                })?;

            let weight_bytes = loaded_model.materialized_bytes.max(1);
            let layer_count = loaded_model.layers.len();

            // 2. Create a ProfiledInferenceSession using the loaded model's
            //    KV-cache topology.  The session is created once here and
            //    stored for downstream use — callers retrieve it via
            //    `take_session()` when they need to run inference.
            let kv_caches: Vec<KvCache> = (0..layer_count)
                .map(|_| {
                    KvCache::new(
                        manifest
                            .kv_cache_contract
                            .page_token_capacity
                            .max(128) as u32,
                        manifest.kv_cache_contract.kv_head_count as u32,
                        manifest.kv_cache_contract.head_dimension as u32,
                        false,
                    )
                })
                .collect();

            let mut session =
                ProfiledInferenceSession::new(path.to_string(), kv_caches);
            session.setup_from_model(&loaded_model);

            let key = WeightResidencyKey {
                cimage_digest: ArtifactDigest(path.to_string()),
                tensor_manifest_digest: ArtifactDigest(format!("tensor:{path}")),
                provider_kind: "runtime:llm".into(),
                dtype_profile: "default".into(),
            };

            // 3. Register the weight segment with the residency manager.
            let segment_id = path.to_string();
            {
                let mut res = self
                    .residency
                    .lock()
                    .map_err(|_| "residency lock poisoned".to_string())?;
                res.request_prefetch(&segment_id, weight_bytes);
                res.bind_segment(&segment_id)
                    .map_err(|e| format!("residency bind failed: {e}"))?;
            }

            // 4. Store loaded artifacts.
            *self
                .model
                .lock()
                .map_err(|_| "model lock poisoned".to_string())? =
                Some(loaded_model);
            *self
                .session
                .lock()
                .map_err(|_| "session lock poisoned".to_string())? =
                Some(session);
            *self
                .loaded_key
                .lock()
                .map_err(|_| "loaded key lock poisoned".to_string())? =
                Some(key.clone());

            Ok(key)
        }

        /// Pins a weight residency for the given session, incrementing the
        /// lease counter.
        ///
        /// Idempotent: calling multiple times with the same session is a
        /// no-op. Delegates to [`ResidencyManager::mark_in_flight`] so the
        /// segment lifecycle reflects the active lease.
        pub fn pin_for_session(
            &self,
            key: &WeightResidencyKey,
            session_id: &SessionId,
        ) -> Result<(), String> {
            let mut leases = self
                .session_leases
                .lock()
                .map_err(|_| "session leases lock poisoned".to_string())?;

            if leases.insert(*session_id) {
                let mut count = self
                    .lease_count
                    .lock()
                    .map_err(|_| "lease count lock poisoned".to_string())?;
                *count += 1;

                let mut res = self
                    .residency
                    .lock()
                    .map_err(|_| "residency lock poisoned".to_string())?;
                let _ = res.mark_in_flight(&key.cimage_digest.0);
            }
            Ok(())
        }

        /// Releases a session's lease on a weight residency, decrementing
        /// the lease counter.
        ///
        /// Returns an error if the session was not holding a lease on this
        /// key. When the last lease is released the segment is retired from
        /// the residency manager via [`ResidencyManager::retire`].
        pub fn release_session(
            &self,
            key: &WeightResidencyKey,
            session_id: &SessionId,
        ) -> Result<(), String> {
            let mut leases = self
                .session_leases
                .lock()
                .map_err(|_| "session leases lock poisoned".to_string())?;

            if !leases.remove(session_id) {
                return Err(
                    "session not holding a lease on this weight residency"
                        .to_string(),
                );
            }

            let mut count = self
                .lease_count
                .lock()
                .map_err(|_| "lease count lock poisoned".to_string())?;
            *count = count.saturating_sub(1);

            if *count == 0 {
                let mut res = self
                    .residency
                    .lock()
                    .map_err(|_| "residency lock poisoned".to_string())?;
                res.retire(&key.cimage_digest.0);
            }

            Ok(())
        }

        /// Returns a [`WeightResidencyReceipt`] for the given key.
        ///
        /// Reports `cache_hit: true` when a model is loaded (the most recent
        /// load covering this key), together with the materialized byte count
        /// and current lease count.
        pub fn get_residency_receipt(
            &self,
            key: &WeightResidencyKey,
        ) -> WeightResidencyReceipt {
            let model_guard =
                self.model.lock().unwrap_or_else(|e| e.into_inner());
            let count = self
                .lease_count
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            let cache_hit = model_guard.is_some();
            let weight_bytes = model_guard
                .as_ref()
                .map(|m| m.materialized_bytes)
                .unwrap_or(0);

            WeightResidencyReceipt {
                cimage_digest: key.cimage_digest.clone(),
                cache_hit,
                initial_load_bytes: weight_bytes,
                decode_step_reload_count: 0,
                active_weight_leases: *count,
                metal_visible: true,
                accelerate_visible: true,
                coreml_auxiliary_visibility: CoreMlVisibilityState::Full,
                materialization_events: Vec::new(),
                eviction_status: if cache_hit {
                    WeightEvictionStatus::Retained
                } else {
                    WeightEvictionStatus::Ineligible
                },
            }
        }

        /// Returns all weight residency keys whose lease count is zero,
        /// making them eligible for eviction.
        pub fn eviction_eligible(&self) -> Vec<WeightResidencyKey> {
            let key_guard = self
                .loaded_key
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            let count = self
                .lease_count
                .lock()
                .unwrap_or_else(|e| e.into_inner());

            if *count == 0 {
                key_guard.iter().cloned().collect()
            } else {
                Vec::new()
            }
        }

        /// Takes the stored [`ProfiledInferenceSession`], leaving `None` in
        /// its place.  Callers that need to run inference on the loaded model
        /// acquire the session via this method and return it with
        /// [`return_session`].
        pub fn take_session(&self) -> Option<ProfiledInferenceSession> {
            self.session
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
        }

        /// Returns a [`ProfiledInferenceSession`] back to the manager after
        /// inference is complete.
        pub fn return_session(&self, session: ProfiledInferenceSession) {
            *self.session.lock().unwrap_or_else(|e| e.into_inner()) =
                Some(session);
        }

        /// Returns a reference to the stored loaded model, if any.
        pub fn model_ref(
            &self,
        ) -> Option<std::sync::MutexGuard<'_, Option<LoadedProfiledModel>>> {
            self.model.lock().ok()
        }

        /// Returns the memory budget for this manager.
        pub fn memory_budget(&self) -> u64 {
            self.memory_budget_bytes
        }

        /// Returns the safety reserve for this manager.
        pub fn safety_reserve(&self) -> u64 {
            self.safety_reserve_bytes
        }
    }
}
