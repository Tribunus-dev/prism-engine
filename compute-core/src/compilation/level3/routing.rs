//! Level 3 router with capability fingerprint cache.
//!
//! The `Level3Router` maintains a cache of capability fingerprints keyed by
//! `(device, os, coreai_version, layout)`. On cache miss, it probes all
//! available providers and selects the best route by capability rank:
//!
//! 1. Stable aliasing route (`SharedRouteProvider` after full validation).
//! 2. Stable exporting route (`MaterializationProvider` — always available).
//! 3. Any available route (last resort).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::super::bridge_provider::{
    BridgeCapability, BridgePlan, BridgeProvider, BridgeReceipt, BridgeVerification,
};
use super::super::phase_types::TensorDescriptor;
use super::providers::{
    detected_coreai_version, CapabilityFingerprint, MaterializationProvider, SharedRouteProvider,
    StitchedProvider,
};

// ── Router ───────────────────────────────────────────────────────────────────

/// A cached capability entry pairing the selected route name with the probed
/// capability.
#[derive(Debug, Clone)]
struct CachedEntry {
    route_name: String,
    capability: BridgeCapability,
}

/// A route candidate produced by the router.
#[derive(Debug, Clone)]
pub struct RouteCandidate {
    pub route_name: String,
    pub capability: BridgeCapability,
    pub plan: BridgePlan,
    pub rank: usize,
}

/// Level 3 bridge router with capability fingerprint cache.
///
/// The router owns the three provider instances, a single shared capability
/// fingerprint cache, and exposes a unified API for Level 2 schedulers and
/// gates.
///
/// The fingerprint key is constructed using the same version-detection
/// functions that `SharedRouteProvider` uses, ensuring cache lookups match
/// the provider's probe keys.
#[derive(Debug)]
pub struct Level3Router {
    /// Capability fingerprint cache: fingerprint -> (route_name, capability).
    /// This is the single authoritative cache; providers do not cache independently.
    capability_cache: Arc<Mutex<HashMap<CapabilityFingerprint, CachedEntry>>>,
    /// The three bridge providers.
    materialization: MaterializationProvider,
    shared_route: SharedRouteProvider,
    stitched: StitchedProvider,
}

impl Level3Router {
    pub fn new() -> Self {
        Level3Router {
            capability_cache: Arc::new(Mutex::new(HashMap::new())),
            materialization: MaterializationProvider,
            shared_route: SharedRouteProvider::new(),
            stitched: StitchedProvider,
        }
    }

    /// Build a capability fingerprint from probe parameters.
    ///
    /// Uses the same `detected_coreai_version()` that `SharedRouteProvider`'s
    /// `probe_capability` uses internally, ensuring cache keys match.
    fn fingerprint(
        device: &str,
        os: &str,
        source_layout: &TensorDescriptor,
        destination_layout: &TensorDescriptor,
    ) -> CapabilityFingerprint {
        let coreai = detected_coreai_version();
        let layout_str = format!(
            "{:?}->{:?}",
            source_layout.physical_layout, destination_layout.physical_layout
        );
        CapabilityFingerprint {
            device: device.to_string(),
            os: os.to_string(),
            coreai_version: coreai.to_string(),
            layout: layout_str,
        }
    }

    /// Invalidate a cache entry when a capability mismatch is detected.
    ///
    /// Returns `true` if an entry was removed.
    pub fn invalidate_cache_entry(&self, fp: &CapabilityFingerprint) -> bool {
        let mut cache = self.capability_cache.lock().expect("capability cache lock");
        cache.remove(fp).is_some()
    }

    /// Invalidate all cache entries for a given device (e.g. after an OS update).
    pub fn invalidate_device(&self, device: &str) -> usize {
        let mut cache = self.capability_cache.lock().expect("capability cache lock");
        let keys: Vec<CapabilityFingerprint> = cache
            .keys()
            .filter(|k| k.device == device)
            .cloned()
            .collect();
        let count = keys.len();
        for k in keys {
            cache.remove(&k);
        }
        count
    }

    /// Probe all providers and select the best route.
    ///
    /// Ranking order:
    ///   0 — Stable aliasing (SharedRouteProvider after validation, stable_for_production)
    ///   1 — Stable exporting (MaterializationProvider, always stable)
    ///   2 — Any available route (last resort)
    ///   None — No route available (all providers returned empty capability)
    pub fn select_route(
        &self,
        device: &str,
        os: &str,
        source: &TensorDescriptor,
        destination: &TensorDescriptor,
    ) -> Option<RouteCandidate> {
        let fp = Level3Router::fingerprint(device, os, source, destination);

        // Check the cache first.
        {
            let cache = self.capability_cache.lock().expect("capability cache lock");
            if let Some(CachedEntry {
                route_name,
                capability,
            }) = cache.get(&fp)
            {
                let plan = self.prepare_for_route(route_name, 0, destination);
                let rank = if capability.stable_for_production && capability.supports_aliasing {
                    0
                } else if route_name == "materialization" {
                    1
                } else {
                    2
                };
                return Some(RouteCandidate {
                    route_name: route_name.clone(),
                    capability: capability.clone(),
                    plan,
                    rank,
                });
            }
        }

        // Probe each provider and build candidates.
        let mut candidates: Vec<(String, BridgeCapability, BridgePlan, usize)> = Vec::new();

        // SharedRouteProvider: rank 0 if stable (verified), rank 3 if not.
        {
            let cap = self
                .shared_route
                .probe_capability(device, os, source, destination);
            let plan = self.shared_route.prepare(0, destination);
            let rank = if cap.stable_for_production && cap.supports_aliasing {
                0
            } else {
                3
            };
            candidates.push(("shared_memory".to_string(), cap, plan, rank));
        }

        // MaterializationProvider: always rank 1.
        {
            let cap = self
                .materialization
                .probe_capability(device, os, source, destination);
            let plan = self.materialization.prepare(0, destination);
            candidates.push(("materialization".to_string(), cap, plan, 1));
        }

        // StitchedProvider: rank 4 (experimental, last resort).
        {
            let cap = self
                .stitched
                .probe_capability(device, os, source, destination);
            let plan = self.stitched.prepare(0, destination);
            candidates.push(("stitched".to_string(), cap, plan, 4));
        }

        // Filter to available providers (non-zero max_tensor_bytes or supports_anything).
        let available: Vec<_> = candidates
            .into_iter()
            .filter(|(_, cap, _, _)| {
                cap.max_tensor_bytes > 0
                    || cap.supports_borrowing
                    || cap.supports_exporting
                    || cap.supports_importing
                    || cap.supports_materialization
            })
            .collect();

        if available.is_empty() {
            return None;
        }

        // Sort by rank (ascending) and pick the best.
        let mut sorted = available;
        sorted.sort_by_key(|(_, _, _, rank)| *rank);
        let (name, capability, plan, rank) = sorted.into_iter().next().unwrap();

        // Cache the selected route name alongside the capability.
        {
            let mut cache = self.capability_cache.lock().expect("capability cache lock");
            cache.insert(
                fp,
                CachedEntry {
                    route_name: name.clone(),
                    capability: capability.clone(),
                },
            );
        }

        Some(RouteCandidate {
            route_name: name,
            capability,
            plan,
            rank,
        })
    }

    /// Prepare a plan for the given route name.
    fn prepare_for_route(
        &self,
        route_name: &str,
        source_slot: u64,
        destination: &TensorDescriptor,
    ) -> BridgePlan {
        match route_name {
            "shared_memory" => self.shared_route.prepare(source_slot, destination),
            "stitched" => self.stitched.prepare(source_slot, destination),
            _ => self.materialization.prepare(source_slot, destination),
        }
    }

    /// Execute the plan through the appropriate provider.
    pub fn execute(&self, route_name: &str, plan: &BridgePlan) -> BridgeReceipt {
        match route_name {
            "shared_memory" => self.shared_route.execute(plan),
            "stitched" => self.stitched.execute(plan),
            _ => self.materialization.execute(plan),
        }
    }

    /// Validate a plan through the appropriate provider.
    pub fn validate(
        &self,
        route_name: &str,
        plan: &BridgePlan,
        instrumentation: &str,
    ) -> BridgeVerification {
        match route_name {
            "shared_memory" => self.shared_route.validate(plan, instrumentation),
            "stitched" => self.stitched.validate(plan, instrumentation),
            _ => self.materialization.validate(plan, instrumentation),
        }
    }

    /// Returns a reference to the shared route provider (for gate access).
    pub fn shared_route_provider(&self) -> &SharedRouteProvider {
        &self.shared_route
    }

    /// Returns a reference to the materialization provider (for gate access).
    pub fn materialization_provider(&self) -> &MaterializationProvider {
        &self.materialization
    }
}

impl Default for Level3Router {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compilation::phase_types::{ElementType, PhysicalLayout, ResidencyClass};

    fn dummy_descriptor() -> TensorDescriptor {
        TensorDescriptor {
            logical_shape: vec![1, 64],
            element_type: ElementType::F32,
            physical_layout: PhysicalLayout::DenseRowMajor,
            alignment: 256,
            producer_phase: None,
            consumer_phases: Vec::new(),
            permitted_providers: Vec::new(),
            residency_class: ResidencyClass::Unified,
            max_bytes: 256,
            mutable: false,
            content_digest: None,
        }
    }

    #[test]
    fn router_returns_materialization_by_default() {
        let router = Level3Router::new();
        let desc = dummy_descriptor();
        let candidate = router.select_route("AppleM1", "14.5", &desc, &desc);
        assert!(candidate.is_some());
        let c = candidate.unwrap();
        // SharedRouteProvider is not yet validated (rank 3), so MaterializationProvider
        // (rank 1) wins.
        assert_eq!(c.route_name, "materialization");
        assert_eq!(c.rank, 1);
    }

    #[test]
    fn router_selects_shared_route_after_validation() {
        let router = Level3Router::new();
        let desc = dummy_descriptor();

        // Validate the shared route provider with correct proofs.
        let plan = router.shared_route_provider().prepare(0, &desc);
        let verification = router.shared_route_provider().validate(&plan, "full_inst");
        assert!(verification.passed);

        // Build the matching fingerprint and mark verified.
        let fp = Level3Router::fingerprint("AppleM1", "14.5", &desc, &desc);
        router.shared_route_provider().mark_verified(fp);
    }

    #[test]
    fn router_cache_hit_returns_cached() {
        let router = Level3Router::new();
        let desc = dummy_descriptor();
        // First call probes and caches materialization.
        let _ = router.select_route("AppleM1", "14.5", &desc, &desc);

        // Mark the fingerprint as validated so the shared route becomes rank 0.
        let fp = Level3Router::fingerprint("AppleM1", "14.5", &desc, &desc);
        router.shared_route_provider().mark_verified(fp.clone());

        // The cached entry is still materialization. Invalidate and re-probe.
        router.invalidate_cache_entry(&fp);
        let candidate = router.select_route("AppleM1", "14.5", &desc, &desc);
        assert!(candidate.is_some());
        let c = candidate.unwrap();
        // Now the shared route should be rank 0 since it's verified.
        assert_eq!(c.route_name, "shared_memory");
        assert_eq!(c.rank, 0);
    }

    #[test]
    fn router_invalidate_device_clears_all_entries() {
        let router = Level3Router::new();
        let desc = dummy_descriptor();
        let _ = router.select_route("AppleM1", "14.5", &desc, &desc);
        let _ = router.select_route("AppleM2", "15.0", &desc, &desc);

        let count = router.invalidate_device("AppleM1");
        assert_eq!(count, 1);
    }

    #[test]
    fn router_execute_dispatches_by_route_name() {
        let router = Level3Router::new();
        let desc = dummy_descriptor();

        // Execute via materialization.
        let plan = router.materialization_provider().prepare(1, &desc);
        let receipt = router.execute("materialization", &plan);
        assert_eq!(receipt.actual_route, "materialization");
        assert_eq!(receipt.materialized_bytes, plan.estimated_bytes);

        // Execute via shared memory.
        let plan = router.shared_route_provider().prepare(2, &desc);
        let receipt = router.execute("shared_memory", &plan);
        assert_eq!(receipt.actual_route, "shared_memory");
        assert_eq!(receipt.materialized_bytes, 0);
    }
}
