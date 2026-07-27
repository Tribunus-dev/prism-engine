//! World resources and extensions used by the compilation pipeline.
//!
//! Single authority: world-level state shape — the resources and extensions
//! that the constitutional compilation pipeline attaches to a [`World`].
//! There is no behavior here; each newtype/struct is a small data carrier
//! read by the stage systems and the orchestrator.
//!
//! Per AGENTS.md, every authority-bearing value should be a newtype, not a
//! raw `String`/`u64`. The newtypes in this file are the canonical form for
//! those values on the constitutional compilation world.

use prism_ecs_core::entity::Entity;
use prism_ecs_source::{CanonicalSource, CanonicalSourceAdapter};

use crate::search::EvaluationStrategy;

// Re-export the event sink so callers can import it from `ecs::resources`
// while the canonical definition lives in the crate root.
pub use crate::VecEventSink;

// ===========================================================================
// Resource types
// ===========================================================================

/// Identifies the session entity in the world.
#[derive(Debug, Clone, Copy)]
pub struct SessionHandle(pub Entity);

/// Full ingress'd source model, stored as a world *extension*.
///
/// We use the extension mechanism (not resources) because [`CanonicalSource`]
/// is `Send + 'static` but not `Sync` (it embeds a
/// `Box<dyn TensorDataProvider>` which is only `Send`).  World extensions
/// require only `Send + 'static`.
///
/// Set by `system_detect_source`; consumed by `system_build_graph`,
/// `system_run_search`, `system_legalize`, and `system_emit_cimage`.
pub struct CurrentSource(pub CanonicalSource);

/// Registered source format adapters.
///
/// Must be set before calling `system_detect_source`.
pub struct SourceAdapterList(pub Vec<Box<dyn CanonicalSourceAdapter + 'static>>);

/// Optional evaluator strategy for the search phase.
pub struct EvaluatorOption(pub Option<Box<dyn EvaluationStrategy + 'static>>);

/// Optional namespaced specialized-model manifest supplied by the caller.
pub struct ModelManifestResource(pub crate::model_manifest::MultiModelManifest);

/// Extension stored on the world to bind an emitted CImage to its
/// producing plan's digest. Read by `system_certify` and by downstream
/// receipts.
#[derive(Debug, Clone)]
pub struct CImagePlanDigest(pub [u8; 32]);

// ===========================================================================
// Tests — each resource must be `Send + 'static` and round-trip through
// `add_resource` / `get_resource` / `set_extension`.
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::entity::EntityKind;
    use prism_ecs_core::world::World;

    #[test]
    fn session_handle_resource_round_trips() {
        let mut world = World::new();
        let spawned = world.spawn(EntityKind::Model, None).expect("spawn entity");
        let handle = SessionHandle(spawned.entity);
        world.add_resource(handle);

        let loaded = world.get_resource::<SessionHandle>();
        assert!(loaded.is_some(), "SessionHandle resource must be readable");
        assert_eq!(loaded.expect("just checked").0, spawned.entity);
    }

    #[test]
    fn source_adapter_list_holds_adapters() {
        // The struct itself is a thin newtype; verify it accepts an empty list
        // and that the list is owned.
        let list = SourceAdapterList(Vec::new());
        assert!(list.0.is_empty());
    }

    #[test]
    fn cimage_plan_digest_carries_bytes() {
        let digest = CImagePlanDigest([0xAAu8; 32]);
        assert_eq!(digest.0.len(), 32);
        assert!(digest.0.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn vec_event_sink_reexport_is_accessible() {
        // Verifies the re-export path resolves through `resources`.
        let sink = VecEventSink::new();
        assert!(sink.events().is_empty());
    }
}
