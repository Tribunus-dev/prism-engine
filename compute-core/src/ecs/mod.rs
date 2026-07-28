#![allow(deprecated)]
pub mod canonical;

pub mod adapter;
pub mod aot;
pub mod compile_session;
pub mod component;
pub mod config;
pub mod entity;
pub mod plan;
pub mod receipt_bus;
pub mod system;
#[cfg(test)]
mod tests;

pub mod agent;
pub mod amd_rocm;
pub mod analysis;
pub mod ane;
pub mod ane_bridge;
pub mod ane_compile;
pub mod ane_keepalive;
pub mod ane_runtime;
pub mod arena;
pub mod arena_info;
pub mod arena_lifecycle;
pub mod arena_pool;
pub mod assessment;
pub mod attention;
pub mod audio_preprocess_accelerate;
pub mod audio_provider;
pub mod autopsy;
pub mod backend;
pub mod benchmark;
pub mod bitnet;
pub mod cache;
pub mod calibration;
pub mod candle_cpu_backend;
pub mod capability;
pub mod cimage;
pub mod cimage_runtime;
pub mod cli;
pub mod compilation;
pub mod compile;
pub mod compile_pipeline;
pub mod compile_progress;
pub mod compile_run;
pub mod compile_state;
pub mod compiler;
pub mod compute_image;
pub mod compute_image_v0;
pub mod compute_ir;
pub mod compute_lane;
pub mod compute_service;
pub mod config_namespace;
// `constitutional/` was deleted in Phase 1 (already absorbed into
// `prism-ecs-constitutional`); see
// `changelogs/2026-07-25-compute-core-absorption-phase-0-1.md`.
pub mod contracts;
pub mod copy_ledger;
pub mod core;
pub mod coreai;
pub mod coreai_audit;
pub mod coreai_bridge;
pub mod coreai_pipeline;
pub mod coreai_state;
pub mod cpu_benchmarks;
pub mod cpu_runtime;
pub mod cpu_worker_pool;
pub mod crash_breadcrumb;
pub mod decode_attribution;
pub mod device;
pub mod diffusion;
pub mod diffusion_provider;
pub mod editing;
pub mod engine;
pub mod engine_error;
pub mod engine_policy;
pub mod engine_receipts;
pub mod errors;
pub mod evidence;
pub mod execution_profile;
pub mod executor;
pub mod executor_projection;
pub mod exo;
pub mod experiment;
pub mod external_array;
pub mod ffi;
pub mod fusion_region;
pub mod gemma;
pub mod generation;
pub mod gguf;
pub mod gpu_memory;
pub mod gpu_worker;
pub mod heterogeneous;
pub mod hybrid_profile;
pub mod image_provider;
// `inference/` was deleted in the engine-subsystem deletion pass; the
// canonical home for per-image / per-session / per-step inference state
// is `prism_ecs_server::inference_state` (see
// `changelogs/2026-07-27-engine-subsystem-deletion-inference.md`).
// `inference_profile/` (TAIP) was deleted in Phase 1; superseded by the
// constitutional phase-graph surface. See
// `changelogs/2026-07-25-compute-core-absorption-phase-0-1.md`.
pub mod integration;
pub mod kv_arena;
// `kv_cache/` was deleted in Phase 1 (already absorbed into `prism-kv-cache`).
pub mod kv_cache_types;
pub mod layout_compiler;
pub mod layout_transform;
pub mod loader;
pub mod logging;
pub mod lora;
pub mod lut;
pub mod mapped_image;
pub mod memory;
pub mod metal_backend;
pub mod metal_capture;
pub mod metal_launcher;
pub mod mlir;
// Metal backend compiler — gated behind macOS + metal-dispatch
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub mod metrics;
pub mod mil_builder;
pub mod mlpackage;
pub mod mlx_api_compat;
pub mod mlx_executor;
pub mod mlx_inventory;
pub mod mlx_patch_register;
pub mod mlx_runtime_probe;
pub mod model;
pub mod model_cache;
pub mod model_runtime;
pub mod model_store;
pub mod mtp;
pub mod native_kernel;
pub mod nf4tile640;
pub mod operation_catalog;
pub mod parsing;
pub mod pg_receipt_subscriber;
pub mod pipeline_parity;
pub mod placement_profile;
pub mod plugin;
pub mod primitives;
pub mod profile_compiler;
pub mod profiled_executor;
pub mod profiled_model;
pub mod projection_executor;
pub mod projection_identity;
pub mod projection_tests;
pub mod projection_types;
#[cfg(test)]
pub mod quant_abi_test;
// `quantization/` was deleted in Phase 1 (already absorbed into
// `prism-ecs-quantization`); see
// `changelogs/2026-07-25-compute-core-absorption-phase-0-1.md`.
pub mod quantized;
pub mod readiness_gates;
pub mod reasoning_evidence;
pub mod receipt;
pub mod receipts;
pub mod registry;
pub mod replay_projection;
pub mod requalification;
pub mod research;
pub mod research_contracts;
pub mod research_metrics;
pub mod research_trace;
pub mod residency;
pub mod ring;
pub mod runtime;
pub mod runtime_contract;
pub mod runtime_orchestration;
pub mod runtime_trace;
pub mod scheduling;
pub mod server;
pub mod session;
pub mod sidecar;
pub mod speculative;
pub mod state_store;
pub mod storage_adapters;
pub mod storage_kernel;
pub mod streaming;
pub mod supervisor_crash;
pub mod ternary;
pub mod tokenizer;
pub mod toolchain_attest;
pub mod tools;
pub mod training_target;
pub mod transform_recipe;
pub mod treatment;
pub mod tts;
pub mod validator;
pub mod valkey_projection;
pub mod video;
pub mod video_provider;
pub mod vision;
pub mod weight_codec;
pub mod worker_crash_ledger;
pub mod worker_dispatch;
pub mod worker_memory;
pub mod worker_protocol;
pub use component::aot::*;
pub use component::backend::*;
pub use component::executor::*;
pub use component::memory::*;
pub use component::quality::*;
pub use component::tensor::*;
pub use prism_ecs_core::*;

use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::schema::SchemaCatalogue;
pub use crate::ecs::constitutional::world_txn::{
    CommitReceipt, CommittedEpoch, ComponentChange, PreparedWorldTxn, WorldTxn, WorldTxnError,
};
#[cfg(feature = "mlx-backend")]
pub use core::bridge;

#[deprecated(note = "use Entity(u64, u32) for generation safety")]
pub type EntityId = u64;

/// Legacy alias for compatibility during migration.
#[deprecated(note = "use World instead")]
pub type CompWorld = World;

// EntityRef is now in prism-ecs-core (re-exported via prism_ecs_core::* above).
// World is now in prism-ecs-core (re-exported via prism_ecs_core::* above).

/// Phase in the compiler pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulePhase {
    ModelLoading,
    Quantization,
    QuantizationPlanning,
    MemoryPlanning,
    FusionDispatch,
    KernelGeneration,
    Compilation,
    Packaging,
    Validation,
    Execution,
}

/// A compiler pass over the ECS world.
pub trait CompilerSystem: Send + Sync {
    fn name(&self) -> &str;
    fn phase(&self) -> SchedulePhase;
    fn run(&self, world: &mut World) -> anyhow::Result<()>;
}

// ═════════════════════════════════════════════════════════════════════════════
// Extension traits — compute-core-specific World methods stored via
// type-erased extensions on prism_ecs_core::World.
// ═════════════════════════════════════════════════════════════════════════════

use prism_ecs_constitutional::WorldTransitExt;

/// Extension trait providing system-management methods on [`World`].
///
/// Systems are stored as `Vec<Box<dyn CompilerSystem>>` via the type-erased
/// extension mechanism (`World::set_extension` / `World::get_extension`).
pub trait WorldSystemsExt {
    fn add_system(&mut self, system: Box<dyn CompilerSystem>);
    fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()>;
    fn system_count(&self) -> usize;
}

impl WorldSystemsExt for World {
    fn add_system(&mut self, system: Box<dyn CompilerSystem>) {
        if self
            .get_extension::<Vec<Box<dyn CompilerSystem>>>()
            .is_none()
        {
            self.set_extension(Vec::<Box<dyn CompilerSystem>>::new());
        }
        self.get_extension_mut::<Vec<Box<dyn CompilerSystem>>>()
            .expect("systems extension just initialized")
            .push(system);
    }

    fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()> {
        // Take systems, partition, restore unmatched — then drop the borrow
        let matched = {
            let systems = self
                .get_extension_mut::<Vec<Box<dyn CompilerSystem>>>()
                .expect("run_phase requires systems extension — call add_system first");
            let prev_systems = std::mem::take(systems);
            let (matched, unmatched): (Vec<_>, Vec<_>) =
                prev_systems.into_iter().partition(|s| s.phase() == phase);
            *systems = unmatched;
            matched
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for system in &matched {
                system.run(self)?;
            }
            self.commit_stage();
            Ok::<_, anyhow::Error>(())
        }));
        // Re-borrow self to push matched systems back
        let systems = self
            .get_extension_mut::<Vec<Box<dyn CompilerSystem>>>()
            .expect("systems extension just initialized");
        for sys in matched {
            systems.push(sys);
        }
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => {
                self.staging.clear();
                Err(e.context("system returned error (deferred component inserts discarded)"))
            }
            Err(panic) => {
                self.staging.clear();
                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(anyhow::anyhow!(
                    "System panicked (staged inserts discarded): {msg}"
                ))
            }
        }
    }

    fn system_count(&self) -> usize {
        self.get_extension::<Vec<Box<dyn CompilerSystem>>>()
            .map(|s| s.len())
            .unwrap_or(0)
    }
}

/// Extension trait providing transactional/constitutional methods on [`World`].
///
/// Epoch, journal, and committed-events state are stored via the type-erased
/// extension mechanism so that [`prism_ecs_core::World`] has no direct
/// dependency on constitutional types.
pub trait WorldConstitutionalExt {
    fn last_journal(&self) -> &[ComponentChange];
    fn last_committed_events(&self) -> &[DomainEvent];
    fn drain_committed_events(&mut self) -> Vec<DomainEvent>;
    fn transit(&mut self, txn: WorldTxn) -> Result<CommittedEpoch, WorldTxnError>;
    fn prepare(
        &self,
        txn: WorldTxn,
        catalogue: Option<&SchemaCatalogue>,
    ) -> Result<PreparedWorldTxn, WorldTxnError>;
    fn apply_prepared(&mut self, prepared: PreparedWorldTxn) -> CommitReceipt;
}

impl WorldConstitutionalExt for World {
    fn last_journal(&self) -> &[ComponentChange] {
        self.get_extension::<Vec<ComponentChange>>()
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    fn last_committed_events(&self) -> &[DomainEvent] {
        self.get_extension::<Vec<DomainEvent>>()
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    fn drain_committed_events(&mut self) -> Vec<DomainEvent> {
        self.get_extension_mut::<Vec<DomainEvent>>()
            .map(std::mem::take)
            .unwrap_or_default()
    }

    fn transit(&mut self, txn: WorldTxn) -> Result<CommittedEpoch, WorldTxnError> {
        let prepared = WorldTransitExt::prepare(self, txn, None)?;
        let receipt = WorldTransitExt::apply_prepared(self, prepared);
        Ok(CommittedEpoch(receipt.committed_epoch))
    }

    fn prepare(
        &self,
        txn: WorldTxn,
        catalogue: Option<&SchemaCatalogue>,
    ) -> Result<PreparedWorldTxn, WorldTxnError> {
        <World as WorldTransitExt>::prepare(self, txn, catalogue)
    }

    /// Atomically apply a validated, prepared transaction.
    ///
    /// # Panics
    /// - If the world epoch does not match the prepared transaction's expected epoch.
    ///
    /// After the first mutation, no recoverable errors remain — all invariants
    /// were validated during ::prepare().
    fn apply_prepared(&mut self, prepared: PreparedWorldTxn) -> CommitReceipt {
        <World as WorldTransitExt>::apply_prepared(self, prepared)
    }
}

#[cfg(test)]
mod entity_tests {
    use super::*;

    #[test]
    fn zero_entity_is_invalid() {
        let zero = Entity(0, 0);
        assert_eq!(zero.id(), 0);
        assert_eq!(zero.generation(), 0);
    }

    #[test]
    fn entity_id_and_generation_accessors() {
        let e = Entity(42, 3);
        assert_eq!(e.id(), 42);
        assert_eq!(e.generation(), 3);
    }

    #[test]
    fn entity_serialization_round_trip() {
        let e = Entity(100, 5);
        let json = serde_json::to_string(&e).unwrap();
        let deserialized: Entity = serde_json::from_str(&json).unwrap();
        assert_eq!(e, deserialized);
        assert_eq!(deserialized.id(), 100);
        assert_eq!(deserialized.generation(), 5);
    }

    #[test]
    fn entity_partial_eq_by_id_and_generation() {
        let a = Entity(1, 0);
        let b = Entity(1, 1);
        let c = Entity(2, 0);
        assert_ne!(a, b, "different generations must not be equal");
        assert_ne!(a, c, "different IDs must not be equal");
        assert_eq!(a, Entity(1, 0));
    }

    #[test]
    fn entity_hash_consistency() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Entity(10, 0));
        set.insert(Entity(10, 1));
        set.insert(Entity(20, 0));
        assert_eq!(set.len(), 3);
        assert!(set.contains(&Entity(10, 0)));
        assert!(set.contains(&Entity(10, 1)));
    }

    #[test]
    fn entity_copy_trait() {
        let a = Entity(5, 2);
        let b = a;
        assert_eq!(a, b);
    }

    #[test]
    fn entity_from_comp_entity_sets_generation_zero() {
        let ce = CompEntity(42);
        let e = Entity::from(ce);
        assert_eq!(e.id(), 42);
        assert_eq!(e.generation(), 0);
    }
}
