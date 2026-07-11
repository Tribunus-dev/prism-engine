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
pub mod assistant_graph;
pub mod attention;
pub mod audio;
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
pub mod constitutional;
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
pub mod duckdb_projection;
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
pub mod inference;
pub mod inference_profile;
pub mod integration;
pub mod kv_arena;
pub mod kv_cache;
pub mod kv_cache_types;
pub mod layout_compiler;
pub mod layout_transform;
pub mod loader;
pub mod logging;
pub mod lora;
pub mod lut;
pub mod mapped_image;
pub mod memory;
pub mod metal_capture;
pub mod metal_launcher;
pub mod metal_runtime;
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
pub mod models;
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
pub mod quant_abi_test;
pub mod quantization;
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

use crate::ecs::constitutional::command::DomainEvent;
use crate::ecs::constitutional::types::{SchemaVersion, WorldEpoch};
use crate::ecs::constitutional::world_txn::{
    ChangeType, CommittedEpoch, ComponentChange, WorldTxn, WorldTxnError,
};
use serde::{Deserialize, Serialize};
use std::any::{Any, TypeId};
use std::collections::HashMap;

pub type EntityId = u64;

/// Opaque entity handle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompEntity(pub EntityId);

/// Tag trait for data attached to entities.
pub trait Component: std::fmt::Debug + Send + Sync + 'static {}

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
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()>;
}

/// The ECS world — all entities, components, and systems.
pub struct CompWorld {
    component_store: ComponentStore,
    resource_store: ResourceStore,
    systems: Vec<Box<dyn CompilerSystem>>,
    entity_meta: Vec<EntityMeta>,
    next_id: u64,
    staging: Vec<Box<dyn FnOnce(&mut ComponentStore) + Send + 'static>>,
    epoch: WorldEpoch,
    journal: Vec<ComponentChange>,
    component_versions: std::collections::HashMap<u64, u64>,
    committed_events: Vec<DomainEvent>,
}

impl std::fmt::Debug for CompWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompWorld")
            .field("entity_count", &self.entity_meta.len())
            .field("system_count", &self.systems.len())
            .field("staged_changes", &self.staging.len())
            .finish()
    }
}

/// Manages the lifecycle of a single executable argument region.
#[derive(Debug)]
struct EntityMeta {
    kind: EntityKind,
    #[allow(dead_code)]
    generation: u32,
    name: Option<String>,
}

impl Default for EntityMeta {
    fn default() -> Self {
        Self {
            kind: EntityKind::Model,
            #[allow(dead_code)]
            generation: 0,
            name: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    Model,
    Tensor,
    Layer,
    Expert,
    Dispatch,
    Kernel,
    KernelVariant,
    Buffer,
    CommandBuffer,
    Executable,
    Fence,
    Session,
    Artifact,
    Device,
    Residency,
    Agent,
}

/// Type-erased storage for components, indexed by (TypeId, EntityId).
#[derive(Debug)]
pub struct ComponentStore {
    data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

/// Type-erased storage for global resources (not per-entity).
#[derive(Debug)]
pub struct ResourceStore {
    data: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl Default for ResourceStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl Default for ComponentStore {
    fn default() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
}

impl CompWorld {
    /// Spawn entity with kind and optional name.
    pub fn spawn(&mut self, kind: EntityKind, name: Option<String>) -> CompEntity {
        let entity = self.spawn_entity(kind);
        if let Some(n) = name {
            let idx = (entity.0 - 1) as usize;
            if let Some(meta) = self.entity_meta.get_mut(idx) {
                meta.name = Some(n);
            }
        }
        entity
    }

    /// Get the name of an entity.
    pub fn name(&self, entity: CompEntity) -> Option<&str> {
        if entity.0 == 0 {
            return None;
        }
        let idx = (entity.0 - 1) as usize;
        self.entity_meta.get(idx).and_then(|m| m.name.as_deref())
    }

    /// Find all entities of a given kind.
    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<CompEntity> {
        self.entity_meta
            .iter()
            .enumerate()
            .filter(|(_, meta)| meta.kind == kind)
            .map(|(i, _)| CompEntity((i + 1) as u64))
            .collect()
    }

    /// Get the kind of an entity (alias for entity_kind).
    pub fn kind(&self, entity: CompEntity) -> Option<EntityKind> {
        if entity.0 == 0 {
            return None;
        }
        self.entity_kind(entity)
    }

    pub fn remove_component<T: Component>(&mut self, entity: CompEntity) -> Option<T> {
        let store = &mut self.component_store;
        let type_id = TypeId::of::<T>();
        store
            .data
            .get_mut(&type_id)?
            .downcast_mut::<HashMap<EntityId, T>>()
            .and_then(|map| map.remove(&entity.0))
    }

    pub fn new() -> Self {
        Self {
            component_store: ComponentStore::default(),
            resource_store: ResourceStore::default(),
            systems: Vec::new(),
            entity_meta: Vec::new(),
            next_id: 1,
            staging: Vec::new(),
            epoch: WorldEpoch(1),
            journal: Vec::new(),
            component_versions: std::collections::HashMap::new(),
            committed_events: Vec::new(),
        }
    }

    /// Returns the next entity ID that will be assigned, without consuming it.
    pub fn next_entity_id(&self) -> u64 {
        self.next_id
    }

    /// Spawn an entity at a specific reserved ID (used by WorldTxn during commit).
    ///
    /// Idempotent: if the entity slot already exists at this ID, the call is
    /// a no-op. This allows phase 1aa (reservation) and phase 3c (apply) to
    /// both call it without double-pushing entity metadata.
    pub fn spawn_entity_with_id(&mut self, id: u64, kind: EntityKind) -> CompEntity {
        let idx = (id - 1) as usize;
        if idx < self.entity_meta.len() {
            // Already reserved — idempotent.
            return CompEntity(id);
        }
        // Fill gap if needed (spawns may be out of order from reservation)
        while self.entity_meta.len() <= idx {
            self.entity_meta.push(EntityMeta {
                kind,
                generation: 0,
                name: None,
            });
        }
        if id > self.next_id {
            self.next_id = id + 1;
        }
        CompEntity(id)
    }

    pub fn spawn_entity(&mut self, kind: EntityKind) -> CompEntity {
        let id = self.next_id;
        self.next_id += 1;
        self.entity_meta.push(EntityMeta {
            kind,
            generation: 0,
            name: None,
        });
        CompEntity(id)
    }

    pub fn entity_kind(&self, entity: CompEntity) -> Option<EntityKind> {
        if entity.0 == 0 {
            return None;
        }
        let idx = (entity.0 - 1) as usize;
        self.entity_meta.get(idx).map(|m| m.kind)
    }

    pub fn add_component<T: Component>(&mut self, entity: CompEntity, component: T) {
        let store = &mut self.component_store;
        let type_id = TypeId::of::<T>();
        let map: &mut HashMap<EntityId, T> = store
            .data
            .entry(type_id)
            .or_insert_with(|| Box::new(HashMap::<EntityId, T>::new()))
            .downcast_mut::<HashMap<EntityId, T>>()
            .expect("type mismatch in ComponentStore");
        map.insert(entity.0, component);
    }

    pub fn stage_component<T: Component>(&mut self, entity: CompEntity, component: T) {
        self.staging
            .push(Box::new(move |store: &mut ComponentStore| {
                let type_id = TypeId::of::<T>();
                let map: &mut HashMap<EntityId, T> = store
                    .data
                    .entry(type_id)
                    .or_insert_with(|| Box::new(HashMap::<EntityId, T>::new()))
                    .downcast_mut::<HashMap<EntityId, T>>()
                    .expect("type mismatch in ComponentStore");
                map.insert(entity.0, component);
            }));
    }

    pub fn commit_stage(&mut self) {
        let staging = std::mem::take(&mut self.staging);
        for op in staging {
            op(&mut self.component_store);
        }
    }

    /// Discard deferred component insert operations added via [`stage_component`].
    ///
    /// **This is NOT a transactional world rollback.** It only clears the staging
    /// queue. Systems that performed direct mutations via [`add_component`],
    /// [`remove_component`], [`get_component_mut`], or [`spawn`] before returning
    /// an error are NOT reverted. Use [`WorldTxn`] (when available) for atomic
    /// state transitions.
    pub fn rollback_stage(&mut self) {
        self.staging.clear();
    }

    pub fn get_component<T: Component>(&self, entity: CompEntity) -> Option<&T> {
        let store = &self.component_store;
        let type_id = TypeId::of::<T>();
        store
            .data
            .get(&type_id)?
            .downcast_ref::<HashMap<EntityId, T>>()
            .and_then(|map| map.get(&entity.0))
    }

    pub fn get_component_mut<T: Component>(&mut self, entity: CompEntity) -> Option<&mut T> {
        let store = &mut self.component_store;
        let type_id = TypeId::of::<T>();
        store
            .data
            .get_mut(&type_id)?
            .downcast_mut::<HashMap<EntityId, T>>()
            .and_then(|map| map.get_mut(&entity.0))
    }

    pub fn add_resource<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.resource_store
            .data
            .insert(TypeId::of::<T>(), Box::new(resource));
    }

    pub fn get_resource<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.resource_store
            .data
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    pub fn get_resource_mut<T: 'static + Send + Sync>(&mut self) -> Option<&mut T> {
        self.resource_store
            .data
            .get_mut(&TypeId::of::<T>())
            .and_then(|b| b.downcast_mut::<T>())
    }

    pub fn add_system(&mut self, system: Box<dyn CompilerSystem>) {
        self.systems.push(system);
    }

    pub fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()> {
        let prev_systems = std::mem::take(&mut self.systems);
        let (matched, unmatched): (Vec<_>, Vec<_>) =
            prev_systems.into_iter().partition(|s| s.phase() == phase);
        self.systems = unmatched;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for system in &matched {
                system.run(self)?;
            }
            self.commit_stage();
            Ok::<_, anyhow::Error>(())
        }));
        for sys in matched {
            self.systems.push(sys);
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

    pub fn entity_count(&self) -> usize {
        self.entity_meta.len()
    }

    pub fn system_count(&self) -> usize {
        self.systems.len()
    }
}

impl CompWorld {
    pub fn current_epoch(&self) -> WorldEpoch {
        self.epoch
    }

    pub fn last_journal(&self) -> &[ComponentChange] {
        &self.journal
    }

    pub fn last_committed_events(&self) -> &[DomainEvent] {
        &self.committed_events
    }

    /// Check whether an entity exists in the world.
    pub fn has_entity(&self, entity: CompEntity) -> bool {
        if entity.0 == 0 {
            return false;
        }
        let idx = (entity.0 - 1) as usize;
        idx < self.entity_meta.len()
    }

    pub fn drain_committed_events(&mut self) -> Vec<DomainEvent> {
        std::mem::take(&mut self.committed_events)
    }
    pub fn transit(&mut self, txn: WorldTxn) -> Result<CommittedEpoch, WorldTxnError> {
        // 1a. Validate epoch
        if self.epoch != txn.expected_epoch {
            return Err(WorldTxnError::StaleEpoch {
                expected: txn.expected_epoch,
                current: self.epoch,
            });
        }

        // 1aa. Pre-validate staged spawns and reserve entity IDs so
        //      that step 1c finds spawned entities during existence checks.
        for spawn in &txn.spawns {
            (spawn.preflight)(self)?;
            // Reserve the entity slot so subsequent insert/remove checks find it.
            // spawn_entity_with_id is idempotent for already-reserved slots.
            self.spawn_entity_with_id(spawn.entity, spawn.kind);
        }

        // 1b. Validate read dependencies
        for dep in &txn.read_deps {
            let current_ver = self
                .component_versions
                .get(&dep.entity)
                .copied()
                .unwrap_or(0);
            if current_ver != dep.observed_version {
                return Err(WorldTxnError::StaleRead {
                    entity: dep.entity,
                    schema_id: dep.schema_id,
                    observed: dep.observed_version,
                    current: current_ver,
                });
            }
        }

        // 1c. Validate entity existence for every staged operation
        for insert in &txn.inserts {
            let idx = (insert.entity as usize).wrapping_sub(1);
            if idx >= self.entity_meta.len() {
                return Err(WorldTxnError::InvalidEntity(insert.entity));
            }
            // Full generation validation comes with CompEntity refactor;
            // for now the handle is just a 1-based index and generation
            // is always 0 in append-only mode.
        }
        for remove in &txn.removes {
            let idx = (remove.entity as usize).wrapping_sub(1);
            if idx >= self.entity_meta.len() {
                return Err(WorldTxnError::InvalidEntity(remove.entity));
            }
        }

        // 1d. Validate staged operations via preflight closures
        for insert in &txn.inserts {
            (insert.preflight)(&self.component_store)?;
        }
        for remove in &txn.removes {
            (remove.preflight)(&self.component_store)?;
        }

        // -- PHASE 2: Build journal with the new epoch -----------------

        let next_epoch = WorldEpoch(self.epoch.0 + 1);
        let mut journal = Vec::new();
        for insert in &txn.inserts {
            journal.push(ComponentChange {
                entity: insert.entity,
                schema_id: insert.schema_id,
                schema_version: insert.schema_version,
                change_type: ChangeType::Insert,
                before_hash: None,
                after_hash: None,
                world_epoch: next_epoch,
            });
        }
        for remove in &txn.removes {
            journal.push(ComponentChange {
                entity: remove.entity,
                schema_id: remove.schema_id,
                schema_version: SchemaVersion(0),
                change_type: ChangeType::Remove,
                before_hash: None,
                after_hash: None,
                world_epoch: next_epoch,
            });
        }

        // -- PHASE 3: Apply all mutations ------------------------------

        for insert in txn.inserts {
            (insert.apply)(&mut self.component_store);
            *self.component_versions.entry(insert.entity).or_insert(0) += 1;
        }
        for remove in txn.removes {
            (remove.apply)(&mut self.component_store);
            *self.component_versions.entry(remove.entity).or_insert(0) += 1;
        }

        // 3c. Apply staged spawns
        for spawn in txn.spawns {
            (spawn.apply)(self);
        }

        // -- PHASE 4: Advance epoch AFTER all mutations succeed --------

        self.epoch = next_epoch;
        self.journal = journal;
        self.committed_events = txn.events;

        Ok(CommittedEpoch(next_epoch))
    }
}

impl ResourceStore {
    pub fn insert<T: 'static + Send + Sync>(&mut self, resource: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(resource));
    }

    pub fn get<T: 'static + Send + Sync>(&self) -> Option<&T> {
        self.data
            .get(&TypeId::of::<T>())
            .and_then(|b| b.downcast_ref::<T>())
    }

    pub fn contains<T: 'static + Send + Sync>(&self) -> bool {
        self.data.contains_key(&TypeId::of::<T>())
    }

    pub fn remove<T: 'static + Send + Sync>(&mut self) -> Option<T> {
        self.data
            .remove(&TypeId::of::<T>())
            .and_then(|b| b.downcast::<T>().ok().map(|b| *b))
    }
}

impl ComponentStore {
    pub fn contains<T: Component>(&self, entity: CompEntity) -> bool {
        let type_id = TypeId::of::<T>();
        self.data
            .get(&type_id)
            .and_then(|b| b.downcast_ref::<HashMap<EntityId, T>>())
            .map_or(false, |map| map.contains_key(&entity.0))
    }

    pub fn remove<T: Component>(&mut self, entity: CompEntity) -> Option<T> {
        let type_id = TypeId::of::<T>();
        self.data
            .get_mut(&type_id)
            .and_then(|b| b.downcast_mut::<HashMap<EntityId, T>>())
            .and_then(|map| map.remove(&entity.0))
    }
}
