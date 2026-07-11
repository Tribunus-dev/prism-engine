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

use anyhow;
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
    generation: u32,
    name: Option<String>,
}

impl Default for EntityMeta {
    fn default() -> Self {
        Self {
            kind: EntityKind::Model,
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
        let id = self.next_id;
        self.next_id += 1;
        self.entity_meta.push(EntityMeta {
            kind,
            generation: 0,
            name,
        });
        CompEntity(id)
    }

    /// Get the name of an entity.
    pub fn name(&self, entity: CompEntity) -> Option<&str> {
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

    /// Remove a component from an entity.
    /// Get the kind of an entity (alias for entity_kind).
    pub fn kind(&self, entity: CompEntity) -> Option<EntityKind> {
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
        }
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
        let matched: Vec<_> = prev_systems
            .into_iter()
            .filter(|s| s.phase() == phase)
            .collect();
        let unmatched: Vec<_> = std::mem::take(&mut self.systems);
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
                self.rollback_stage();
                Err(e)
            }
            Err(panic) => {
                self.rollback_stage();
                let msg = if let Some(s) = panic.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(anyhow::anyhow!("System panic (rolled back): {msg}"))
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
