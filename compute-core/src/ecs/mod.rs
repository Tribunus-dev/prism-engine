pub mod adapter;
pub mod aot;
pub mod compile_session;
pub mod component;
pub mod config;
pub mod entity;
pub mod plan;
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
    ModelLoading,     // Phase A
    Quantization,     // Phase B
    MemoryPlanning,   // Phase C
    FusionDispatch,   // Phase D
    KernelGeneration, // Phase E
    Compilation,      // Phase F
    Packaging,        // Phase G
    Validation,       // Phase H — final admission gates and completeness checks
    Execution,        // Phase I — runtime execution, scheduling, and dispatch
}

/// A compiler pass over the ECS world.
pub trait CompilerSystem: Send + Sync {
    fn name(&self) -> &str;
    fn phase(&self) -> SchedulePhase;
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()>;
}

/// The ECS world — all entities, components, and systems.
pub struct CompWorld {
    next_id: EntityId,
    entities: Vec<EntityMeta>,
    components: ComponentStore,
    systems: Vec<Box<dyn CompilerSystem>>,
    resources: ResourceStore,
}

impl std::fmt::Debug for CompWorld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompWorld")
            .field("entity_count", &self.entities.len())
            .field("system_count", &self.systems.len())
            .field("component_type_count", &self.components.data.len())
            .finish()
    }
}

/// Manages the lifecycle of a single executable argument region.
#[derive(Debug)]
struct EntityMeta {
    name: Option<String>,
    kind: EntityKind,
}

impl Default for EntityMeta {
    fn default() -> Self {
        Self {
            name: None,
            kind: EntityKind::Model,
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
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entities: Vec::new(),
            components: ComponentStore::default(),
            systems: Vec::new(),
            resources: ResourceStore::default(),
        }
    }

    pub fn add_resource<T: Send + Sync + 'static>(&mut self, resource: T) {
        self.resources.insert(resource);
    }

    pub fn get_resource<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.resources.get()
    }

    pub fn get_resource_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        self.resources.get_mut()
    }

    pub fn spawn(&mut self, kind: EntityKind, name: Option<String>) -> CompEntity {
        let id = self.next_id;
        self.next_id += 1;
        self.entities.push(EntityMeta { name, kind });
        CompEntity(id)
    }

    pub fn kind(&self, entity: CompEntity) -> Option<EntityKind> {
        self.entities.get(entity.0 as usize - 1).map(|m| m.kind)
    }

    pub fn name(&self, entity: CompEntity) -> Option<&str> {
        self.entities
            .get(entity.0 as usize - 1)
            .and_then(|m| m.name.as_deref())
    }

    pub fn add_component<T: Component>(&mut self, entity: CompEntity, component: T) {
        self.components.insert(entity, component);
    }

    pub fn get_component<T: Component>(&self, entity: CompEntity) -> Option<&T> {
        self.components.get(entity)
    }

    pub fn get_component_mut<T: Component>(&mut self, entity: CompEntity) -> Option<&mut T> {
        self.components.get_mut(entity)
    }

    pub fn remove_component<T: Component>(&mut self, entity: CompEntity) -> Option<T> {
        self.components.remove(entity)
    }

    pub fn add_system(&mut self, system: Box<dyn CompilerSystem>) {
        self.systems.push(system);
    }

    pub fn run_phase(&mut self, phase: SchedulePhase) -> anyhow::Result<()> {
        // Take systems out of self to avoid borrow conflict, then restore.
        let systems = std::mem::take(&mut self.systems);
        for system in &systems {
            if system.phase() == phase {
                system.run(self)?;
            }
        }
        self.systems = systems;
        Ok(())
    }

    pub fn run_all(&mut self) -> anyhow::Result<()> {
        let phases = [
            SchedulePhase::ModelLoading,
            SchedulePhase::Quantization,
            SchedulePhase::MemoryPlanning,
            SchedulePhase::FusionDispatch,
            SchedulePhase::KernelGeneration,
            SchedulePhase::Compilation,
            SchedulePhase::Packaging,
            SchedulePhase::Validation,
        ];
        for phase in phases {
            self.run_phase(phase)?;
        }
        Ok(())
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn entities_of_kind(&self, kind: EntityKind) -> Vec<CompEntity> {
        self.entities
            .iter()
            .enumerate()
            .filter(|(_, m)| m.kind == kind)
            .map(|(i, _)| CompEntity(i as u64 + 1))
            .collect()
    }
}

impl ResourceStore {
    fn insert<T: Send + Sync + 'static>(&mut self, resource: T) {
        let tid = TypeId::of::<T>();
        self.data.insert(tid, Box::new(resource));
    }

    fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        let tid = TypeId::of::<T>();
        self.data
            .get(&tid)
            .and_then(|slot| slot.downcast_ref::<T>())
    }

    fn get_mut<T: Send + Sync + 'static>(&mut self) -> Option<&mut T> {
        let tid = TypeId::of::<T>();
        self.data
            .get_mut(&tid)
            .and_then(|slot| slot.downcast_mut::<T>())
    }
}

impl ComponentStore {
    fn insert<T: Component>(&mut self, entity: CompEntity, component: T) {
        let tid = TypeId::of::<T>();
        let slot = self
            .data
            .entry(tid)
            .or_insert_with(|| Box::<HashMap<EntityId, T>>::default());
        if let Some(map) = slot.downcast_mut::<HashMap<EntityId, T>>() {
            map.insert(entity.0, component);
        }
    }

    fn get<T: Component>(&self, entity: CompEntity) -> Option<&T> {
        let tid = TypeId::of::<T>();
        self.data
            .get(&tid)
            .and_then(|slot| slot.downcast_ref::<HashMap<EntityId, T>>())
            .and_then(|map| map.get(&entity.0))
    }

    fn get_mut<T: Component>(&mut self, entity: CompEntity) -> Option<&mut T> {
        let tid = TypeId::of::<T>();
        self.data
            .get_mut(&tid)
            .and_then(|slot| slot.downcast_mut::<HashMap<EntityId, T>>())
            .and_then(|map| map.get_mut(&entity.0))
    }

    fn remove<T: Component>(&mut self, entity: CompEntity) -> Option<T> {
        let tid = TypeId::of::<T>();
        self.data
            .get_mut(&tid)
            .and_then(|slot| slot.downcast_mut::<HashMap<EntityId, T>>())
            .and_then(|map| map.remove(&entity.0))
    }
}
