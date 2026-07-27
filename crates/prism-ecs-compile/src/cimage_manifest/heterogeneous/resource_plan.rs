//! Compiler-owned activation resource plan.
//!
//! This module owns the **memory layout contract** for a
//! heterogeneous image: every arena, slot, alias, materialization
//! node, and resource lifetime interval. The runtime does not guess
//! resource ownership — it consumes this plan and provisions the
//! declared arenas at load time.
//!
//! A `SlotBacking` describes how a slot's bytes are physically
//! provisioned (IOSurface, host pointer, Metal private); a
//! `ConcurrencyClass` describes the access discipline the executor
//! must enforce.

use serde::{Deserialize, Serialize};

use super::backend_plan::MaterializationPlan;
use super::phase_ir::PhaseId;
use super::shared::ActivationAbi;

/// The compiler-owned activation resource plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledResourcePlan {
    pub arenas: Vec<ArenaPlan>,
    pub slots: Vec<CompiledSlot>,
    pub aliases: Vec<SlotAlias>,
    pub materializations: Vec<MaterializationNode>,
    pub lifetime_intervals: Vec<ResourceLifetime>,
}

/// Describes one activation arena (IOSurface pool or host heap).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaPlan {
    pub arena_id: ArenaId,
    pub byte_size: u64,
    pub alignment: u64,
    pub backing: ArenaBacking,
    pub ring_depth: u32,
}

/// How an arena is backed at runtime.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArenaBacking {
    /// IOSurface — the standard shared activation backing where
    /// Metal and Core ML interoperate.
    IOSurface,
    /// Host heap allocation (CPU-pointer accessible).
    HostHeap,
    /// Metal buffer allocation.
    MetalBuffer,
}

/// A compiled slot — describes a single activation or tensor binding
/// within an arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledSlot {
    pub slot_id: SlotId,
    pub arena_id: ArenaId,
    pub activation_abi: ActivationAbi,
    pub byte_length: u64,
    pub alignment: u64,
    pub backing: SlotBacking,
    pub producer_phase: PhaseId,
    pub consumer_phases: Vec<PhaseId>,
    pub concurrency_class: ConcurrencyClass,
}

/// How a slot's backing memory is provisioned.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum SlotBacking {
    /// IOSurface — shared between Metal and Core ML.
    IOSurface,
    /// Host pointer — CPU-accessible, usable by Accelerate.
    HostPointer,
    /// Metal private buffer — GPU-only.
    MetalPrivate,
}

/// ABI binding mode for the slot.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ConcurrencyClass {
    Exclusive,
    SharedRead,
    ProducerConsumer,
}

/// A slot alias — two slot ids that share the same backing memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlotAlias {
    pub alias_id: SlotAliasId,
    pub primary_slot: SlotId,
    pub secondary_slot: SlotId,
    pub offset_bytes: u64,
}

/// A materialization node — inserted where data must cross device
/// boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterializationNode {
    pub materialization_id: MaterializationId,
    pub from_slot: SlotId,
    pub to_slot: SlotId,
    pub plan: MaterializationPlan,
    pub estimated_cost_ns: u64,
}

/// A resource lifetime interval — when a slot is live.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLifetime {
    pub slot_id: SlotId,
    pub first_phase: PhaseId,
    pub last_phase: PhaseId,
}

pub type ArenaId = u64;
pub type SlotId = u64;
pub type SlotAliasId = u64;
pub type MaterializationId = u64;
