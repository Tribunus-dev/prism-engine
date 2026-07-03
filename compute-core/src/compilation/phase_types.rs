//! Distill-compiler PhaseIR types — immutable phase descriptors shared across
//! all execution levels (1–3).
//!
//! These types describe what a compile phase is, what tensors it reads and
//! writes, which provider executes it, and its place in the phase DAG.
//! They are the universal vocabulary of the activation-arena contract and the
//! receipt manifest.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export core identities from the existing phase_ir module.
pub use super::phase_ir::{CompilationId, DeviceSignature, PhaseId};

// ── Compile phase type ──────────────────────────────────────────────────────

/// The 14 primitive phase types of the distill-compiler.
///
/// Every phase has immutable input and output tensor descriptors. The phase
/// type determines the logical operation; the provider (Metal / Core ML /
/// Accelerate / …) determines the execution route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhaseType {
    LoadTeacherRegion,
    LoadStudentCandidate,
    TeacherForward,
    StudentForward,
    CompareActivations,
    ProbeAttention,
    ProbeResidual,
    ProbeNorm,
    ScaleSolve,
    TritCommit,
    SidecarAllocate,
    AdvanceFrontier,
    SealStudentRegion,
    SealReceipt,
}

// ── Element type ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ElementType {
    F32,
    F16,
    BF16,
    I8,
    I32,
    U8,
    U32,
}

// ── Physical layout ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicalLayout {
    DenseRowMajor,
    DenseColMajor,
    Blocked(usize),
    Tile640,
    NCHW,
    NHWC,
}

// ── Provider kind ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProviderKind {
    Metal,
    Accelerate,
    CoreML,
    Cpu,
    Disk,
}

// ── Residency class ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResidencyClass {
    Unified,
    Discrete,
    CpuOnly,
    DiskBacked,
}

// ── Tensor descriptor ───────────────────────────────────────────────────────

/// Logical (provider-independent) tensor descriptor.
///
/// Carries the shape, element type, physical layout, alignment, provenance
/// (producer + consumer phases), permitted providers, residency class,
/// maximum byte budget, mutability, and — once materialized — a content
/// digest verified by the arena.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub logical_shape: Vec<usize>,
    pub element_type: ElementType,
    pub physical_layout: PhysicalLayout,
    pub alignment: usize,
    pub producer_phase: Option<PhaseId>,
    pub consumer_phases: Vec<PhaseId>,
    pub permitted_providers: Vec<ProviderKind>,
    pub residency_class: ResidencyClass,
    pub max_bytes: u64,
    pub mutable: bool,
    pub content_digest: Option<[u8; 32]>,
}

impl TensorDescriptor {
    /// Compute the number of bytes of a dense linearised tensor with this type.
    pub fn element_size(&self) -> usize {
        match self.element_type {
            ElementType::F32 | ElementType::I32 | ElementType::U32 => 4,
            ElementType::F16 | ElementType::BF16 => 2,
            ElementType::I8 | ElementType::U8 => 1,
        }
    }

    /// Flat (contiguous) element count.
    pub fn flat_elements(&self) -> usize {
        self.logical_shape.iter().product::<usize>().max(1)
    }

    /// Minimum contiguous bytes for this tensor.
    pub fn min_bytes(&self) -> u64 {
        (self.flat_elements() * self.element_size()) as u64
    }
}

// ── Phase descriptor ────────────────────────────────────────────────────────

/// An immutable compile-phase descriptor.
///
/// Each phase has a unique id, a type tag, a fixed set of input and output
/// tensor descriptors, an assigned provider, a sequence number (for ordering
/// within the scheduler), and arbitrary metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase {
    pub phase_id: PhaseId,
    pub phase_type: PhaseType,
    pub inputs: Vec<TensorDescriptor>,
    pub outputs: Vec<TensorDescriptor>,
    pub provider: ProviderKind,
    pub sequence_number: u64,
    pub metadata: HashMap<String, String>,
}
