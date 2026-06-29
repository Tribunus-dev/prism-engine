//! ECS component types for Prism Engine agent management.
//!
//! Each component represents one axis of an agent's execution state.
//! The hot-path state machine (AgentSlot) bundles atomic transitions
//! with cache-line alignment.  Cold-path data (payloads, KV references,
//! tools) lives in separate components for sparse iteration.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::runtime::agent_slot::STATE_IDLE;

// ---------------------------------------------------------------------------
// Hot-path — agent slot state machine (cache-line-aligned, atomic)
// ---------------------------------------------------------------------------

/// Per-agent execution slot.  Identical to the existing `AgentSlot` struct.
/// The `#[repr(align(64))]` prevents false sharing between the E-core
/// prefetch pump (DSTREAM) and the P-core ANE multiplexer.
///
/// Systems access the atomic `state` field lock-free; the rest of the
/// struct is written at init time and read-only afterwards.
#[repr(align(64))]
#[derive(Debug)]
pub struct AgentSlot {
    /// Atomic state machine: IDLE → PREFETCHING → READY → EXECUTING → IDLE
    pub state: AtomicU8,
    /// Agent surface / slot index.
    pub surface_id: u32,
    /// Byte offset into the .cimage mmap where this agent's weights begin.
    pub weight_offset: usize,
    /// Ping-pong phase selector (0 = Attention, 1 = MLP).
    pub prefetch_phase: u8,
}

// SAFETY: AtomicU8 provides interior mutability, so AgentSlot is Sync
// despite the non-atomic fields being conceptually immutable after init.
unsafe impl Sync for AgentSlot {}

impl AgentSlot {
    pub fn new(surface_id: u32, weight_offset: usize) -> Self {
        Self {
            state: AtomicU8::new(STATE_IDLE),
            surface_id,
            weight_offset,
            prefetch_phase: 0,
        }
    }

    /// Transition from `expected` to `target` using the standard acquire/
    /// release ordering.  Returns true on success.
    #[inline]
    pub fn try_transition(&self, expected: u8, target: u8) -> bool {
        self.state
            .compare_exchange(expected, target, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    /// Load the current state.
    #[inline]
    pub fn load_state(&self) -> u8 {
        self.state.load(Ordering::Acquire)
    }

    /// Store a new state (release ordering).
    #[inline]
    pub fn store_state(&self, state: u8) {
        self.state.store(state, Ordering::Release);
    }
}

impl Clone for AgentSlot {
    fn clone(&self) -> Self {
        Self {
            state: AtomicU8::new(self.state.load(Ordering::Relaxed)),
            surface_id: self.surface_id,
            weight_offset: self.weight_offset,
            prefetch_phase: self.prefetch_phase,
        }
    }
}

// ---------------------------------------------------------------------------
// Cold-path — per-agent data payloads
// ---------------------------------------------------------------------------

/// The modality and tokenized prompt for an agent's current task.
#[derive(Debug, Clone)]
pub struct AgentPayload {
    /// Tokenized prompt tensor.
    pub prompt_tokens: Vec<u16>,
    /// Image surface handle (IOSurface ID), 0 if none.
    pub image_surface_id: u32,
    /// Audio surface handle (IOSurface ID), 0 if none.
    pub audio_surface_id: u32,
}

impl AgentPayload {
    pub fn is_multimodal(&self) -> bool {
        self.image_surface_id != 0 || self.audio_surface_id != 0
    }
}

/// Reference to the KV cache pages allocated for this agent.
/// Filled by the ANE prefill system, consumed by the GPU decode system.
#[derive(Debug, Clone)]
pub struct KVCacheRef {
    /// Page indices into the shared KV arena.
    pub page_indices: Vec<u32>,
    /// Current sequence length (prefilled tokens).
    pub seq_len: u32,
    /// Maximum sequence length (budget-driven).
    pub max_seq_len: u32,
}

impl KVCacheRef {
    pub fn new(max_seq_len: u32) -> Self {
        Self {
            page_indices: Vec::new(),
            seq_len: 0,
            max_seq_len,
        }
    }

    pub fn is_full(&self) -> bool {
        self.seq_len >= self.max_seq_len
    }
}

/// Tools available to this agent session.
#[derive(Debug, Clone)]
pub struct ToolRegistry {
    pub tools: Vec<ToolDef>,
}

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// A pending agent action — either generating tokens or executing a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentStatus {
    /// Awaiting a prompt or tool invocation.
    Idle,
    /// Tokens being generated.
    Generating,
    /// Waiting on a tool to complete.
    AwaitingTool,
    /// Awaiting memory budget allocation.
    WaitingForBudget,
}
