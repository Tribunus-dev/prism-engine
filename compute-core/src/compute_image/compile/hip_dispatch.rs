//! HIP compute dispatch wrappers — host-side dispatch for each AMD ROCm kernel class.
//!
//! Each dispatcher is a stateless struct holding only an `Arc<parking_lot::Mutex<HipKernelRegistry>>`.
//! The `dispatch()` method encodes a HIP kernel launch on the provided stream.
//!
//! Follows the same structural pattern as `kernel_dispatch.rs` (Metal dispatchers)
//! but targets HIP runtime for AMD MI300X on ROCm.

use parking_lot::Mutex;
use std::sync::Arc;

// ── Stub HIP runtime types (replace with real FFI bindings when wired) ─────

/// Minimal placeholder HIP runtime types for structural compilation.
/// These will be replaced with actual HIP FFI bindings in Phase 2.
mod hiprt {
    /// Opaque HIP stream handle.
    pub struct Stream;
    /// Opaque HIP device buffer.
    pub struct Buffer;
}

// ── Registry type alias ──────────────────────────────────────────────────────

/// Placeholder for the HIP kernel registry.
///
/// In a fully wired implementation this would cache loaded HIP modules (`.code`)
/// and functions with their associated launch configuration. For compilation
/// scaffolding the struct is empty; the pointer serves as a token for the
/// shared-registry pattern.
pub struct HipKernelRegistry;

/// Shared handle to the HIP kernel registry.
pub type RegistryRef = Arc<Mutex<HipKernelRegistry>>;

// ═════════════════════════════════════════════════════════════════════════════
// Q8_0GemvDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the Q8_0 block-f32 scale + int8 GEMV kernel (32-element blocks).
///
/// Kernel template: `templates/gemv_q8_0.hip`
/// Layout: per-block 1xf32 scale + 32xint8 weights, accumulated into f32.
pub struct Q8_0GemvDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl Q8_0GemvDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        Q8_0GemvDispatcher {
            registry,
            kernel_name: "gemv_q8_0",
        }
    }

    /// Encode a Q8_0 GEMV dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _weights: &hiprt::Buffer,
        _input: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _in_dim: u32,
        _out_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Q4_KGemvDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the Q4_K K-quant 4-bit GEMV kernel (256-element super-blocks).
///
/// Kernel template: `templates/gemv_q4_k.hip`
pub struct Q4_KGemvDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl Q4_KGemvDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        Q4_KGemvDispatcher {
            registry,
            kernel_name: "gemv_q4_k",
        }
    }

    /// Encode a Q4_K GEMV dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _weights: &hiprt::Buffer,
        _input: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _in_dim: u32,
        _out_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Q2_KGemvDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the Q2_K K-quant 2-bit GEMV kernel (256-element super-blocks).
///
/// Kernel template: `templates/gemv_q2_k.hip`
pub struct Q2_KGemvDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl Q2_KGemvDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        Q2_KGemvDispatcher {
            registry,
            kernel_name: "gemv_q2_k",
        }
    }

    /// Encode a Q2_K GEMV dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _weights: &hiprt::Buffer,
        _input: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _in_dim: u32,
        _out_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// IQ2_XXSGemvDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the IQ2_XXS importance-weighted 2-bit GEMV kernel with codebook.
///
/// Kernel template: `templates/gemv_iq2_xxs.hip`
pub struct IQ2_XXSGemvDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl IQ2_XXSGemvDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        IQ2_XXSGemvDispatcher {
            registry,
            kernel_name: "gemv_iq2_xxs",
        }
    }

    /// Encode an IQ2_XXS GEMV dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _weights: &hiprt::Buffer,
        _input: &hiprt::Buffer,
        _codebook: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _in_dim: u32,
        _out_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// MoeRouterDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the MoE expert router (top-k gating) kernel.
///
/// Kernel template: `templates/moe_router.hip`
pub struct MoeRouterDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl MoeRouterDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        MoeRouterDispatcher {
            registry,
            kernel_name: "moe_router",
        }
    }

    /// Encode a MoE router dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _hidden_states: &hiprt::Buffer,
        _router_weights: &hiprt::Buffer,
        _expert_indices: &hiprt::Buffer,
        _expert_scores: &hiprt::Buffer,
        _num_tokens: u32,
        _num_experts: u32,
        _top_k: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// MoeSparseMatmulDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the MoE sparse matmul kernel (selected expert forward pass).
///
/// Kernel template: `templates/moe_sparse_matmul.hip`
pub struct MoeSparseMatmulDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl MoeSparseMatmulDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        MoeSparseMatmulDispatcher {
            registry,
            kernel_name: "moe_sparse_matmul",
        }
    }

    /// Encode a MoE sparse matmul dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _expert_weights: &hiprt::Buffer,
        _expert_indices: &hiprt::Buffer,
        _input: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _hidden_dim: u32,
        _intermediate_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SharedExpertMlpDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the shared expert MLP kernel (dense MLP shared across tokens).
///
/// Kernel template: `templates/shared_expert_mlp.hip`
pub struct SharedExpertMlpDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl SharedExpertMlpDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        SharedExpertMlpDispatcher {
            registry,
            kernel_name: "shared_expert_mlp",
        }
    }

    /// Encode a shared expert MLP dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _gate_weight: &hiprt::Buffer,
        _up_weight: &hiprt::Buffer,
        _down_weight: &hiprt::Buffer,
        _input: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _hidden_dim: u32,
        _intermediate_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// CompressedAttentionDispatcher
// ═════════════════════════════════════════════════════════════════════════════

/// Dispatches the compressed attention kernel (grouped-query with latent
/// compression).
///
/// Kernel template: `templates/compressed_attention.hip`
pub struct CompressedAttentionDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

impl CompressedAttentionDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        CompressedAttentionDispatcher {
            registry,
            kernel_name: "compressed_attention",
        }
    }

    /// Encode a compressed attention dispatch on the given HIP stream.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        _stream: &hiprt::Stream,
        _query: &hiprt::Buffer,
        _key: &hiprt::Buffer,
        _value: &hiprt::Buffer,
        _output: &hiprt::Buffer,
        _num_heads: u32,
        _seq_len: u32,
        _head_dim: u32,
    ) {
        let _ = &self.registry;
        let _ = self.kernel_name;
    }
}
