//! Kernel ABI — the complete interface contract for a compiled kernel.
//!
//! This module owns the canonical authority for the kernel ABI as
//! observed by the evaluator: version, buffer bindings, function
//! constants, threadgroup memory, dispatch geometry, and
//! threadgroup size. The full engine-local `KernelAbi` lives at
//! `compute-core/src/ecs/canonical/kernel_abi.rs` (305 LOC); this
//! type captures the subset the evaluation surface needs to
//! identify and bind a generated executable.

use serde::{Deserialize, Serialize};

/// A buffer binding slot in a kernel's ABI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferBinding {
    /// Backend-specific buffer slot index (e.g. `[[buffer(N)]]`).
    pub slot: u32,
    /// Logical name of the buffer.
    pub name: String,
    /// Byte size of the binding.
    pub byte_size: u64,
    /// Whether this binding is optional.
    pub optional: bool,
}

/// A function constant binding in a kernel's ABI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstantBinding {
    /// Backend-specific constant index.
    pub index: u32,
    /// Logical name.
    pub name: String,
    /// The constant value, if fixed at compile time.
    pub default_value: Option<u32>,
}

/// A threadgroup memory allocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThreadgroupAllocation {
    /// Byte size of threadgroup memory.
    pub byte_size: u32,
}

/// How dispatch geometry is determined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DispatchGeometryPolicy {
    /// Fixed grid dimensions (width, height, depth).
    Fixed(u32, u32, u32),
    /// Derived from buffer sizes (typically output dimension).
    FromOutputBuffer,
    /// Dynamic via function constant.
    FromConstant,
}

/// KernelAbi — the complete interface contract for a compiled kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelAbi {
    pub version: u32,
    pub buffers: Vec<BufferBinding>,
    pub constants: Vec<ConstantBinding>,
    pub threadgroup_memory: Vec<ThreadgroupAllocation>,
    pub dispatch_geometry: DispatchGeometryPolicy,
    pub threads_per_threadgroup: (u32, u32, u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_abi_is_constructible_and_serializable() {
        let abi = KernelAbi {
            version: 1,
            buffers: vec![BufferBinding {
                slot: 0,
                name: "input".to_string(),
                byte_size: 1024,
                optional: false,
            }],
            constants: vec![ConstantBinding {
                index: 0,
                name: "tile_m".to_string(),
                default_value: Some(64),
            }],
            threadgroup_memory: vec![ThreadgroupAllocation { byte_size: 4096 }],
            dispatch_geometry: DispatchGeometryPolicy::Fixed(1, 1, 1),
            threads_per_threadgroup: (32, 1, 1),
        };

        let json = serde_json::to_string(&abi).expect("serialize");
        let restored: KernelAbi = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, abi);
    }

    #[test]
    fn dispatch_geometry_variants_are_distinct() {
        let fixed = DispatchGeometryPolicy::Fixed(8, 4, 1);
        let from_buf = DispatchGeometryPolicy::FromOutputBuffer;
        let from_const = DispatchGeometryPolicy::FromConstant;
        assert_ne!(fixed, from_buf);
        assert_ne!(from_buf, from_const);
        assert_ne!(fixed, from_const);
    }
}
