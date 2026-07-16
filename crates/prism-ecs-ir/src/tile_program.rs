//! Triton IR dialect — block-level tensor program operations.
//!
//! Provides operations for Triton-style block-level tensor programming:
//! memory ops, compute primitives, and program-level grid specification.
//! Modeled on the Triton language's MLIR operations.
//!
//! All operations define a `TritonOp` component on the entity alongside
//! the standard `OpMarker`, `OpName`, `Operands`, `Results`, and
//! `OpAttributes` components. Program-level ops also carry a `RegionRef`
//! for their kernel body.
//!
//! # Operation semantics
//!
//! - `triton.load`: Load a block from memory at ptr, with optional mask and
//!   other (e.g., padding value). 3 operands (ptr, mask, other), 1 result.
//!
//! - `triton.store`: Store a block to memory. 3 operands (ptr, value, mask),
//!   0 results.
//!
//! - `triton.dot`: Block-level matrix multiply-accumulate: `D = A * B + C`.
//!   The critical performance primitive. 3 operands (a, b, acc), 1 result.
//!
//! - `triton.reduce`: Block-level reduction along an axis.
//!   1 operand (the block), 1 result.
//!
//! - `triton.scan`: Block-level exclusive/inclusive scan.
//!   1 operand, 1 result.
//!
//! - `triton.atomic_add`: Atomic add to shared memory.
//!   2 operands (ptr, val), 0 results.
//!
//! - `triton.atomic_cas`: Atomic compare-and-swap to shared memory.
//!   3 operands (ptr, cmp, val), 1 result (old value).
//!
//! - `triton.program`: Top-level kernel definition. Carries a `GridDim`
//!   component for 3D grid launch dimensions and a `RegionRef` for the
//!   kernel body. 0 operands, 0 results.
//!
//! - `triton.splat`: Broadcast a scalar to a block with all-elements-equal.
//!   1 operand (scalar), 1 result (block).
//!
//! - `triton.broadcast`: Broadcast along specified axes.
//!   1 operand, 1 result.
//!
//! - `triton.expand_dims`: Add a dimension of size 1.
//!   1 operand, 1 result.
//!
//! - `triton.trans`: Transpose (permute dimensions).
//!   1 operand, 1 result.
//!
//! - `triton.reshape`: Reshape a block (view semantics).
//!   1 operand, 1 result.
//!
//! - `triton.cat`: Concatenate two blocks along an axis.
//!   2 operands, 1 result.
//!
//! - `triton.gather`: Gather elements by indices.
//!   2 operands (block, indices), 1 result.
//!
//! - `triton.where`: Element-wise select from two blocks based on a
//!   condition mask. 3 operands (cond, x, y), 1 result.
//!
//! - `triton.sort`: Sort elements along an axis.
//!   1 operand, 1 result.

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpInfo, OpRegistry, OpVerifierContext};

// ── Op kind ──────────────────────────────────────────────────────────────────

/// Specific Triton IR operation variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TritonOpKind {
    /// Load a block from memory: ptr, mask, other -> block.
    Load,
    /// Store a block to memory: ptr, value, mask.
    Store,
    /// Matrix multiply-accumulate: a * b + acc -> result.
    Dot,
    /// Block reduction along an axis.
    Reduce,
    /// Block exclusive/inclusive scan.
    Scan,
    /// Atomic add to shared memory.
    AtomicAdd,
    /// Atomic compare-and-swap.
    AtomicCAS,
    /// Top-level kernel program with grid dimensions.
    Program,
    /// Broadcast scalar to block.
    Splat,
    /// Broadcast along axes.
    Broadcast,
    /// Expand dims (insert size-1 axis).
    ExpandDims,
    /// Transpose (permute dimensions).
    Trans,
    /// Reshape (view semantics).
    Reshape,
    /// Concatenate two blocks.
    Cat,
    /// Gather elements by indices.
    Gather,
    /// Element-wise select from two blocks.
    Where,
    /// Sort elements along an axis.
    Sort,
}

impl TritonOpKind {
    /// MLIR-style operation name for this kind.
    pub fn op_name(&self) -> &'static str {
        match self {
            TritonOpKind::Load => "triton.load",
            TritonOpKind::Store => "triton.store",
            TritonOpKind::Dot => "triton.dot",
            TritonOpKind::Reduce => "triton.reduce",
            TritonOpKind::Scan => "triton.scan",
            TritonOpKind::AtomicAdd => "triton.atomic_add",
            TritonOpKind::AtomicCAS => "triton.atomic_cas",
            TritonOpKind::Program => "triton.program",
            TritonOpKind::Splat => "triton.splat",
            TritonOpKind::Broadcast => "triton.broadcast",
            TritonOpKind::ExpandDims => "triton.expand_dims",
            TritonOpKind::Trans => "triton.trans",
            TritonOpKind::Reshape => "triton.reshape",
            TritonOpKind::Cat => "triton.cat",
            TritonOpKind::Gather => "triton.gather",
            TritonOpKind::Where => "triton.where",
            TritonOpKind::Sort => "triton.sort",
        }
    }

    /// Number of required operands for this kind.
    pub fn operand_count(&self) -> usize {
        match self {
            TritonOpKind::Program => 0,
            TritonOpKind::Reduce
            | TritonOpKind::Scan
            | TritonOpKind::Splat
            | TritonOpKind::Broadcast
            | TritonOpKind::ExpandDims
            | TritonOpKind::Trans
            | TritonOpKind::Reshape
            | TritonOpKind::Sort => 1,
            TritonOpKind::AtomicAdd | TritonOpKind::Cat | TritonOpKind::Gather => 2,
            TritonOpKind::Load
            | TritonOpKind::Store
            | TritonOpKind::Dot
            | TritonOpKind::AtomicCAS
            | TritonOpKind::Where => 3,
        }
    }

    /// Number of results for this kind.
    pub fn result_count(&self) -> usize {
        match self {
            TritonOpKind::Store | TritonOpKind::AtomicAdd | TritonOpKind::Program => 0,
            _ => 1,
        }
    }
}

// ── Components ───────────────────────────────────────────────────────────────

/// Component attaching a Triton IR op kind to an operation entity.
///
/// Every entity representing a Triton IR operation carries this component
/// so dialects and passes can discriminate Triton operations from other
/// dialects or from general `OpName`-only queries.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TritonOp(pub TritonOpKind);
impl Component for TritonOp {}

/// Grid dimensions for a Triton program (kernel launch grid).
///
/// Carried by `triton.program` operations to specify the 3D launch grid
/// `(grid_x, grid_y, grid_z)`. Each dimension represents the number of
/// program instances launched along that axis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridDim(pub (u32, u32, u32));
impl Component for GridDim {}

// ── Verifiers ────────────────────────────────────────────────────────────────

/// Verify a triton.load operation: 3 operands (ptr, mask, other), 1 result.
pub fn verify_load(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 3 {
        errors.push(format!(
            "triton.load expects 3 operands (ptr, mask, other), got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!(
            "triton.load expects 1 result, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a triton.store operation: 3 operands (ptr, value, mask), 0 results.
pub fn verify_store(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 3 {
        errors.push(format!(
            "triton.store expects 3 operands (ptr, value, mask), got {}",
            ctx.operand_types.len()
        ));
    }
    if !ctx.result_types.is_empty() {
        errors.push(format!(
            "triton.store expects 0 results, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a triton.dot operation: 3 operands (a, b, acc), 1 result.
pub fn verify_dot(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 3 {
        errors.push(format!(
            "triton.dot expects 3 operands (a, b, acc), got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!(
            "triton.dot expects 1 result, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a triton.program operation: 0 operands, 0 results.
pub fn verify_program(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if !ctx.operand_types.is_empty() {
        errors.push(format!(
            "triton.program expects 0 operands, got {}",
            ctx.operand_types.len()
        ));
    }
    if !ctx.result_types.is_empty() {
        errors.push(format!(
            "triton.program expects 0 results, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a unary block op: 1 operand, 1 result.
fn verify_unary_block(ctx: &OpVerifierContext, name: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 1 {
        errors.push(format!(
            "{} expects 1 operand, got {}",
            name,
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!(
            "{} expects 1 result, got {}",
            name,
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify a binary block op: 2 operands, 1 result.
fn verify_binary_block(ctx: &OpVerifierContext, name: &str) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 2 {
        errors.push(format!(
            "{} expects 2 operands, got {}",
            name,
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!(
            "{} expects 1 result, got {}",
            name,
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify triton.reduce: 1 operand, 1 result.
pub fn verify_reduce(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.reduce")
}

/// Verify triton.scan: 1 operand, 1 result.
pub fn verify_scan(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.scan")
}

/// Verify triton.atomic_add: 2 operands (ptr, val), 0 results.
pub fn verify_atomic_add(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 2 {
        errors.push(format!(
            "triton.atomic_add expects 2 operands (ptr, val), got {}",
            ctx.operand_types.len()
        ));
    }
    if !ctx.result_types.is_empty() {
        errors.push(format!(
            "triton.atomic_add expects 0 results, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify triton.atomic_cas: 3 operands (ptr, cmp, val), 1 result.
pub fn verify_atomic_cas(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 3 {
        errors.push(format!(
            "triton.atomic_cas expects 3 operands (ptr, cmp, val), got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!(
            "triton.atomic_cas expects 1 result, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify triton.splat: 1 operand, 1 result.
pub fn verify_splat(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.splat")
}

/// Verify triton.broadcast: 1 operand, 1 result.
pub fn verify_broadcast(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.broadcast")
}

/// Verify triton.expand_dims: 1 operand, 1 result.
pub fn verify_expand_dims(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.expand_dims")
}

/// Verify triton.trans: 1 operand, 1 result.
pub fn verify_trans(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.trans")
}

/// Verify triton.reshape: 1 operand, 1 result.
pub fn verify_reshape(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.reshape")
}

/// Verify triton.cat: 2 operands, 1 result.
pub fn verify_cat(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_binary_block(ctx, "triton.cat")
}

/// Verify triton.gather: 2 operands (block, indices), 1 result.
pub fn verify_gather(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_binary_block(ctx, "triton.gather")
}

/// Verify triton.where: 3 operands (cond, x, y), 1 result.
pub fn verify_where(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    if ctx.operand_types.len() != 3 {
        errors.push(format!(
            "triton.where expects 3 operands (cond, x, y), got {}",
            ctx.operand_types.len()
        ));
    }
    if ctx.result_types.len() != 1 {
        errors.push(format!(
            "triton.where expects 1 result, got {}",
            ctx.result_types.len()
        ));
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Verify triton.sort: 1 operand, 1 result.
pub fn verify_sort(ctx: &OpVerifierContext) -> Result<(), Vec<String>> {
    verify_unary_block(ctx, "triton.sort")
}

// ── Type inference ───────────────────────────────────────────────────────────

/// Infer result types for triton.load: result type matches operand 0 element type.
pub fn infer_load(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 && matches!(&operand_types[0], Type::Tensor(_) | Type::MemRef(_)) {
        Some(vec![operand_types[0].clone()])
    } else if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.store: no results.
pub fn infer_store(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    Some(vec![])
}

/// Infer result types for triton.dot: result type matches acc (operand 2).
pub fn infer_dot(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 3 {
        Some(vec![operand_types[2].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.program: no results.
pub fn infer_program(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    Some(vec![])
}

/// Infer result types for triton.reduce: result matches operand.
pub fn infer_reduce(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.scan: result matches operand.
pub fn infer_scan(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.atomic_add: no results.
pub fn infer_atomic_add(_operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    Some(vec![])
}

/// Infer result types for triton.atomic_cas: result matches val (operand 2).
pub fn infer_atomic_cas(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 3 {
        Some(vec![operand_types[2].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.splat: result matches operand.
pub fn infer_splat(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.broadcast: result matches operand.
pub fn infer_broadcast(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.expand_dims: result matches operand (reshaped).
pub fn infer_expand_dims(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.trans: result matches operand.
pub fn infer_trans(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.reshape: result matches operand.
pub fn infer_reshape(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.cat: result matches operand 0.
pub fn infer_cat(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.gather: result matches operand 0 element type.
pub fn infer_gather(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.where: result matches x (operand 1).
pub fn infer_where(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 2 {
        Some(vec![operand_types[1].clone()])
    } else {
        None
    }
}

/// Infer result types for triton.sort: result matches operand.
pub fn infer_sort(operand_types: &[Type], _attributes: &[Attribute]) -> Option<Vec<Type>> {
    if operand_types.len() >= 1 {
        Some(vec![operand_types[0].clone()])
    } else {
        None
    }
}

// ── Registration ─────────────────────────────────────────────────────────────

/// Register all Triton IR dialect operations into the given OpRegistry.
pub fn register_triton_ops(registry: &mut OpRegistry) {
    // Memory ops
    registry.register(OpInfo {
        name: "triton.load",
        description: "Load a block from memory with mask and other",
        verify_fn: Some(verify_load as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_load as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.store",
        description: "Store a block to memory with mask",
        verify_fn: Some(verify_store as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_store as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Compute primitives
    registry.register(OpInfo {
        name: "triton.dot",
        description: "Block-level matrix multiply-accumulate: D = A * B + C",
        verify_fn: Some(verify_dot as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_dot as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.reduce",
        description: "Block reduction along an axis",
        verify_fn: Some(verify_reduce as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_reduce as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.scan",
        description: "Block exclusive/inclusive scan",
        verify_fn: Some(verify_scan as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_scan as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Atomic ops
    registry.register(OpInfo {
        name: "triton.atomic_add",
        description: "Atomic add to shared memory",
        verify_fn: Some(verify_atomic_add as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_atomic_add as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.atomic_cas",
        description: "Atomic compare-and-swap",
        verify_fn: Some(verify_atomic_cas as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_atomic_cas as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Program
    registry.register(OpInfo {
        name: "triton.program",
        description: "Top-level kernel program with grid dimensions",
        verify_fn: Some(verify_program as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_program as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });

    // Block manipulation
    registry.register(OpInfo {
        name: "triton.splat",
        description: "Broadcast scalar to block",
        verify_fn: Some(verify_splat as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_splat as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.broadcast",
        description: "Broadcast along specified axes",
        verify_fn: Some(verify_broadcast as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_broadcast as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.expand_dims",
        description: "Add a dimension of size 1",
        verify_fn: Some(verify_expand_dims as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_expand_dims as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.trans",
        description: "Transpose (permute dimensions)",
        verify_fn: Some(verify_trans as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_trans as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.reshape",
        description: "Reshape a block (view semantics)",
        verify_fn: Some(verify_reshape as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_reshape as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.cat",
        description: "Concatenate two blocks along an axis",
        verify_fn: Some(verify_cat as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_cat as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.gather",
        description: "Gather elements by indices",
        verify_fn: Some(verify_gather as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_gather as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.where",
        description: "Element-wise select from two blocks based on condition mask",
        verify_fn: Some(verify_where as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_where as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
    registry.register(OpInfo {
        name: "triton.sort",
        description: "Sort elements along an axis",
        verify_fn: Some(verify_sort as fn(&OpVerifierContext) -> Result<(), Vec<String>>),
        infer_fn: Some(infer_sort as fn(&[Type], &[Attribute]) -> Option<Vec<Type>>),
    });
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::OpBuilder;
    use crate::ir_types::{Signedness, TensorType, Type};
    use crate::op::{op_name, operands, results};
    use prism_ecs_core::{EntityKind, World};

    // ── Component tests ──────────────────────────────────────────────────

    #[test]
    fn triton_op_kind_op_name() {
        assert_eq!(TritonOpKind::Load.op_name(), "triton.load");
        assert_eq!(TritonOpKind::Store.op_name(), "triton.store");
        assert_eq!(TritonOpKind::Dot.op_name(), "triton.dot");
        assert_eq!(TritonOpKind::Reduce.op_name(), "triton.reduce");
        assert_eq!(TritonOpKind::Scan.op_name(), "triton.scan");
        assert_eq!(TritonOpKind::AtomicAdd.op_name(), "triton.atomic_add");
        assert_eq!(TritonOpKind::AtomicCAS.op_name(), "triton.atomic_cas");
        assert_eq!(TritonOpKind::Program.op_name(), "triton.program");
        assert_eq!(TritonOpKind::Splat.op_name(), "triton.splat");
        assert_eq!(TritonOpKind::Broadcast.op_name(), "triton.broadcast");
        assert_eq!(TritonOpKind::ExpandDims.op_name(), "triton.expand_dims");
        assert_eq!(TritonOpKind::Trans.op_name(), "triton.trans");
        assert_eq!(TritonOpKind::Reshape.op_name(), "triton.reshape");
        assert_eq!(TritonOpKind::Cat.op_name(), "triton.cat");
        assert_eq!(TritonOpKind::Gather.op_name(), "triton.gather");
        assert_eq!(TritonOpKind::Where.op_name(), "triton.where");
        assert_eq!(TritonOpKind::Sort.op_name(), "triton.sort");
    }

    #[test]
    fn triton_op_kind_operand_count() {
        assert_eq!(TritonOpKind::Program.operand_count(), 0);
        assert_eq!(TritonOpKind::Reduce.operand_count(), 1);
        assert_eq!(TritonOpKind::Scan.operand_count(), 1);
        assert_eq!(TritonOpKind::Splat.operand_count(), 1);
        assert_eq!(TritonOpKind::AtomicAdd.operand_count(), 2);
        assert_eq!(TritonOpKind::Cat.operand_count(), 2);
        assert_eq!(TritonOpKind::Gather.operand_count(), 2);
        assert_eq!(TritonOpKind::Load.operand_count(), 3);
        assert_eq!(TritonOpKind::Store.operand_count(), 3);
        assert_eq!(TritonOpKind::Dot.operand_count(), 3);
        assert_eq!(TritonOpKind::AtomicCAS.operand_count(), 3);
        assert_eq!(TritonOpKind::Where.operand_count(), 3);
    }

    #[test]
    fn triton_op_kind_result_count() {
        assert_eq!(TritonOpKind::Store.result_count(), 0);
        assert_eq!(TritonOpKind::AtomicAdd.result_count(), 0);
        assert_eq!(TritonOpKind::Program.result_count(), 0);
        assert_eq!(TritonOpKind::Load.result_count(), 1);
        assert_eq!(TritonOpKind::Dot.result_count(), 1);
        assert_eq!(TritonOpKind::Reduce.result_count(), 1);
    }

    #[test]
    fn triton_op_component_attached() {
        let mut world = World::new();
        let entity: prism_ecs_core::Entity = world
            .spawn(EntityKind::Node, Some("test_triton".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, TritonOp(TritonOpKind::Dot))
            .expect("add TritonOp");
        let retrieved = world.get_component::<TritonOp>(entity);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, TritonOpKind::Dot);
    }

    #[test]
    fn grid_dim_component_attached() {
        let mut world = World::new();
        let entity: prism_ecs_core::Entity = world
            .spawn(EntityKind::Node, Some("test_grid".into()))
            .expect("spawn")
            .into();
        world
            .add_component(entity, GridDim((1, 2, 3)))
            .expect("add GridDim");
        let retrieved = world.get_component::<GridDim>(entity);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().0, (1, 2, 3));
    }

    #[test]
    fn triton_op_serialization_roundtrip() {
        let op = TritonOp(TritonOpKind::Dot);
        let json = serde_json::to_string(&op).unwrap();
        let back: TritonOp = serde_json::from_str(&json).unwrap();
        assert_eq!(op.0, back.0);

        let kind = TritonOpKind::Load;
        let json = serde_json::to_string(&kind).unwrap();
        let back: TritonOpKind = serde_json::from_str(&json).unwrap();
        assert_eq!(kind, back);

        let grid = GridDim((4, 8, 1));
        let json = serde_json::to_string(&grid).unwrap();
        let back: GridDim = serde_json::from_str(&json).unwrap();
        assert_eq!(grid.0, back.0);
    }

    // ── Builder integration tests ────────────────────────────────────────

    #[test]
    fn create_dot_op_via_builder() {
        let mut world = World::new();
        // Create three operand values: a, b, acc (all f32)
        let a = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };
        let b_val = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };
        let acc = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };

        // Create the triton.dot op
        let dot_entity = {
            let mut builder = OpBuilder::new(&mut world);
            let dot = builder
                .create_op("triton.dot", &[a, b_val, acc], &[], &[Type::f32()])
                .unwrap();
            drop(builder);
            // Attach TritonOp component
            world
                .add_component(dot, TritonOp(TritonOpKind::Dot))
                .expect("add TritonOp");
            dot
        };

        // Verify op_name
        let name = op_name(&world, dot_entity);
        assert_eq!(name, Some("triton.dot".to_string()));

        // Verify 3 operands
        let ops = operands(&world, dot_entity);
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0], a);
        assert_eq!(ops[1], b_val);
        assert_eq!(ops[2], acc);

        // Verify 1 result
        let res = results(&world, dot_entity);
        assert_eq!(res.len(), 1);

        // Verify TritonOp component
        let triton_op = world.get_component::<TritonOp>(dot_entity);
        assert!(triton_op.is_some());
        assert_eq!(triton_op.unwrap().0, TritonOpKind::Dot);
    }

    #[test]
    fn create_program_op_with_grid_dim() {
        let mut world = World::new();

        // Create a triton.program op
        let prog_entity = {
            let mut builder = OpBuilder::new(&mut world);
            let prog = builder.create_op("triton.program", &[], &[], &[]).unwrap();
            drop(builder);
            // Attach TritonOp component
            world
                .add_component(prog, TritonOp(TritonOpKind::Program))
                .expect("add TritonOp");
            // Attach GridDim component
            world
                .add_component(prog, GridDim((4, 8, 1)))
                .expect("add GridDim");
            prog
        };

        // Verify op_name
        let name = op_name(&world, prog_entity);
        assert_eq!(name, Some("triton.program".to_string()));

        // Verify 0 operands
        let ops = operands(&world, prog_entity);
        assert_eq!(ops.len(), 0);

        // Verify 0 results
        let res = results(&world, prog_entity);
        assert_eq!(res.len(), 0);

        // Verify TritonOp component
        let triton_op = world.get_component::<TritonOp>(prog_entity);
        assert!(triton_op.is_some());
        assert_eq!(triton_op.unwrap().0, TritonOpKind::Program);

        // Verify GridDim component — the primary contract test
        let grid = world.get_component::<GridDim>(prog_entity);
        assert!(grid.is_some());
        assert_eq!(grid.unwrap().0, (4, 8, 1));
    }

    #[test]
    fn create_store_op_via_builder() {
        let mut world = World::new();
        // Create operands: ptr, value, mask
        let ptr = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::i32()])
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };
        let value = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };
        let mask = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op(
                    "test.produce",
                    &[],
                    &[],
                    &[Type::integer(1, Signedness::Signless)],
                )
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };

        let store_entity = {
            let mut builder = OpBuilder::new(&mut world);
            let store = builder
                .create_op("triton.store", &[ptr, value, mask], &[], &[])
                .unwrap();
            drop(builder);
            world
                .add_component(store, TritonOp(TritonOpKind::Store))
                .expect("add TritonOp");
            store
        };

        assert_eq!(
            op_name(&world, store_entity),
            Some("triton.store".to_string())
        );
        assert_eq!(operands(&world, store_entity).len(), 3);
        assert_eq!(results(&world, store_entity).len(), 0);
    }

    #[test]
    fn create_reduce_op_via_builder() {
        let mut world = World::new();
        let operand = {
            let mut b = OpBuilder::new(&mut world);
            let op = b
                .create_op("test.produce", &[], &[], &[Type::f32()])
                .unwrap();
            drop(b);
            results(&world, op)[0]
        };

        let reduce_entity = {
            let mut builder = OpBuilder::new(&mut world);
            let reduce = builder
                .create_op("triton.reduce", &[operand], &[], &[Type::f32()])
                .unwrap();
            drop(builder);
            world
                .add_component(reduce, TritonOp(TritonOpKind::Reduce))
                .expect("add TritonOp");
            reduce
        };

        assert_eq!(
            op_name(&world, reduce_entity),
            Some("triton.reduce".to_string())
        );
        assert_eq!(operands(&world, reduce_entity).len(), 1);
        assert_eq!(results(&world, reduce_entity).len(), 1);
    }

    // ── Registry tests ────────────────────────────────────────────────────

    #[test]
    fn registry_register_triton_ops() {
        let mut registry = OpRegistry::new();
        register_triton_ops(&mut registry);

        // Verify a few key ops are registered
        let ctx = OpVerifierContext {
            operand_types: vec![Type::f32(), Type::f32(), Type::f32()],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("triton.dot", &ctx).is_ok());

        let inferred = registry.infer_result_types(
            "triton.dot",
            &[Type::f32(), Type::f32(), Type::f32()],
            &[],
        );
        assert_eq!(inferred, Some(vec![Type::f32()]));
    }

    #[test]
    fn registry_verify_triton_ops() {
        let mut registry = OpRegistry::new();
        register_triton_ops(&mut registry);

        // triton.program: 0 operands, 0 results
        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("triton.program", &ctx).is_ok());

        // Wrong operand count should fail
        let bad_ctx = OpVerifierContext {
            operand_types: vec![Type::f32()],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("triton.program", &bad_ctx).is_err());

        // triton.load: 3 operands, 1 result
        let load_ctx = OpVerifierContext {
            operand_types: vec![
                Type::i32(),
                Type::integer(1, Signedness::Signless),
                Type::f32(),
            ],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("triton.load", &load_ctx).is_ok());

        // triton.where: 3 operands, 1 result
        let where_ctx = OpVerifierContext {
            operand_types: vec![
                Type::integer(1, Signedness::Signless),
                Type::f32(),
                Type::f32(),
            ],
            result_types: vec![Type::f32()],
            attributes: vec![],
        };
        assert!(registry.verify("triton.where", &where_ctx).is_ok());
    }

    #[test]
    fn registry_verify_unknown_triton_op() {
        let mut registry = OpRegistry::new();
        register_triton_ops(&mut registry);

        let ctx = OpVerifierContext {
            operand_types: vec![],
            result_types: vec![],
            attributes: vec![],
        };
        assert!(registry.verify("triton.unknown_op", &ctx).is_err());
    }

    #[test]
    fn registry_infer_triton_dot() {
        let mut registry = OpRegistry::new();
        register_triton_ops(&mut registry);

        let result = registry.infer_result_types(
            "triton.dot",
            &[
                Type::Tensor(TensorType::new(vec![16, 32], Type::f32())),
                Type::Tensor(TensorType::new(vec![16, 32], Type::f32())),
                Type::Tensor(TensorType::new(vec![16, 32], Type::f32())),
            ],
            &[],
        );
        // dot infers from acc (operand 2)
        assert_eq!(
            result,
            Some(vec![Type::Tensor(TensorType::new(
                vec![16, 32],
                Type::f32()
            ))])
        );
    }

    #[test]
    fn registry_infer_triton_load() {
        let mut registry = OpRegistry::new();
        register_triton_ops(&mut registry);

        let memref_ty = Type::memref(
            vec![16, 32],
            Type::f32(),
            crate::ir_attrs::Attribute::UnitAttr,
            crate::ir_attrs::Attribute::UnitAttr,
        );
        let result = registry.infer_result_types(
            "triton.load",
            &[
                memref_ty.clone(),
                Type::integer(1, Signedness::Signless),
                Type::f32(),
            ],
            &[],
        );
        assert_eq!(result, Some(vec![memref_ty]));
    }
}
