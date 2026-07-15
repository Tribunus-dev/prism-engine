//! Typed mapping from fixture payloads to executable buffers, surfaces,
//! constants, and outputs.

use serde::{Deserialize, Serialize};

/// Maps fixture fields to executable binding slots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingPlan {
    /// Named buffer bindings: (name, slot_index, byte_size, alignment).
    pub buffers: Vec<BindingSlot>,
    /// Named function constant bindings.
    pub constants: Vec<ConstantSlot>,
    /// Expected output buffer name.
    pub output_buffer: String,
    /// Expected output byte size.
    pub output_size: usize,
}

/// A single buffer binding slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingSlot {
    pub name: String,
    pub slot: u32,
    pub byte_size: usize,
    pub alignment: usize,
}

/// A single function constant binding slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantSlot {
    pub name: String,
    pub index: u32,
    pub value: u32,
}
