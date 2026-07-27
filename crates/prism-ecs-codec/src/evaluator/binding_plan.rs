//! BindingPlan — the typed mapping from fixture payloads to
//! executable buffer bindings, surface bindings, function
//! constants, and outputs.
//!
//! This module owns the canonical authority for the binding plan
//! that maps a fixture's named fields to an executable's binding
//! slots. The plan is part of a [`GeneratedExecutable`](super::GeneratedExecutable)'s
//! identity: changing the binding plan produces a different
//! executable even with the same ABI.

use serde::{Deserialize, Serialize};

/// Maps fixture fields to executable binding slots.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BindingSlot {
    pub name: String,
    pub slot: u32,
    pub byte_size: usize,
    pub alignment: usize,
}

/// A single function constant binding slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstantSlot {
    pub name: String,
    pub index: u32,
    pub value: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binding_plan_is_constructible_and_serializable() {
        let plan = BindingPlan {
            buffers: vec![
                BindingSlot {
                    name: "input".to_string(),
                    slot: 0,
                    byte_size: 1024,
                    alignment: 16,
                },
                BindingSlot {
                    name: "weights".to_string(),
                    slot: 1,
                    byte_size: 2048,
                    alignment: 256,
                },
            ],
            constants: vec![ConstantSlot {
                name: "tile_m".to_string(),
                index: 0,
                value: 64,
            }],
            output_buffer: "output".to_string(),
            output_size: 1024,
        };

        let json = serde_json::to_string(&plan).expect("serialize");
        let restored: BindingPlan = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, plan);
    }

    #[test]
    fn empty_binding_plan_round_trips() {
        let plan = BindingPlan {
            buffers: vec![],
            constants: vec![],
            output_buffer: "out".to_string(),
            output_size: 0,
        };
        let json = serde_json::to_string(&plan).expect("serialize");
        let restored: BindingPlan = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored, plan);
    }
}
