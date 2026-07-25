//! Runtime model — auxiliary accessors (small load-state helpers).
//!
//! This module owns the canonical authority for the small
//! [`RuntimeModel`] accessors that don't fit into the tensor / kernel /
//! UOp / evidence / registry categories. Today this is just
//! [`num_layers`]; future auxiliaries land here.

use super::RuntimeModel;

impl RuntimeModel {
    /// Number of layers in the model (inferred from the manifest's tensor
    /// names or an explicit layer count in the execution plan).
    pub fn num_layers(&self) -> usize {
        // Phase 9: parse layer count from execution_plan or deduce from
        //          tensor-name patterns (e.g. "layers.N.attention").
        0
    }
}
