//! Variant selection refusal — pure data types.

use serde::{Deserialize, Serialize};

/// Reasons a variant selection may be refused.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariantSelectionRefusal {
    /// No variant matches the requested shape class.
    NoMatchingShape,
    /// The variant is not qualified for the runtime.
    UnqualifiedVariant,
    /// The selection policy rejected the variant.
    PolicyRejected,
}
