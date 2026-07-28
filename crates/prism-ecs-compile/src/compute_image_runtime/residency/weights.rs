//! Weight object classification.

use serde::{Deserialize, Serialize};

use super::plan::ResidencyClass;

/// A weight object in the compute image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightObject {
    /// Object identifier.
    pub object_id: String,
    /// Byte size.
    pub byte_size: u64,
    /// Residency class.
    pub residency_class: ResidencyClass,
}

/// Classifier that assigns a [`ResidencyClass`] to weight objects.
#[derive(Debug, Clone, Default)]
pub struct ResidencyClassifier;

impl ResidencyClassifier {
    /// Create a new classifier.
    pub fn new() -> Self {
        Self
    }

    /// Classify a weight object based on its name and size.
    pub fn classify(&self, name: &str, _byte_size: u64) -> ResidencyClass {
        if name.contains("embed") || name.contains("lm_head") || name.contains("norm") {
            ResidencyClass::MandatoryAtSessionStart
        } else {
            ResidencyClass::MandatoryBeforePhase
        }
    }
}
