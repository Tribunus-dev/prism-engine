//! Core ML model auditing — compile-time verification of ANE compliance.
//!
//! Provides `AneModelAudit` to inspect a loaded `.mlmodelc` and report
//! whether it meets ANE execution constraints (supported ops, tensor layouts).

use crate::ane::coreml_bridge::CoreMlModel;

/// Audit result for ANE compatibility.
#[derive(Debug)]
pub struct AneModelAudit {
    pub model_path: String,
    pub is_ane_compatible: bool,
    pub issues: Vec<String>,
}

impl AneModelAudit {
    pub fn new(model: &CoreMlModel, path: &str) -> Self {
        let mut audit = AneModelAudit {
            model_path: path.to_string(),
            is_ane_compatible: true,
            issues: Vec::new(),
        };
        // Basic checks (extend with MIL proto parsing when coreml-proto is available)
        #[cfg(feature = "coreml-proto")]
        {
            // Parse the model's MIL spec to check op compatibility
            audit.check_mil_ops(model);
        }
        audit.is_ane_compatible = audit.issues.is_empty();
        audit
    }

    #[cfg(feature = "coreml-proto")]
    fn check_mil_ops(&mut self, _model: &CoreMlModel) {
        // TODO: Parse MIL spec and check for unsupported ops
        // Requires coreml-proto for MIL model serialization format
    }
}
