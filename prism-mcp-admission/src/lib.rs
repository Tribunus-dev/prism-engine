// prism_mcp_admission — MCP tool handler crate for prism-mcpd
//
// Admission pipeline: tensor analysis, quantization candidate generation,
// calibration, candidate validation, tensor admission, and run comparison.

use prism_mcp_core::{McpHandler, SchedulerHandle};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolDependencies {
    pub artifact_store: Arc<dyn prism_mcp_core::ArtifactRepository>,
    pub evidence_ledger: Arc<dyn prism_mcp_core::EvidenceStore>,
    pub job_manager: Arc<dyn prism_mcp_core::JobStore>,
    pub resource_leases: Arc<dyn prism_mcp_core::LeaseStore>,
    pub scheduler_handle: SchedulerHandle,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

pub mod handlers;

pub fn handlers(_deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::AnalyzeTensorHandler),
        Arc::new(handlers::GenerateAdmissionCandidatesHandler),
        Arc::new(handlers::RunCalibrationHandler),
        Arc::new(handlers::ValidateAdmissionCandidateHandler),
        Arc::new(handlers::AdmitTensorHandler),
        Arc::new(handlers::CompareAdmissionRunsHandler),
    ]
}
