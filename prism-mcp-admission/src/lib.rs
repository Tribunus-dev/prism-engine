// prism_mcp_admission — MCP tool handler crate for prism-mcpd
//
// Admission pipeline: tensor analysis, quantization candidate generation,
// calibration, candidate validation, tensor admission, and run comparison.

use prism_mcp_core::{
    ArtifactStore, DbManager, EvidenceLedger, JobManager, McpHandler, ResourceLeaseManager,
    SchedulerHandle,
};
use std::collections::HashMap;
use std::sync::Arc;

pub struct ToolDependencies {
    pub db: Arc<DbManager>,
    pub artifact_store: ArtifactStore,
    pub evidence_ledger: EvidenceLedger,
    pub job_manager: JobManager,
    pub resource_leases: ResourceLeaseManager,
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
