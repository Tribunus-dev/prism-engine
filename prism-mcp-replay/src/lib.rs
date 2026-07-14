use prism_mcp_core::{McpHandler, SchedulerHandle};
use std::collections::HashMap;
use std::sync::Arc;

pub mod handlers;

pub struct ToolDependencies {
    pub artifact_store: Arc<dyn prism_mcp_core::ArtifactRepository>,
    pub evidence_ledger: Arc<dyn prism_mcp_core::EvidenceStore>,
    pub job_manager: Arc<dyn prism_mcp_core::JobStore>,
    pub resource_leases: Arc<dyn prism_mcp_core::LeaseStore>,
    pub scheduler_handle: SchedulerHandle,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

pub fn handlers(_deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::CaptureReplayHandler),
        Arc::new(handlers::RunReplayHandler),
        Arc::new(handlers::MinimizeReplayHandler),
        Arc::new(handlers::CompareReplaysHandler),
        Arc::new(handlers::ExportReplayHandler),
        Arc::new(handlers::ImportReplayHandler),
    ]
}
