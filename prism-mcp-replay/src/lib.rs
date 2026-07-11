use prism_mcp_core::{
    ArtifactStore, DbManager, EvidenceLedger, JobManager, McpHandler, ResourceLeaseManager,
    SchedulerHandle,
};
use std::collections::HashMap;
use std::sync::Arc;

pub mod handlers;

pub struct ToolDependencies {
    pub db: Arc<DbManager>,
    pub artifact_store: ArtifactStore,
    pub evidence_ledger: EvidenceLedger,
    pub job_manager: JobManager,
    pub resource_leases: ResourceLeaseManager,
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
