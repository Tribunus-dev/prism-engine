use crate::{McpHandler, SchedulerHandle};
use std::collections::HashMap;
use std::sync::Arc;

pub mod handlers;

pub struct ToolDependencies {
    pub artifact_store: Arc<dyn crate::ArtifactRepository>,
    pub evidence_ledger: Arc<dyn crate::EvidenceStore>,
    pub job_manager: Arc<dyn crate::JobStore>,
    pub resource_leases: Arc<dyn crate::LeaseStore>,
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
