// prism_mcp_build — MCP tool handler crate for prism-mcpd
//
// Provides build, check, test, and diff-surface tool handlers
// for the prism-engine workspace.

use std::collections::HashMap;
use std::sync::Arc;

use prism_mcp_core::{
    ArtifactStore, DbManager, EvidenceLedger, JobManager, ResourceLeaseManager, SchedulerHandle,
};
pub use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub mod handlers;

/// Local dependencies mirroring relevant fields from `DaemonState`.
/// Each crate defines its own `ToolDependencies` to avoid coupling
/// to the full daemon state.
pub struct ToolDependencies {
    pub db: Arc<DbManager>,
    pub artifact_store: ArtifactStore,
    pub evidence_ledger: EvidenceLedger,
    pub job_manager: JobManager,
    pub resource_leases: ResourceLeaseManager,
    pub scheduler_handle: SchedulerHandle,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

/// Build the list of registered MCP handler instances.
pub fn handlers(_deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::PlanBuildHandler),
        Arc::new(handlers::BuildComponentHandler),
        Arc::new(handlers::CheckComponentHandler),
        Arc::new(handlers::TestScopeHandler),
        Arc::new(handlers::CompareBuildsHandler),
        Arc::new(handlers::ChangedBuildSurfaceHandler),
    ]
}
