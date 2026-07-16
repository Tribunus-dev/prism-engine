//! prism-mcp-handlers: merged tool handler implementations for prism-mcpd.

use prism_mcp_core::{McpHandler, SchedulerHandle};
use std::collections::HashMap;
use std::sync::Arc;

pub mod browser;
pub mod build;
pub mod kernel;

/// Local dependencies injected into handlers that need crate state.
pub struct ToolDependencies {
    pub artifact_store: Arc<dyn prism_mcp_core::ArtifactRepository>,
    pub evidence_ledger: Arc<dyn prism_mcp_core::EvidenceStore>,
    pub job_manager: Arc<dyn prism_mcp_core::JobStore>,
    pub resource_leases: Arc<dyn prism_mcp_core::LeaseStore>,
    pub scheduler_handle: SchedulerHandle,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

/// Register all handlers from build, kernel, and browser sub-modules.
pub fn handlers(deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    let mut all = Vec::new();
    all.extend(build::handlers(deps));
    all.extend(kernel::handlers(deps));
    all.extend(browser::handlers(deps));
    all
}
