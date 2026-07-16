// prism_mcp_build — MCP tool handler crate for prism-mcpd
//
// Provides build, check, test, and diff-surface tool handlers
// for the prism-engine workspace.

use std::collections::HashMap;
use std::sync::Arc;

use prism_mcp_core::SchedulerHandle;
pub use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};

pub mod handlers;

use crate::ToolDependencies;

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
