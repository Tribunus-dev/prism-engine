// prism_mcp_kernel — MCP tool handler crate for prism-mcpd
//
// Provides 9 kernel-management tools:
//   list_kernel_backends, compile_kernel_recipe, compile_kernel_candidates,
//   inspect_compiled_kernel, disassemble_kernel, analyze_kernel_resources,
//   validate_kernel_abi, compare_kernels, register_kernel

use prism_mcp_core::McpHandler;
use std::collections::HashMap;
use std::sync::Arc;

pub mod handlers;

/// Shared dependencies injected into all kernel handler calls.
pub struct ToolDependencies {
    pub db: Arc<prism_mcp_core::DbManager>,
    pub artifact_store: prism_mcp_core::ArtifactStore,
    pub evidence_ledger: prism_mcp_core::EvidenceLedger,
    pub job_manager: prism_mcp_core::JobManager,
    pub resource_leases: prism_mcp_core::ResourceLeaseManager,
    pub scheduler_handle: prism_mcp_core::SchedulerHandle,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

/// Migration SQL executed when the daemon's database is opened.
/// Creates kernel_recipes and kernel_registry tables.
pub const MIGRATION_SQL: &str = "\
CREATE TABLE IF NOT EXISTS kernel_recipes (
    recipe_name    TEXT NOT NULL,
    backend        TEXT NOT NULL DEFAULT 'metal',
    artifact_hash  BLOB NOT NULL,
    source_preview TEXT,
    compiled_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (recipe_name, backend)
);

CREATE TABLE IF NOT EXISTS kernel_registry (
    name           TEXT PRIMARY KEY,
    backend        TEXT NOT NULL DEFAULT 'metal',
    artifact_hash  BLOB NOT NULL,
    byte_len       INTEGER NOT NULL,
    target         TEXT,
    registered_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
";

/// Register all kernel handlers with the tool registry.
pub fn handlers(_deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::ListKernelBackends::new()),
        Arc::new(handlers::CompileKernelRecipe),
        Arc::new(handlers::CompileKernelCandidates),
        Arc::new(handlers::InspectCompiledKernel),
        Arc::new(handlers::DisassembleKernel),
        Arc::new(handlers::AnalyzeKernelResources),
        Arc::new(handlers::CompareKernels),
        Arc::new(handlers::ValidateKernelAbi),
        Arc::new(handlers::RegisterKernel),
    ]
}
