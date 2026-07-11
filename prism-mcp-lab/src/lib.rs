use prism_mcp_core::McpHandler;
use std::sync::Arc;

pub mod handlers;
pub mod spec;

/// Register all lab handlers.
pub fn handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::CreateExperiment),
        Arc::new(handlers::RunExperiment),
        Arc::new(handlers::GetExperiment),
        Arc::new(handlers::ListExperiments),
        Arc::new(handlers::CancelExperiment),
        Arc::new(handlers::CompareExperiments),
        Arc::new(handlers::PromoteExperimentResult),
        Arc::new(handlers::ResumeExperiment),
    ]
}

/// SQL migration for experiments and experiment_steps tables.
pub const MIGRATION_SQL: &str = handlers::MIGRATION_SQL;
