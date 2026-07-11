use prism_mcp_core::McpHandler;
use std::sync::Arc;
pub mod handlers;
pub fn handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::CreateBenchmarkPlanHandler),
        Arc::new(handlers::RunBenchmarkHandler),
        Arc::new(handlers::CompareBenchmarksHandler),
        Arc::new(handlers::DetectPerformanceRegressionHandler),
        Arc::new(handlers::PromoteBaselineHandler),
    ]
}
