use crate::McpHandler;
use std::sync::Arc;
pub mod handlers;
pub fn handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::StartTraceHandler),
        Arc::new(handlers::StopTraceHandler),
        Arc::new(handlers::CaptureOperationTraceHandler),
        Arc::new(handlers::SummarizeTraceHandler),
        Arc::new(handlers::CompareTracesHandler),
        Arc::new(handlers::FindTraceStallsHandler),
    ]
}
