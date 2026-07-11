use prism_mcp_core::McpHandler;
use std::sync::Arc;

pub mod handlers;

pub fn handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(handlers::InspectModelHandler),
        Arc::new(handlers::ListModelTensorsHandler),
        Arc::new(handlers::GetModelTensorHandler),
        Arc::new(handlers::ClassifyModelTensorsHandler),
        Arc::new(handlers::CompareModelsHandler),
        Arc::new(handlers::EstimateModelMemoryHandler),
        Arc::new(handlers::ValidateModelAssetsHandler),
    ]
}
