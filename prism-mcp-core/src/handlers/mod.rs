//! Tool handler implementations for all absorbed MCP handler crates.
//!
//! Each submodule inlines the handler code from the absorbed crate.
//! Registration functions follow the same signatures as the original crates.

use crate::{McpHandler, SchedulerHandle};
use std::collections::HashMap;
use std::sync::Arc;

pub mod admission;
pub mod bench;
pub mod lab;
pub mod lab_spec;
pub mod model;
pub mod replay;
pub mod trace;

/// Local dependencies injected into handlers that need crate state.
pub struct ToolDependencies {
    pub artifact_store: Arc<dyn crate::ArtifactRepository>,
    pub evidence_ledger: Arc<dyn crate::EvidenceStore>,
    pub job_manager: Arc<dyn crate::JobStore>,
    pub resource_leases: Arc<dyn crate::LeaseStore>,
    pub scheduler_handle: SchedulerHandle,
    pub tools: Arc<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>,
}

// ── Zero-arg registration functions ────────────────────────────────────────

/// Register all model inspector handlers.
pub fn model_handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(model::InspectModelHandler),
        Arc::new(model::ListModelTensorsHandler),
        Arc::new(model::GetModelTensorHandler),
        Arc::new(model::ClassifyModelTensorsHandler),
        Arc::new(model::CompareModelsHandler),
        Arc::new(model::EstimateModelMemoryHandler),
        Arc::new(model::ValidateModelAssetsHandler),
    ]
}

/// Register all benchmark handlers.
pub fn bench_handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(bench::CreateBenchmarkPlanHandler),
        Arc::new(bench::RunBenchmarkHandler),
        Arc::new(bench::CompareBenchmarksHandler),
        Arc::new(bench::DetectPerformanceRegressionHandler),
        Arc::new(bench::PromoteBaselineHandler),
    ]
}

/// Register all trace handlers.
pub fn trace_handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(trace::StartTraceHandler),
        Arc::new(trace::StopTraceHandler),
        Arc::new(trace::CaptureOperationTraceHandler),
        Arc::new(trace::SummarizeTraceHandler),
        Arc::new(trace::CompareTracesHandler),
        Arc::new(trace::FindTraceStallsHandler),
    ]
}

/// Register all lab (experiment lifecycle) handlers.
pub fn lab_handlers() -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(lab::CreateExperiment),
        Arc::new(lab::RunExperiment),
        Arc::new(lab::GetExperiment),
        Arc::new(lab::ListExperiments),
        Arc::new(lab::CancelExperiment),
        Arc::new(lab::CompareExperiments),
        Arc::new(lab::PromoteExperimentResult),
        Arc::new(lab::ResumeExperiment),
    ]
}

// ── Dependency-needing registration functions ──────────────────────────────

/// Register all admission pipeline handlers.
pub fn admission_handlers(_deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(admission::AnalyzeTensorHandler),
        Arc::new(admission::GenerateAdmissionCandidatesHandler),
        Arc::new(admission::RunCalibrationHandler),
        Arc::new(admission::ValidateAdmissionCandidateHandler),
        Arc::new(admission::AdmitTensorHandler),
        Arc::new(admission::CompareAdmissionRunsHandler),
    ]
}

/// Register all replay handlers.
pub fn replay_handlers(_deps: &ToolDependencies) -> Vec<Arc<dyn McpHandler + Sync + Send>> {
    vec![
        Arc::new(replay::CaptureReplayHandler),
        Arc::new(replay::RunReplayHandler),
        Arc::new(replay::MinimizeReplayHandler),
        Arc::new(replay::CompareReplaysHandler),
        Arc::new(replay::ExportReplayHandler),
        Arc::new(replay::ImportReplayHandler),
    ]
}
