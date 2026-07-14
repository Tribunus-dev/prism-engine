use std::collections::HashMap;
use std::sync::Arc;

use prism_mcp_core::{DaemonState, McpHandler};

// ── Inline handler modules ────────────────────────────────────────
mod cimage_handler;
mod doctor_handler;
mod kb_handler;
mod repo_handler;

/// Phase 1: register handlers that do NOT need DaemonState.
/// Returns a `HashMap` for insertion into DaemonState before Phase 2.
pub fn register_basic() -> anyhow::Result<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>>
{
    let mut map: HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>> = HashMap::new();

    // ── Knowledge base operations ──────────────────────────────────
    let search = kb_handler::SearchKbHandler::new()?;
    map.insert(search.name(), Arc::new(search));
    map.insert(
        kb_handler::GetDocumentHandler.name(),
        Arc::new(kb_handler::GetDocumentHandler),
    );
    map.insert(
        kb_handler::ListDocumentsHandler.name(),
        Arc::new(kb_handler::ListDocumentsHandler),
    );

    // ── Doctor ─────────────────────────────────────────────────────
    let doctor = doctor_handler::DoctorHandler::new();
    map.insert(doctor.name(), Arc::new(doctor));

    // ── Repo ───────────────────────────────────────────────────────
    let repo = repo_handler::RepoHandler::new();
    map.insert(repo.name(), Arc::new(repo));

    // ── CImage ─────────────────────────────────────────────────────
    let cimage = cimage_handler::CImageHandler::new();
    map.insert(cimage.name(), Arc::new(cimage));

    // ── Model (zero-arg handlers()) ────────────────────────────────
    for h in prism_mcp_model::handlers() {
        map.insert(h.name(), h);
    }

    // ── Trace (zero-arg handlers()) ────────────────────────────────
    for h in prism_mcp_trace::handlers() {
        map.insert(h.name(), h);
    }

    // ── Bench (zero-arg handlers()) ────────────────────────────────
    for h in prism_mcp_bench::handlers() {
        map.insert(h.name(), h);
    }

    // ── Lab (zero-arg handlers()) ──────────────────────────────────
    for h in prism_mcp_lab::handlers() {
        map.insert(h.name(), h);
    }

    Ok(map)
}

/// Phase 2: register handlers from crates that need &DaemonState
/// (for constructing their ToolDependencies). Called after DaemonState
/// is constructed with the Phase 1 tool map.
pub fn register_stateful(
    state: &DaemonState,
) -> anyhow::Result<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>> {
    let mut map = HashMap::new();

    // ── Build ──────────────────────────────────────────────────────
    {
        use prism_mcp_build::ToolDependencies;
        let deps = ToolDependencies {
            artifact_store: state.artifact_store.clone(),
            evidence_ledger: state.evidence_ledger.clone(),
            job_manager: state.job_manager.clone(),
            resource_leases: state.resource_leases.clone(),
            scheduler_handle: state.scheduler_handle.clone(),
            tools: state.tools.clone(),
        };
        for h in prism_mcp_build::handlers(&deps) {
            map.insert(h.name(), h);
        }
    }

    // ── Kernel ─────────────────────────────────────────────────────
    {
        use prism_mcp_kernel::ToolDependencies;
        let deps = ToolDependencies {
            artifact_store: state.artifact_store.clone(),
            evidence_ledger: state.evidence_ledger.clone(),
            job_manager: state.job_manager.clone(),
            resource_leases: state.resource_leases.clone(),
            scheduler_handle: state.scheduler_handle.clone(),
            tools: state.tools.clone(),
        };
        for h in prism_mcp_kernel::handlers(&deps) {
            map.insert(h.name(), h);
        }
    }

    // ── Admission ──────────────────────────────────────────────────
    {
        use prism_mcp_admission::ToolDependencies;
        let deps = ToolDependencies {
            artifact_store: state.artifact_store.clone(),
            evidence_ledger: state.evidence_ledger.clone(),
            job_manager: state.job_manager.clone(),
            resource_leases: state.resource_leases.clone(),
            scheduler_handle: state.scheduler_handle.clone(),
            tools: state.tools.clone(),
        };
        for h in prism_mcp_admission::handlers(&deps) {
            map.insert(h.name(), h);
        }
    }

    // ── Replay ─────────────────────────────────────────────────────
    {
        use prism_mcp_replay::ToolDependencies;
        let deps = ToolDependencies {
            artifact_store: state.artifact_store.clone(),
            evidence_ledger: state.evidence_ledger.clone(),
            job_manager: state.job_manager.clone(),
            resource_leases: state.resource_leases.clone(),
            scheduler_handle: state.scheduler_handle.clone(),
            tools: state.tools.clone(),
        };
        for h in prism_mcp_replay::handlers(&deps) {
            map.insert(h.name(), h);
        }
    }

    Ok(map)
}
