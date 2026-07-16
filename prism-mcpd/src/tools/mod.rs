use std::collections::HashMap;
use std::sync::Arc;

use crate::tools::subagent_handler::EcsSubagentHandler;
use parking_lot::Mutex;
use prism_ecs_core::World;
use prism_ecs_server::inference::ModelRegistry;
use prism_mcp_core::{DaemonState, McpHandler};

// ── Inline handler modules ────────────────────────────────────────
mod agent_tick_handler;
mod cimage_handler;
mod conversation_handler;
mod coordination_handler;
mod doctor_handler;
mod hf_handler;
mod hw_handler;
mod inference_handler;
mod job_handler;
mod kb_handler;
mod repo_handler;
mod resolve_path_handler;
mod subagent_handler;

/// Phase 1: register handlers that do NOT need DaemonState.
/// Returns a `HashMap` for insertion into DaemonState before Phase 2.
pub fn register_basic(
    registry: Arc<Mutex<ModelRegistry>>,
) -> anyhow::Result<HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>>> {
    let mut map: HashMap<&'static str, Arc<dyn McpHandler + Sync + Send>> = HashMap::new();
    for (name, action) in [
        ("agent_session_start", "start_session"),
        ("agent_session_heartbeat", "heartbeat"),
        ("agent_session_close", "close_session"),
        ("agent_work_create", "create_work"),
        ("agent_work_list", "list_work"),
        ("agent_work_claim", "claim_work"),
        ("agent_work_release", "release_claim"),
        ("agent_path_lock", "acquire_path"),
        ("agent_path_unlock", "release_path"),
        ("agent_coordination_recover", "recover"),
        ("agent_work_handoff", "handoff"),
        ("agent_coordination_event", "event"),
        ("agent_coordination_status", "status"),
    ] {
        map.insert(
            name,
            Arc::new(coordination_handler::CoordinationHandler { name, action }),
        );
    }

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
    map.insert(
        "resolve_path",
        Arc::new(resolve_path_handler::ResolvePathHandler),
    );

    // ── CImage ─────────────────────────────────────────────────────
    let cimage = cimage_handler::CImageHandler::new();
    map.insert(cimage.name(), Arc::new(cimage));

    // ── Inference ─────────────────────────────────────────────────
    let inference = inference_handler::InferenceHandler::new(registry);
    map.insert(inference.name(), Arc::new(inference));

    // ── Subagent ────────────────────────────────────────────────
    // Subagent needs a World — registered in register_stateful (Phase 2)

    // ── Agent Tick ─────────────────────────────────────────────
    let agent_tick = agent_tick_handler::AgentTickHandler::new();
    map.insert(agent_tick.name(), Arc::new(agent_tick));

    // ── Conversation ───────────────────────────────────────────────
    let conversation = conversation_handler::ConversationHandler;
    map.insert("conversation", Arc::new(conversation));

    // ── Model (zero-arg handlers()) ────────────────────────────────
    // ── Hardware Probe ─────────────────────────────────────────────
    let hw_probe = hw_handler::HwProbeHandler::new();
    map.insert(hw_probe.name(), Arc::new(hw_probe));

    // ── HuggingFace ────────────────────────────────────────────────
    let hf = hf_handler::HfHandler::new();
    map.insert(hf.name(), Arc::new(hf));

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

    // ── Browser ────────────────────────────────────────────────────
    {
        let deps = prism_mcp_browser::ToolDependencies {
            artifact_store: state.artifact_store.clone(),
            evidence_ledger: state.evidence_ledger.clone(),
            resource_leases: state.resource_leases.clone(),
            tools: state.tools.clone(),
        };
        for h in prism_mcp_browser::handlers(&deps) {
            map.insert(h.name(), h);
        }
    }

    // ── Coordination jobs ──────────────────────────────────────────
    map.insert(
        "run_job",
        Arc::new(job_handler::JobHandler::new(state.job_manager.clone())),
    );
    map.insert(
        "test_scope",
        Arc::new(job_handler::TestScopeJobHandler::new(
            state.job_manager.clone(),
        )),
    );

    // ── Subagent (ECS-backed) ────────────────────────────────────
    let ecs_world = Arc::new(Mutex::new(World::new()));
    map.insert("subagent", Arc::new(EcsSubagentHandler::new(ecs_world)));

    Ok(map)
}
