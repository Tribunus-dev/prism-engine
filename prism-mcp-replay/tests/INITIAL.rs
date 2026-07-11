use prism_mcp_core::McpHandler;

// ── capture_replay ─────────────────────────────────────────────────────────

#[test]
fn test_capture_replay_handler_metadata() {
    let h = prism_mcp_replay::handlers::CaptureReplayHandler;
    assert_eq!(h.name(), "capture_replay");
    assert!(!h.description().is_empty());
    let schema = h.input_schema();
    assert!(schema.is_object());
    assert!(schema["properties"]["invocation_id"].is_object());
}

// ── run_replay ─────────────────────────────────────────────────────────────

#[test]
fn test_run_replay_handler_metadata() {
    let h = prism_mcp_replay::handlers::RunReplayHandler;
    assert_eq!(h.name(), "run_replay");
    assert!(!h.description().is_empty());
    let schema = h.input_schema();
    assert!(schema.is_object());
}

// ── minimize_replay ────────────────────────────────────────────────────────

#[test]
fn test_minimize_replay_handler_metadata() {
    let h = prism_mcp_replay::handlers::MinimizeReplayHandler;
    assert_eq!(h.name(), "minimize_replay");
    assert!(!h.description().is_empty());
    let schema = h.input_schema();
    assert!(schema.is_object());
}

// ── compare_replays ────────────────────────────────────────────────────────

#[test]
fn test_compare_replays_handler_metadata() {
    let h = prism_mcp_replay::handlers::CompareReplaysHandler;
    assert_eq!(h.name(), "compare_replays");
    assert!(!h.description().is_empty());
    let schema = h.input_schema();
    assert!(schema.is_object());
}

// ── export_replay ──────────────────────────────────────────────────────────

#[test]
fn test_export_replay_handler_metadata() {
    let h = prism_mcp_replay::handlers::ExportReplayHandler;
    assert_eq!(h.name(), "export_replay");
    assert!(!h.description().is_empty());
    let schema = h.input_schema();
    assert!(schema.is_object());
}

// ── import_replay ──────────────────────────────────────────────────────────

#[test]
fn test_import_replay_handler_metadata() {
    let h = prism_mcp_replay::handlers::ImportReplayHandler;
    assert_eq!(h.name(), "import_replay");
    assert!(!h.description().is_empty());
    let schema = h.input_schema();
    assert!(schema.is_object());
}

// ── handler factory — unique names ─────────────────────────────────────────

#[test]
fn test_handler_unique_names() {
    let names: Vec<&str> = vec![
        prism_mcp_replay::handlers::CaptureReplayHandler.name(),
        prism_mcp_replay::handlers::RunReplayHandler.name(),
        prism_mcp_replay::handlers::MinimizeReplayHandler.name(),
        prism_mcp_replay::handlers::CompareReplaysHandler.name(),
        prism_mcp_replay::handlers::ExportReplayHandler.name(),
        prism_mcp_replay::handlers::ImportReplayHandler.name(),
    ];
    let mut sorted = names.clone();
    sorted.sort();
    let mut deduped = sorted.clone();
    deduped.dedup();
    assert_eq!(sorted, deduped, "all handler names must be unique");
    assert_eq!(names.len(), 6, "expected exactly 6 handlers");
}
