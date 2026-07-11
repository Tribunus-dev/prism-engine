/// Basic sanity tests for prism-mcp-trace handler crate.
/// Tests skip full DaemonState construction and focus on handler metadata
/// and the top-level `handlers()` export.

#[test]
fn test_handlers_export() {
    let h = prism_mcp_trace::handlers();
    assert_eq!(h.len(), 6);
    let names: Vec<&str> = h.iter().map(|h| h.name()).collect();
    assert!(names.contains(&"start_trace"));
    assert!(names.contains(&"stop_trace"));
    assert!(names.contains(&"capture_operation_trace"));
    assert!(names.contains(&"summarize_trace"));
    assert!(names.contains(&"compare_traces"));
    assert!(names.contains(&"find_trace_stalls"));
}

#[test]
fn test_handler_descriptions_are_nonempty() {
    let h = prism_mcp_trace::handlers();
    for handler in &h {
        assert!(
            !handler.description().is_empty(),
            "{} has empty description",
            handler.name()
        );
    }
}

#[test]
fn test_handler_names_are_unique() {
    let h = prism_mcp_trace::handlers();
    let mut names: Vec<&str> = h.iter().map(|h| h.name()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 6, "handler names must be unique");
}
