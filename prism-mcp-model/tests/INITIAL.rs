use prism_mcp_model::handlers;

#[test]
fn test_all_model_handlers_registered() {
    let hs = handlers();
    assert_eq!(hs.len(), 7);
    let names: Vec<&str> = hs.iter().map(|h| h.name()).collect();
    assert!(names.contains(&"inspect_model"));
    assert!(names.contains(&"list_model_tensors"));
    assert!(names.contains(&"get_model_tensor"));
    assert!(names.contains(&"classify_model_tensors"));
    assert!(names.contains(&"compare_models"));
    assert!(names.contains(&"estimate_model_memory"));
    assert!(names.contains(&"validate_model_assets"));
}
