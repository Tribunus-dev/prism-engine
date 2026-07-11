// Initial verification test for prism-mcp-admission
//
// Tests handler names, descriptions, and input_schema shapes.
// `call()` validation requires DaemonState construction from prism-mcp-core,
// which depends on real backend components — covered by integration tests.

use prism_mcp_core::McpHandler;

/// Assert handler metadata is well-formed.
fn check_handler(h: &dyn McpHandler, expected_name: &str) {
    assert_eq!(h.name(), expected_name, "name mismatch");
    let desc = h.description();
    assert!(
        !desc.is_empty(),
        "handler '{expected_name}' has empty description"
    );
    let schema = h.input_schema();
    assert!(
        schema.is_object(),
        "handler '{expected_name}' input_schema is not an object"
    );
    assert_eq!(
        schema.get("type").and_then(|v| v.as_str()),
        Some("object"),
        "handler '{expected_name}' input_schema.type != 'object'"
    );
    assert!(
        schema.get("required").and_then(|v| v.as_array()).is_some(),
        "handler '{expected_name}' input_schema has no 'required' array"
    );
    // Every required field should also appear in properties
    let required = schema["required"].as_array().unwrap();
    let props = schema
        .get("properties")
        .and_then(|v| v.as_object())
        .unwrap();
    for field in required {
        let name = field.as_str().unwrap_or("<non-string>");
        assert!(
            props.contains_key(name),
            "handler '{expected_name}' required field '{name}' missing from properties"
        );
    }
}

#[test]
fn test_handler_metadata() {
    use prism_mcp_admission::handlers::*;

    let handlers: Vec<(&str, &dyn McpHandler)> = vec![
        ("analyze_tensor", &AnalyzeTensorHandler),
        (
            "generate_admission_candidates",
            &GenerateAdmissionCandidatesHandler,
        ),
        ("run_calibration", &RunCalibrationHandler),
        (
            "validate_admission_candidate",
            &ValidateAdmissionCandidateHandler,
        ),
        ("admit_tensor", &AdmitTensorHandler),
        ("compare_admission_runs", &CompareAdmissionRunsHandler),
    ];

    assert_eq!(handlers.len(), 6, "expected 6 admission handlers");

    for (name, handler) in &handlers {
        check_handler(*handler, name);
    }
}
