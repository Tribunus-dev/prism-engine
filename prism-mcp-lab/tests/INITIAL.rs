use prism_mcp_core::McpHandler;
use prism_mcp_lab::handlers;
use prism_mcp_lab::spec::*;

#[test]
fn test_handler_names_unique() {
    let names = vec![
        handlers::CreateExperiment.name(),
        handlers::RunExperiment.name(),
        handlers::GetExperiment.name(),
        handlers::ListExperiments.name(),
        handlers::CancelExperiment.name(),
        handlers::CompareExperiments.name(),
        handlers::PromoteExperimentResult.name(),
        handlers::ResumeExperiment.name(),
    ];
    let mut seen = std::collections::HashSet::new();
    for n in &names {
        assert!(seen.insert(n), "duplicate handler name: {}", n);
    }
    assert_eq!(names.len(), 8);
}

#[test]
fn test_spec_default_states() {
    let spec = ExperimentSpec {
        name: "test".into(),
        description: "".into(),
        steps: vec![],
        state: ExperimentState::default(),
        result: None,
        tags: vec![],
    };
    assert_eq!(spec.state, ExperimentState::Pending);
    assert!(spec.is_terminal());
}

#[test]
fn test_spec_ready_steps() {
    let spec = ExperimentSpec {
        name: "test".into(),
        description: "".into(),
        state: ExperimentState::Pending,
        result: None,
        tags: vec![],
        steps: vec![
            ExperimentStep {
                name: "a".into(),
                tool_name: "t1".into(),
                args: serde_json::json!({}),
                depends_on: vec![],
                gates: vec![],
                state: StepState::Pending,
                result_summary: None,
            },
            ExperimentStep {
                name: "b".into(),
                tool_name: "t2".into(),
                args: serde_json::json!({}),
                depends_on: vec!["a".into()],
                gates: vec![],
                state: StepState::Pending,
                result_summary: None,
            },
        ],
    };
    let ready = spec.ready_steps();
    assert_eq!(ready, vec!["a"]);
}

#[test]
fn test_spec_is_terminal() {
    let spec = ExperimentSpec {
        name: "test".into(),
        description: "".into(),
        state: ExperimentState::Running,
        result: None,
        tags: vec![],
        steps: vec![
            ExperimentStep {
                name: "a".into(),
                tool_name: "t1".into(),
                args: serde_json::json!({}),
                depends_on: vec![],
                gates: vec![],
                state: StepState::Passed,
                result_summary: None,
            },
            ExperimentStep {
                name: "b".into(),
                tool_name: "t2".into(),
                args: serde_json::json!({}),
                depends_on: vec![],
                gates: vec![],
                state: StepState::Failed,
                result_summary: None,
            },
        ],
    };
    assert!(spec.is_terminal());
}
