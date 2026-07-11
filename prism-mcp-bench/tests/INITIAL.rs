// prism_mcp_bench — initial integration smoke test
//
// Verifies that all 5 handlers are registered with the correct names.

use prism_mcp_bench::handlers::{
    CompareBenchmarksHandler, CreateBenchmarkPlanHandler, DetectPerformanceRegressionHandler,
    PromoteBaselineHandler, RunBenchmarkHandler,
};
use prism_mcp_core::McpHandler;

#[test]
fn handler_names_match_spec() {
    let handlers: Vec<Box<dyn McpHandler>> = vec![
        Box::new(CreateBenchmarkPlanHandler),
        Box::new(RunBenchmarkHandler),
        Box::new(CompareBenchmarksHandler),
        Box::new(DetectPerformanceRegressionHandler),
        Box::new(PromoteBaselineHandler),
    ];

    let expected = &[
        "create_benchmark_plan",
        "run_benchmark",
        "compare_benchmarks",
        "detect_performance_regression",
        "promote_baseline",
    ];

    for (h, exp) in handlers.iter().zip(expected) {
        assert_eq!(h.name(), *exp, "handler name mismatch");
    }

    assert_eq!(handlers.len(), expected.len(), "wrong number of handlers");
}

#[test]
fn each_handler_has_description() {
    let handlers: Vec<Box<dyn McpHandler>> = vec![
        Box::new(CreateBenchmarkPlanHandler),
        Box::new(RunBenchmarkHandler),
        Box::new(CompareBenchmarksHandler),
        Box::new(DetectPerformanceRegressionHandler),
        Box::new(PromoteBaselineHandler),
    ];

    for h in &handlers {
        let desc = h.description();
        assert!(
            !desc.is_empty(),
            "handler {} has empty description",
            h.name()
        );
    }
}

#[test]
fn each_handler_has_input_schema() {
    let handlers: Vec<Box<dyn McpHandler>> = vec![
        Box::new(CreateBenchmarkPlanHandler),
        Box::new(RunBenchmarkHandler),
        Box::new(CompareBenchmarksHandler),
        Box::new(DetectPerformanceRegressionHandler),
        Box::new(PromoteBaselineHandler),
    ];

    for h in &handlers {
        let schema = h.input_schema();
        assert!(
            schema.is_object(),
            "handler {} input_schema is not an object",
            h.name()
        );
        assert!(
            schema.get("properties").is_some(),
            "handler {} input_schema missing properties",
            h.name()
        );
    }
}

#[test]
fn migration_constant_is_valid_sql() {
    let sql = prism_mcp_bench::handlers::MIGRATION;
    assert!(
        sql.contains("CREATE TABLE"),
        "MIGRATION must create a table"
    );
    assert!(
        sql.contains("benchmark_baselines"),
        "MIGRATION must reference benchmark_baselines"
    );
    assert!(
        sql.contains("baseline_name TEXT PRIMARY KEY"),
        "MIGRATION missing baseline_name PK"
    );
}
