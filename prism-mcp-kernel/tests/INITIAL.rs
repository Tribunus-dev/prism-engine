// prism-mcp-kernel smoke tests
//
// Validates handler naming, descriptions, and schema structure by
// instantiating handler structs directly — no runtime deps needed.

use prism_mcp_core::McpHandler;
use prism_mcp_kernel::handlers::{
    AnalyzeKernelResources, CompareKernels, CompileKernelCandidates, CompileKernelRecipe,
    DisassembleKernel, InspectCompiledKernel, ListKernelBackends, RegisterKernel,
    ValidateKernelAbi,
};
use prism_mcp_kernel::MIGRATION_SQL;

fn all_handler_names() -> Vec<&'static str> {
    vec![
        ListKernelBackends::new().name(),
        CompileKernelRecipe.name(),
        CompileKernelCandidates.name(),
        InspectCompiledKernel.name(),
        DisassembleKernel.name(),
        AnalyzeKernelResources.name(),
        ValidateKernelAbi.name(),
        CompareKernels.name(),
        RegisterKernel.name(),
    ]
}

#[test]
fn test_handler_names_are_unique() {
    let names = all_handler_names();
    assert_eq!(names.len(), 9, "expected exactly 9 unique handler names");
    let mut sorted = names.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 9, "duplicate handler names detected");
}

#[test]
fn test_handler_names_use_underscores() {
    for name in all_handler_names() {
        assert!(
            !name.contains('-'),
            "handler name '{}' uses hyphens instead of underscores",
            name
        );
        assert!(
            name.contains('_'),
            "handler name '{}' should use underscores",
            name
        );
    }
}

#[test]
fn test_descriptions_are_nonempty() {
    let handlers: Vec<Box<dyn prism_mcp_core::McpHandler + Sync>> = vec![
        Box::new(ListKernelBackends::new()),
        Box::new(CompileKernelRecipe),
        Box::new(CompileKernelCandidates),
        Box::new(InspectCompiledKernel),
        Box::new(DisassembleKernel),
        Box::new(AnalyzeKernelResources),
        Box::new(ValidateKernelAbi),
        Box::new(CompareKernels),
        Box::new(RegisterKernel),
    ];
    for h in &handlers {
        assert!(
            !h.description().is_empty(),
            "handler '{}' has empty description",
            h.name()
        );
    }
}

#[test]
fn test_input_schema_is_valid_object() {
    let handlers: Vec<Box<dyn prism_mcp_core::McpHandler + Sync>> = vec![
        Box::new(ListKernelBackends::new()),
        Box::new(CompileKernelRecipe),
        Box::new(CompileKernelCandidates),
        Box::new(InspectCompiledKernel),
        Box::new(DisassembleKernel),
        Box::new(AnalyzeKernelResources),
        Box::new(ValidateKernelAbi),
        Box::new(CompareKernels),
        Box::new(RegisterKernel),
    ];
    for h in &handlers {
        let schema = h.input_schema();
        assert!(
            schema.is_object(),
            "handler '{}' input_schema is not an object",
            h.name()
        );
        let obj = schema.as_object().unwrap();
        assert!(
            obj.contains_key("type"),
            "handler '{}' schema missing 'type'",
            h.name()
        );
        assert!(
            obj.contains_key("properties"),
            "handler '{}' schema missing 'properties'",
            h.name()
        );
    }
}

#[test]
fn test_migration_sql_has_both_tables() {
    assert!(!MIGRATION_SQL.is_empty(), "MIGRATION_SQL must not be empty");
    assert!(
        MIGRATION_SQL.contains("CREATE TABLE IF NOT EXISTS kernel_recipes"),
        "MIGRATION_SQL must create kernel_recipes"
    );
    assert!(
        MIGRATION_SQL.contains("CREATE TABLE IF NOT EXISTS kernel_registry"),
        "MIGRATION_SQL must create kernel_registry"
    );
}
