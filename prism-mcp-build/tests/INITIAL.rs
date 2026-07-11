// prism_mcp_build: compile-time smoke test

/// Verify the crate exports the expected handler types and function.
#[test]
fn test_module_structure() {
    // This test verifies the crate compiles and exports its public API.
    // ToolDependencies construction requires the daemon runtime, so we
    // just verify the function pointer type checks out.
    let _: fn(
        &prism_mcp_build::ToolDependencies,
    ) -> Vec<std::sync::Arc<dyn prism_mcp_build::McpHandler + Sync + Send>> =
        prism_mcp_build::handlers;

    // If the above compiles, our public API is correct.
}
