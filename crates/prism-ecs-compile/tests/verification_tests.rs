use prism_ecs_compile::{CompilationStage, CompileConfig};
#[test]
fn compiler_contract_has_source_stage() {
    assert_eq!(CompilationStage::SourceDetection as u8, 0);
    let _ = CompileConfig::default();
}
