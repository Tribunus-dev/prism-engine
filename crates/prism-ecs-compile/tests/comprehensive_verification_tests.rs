use prism_ecs_compile::{CompilationStage, SearchConfig};
#[test]
fn search_config_and_pipeline_are_public() {
    let _ = SearchConfig::default();
    assert!(matches!(
        CompilationStage::Certify,
        CompilationStage::Certify
    ));
}
