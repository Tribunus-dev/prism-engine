use crate::mil_builder::MilBuilder;
use crate::mlpackage::{self, ModelMeta};
use coreml_proto::proto::mil_spec::DataType;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub struct StatelessTernaryTileSpec {
    pub input_width: usize,
    pub output_width: usize,
}

pub fn compile_stateless_ternary_tile(
    spec: StatelessTernaryTileSpec,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    if spec.input_width == 0 || spec.output_width == 0 {
        return Err("ternary tile dimensions must be nonzero".into());
    }
    let program = MilBuilder::new("main")
        .input("activation", DataType::Float32, &[spec.input_width as i64])
        .input(
            "ternary_weights",
            DataType::Float32,
            &[spec.input_width as i64, spec.output_width as i64],
        )
        .matmul("activation", "ternary_weights")
        .output("matmul_0")
        .build()
        .map_err(|e| e.to_string())?;
    mlpackage::write_mlpackage(
        program,
        output_dir,
        &ModelMeta {
            model_name: "prism_ternary_tile".into(),
            function_name: "main".into(),
            short_description: "Prism ternary evaluator tile".into(),
            version: "1".into(),
            author: "Prism".into(),
            output_name: "matmul_0".into(),
            inputs: vec![
                ("activation".into(), vec![spec.input_width as i64]),
                (
                    "ternary_weights".into(),
                    vec![spec.input_width as i64, spec.output_width as i64],
                ),
            ],
            outputs: vec![("matmul_0".into(), vec![spec.output_width as i64])],
        },
    )
}
