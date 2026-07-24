use crate::mil_builder::MilBuilder;
use crate::mlpackage::{self, ModelMeta};
use coreml_proto::proto::mil_spec::DataType;
use std::path::{Path, PathBuf};

pub fn compile_stateless_planar_add(
    rows: usize,
    columns: usize,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    if rows == 0 || columns == 0 {
        return Err("planar dimensions must be nonzero".into());
    }
    let program = MilBuilder::new("main")
        .input(
            "activation",
            DataType::Float32,
            &[rows as i64, columns as i64],
        )
        .input("bias", DataType::Float32, &[rows as i64, columns as i64])
        .add("activation", "bias")
        .output("add_0")
        .build()
        .map_err(|e| e.to_string())?;
    mlpackage::write_mlpackage(
        program,
        output_dir,
        &ModelMeta {
            model_name: "prism_planar_add".into(),
            function_name: "main".into(),
            short_description: "Prism planar evaluator".into(),
            version: "1".into(),
            author: "Prism".into(),
            output_name: "add_0".into(),
            inputs: vec![
                ("activation".into(), vec![rows as i64, columns as i64]),
                ("bias".into(), vec![rows as i64, columns as i64]),
            ],
            outputs: vec![("add_0".into(), vec![rows as i64, columns as i64])],
        },
    )
}
