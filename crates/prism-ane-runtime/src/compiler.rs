//! ANE compiler — compiles MIL programs into loadable ANE models via CoreML.
//!
//! Behind the `coreml` feature, this module invokes CoreML's MLModel serialization
//! path to compile a MIL program string into a model package that the ANE can load.
//! Without the feature, all operations return a clear "not available" error.

/// A compiled ANE model, ready for dispatch.
#[derive(Debug, Clone)]
pub struct AneBinary {
    /// The raw compiled model bytes (mlmodelc package or protobuf blob).
    pub binary: Vec<u8>,
    /// Entry point name within the model.
    pub entry_point: String,
}

/// Compile a MIL program string into an [`AneBinary`].
///
/// Requires the `coreml` feature and macOS. On other platforms or when the
/// feature is disabled, returns a clear "not available" error.
///
/// # Errors
///
/// - Returns an error when the CoreML toolchain is unavailable.
/// - Returns an error when the MIL program cannot be parsed.
/// - Returns an error when compilation fails.
#[cfg(feature = "coreml")]
pub fn compile_mil(mil_source: &str) -> Result<AneBinary, String> {
    use prism_ane::{
        mil_builder::MilBuilder,
        mlpackage::{self, ModelMeta},
    };
    let dims = mil_source
        .split_whitespace()
        .find_map(|token| token.strip_prefix("matmul_"))
        .and_then(|v| {
            let mut p = v.split('x').filter_map(|x| x.parse::<i64>().ok());
            Some((p.next()?, p.next()?, p.next()?))
        })
        .ok_or_else(|| "ANE MIL source must contain a matmul_MxKxN declaration".to_string())?;
    let (m, k, n) = dims;
    let program = MilBuilder::new("main")
        .input(
            "a",
            coreml_proto::proto::mil_spec::DataType::Float32,
            &[m, k],
        )
        .input(
            "b",
            coreml_proto::proto::mil_spec::DataType::Float32,
            &[k, n],
        )
        .matmul("a", "b")
        .output("matmul_0")
        .build()
        .map_err(|e| format!("build MIL program: {e}"))?;
    let temp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let meta = ModelMeta {
        model_name: "prism_ane_kernel".into(),
        function_name: "main".into(),
        short_description: "Prism ANE evaluator kernel".into(),
        version: "1.0".into(),
        author: "Prism".into(),
        output_name: "matmul_0".into(),
        inputs: vec![("a".into(), vec![m, k]), ("b".into(), vec![k, n])],
        outputs: vec![("matmul_0".into(), vec![m, n])],
    };
    let package = mlpackage::write_mlpackage(program, temp.path(), &meta)?;
    let out = temp.path().join("compiled");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let status = std::process::Command::new("xcrun")
        .args(["coremlcompiler", "compile"])
        .arg(&package)
        .arg(&out)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("coremlcompiler failed".into());
    }
    let compiled = out.join("prism_ane_kernel.mlmodelc");
    if !compiled.exists() {
        return Err(format!("missing compiled model {}", compiled.display()));
    }
    Ok(AneBinary {
        binary: prism_ane::pack_mlmodelc(&compiled)?,
        entry_point: "main".into(),
    })
}

/// Compile the stateless INT8 matmul contract used by the bounded evaluator.
/// Inputs and outputs remain explicitly typed so an IOSurface-backed arena can
/// be bound without an implicit float staging buffer.
#[cfg(feature = "coreml")]
pub fn compile_mil_int8(mil_source: &str) -> Result<AneBinary, String> {
    use prism_ane::{
        mil_builder::MilBuilder,
        mlpackage::{self, ModelMeta},
    };
    let dims = mil_source
        .split_whitespace()
        .find_map(|token| token.strip_prefix("matmul_"))
        .and_then(|v| {
            let mut p = v.split('x').filter_map(|x| x.parse::<i64>().ok());
            Some((p.next()?, p.next()?, p.next()?))
        })
        .ok_or_else(|| "ANE INT8 MIL source must contain a matmul_MxKxN declaration".to_string())?;
    let (m, k, n) = dims;
    let program = MilBuilder::new("main")
        .input("a", coreml_proto::proto::mil_spec::DataType::Int8, &[m, k])
        .input("b", coreml_proto::proto::mil_spec::DataType::Int8, &[k, n])
        .matmul("a", "b")
        .output("matmul_0")
        .build()
        .map_err(|e| format!("build INT8 MIL program: {e}"))?;
    let temp = tempfile::tempdir().map_err(|e| e.to_string())?;
    let meta = ModelMeta {
        model_name: "prism_ane_int8_kernel".into(),
        function_name: "main".into(),
        short_description: "Prism stateless planar INT8 evaluator kernel".into(),
        version: "1.0".into(),
        author: "Prism".into(),
        output_name: "matmul_0".into(),
        inputs: vec![("a".into(), vec![m, k]), ("b".into(), vec![k, n])],
        outputs: vec![("matmul_0".into(), vec![m, n])],
    };
    let package = mlpackage::write_mlpackage(program, temp.path(), &meta)?;
    let out = temp.path().join("compiled");
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let status = std::process::Command::new("xcrun")
        .args(["coremlcompiler", "compile"])
        .arg(&package)
        .arg(&out)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("coremlcompiler failed for INT8 MIL".into());
    }
    let compiled = out.join("prism_ane_int8_kernel.mlmodelc");
    if !compiled.exists() {
        return Err(format!("missing compiled model {}", compiled.display()));
    }
    Ok(AneBinary {
        binary: prism_ane::pack_mlmodelc(&compiled)?,
        entry_point: "main".into(),
    })
}

#[cfg(not(feature = "coreml"))]
pub fn compile_mil_int8(_mil_source: &str) -> Result<AneBinary, String> {
    Err("ANE INT8 compilation requires the 'coreml' feature and macOS".into())
}

#[cfg(not(feature = "coreml"))]
pub fn compile_mil(_mil_source: &str) -> Result<AneBinary, String> {
    Err("ANE compilation requires the 'coreml' feature and macOS. \
         Enable prism-ane-runtime/coreml in your Cargo.toml"
        .into())
}

#[cfg(all(test, feature = "coreml", target_os = "macos"))]
mod tests {
    #[test]
    fn compiles_small_coreml_matmul() {
        let artifact = super::compile_mil("MIL PROGRAM matmul_2x3x1").expect("Core ML compiler");
        assert!(!artifact.binary.is_empty());
        assert_eq!(artifact.entry_point, "main");
    }
}
