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
    // Parse the MIL pseudo-code and construct a CoreML Model protobuf.
    // This is a placeholder — real implementation will parse MIL directives
    // and generate the corresponding NeuralNetwork layers.
    let _ = mil_source;
    Err("ANE compile_mil: CoreML compilation not yet implemented".into())
}

#[cfg(not(feature = "coreml"))]
pub fn compile_mil(_mil_source: &str) -> Result<AneBinary, String> {
    Err("ANE compilation requires the 'coreml' feature and macOS. \
         Enable prism-ane-runtime/coreml in your Cargo.toml"
        .into())
}
