use std::path::Path;
use anyhow::Result;
use super::{Sd3Variant, Metadata};

pub struct Sd3Compiler;

impl Sd3Compiler {
    pub fn compile(_gguf_path: &Path, _variant: Sd3Variant) -> Result<Metadata> {
        Ok(Metadata {})
    }
}
