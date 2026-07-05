use std::path::Path;
use anyhow::Result;
use super::{SdxlVariant, Metadata};

pub struct SdxlCompiler;

impl SdxlCompiler {
    pub fn compile(_gguf_path: &Path, _variant: SdxlVariant) -> Result<Metadata> {
        Ok(Metadata {})
    }
}
