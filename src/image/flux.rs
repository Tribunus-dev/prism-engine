use std::path::Path;
use anyhow::Result;
use super::{FluxVariant, Metadata};

pub struct FluxCompiler;

impl FluxCompiler {
    pub fn compile(_gguf_path: &Path, _variant: FluxVariant) -> Result<Metadata> {
        Ok(Metadata {})
    }
}
