//! Canonical model source — format-independent abstraction over a loaded model.
//!
//! Combines the architecture graph, tensor provider, and provenance manifest
//! into a single source object used by the compilation and admission pipelines.

use prism_ecs_core::identity::{SourceManifest, TensorProvider};
use prism_ecs_ir::model_graph::{ArchitectureFamily, ModelGraph};

use crate::compile_config::CanonicalModelConfig;

/// A format-independent model source ready for search and compilation.
pub struct CanonicalModelSource {
    pub architecture: ArchitectureFamily,
    pub config: CanonicalModelConfig,
    pub graph: ModelGraph,
    pub tensors: Box<dyn TensorProvider>,
    pub source_manifest: SourceManifest,
}
