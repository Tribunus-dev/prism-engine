use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use uuid::Uuid;

/// Identifies a single tool invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InvocationId(pub Uuid);

impl InvocationId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for InvocationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for InvocationId {
    fn default() -> Self {
        Self::new()
    }
}

/// Identifies an async job tracked by JobManager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(pub Uuid);

impl JobId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies an experiment in prism-lab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExperimentId(pub Uuid);

impl ExperimentId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl fmt::Display for ExperimentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Identifies a build target (e.g. "aarch64-apple-darwin", "gfx942").
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetId(pub String);

impl TargetId {
    pub fn new(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Identifies a model manifest.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

/// Identifies a tensor within a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorId(pub String);

/// Identifies a kernel recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelRecipeId(pub String);

/// Identifies a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BenchmarkRunId(pub Uuid);

/// Identifies a replay bundle.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayId(pub String);

/// An input source: either a stored artifact or a raw filesystem path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputSource {
    Artifact(crate::ArtifactId),
    Path(PathBuf),
}

/// A single diagnostic message from a tool operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: String,
    pub message: String,
    pub location: Option<String>,
}

/// A named metric with typed value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    F64(f64),
    I64(i64),
    Str(String),
}
