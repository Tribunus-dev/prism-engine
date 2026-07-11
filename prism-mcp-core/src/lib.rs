//! prism-mcp-core: shared framework for the prism-mcpd multi-tool MCP daemon.

pub mod artifact;
pub mod db;
pub mod evidence;
pub mod file_lock;
pub mod ident;
pub mod job;
pub mod lease;
pub mod log;
pub mod protocol;
pub mod scheduler;
pub mod schema;
pub mod subprocess;
pub mod work_journal;

pub use artifact::{ArtifactId, ArtifactKind, ArtifactRecord, ArtifactStore};
pub use db::{DbManager, ReaderGuard};
pub use evidence::{EvidenceLedger, EvidenceReceipt, EvidenceStatus, MetricSet, ToolInvocationId};
pub use file_lock::{FileLock, FileLockGuard};
pub use ident::{
    BenchmarkRunId, Diagnostic, ExperimentId, InputSource, InvocationId, JobId, KernelRecipeId,
    ModelId, ReplayId, TargetId, TensorId,
};
pub use job::{JobEvent, JobManager, JobProgress, JobRecord, JobState};
pub use lease::{ResourceClass, ResourceLease, ResourceLeaseManager, ResourceRequest};
pub use log::init_logging;
pub use protocol::{
    ConnectionId, DaemonState, McpError, McpHandler, McpRequest, McpResponse, McpStatus,
    RequestContext, RequestEnvelope, ResponseFrame, ToolRequest, ToolResult,
};
pub use scheduler::{Scheduler, SchedulerHandle, ToolConcurrencyPolicy, ToolLimit};
pub use schema::SchemaHeader;
pub use subprocess::{run_with_timeout, ProcessCache};
pub use work_journal::{JournalEntry, JournalPhase, WorkJournal};
