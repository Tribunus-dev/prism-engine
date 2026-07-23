//! prism-mcp-core: shared framework for the prism-mcpd multi-tool MCP daemon.

pub mod artifact;
pub mod db;
pub mod coordination;
pub mod evidence;
pub mod file_lock;
pub mod graph_query;
pub mod ident;
pub mod job;
pub mod lease;
pub mod log;
pub mod protocol;
pub mod provenance;
pub mod scheduler;
pub mod schema;
pub mod semantic;
pub mod storage;
pub mod subprocess;
pub mod work_journal;

pub use artifact::{ArtifactId, ArtifactKind, ArtifactRecord};
pub use db::DbManager;
pub use coordination::{
    ClaimResult, CoordinationEvent, CoordinationSession, CoordinationStore, LockResult, PathLock,
    WorkItem,
};
pub use evidence::{EvidenceReceipt, EvidenceStatus, MetricSet, ToolInvocationId};
pub use file_lock::{FileLock, FileLockGuard};
pub use graph_query::{
    FederatedGraphQuery, GraphAuthority, GraphEdge, GraphEvidence, GraphNode, GraphOperation,
    GraphProjection, TraversalResult,
};
pub use ident::{
    BenchmarkRunId, Diagnostic, ExperimentId, InputSource, InvocationId, JobId, KernelRecipeId,
    ModelId, ReplayId, TargetId, TensorId,
};
pub use job::{JobEvent, JobProgress, JobRecord, JobState};
pub use lease::{ResourceClass, ResourceLease, ResourceLeaseManager, ResourceRequest};
pub use log::init_logging;
pub use protocol::{
    ConnectionId, DaemonState, McpError, McpHandler, McpRequest, McpResponse, McpStatus,
    NormalizedToolCall, RequestContext, RequestEnvelope, ResponseFrame,
    ToolCallNormalizationError, ToolCallNormalizationReceipt, ToolRequest, ToolResult,
};
pub use provenance::{
    ProvenanceDomain, ProvenanceEdge, ProvenanceGraphStore, ProvenanceKind, ProvenanceNode,
    ProvenanceQuery, ProvenanceSubgraph,
};
pub use scheduler::{Scheduler, SchedulerHandle, ToolConcurrencyPolicy, ToolLimit};
pub use schema::SchemaHeader;
pub use semantic::{
    validate_typed_arguments, ActionContract, LoopBudget, LoopDecision, LoopGuard, Ontology,
    OntologyEntity, OntologyRelation, SemanticAdmission, SemanticViolation, StagedAction,
    StagedActionPhase, ValidationReceipt,
};
pub use storage::{
    ArtifactRepository, BenchmarkStore, ConversationStore, EvidenceStore, ExperimentStore,
    JobStore, KnowledgeDocument, KnowledgeListRow, KnowledgeSearchResult, KnowledgeStore,
    LeaseStore, ProjectionStore,
};
pub use subprocess::{run_with_timeout, ProcessCache};
pub use work_journal::{JournalEntry, JournalPhase, WorkJournal};
