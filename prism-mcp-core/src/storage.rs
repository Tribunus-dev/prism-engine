//! Backend-neutral persistence contracts for the daemon.
//!
//! The SQLite implementations remain the compatibility adapter for local and
//! test profiles. Production adapters can implement these contracts without
//! making protocol handlers depend on a particular database client.

use anyhow::Result;

use crate::artifact::{ArtifactId, ArtifactKind, ArtifactRecord};
use crate::evidence::EvidenceReceipt;
use crate::evidence::ToolInvocationId;
use crate::ident::JobId;
use crate::job::{JobEvent, JobProgress, JobRecord, JobState};

pub trait JobStore: Send + Sync {
    fn create_job(&self, tool: &str, operation: &str) -> Result<JobId>;
    fn update_state(&self, id: &JobId, state: JobState) -> Result<()>;
    fn update_progress(&self, id: &JobId, progress: JobProgress) -> Result<()>;
    fn get_job(&self, id: &JobId) -> Result<JobRecord>;
    fn list_jobs(&self, tool: Option<&str>) -> Result<Vec<JobRecord>>;
    fn cancel_job(&self, id: &JobId) -> Result<()>;
    fn push_event(&self, job_id: &JobId, event_type: &str, message: &str) -> Result<()>;
    fn get_events(&self, job_id: &JobId) -> Result<Vec<JobEvent>>;
}

pub trait EvidenceStore: Send + Sync {
    fn record(&self, receipt: &EvidenceReceipt) -> Result<()>;
    fn query(
        &self,
        tool: &str,
        operation: Option<&str>,
        limit: usize,
    ) -> Result<Vec<EvidenceReceipt>>;
}

pub trait LeaseStore: Send + Sync {
    fn acquire(&self, key: &str, owner: &str, ttl_seconds: u64) -> Result<bool>;
    fn release(&self, key: &str, owner: &str) -> Result<()>;
}

pub trait ProjectionStore: Send + Sync {
    fn record_benchmark(
        &self,
        report_id: &str,
        plan_id: &str,
        elapsed_ms: f64,
        exit_code: i32,
        output: &str,
    ) -> Result<()>;
    fn put_trace(&self, trace_id: &str, snapshot: &serde_json::Value) -> Result<()>;
    fn get_trace(&self, trace_id: &str) -> Result<Option<serde_json::Value>>;
    fn record_kernel(
        &self,
        name: &str,
        backend: &str,
        artifact_hash: &str,
        byte_len: u64,
        target: &str,
    ) -> Result<()>;
    fn put_replay(&self, replay_id: &str, status: &str, payload: &serde_json::Value) -> Result<()>;
    fn get_replay(&self, replay_id: &str) -> Result<Option<(String, serde_json::Value)>>;
}

pub trait ExperimentStore: Send + Sync {
    fn put_experiment(&self, experiment_id: &str, document: &serde_json::Value) -> Result<()>;
    fn get_experiment(&self, experiment_id: &str) -> Result<Option<serde_json::Value>>;
    fn list_experiments(&self) -> Result<Vec<(String, serde_json::Value)>>;
}

pub trait BenchmarkStore: Send + Sync {
    fn put_plan(&self, plan_id: &str, name: &str, spec: &serde_json::Value) -> Result<()>;
    fn get_plan(&self, plan_id: &str) -> Result<Option<serde_json::Value>>;
    fn get_report(&self, report_id: &str) -> Result<Option<(f64, i32)>>;
    fn get_baseline(&self, name: &str) -> Result<Option<String>>;
    fn put_baseline(&self, name: &str, report_id: &str) -> Result<()>;
}

pub trait ArtifactRepository: Send + Sync {
    fn put(
        &self,
        data: &[u8],
        kind: ArtifactKind,
        producer: &ToolInvocationId,
    ) -> Result<ArtifactId>;
    /// Retrieve artifact data by id.
    fn get(&self, id: &ArtifactId) -> Result<Option<Vec<u8>>>;
    fn list(&self, kind: Option<&ArtifactKind>) -> Result<Vec<ArtifactRecord>>;
}

#[derive(Debug, Clone)]
pub struct KnowledgeSearchResult {
    pub section_id: String,
    pub document_id: String,
    pub heading: String,
    pub word_count: i64,
    pub doc_title: String,
    pub doc_type: String,
    pub snippet: String,
    pub rank: f64,
}
#[derive(Debug, Clone)]
pub struct KnowledgeDocument {
    pub id: String,
    pub title: String,
    pub doc_type: String,
    pub content: String,
    pub version: i64,
    pub status: String,
    pub created_at: String,
}
#[derive(Debug, Clone)]
pub struct KnowledgeListRow {
    pub id: String,
    pub title: String,
    pub doc_type: String,
    pub version: i64,
    pub status: String,
    pub updated_at: String,
}
pub trait KnowledgeStore: Send + Sync {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<KnowledgeSearchResult>>;
    fn get_document(&self, id: &str) -> Result<Option<KnowledgeDocument>>;
    fn list_documents(&self, doc_type: Option<&str>, limit: usize)
        -> Result<Vec<KnowledgeListRow>>;
}

pub trait ConversationStore: Send + Sync {
    fn append(&self, _conversation_id: &str, _role: &str, _content: &str) -> Result<()> { Ok(()) }
    fn list(&self, _conversation_id: &str) -> Result<Vec<(String, String)>> { Ok(Vec::new()) }
}
