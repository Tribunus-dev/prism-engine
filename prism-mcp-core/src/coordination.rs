use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationSession { pub session_id: String, pub agent_id: String, pub status: String, pub last_heartbeat_at: String }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem { pub work_id: String, pub title: String, pub status: String, pub priority: i32 }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResult { pub claimed: bool, pub claim_id: Option<String>, pub conflict_session_id: Option<String> }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathLock { pub lock_id: String, pub path: String, pub lock_kind: String, pub session_id: String, pub expires_at: String }
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockResult { pub acquired: bool, pub locks: Vec<PathLock>, pub conflicts: Vec<PathLock> }

pub trait CoordinationStore: Send + Sync {
    fn start_session(&self, session_id: &str, agent_id: &str, purpose: Option<&str>) -> Result<CoordinationSession>;
    fn heartbeat(&self, session_id: &str) -> Result<()>;
    fn close_session(&self, session_id: &str) -> Result<()>;
    fn create_work(&self, work_id: &str, title: &str, priority: i32, session_id: Option<&str>) -> Result<WorkItem>;
    fn list_work(&self, status: Option<&str>) -> Result<Vec<WorkItem>>;
    fn claim_work(&self, work_id: &str, session_id: &str, ttl_seconds: i64) -> Result<ClaimResult>;
    fn release_claim(&self, claim_id: &str, session_id: &str) -> Result<()>;
    fn acquire_path(&self, session_id: &str, path: &str, kind: &str, ttl_seconds: i64) -> Result<LockResult>;
    fn release_path(&self, lock_id: &str, session_id: &str) -> Result<()>;
    fn recover_expired(&self) -> Result<serde_json::Value>;
}
