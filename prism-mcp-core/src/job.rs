use anyhow::Result;
use chrono::{DateTime, Utc};
use std::sync::Arc;

use crate::db::DbManager;
use crate::ident::JobId;

/// Possible states of an async job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    Queued,
    WaitingForResource,
    Running,
    Cancelling,
    Succeeded,
    Failed(String),
    Cancelled,
}

impl JobState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "Queued",
            Self::WaitingForResource => "WaitingForResource",
            Self::Running => "Running",
            Self::Cancelling => "Cancelling",
            Self::Succeeded => "Succeeded",
            Self::Failed(_) => "Failed",
            Self::Cancelled => "Cancelled",
        }
    }
}

/// Progress update for a running job.
#[derive(Debug, Clone)]
pub struct JobProgress {
    pub message: String,
    pub percent: f64,
}

/// Full record for a tracked job.
#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: JobId,
    pub tool: String,
    pub operation: String,
    pub state: JobState,
    pub progress: Option<JobProgress>,
    pub receipt_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// An event emitted during a job's lifecycle.
#[derive(Debug, Clone)]
pub struct JobEvent {
    pub event_type: String,
    pub message: String,
    pub created_at: DateTime<Utc>,
}

/// Tracks async jobs in the shared database.
#[derive(Clone)]
pub struct JobManager {
    db: Arc<DbManager>,
}

impl JobManager {
    pub fn new(db: Arc<DbManager>) -> Result<Self> {
        db.with_writer(|conn| {
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    tool TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'Queued',
                    progress_msg TEXT,
                    progress_pct REAL,
                    receipt_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id);",
            )?;
            Ok(())
        })?;
        Ok(Self { db })
    }

    pub fn create_job(&self, tool: &str, operation: &str) -> Result<JobId> {
        let id = JobId::new();
        let now = Utc::now().to_rfc3339();
        self.db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO jobs (id, tool, operation, state, created_at, updated_at) VALUES (?1, ?2, ?3, 'Queued', ?4, ?4)",
                rusqlite::params![id.to_string(), tool, operation, now],
            )?;
            Ok(())
        })?;
        Ok(id)
    }

    pub fn update_state(&self, id: &JobId, state: JobState) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        let state_str = state.as_str();
        let detail = match &state {
            JobState::Failed(msg) => msg.clone(),
            _ => String::new(),
        };
        let extra = if state_str == "Failed" {
            ", status = ?3"
        } else {
            ""
        };
        let sql = format!(
            "UPDATE jobs SET state = ?1, updated_at = ?2{} WHERE id = ?4",
            extra
        );
        self.db.with_writer(|conn| {
            conn.execute(
                &sql,
                rusqlite::params![state_str, now, detail, id.to_string()],
            )?;
            Ok(())
        })?;
        self.push_event(id, "state_change", &format!("→ {}", state_str))?;
        Ok(())
    }

    pub fn update_progress(&self, id: &JobId, progress: JobProgress) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.db.with_writer(|conn| {
            conn.execute(
                "UPDATE jobs SET progress_msg = ?1, progress_pct = ?2, updated_at = ?3 WHERE id = ?4",
                rusqlite::params![progress.message, progress.percent, now, id.to_string()],
            )?;
            Ok(())
        })
    }

    pub fn get_job(&self, id: &JobId) -> Result<JobRecord> {
        self.db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, tool, operation, state, progress_msg, progress_pct, receipt_id, created_at, updated_at FROM jobs WHERE id = ?1",
            )?;
            let mut rows = stmt.query(rusqlite::params![id.to_string()])?;
            match rows.next()? {
                Some(row) => Ok(JobRecord {
                    id: JobId(id.to_string().parse().unwrap_or_default()),
                    tool: row.get(1)?,
                    operation: row.get(2)?,
                    state: parse_state(&row.get::<_, String>(3)?, &row.get::<_, Option<String>>(4)?),
                    progress: match (row.get::<_, Option<String>>(4)?, row.get::<_, Option<f64>>(5)?) {
                        (Some(msg), Some(pct)) => Some(JobProgress { message: msg, percent: pct }),
                        _ => None,
                    },
                    receipt_id: row.get(6)?,
                    created_at: row.get::<_, String>(7)?.parse().unwrap_or_else(|_| Utc::now()),
                    updated_at: row.get::<_, String>(8)?.parse().unwrap_or_else(|_| Utc::now()),
                }),
                None => anyhow::bail!("job not found: {}", id),
            }
        })
    }

    pub fn list_jobs(&self, tool: Option<&str>) -> Result<Vec<JobRecord>> {
        self.db.with_reader(|conn| {
            let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(t) = tool {
                ("SELECT id, tool, operation, state, progress_msg, progress_pct, receipt_id, created_at, updated_at FROM jobs WHERE tool = ?1 ORDER BY created_at DESC".into(),
                 vec![Box::new(t.to_string())])
            } else {
                ("SELECT id, tool, operation, state, progress_msg, progress_pct, receipt_id, created_at, updated_at FROM jobs ORDER BY created_at DESC".into(),
                 vec![])
            };
            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
                Ok(JobRecord {
                    id: JobId(row.get::<_, String>(0)?.parse().unwrap_or_default()),
                    tool: row.get(1)?,
                    operation: row.get(2)?,
                    state: parse_state(&row.get::<_, String>(3)?, &row.get::<_, Option<String>>(4)?),
                    progress: match (row.get::<_, Option<String>>(4)?, row.get::<_, Option<f64>>(5)?) {
                        (Some(msg), Some(pct)) => Some(JobProgress { message: msg, percent: pct }),
                        _ => None,
                    },
                    receipt_id: row.get(6)?,
                    created_at: row.get::<_, String>(7)?.parse().unwrap_or_else(|_| Utc::now()),
                    updated_at: row.get::<_, String>(8)?.parse().unwrap_or_else(|_| Utc::now()),
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
        })
    }

    pub fn cancel_job(&self, id: &JobId) -> Result<()> {
        self.update_state(id, JobState::Cancelling)
    }

    pub fn push_event(&self, job_id: &JobId, event_type: &str, message: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.db.with_writer(|conn| {
            conn.execute(
                "INSERT INTO job_events (job_id, event_type, message, created_at) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![job_id.to_string(), event_type, message, now],
            )?;
            Ok(())
        })
    }

    pub fn get_events(&self, job_id: &JobId) -> Result<Vec<JobEvent>> {
        self.db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT event_type, message, created_at FROM job_events WHERE job_id = ?1 ORDER BY id",
            )?;
            let rows = stmt.query_map(rusqlite::params![job_id.to_string()], |row| {
                Ok(JobEvent {
                    event_type: row.get(0)?,
                    message: row.get(1)?,
                    created_at: row.get::<_, String>(2)?.parse().unwrap_or_else(|_| Utc::now()),
                })
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
        })
    }
}

fn parse_state(state: &str, _detail: &Option<String>) -> JobState {
    match state {
        "Queued" => JobState::Queued,
        "WaitingForResource" => JobState::WaitingForResource,
        "Running" => JobState::Running,
        "Cancelling" => JobState::Cancelling,
        "Succeeded" => JobState::Succeeded,
        "Failed" => JobState::Failed(_detail.clone().unwrap_or_default()),
        "Cancelled" => JobState::Cancelled,
        _ => JobState::Queued,
    }
}
