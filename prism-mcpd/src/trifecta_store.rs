#[cfg(feature = "trifecta")]
use anyhow::Result;
#[cfg(feature = "trifecta")]
use parking_lot::Mutex;
#[cfg(feature = "trifecta")]
use prism_mcp_core::{
    ArtifactId, ArtifactKind, ArtifactRecord, ArtifactRepository, EvidenceReceipt, EvidenceStatus,
    EvidenceStore, JobEvent, JobId, JobProgress, JobRecord, JobState, JobStore, ToolInvocationId,
};
#[cfg(feature = "trifecta")]
use prism_mcp_core::{ClaimResult, CoordinationSession, CoordinationStore, LockResult, PathLock, WorkItem};
#[cfg(feature = "trifecta")]
use prism_mcp_core::{KnowledgeDocument, KnowledgeListRow, KnowledgeSearchResult, KnowledgeStore};
#[cfg(feature = "trifecta")]
use std::sync::Arc;

#[cfg(feature = "trifecta")]
pub struct PostgresEvidenceStore {
    runtime: tokio::runtime::Runtime,
    client: Mutex<tokio_postgres::Client>,
}

#[cfg(feature = "trifecta")]
pub struct PostgresKnowledgeStore {
    runtime: tokio::runtime::Runtime,
    client: Mutex<tokio_postgres::Client>,
}

#[cfg(feature = "trifecta")]
pub struct PostgresJobStore {
    runtime: tokio::runtime::Runtime,
    client: Mutex<tokio_postgres::Client>,
}
#[cfg(feature = "trifecta")]
pub struct PostgresCoordinationStore { runtime: tokio::runtime::Runtime, client: Mutex<tokio_postgres::Client> }

#[cfg(feature = "trifecta")]
pub struct PostgresExperimentStore {
    runtime: tokio::runtime::Runtime,
    client: Mutex<tokio_postgres::Client>,
}

#[cfg(feature = "trifecta")]
pub struct PostgresBenchmarkStore {
    runtime: tokio::runtime::Runtime,
    client: Mutex<tokio_postgres::Client>,
}

#[cfg(feature = "trifecta")]
pub struct ValkeyLeaseStore {
    connection: Mutex<redis::Connection>,
}

#[cfg(feature = "trifecta")]
pub struct DuckDbProjectionStore {
    connection: Mutex<duckdb::Connection>,
}

#[cfg(feature = "trifecta")]
#[cfg(feature = "trifecta")]
pub struct PostgresArtifactRepository {
    base: std::path::PathBuf,
    runtime: tokio::runtime::Runtime,
    client: Mutex<tokio_postgres::Client>,
}

#[cfg(feature = "trifecta")]
impl PostgresEvidenceStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let (client, connection) =
            runtime.block_on(tokio_postgres::connect(url, tokio_postgres::NoTls))?;
        runtime.spawn(async move {
            let _ = connection.await;
        });
        Ok(Arc::new(Self {
            runtime,
            client: Mutex::new(client),
        }))
    }
}

#[cfg(feature = "trifecta")]
impl PostgresKnowledgeStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let (client, connection) =
            runtime.block_on(tokio_postgres::connect(url, tokio_postgres::NoTls))?;
        runtime.spawn(async move {
            let _ = connection.await;
        });
        Ok(Arc::new(Self {
            runtime,
            client: Mutex::new(client),
        }))
    }
}

#[cfg(feature = "trifecta")]
fn connect_postgres(url: &str) -> Result<(tokio::runtime::Runtime, tokio_postgres::Client)> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let (client, connection) =
        runtime.block_on(tokio_postgres::connect(url, tokio_postgres::NoTls))?;
    runtime.spawn(async move {
        let _ = connection.await;
    });
    Ok((runtime, client))
}

#[cfg(feature = "trifecta")]
impl PostgresCoordinationStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> { let (runtime, client) = connect_postgres(url)?; Ok(Arc::new(Self { runtime, client: Mutex::new(client) })) }
}

#[cfg(feature = "trifecta")]
impl CoordinationStore for PostgresCoordinationStore {
    fn start_session(&self, id: &str, agent: &str, purpose: Option<&str>) -> Result<CoordinationSession> {
        let c = self.client.lock(); let now = chrono::Utc::now();
        let session_id = id.to_string();
        let agent_id = agent.to_string();
        let purpose_value = purpose.unwrap_or("").to_string();
        self.runtime.block_on(c.execute("INSERT INTO prism_coord_sessions(session_id,agent_id,purpose,last_heartbeat_at) VALUES($1,$2,$3,$4) ON CONFLICT(session_id) DO UPDATE SET status='active',last_heartbeat_at=EXCLUDED.last_heartbeat_at", &[&session_id,&agent_id,&purpose_value,&now]))?;
        Ok(CoordinationSession { session_id:id.into(), agent_id:agent.into(), status:"active".into(), last_heartbeat_at:now.to_rfc3339() })
    }
    fn heartbeat(&self, id: &str) -> Result<()> { let c=self.client.lock(); let now=chrono::Utc::now(); self.runtime.block_on(c.execute("UPDATE prism_coord_sessions SET last_heartbeat_at=$1,status='active' WHERE session_id=$2 AND status <> 'closed'", &[&now,&id]))?; Ok(()) }
    fn close_session(&self, id: &str) -> Result<()> { let c=self.client.lock(); self.runtime.block_on(c.execute("UPDATE prism_coord_sessions SET status='closed',closed_at=now() WHERE session_id=$1", &[&id]))?; Ok(()) }
    fn create_work(&self,id:&str,title:&str,priority:i32,session:Option<&str>)->Result<WorkItem>{let c=self.client.lock();let work_id=id.to_string();let work_title=title.to_string();let created_by=session.unwrap_or("").to_string();self.runtime.block_on(c.execute("INSERT INTO prism_coord_work(work_id,title,priority,created_by) VALUES($1,$2,$3,$4)", &[&work_id,&work_title,&priority,&created_by]))?;Ok(WorkItem{work_id:id.into(),title:title.into(),status:"queued".into(),priority})}
    fn list_work(&self,status:Option<&str>)->Result<Vec<WorkItem>>{let c=self.client.lock();let rows=self.runtime.block_on(c.query("SELECT work_id,title,status,priority FROM prism_coord_work WHERE ($1::text IS NULL OR status=$1) ORDER BY priority DESC,created_at", &[&status]))?;Ok(rows.into_iter().map(|r|WorkItem{work_id:r.get(0),title:r.get(1),status:r.get(2),priority:r.get(3)}).collect())}
    fn claim_work(&self, work:&str, session:&str, ttl:i64)->Result<ClaimResult> {
        let mut c=self.client.lock();
        let tx=self.runtime.block_on(c.transaction())?;
        let valid=self.runtime.block_on(tx.query_opt("SELECT work_id FROM prism_coord_work WHERE work_id=$1 AND status IN ('queued','blocked') FOR UPDATE", &[&work]))?.is_some();
        if !valid {
            let owner=self.runtime.block_on(tx.query_opt("SELECT session_id FROM prism_coord_claims WHERE work_id=$1 AND status='active'", &[&work]))?.map(|row| row.get(0));
            self.runtime.block_on(tx.commit())?;
            return Ok(ClaimResult{claimed:false,claim_id:None,conflict_session_id:owner});
        }
        let id=uuid::Uuid::new_v4().to_string();
        let claim_work_id = work.to_string();
        let claim_session_id = session.to_string();
        let ttl = ttl.clamp(1, 86_400);
        let claim_sql = format!("INSERT INTO prism_coord_claims(claim_id,work_id,session_id,expires_at) VALUES($1,$2,$3,now()+{} * interval '1 second') ON CONFLICT (work_id) WHERE status='active' DO NOTHING", ttl);
        let inserted=self.runtime.block_on(tx.execute(&claim_sql, &[&id,&claim_work_id,&claim_session_id]))?;
        if inserted == 0 { let owner=self.runtime.block_on(tx.query_one("SELECT session_id FROM prism_coord_claims WHERE work_id=$1 AND status='active'", &[&work]))?; self.runtime.block_on(tx.commit())?; return Ok(ClaimResult{claimed:false,claim_id:None,conflict_session_id:Some(owner.get(0))}); }
        self.runtime.block_on(tx.execute("UPDATE prism_coord_work SET status='claimed' WHERE work_id=$1", &[&work]))?;
        self.runtime.block_on(tx.commit())?;
        Ok(ClaimResult{claimed:true,claim_id:Some(id),conflict_session_id:None})
    }
    fn release_claim(&self,id:&str,session:&str)->Result<()> {let c=self.client.lock();self.runtime.block_on(c.execute("UPDATE prism_coord_claims SET status='released',released_at=now() WHERE claim_id=$1 AND session_id=$2 AND status='active'", &[&id,&session]))?;Ok(())}
    fn acquire_path(&self,session:&str,path:&str,kind:&str,ttl:i64)->Result<LockResult>{let c=self.client.lock();self.runtime.block_on(c.query_one("SELECT pg_advisory_xact_lock(hashtextextended($1, 1))", &[&path]))?;let conflict=self.runtime.block_on(c.query_opt("SELECT lock_id,session_id,lock_kind,expires_at::text FROM prism_coord_locks WHERE path=$1 AND status='active' AND expires_at>now() AND session_id<>$2 AND ($3='write' OR lock_kind='write')", &[&path.to_string(),&session.to_string(),&kind.to_string()]))?;if let Some(r)=conflict{return Ok(LockResult{acquired:false,locks:vec![],conflicts:vec![PathLock{lock_id:r.get(0),path:path.into(),lock_kind:r.get(2),session_id:r.get(1),expires_at:r.get(3)}]});}let id=uuid::Uuid::new_v4().to_string();self.runtime.block_on(c.execute("INSERT INTO prism_coord_locks(lock_id,path,lock_kind,session_id,expires_at) VALUES($1,$2,$3,$4,now()+$5 * interval '1 second')",&[&id,&path.to_string(),&kind.to_string(),&session.to_string(),&ttl.to_string()]))?;Ok(LockResult{acquired:true,locks:vec![PathLock{lock_id:id,path:path.into(),lock_kind:kind.into(),session_id:session.into(),expires_at:(chrono::Utc::now()+chrono::Duration::seconds(ttl)).to_rfc3339()}],conflicts:vec![]})}
    fn release_path(&self,id:&str,session:&str)->Result<()> {let c=self.client.lock();self.runtime.block_on(c.execute("UPDATE prism_coord_locks SET status='released',released_at=now() WHERE lock_id=$1 AND session_id=$2 AND status='active'", &[&id,&session]))?;Ok(())}
    fn recover_expired(&self)->Result<serde_json::Value>{let c=self.client.lock();let claims=self.runtime.block_on(c.execute("UPDATE prism_coord_claims SET status='expired',released_at=now() WHERE status='active' AND expires_at<=now()", &[]))?;let locks=self.runtime.block_on(c.execute("UPDATE prism_coord_locks SET status='expired',released_at=now() WHERE status='active' AND expires_at<=now()", &[]))?;Ok(serde_json::json!({"expired_claims":claims,"expired_locks":locks}))}
    fn handoff(&self, work:&str, from:&str, to:&str, context:&serde_json::Value)->Result<()> { let c=self.client.lock(); let id=uuid::Uuid::new_v4().to_string(); self.runtime.block_on(c.execute("INSERT INTO prism_coord_handoffs(handoff_id,work_id,from_session,to_session,context) VALUES($1,$2,$3,$4,$5)",&[&id,&work,&from,&to,context]))?; Ok(()) }
    fn append_event(&self, kind:&str, session:&str, payload:&serde_json::Value)->Result<prism_mcp_core::CoordinationEvent> { let c=self.client.lock(); let row=self.runtime.block_on(c.query_one("INSERT INTO prism_coord_events(event_type,session_id,payload) VALUES($1,$2,$3) RETURNING sequence", &[&kind,&session,payload]))?; Ok(prism_mcp_core::CoordinationEvent{sequence:row.get(0),event_type:kind.into(),session_id:session.into(),payload:payload.clone()}) }
    fn status(&self)->Result<serde_json::Value> { let c=self.client.lock(); let sessions=self.runtime.block_on(c.query_one("SELECT count(*) FROM prism_coord_sessions WHERE status='active'",&[]))?; let work=self.runtime.block_on(c.query_one("SELECT count(*) FROM prism_coord_work WHERE status IN ('queued','claimed','running','blocked')",&[]))?; let locks=self.runtime.block_on(c.query_one("SELECT count(*) FROM prism_coord_locks WHERE status='active' AND expires_at>now()",&[]))?; Ok(serde_json::json!({"active_sessions":sessions.get::<_,i64>(0),"open_work":work.get::<_,i64>(0),"active_locks":locks.get::<_,i64>(0)})) }
}

#[cfg(feature = "trifecta")]
impl PostgresExperimentStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> {
        let (runtime, client) = connect_postgres(url)?;
        Ok(Arc::new(Self {
            runtime,
            client: Mutex::new(client),
        }))
    }
}

#[cfg(feature = "trifecta")]
impl prism_mcp_core::ExperimentStore for PostgresExperimentStore {
    fn put_experiment(&self, id: &str, document: &serde_json::Value) -> Result<()> {
        let client = self.client.lock();
        self.runtime.block_on(client.execute("INSERT INTO prism_experiments (id,document) VALUES ($1,$2) ON CONFLICT (id) DO UPDATE SET document=EXCLUDED.document,updated_at=now()", &[&id, document]))?;
        Ok(())
    }
    fn get_experiment(&self, id: &str) -> Result<Option<serde_json::Value>> {
        let client = self.client.lock();
        Ok(self
            .runtime
            .block_on(
                client.query_opt("SELECT document FROM prism_experiments WHERE id=$1", &[&id]),
            )?
            .map(|row| row.get(0)))
    }
    fn list_experiments(&self) -> Result<Vec<(String, serde_json::Value)>> {
        let client = self.client.lock();
        let rows = self.runtime.block_on(client.query(
            "SELECT id,document FROM prism_experiments ORDER BY updated_at DESC",
            &[],
        ))?;
        Ok(rows
            .into_iter()
            .map(|row| (row.get(0), row.get(1)))
            .collect())
    }
}

#[cfg(feature = "trifecta")]
impl PostgresBenchmarkStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> {
        let (runtime, client) = connect_postgres(url)?;
        Ok(Arc::new(Self {
            runtime,
            client: Mutex::new(client),
        }))
    }
}

#[cfg(feature = "trifecta")]
impl prism_mcp_core::BenchmarkStore for PostgresBenchmarkStore {
    fn put_plan(&self, id: &str, name: &str, spec: &serde_json::Value) -> Result<()> {
        let client = self.client.lock();
        self.runtime.block_on(client.execute("INSERT INTO prism_benchmark_plans (id,name,spec) VALUES ($1,$2,$3) ON CONFLICT (id) DO UPDATE SET name=EXCLUDED.name,spec=EXCLUDED.spec,updated_at=now()", &[&id, &name, spec]))?;
        Ok(())
    }
    fn get_plan(&self, id: &str) -> Result<Option<serde_json::Value>> {
        let client = self.client.lock();
        Ok(self
            .runtime
            .block_on(
                client.query_opt("SELECT spec FROM prism_benchmark_plans WHERE id=$1", &[&id]),
            )?
            .map(|row| row.get(0)))
    }
    fn get_report(&self, id: &str) -> Result<Option<(f64, i32)>> {
        let client = self.client.lock();
        Ok(self
            .runtime
            .block_on(client.query_opt(
                "SELECT elapsed_ms,exit_code FROM prism_benchmark_reports WHERE id=$1",
                &[&id],
            ))?
            .map(|row| (row.get(0), row.get(1))))
    }
    fn get_baseline(&self, name: &str) -> Result<Option<String>> {
        let client = self.client.lock();
        Ok(self
            .runtime
            .block_on(client.query_opt(
                "SELECT report_id FROM prism_benchmark_baselines WHERE name=$1",
                &[&name],
            ))?
            .map(|row| row.get(0)))
    }
    fn put_baseline(&self, name: &str, report_id: &str) -> Result<()> {
        let client = self.client.lock();
        self.runtime.block_on(client.execute("INSERT INTO prism_benchmark_baselines (name,report_id) VALUES ($1,$2) ON CONFLICT (name) DO UPDATE SET report_id=EXCLUDED.report_id,updated_at=now()", &[&name, &report_id]))?;
        Ok(())
    }
}

#[cfg(feature = "trifecta")]
impl KnowledgeStore for PostgresKnowledgeStore {
    fn search(&self, query: &str, limit: usize) -> Result<Vec<KnowledgeSearchResult>> {
        let client = self.client.lock();
        let rows=self.runtime.block_on(client.query("SELECT s.id,s.document_id,s.heading,s.word_count,d.title,d.doc_type,ts_headline('english',s.content,plainto_tsquery('english',$1)),ts_rank(to_tsvector('english',s.content),plainto_tsquery('english',$1)) FROM prism_document_sections s JOIN prism_documents d ON d.id=s.document_id WHERE to_tsvector('english',s.content) @@ plainto_tsquery('english',$1) ORDER BY 8 DESC LIMIT $2", &[&query,&(limit as i64)]))?;
        Ok(rows
            .into_iter()
            .map(|r| KnowledgeSearchResult {
                section_id: r.get(0),
                document_id: r.get(1),
                heading: r.get(2),
                word_count: r.get(3),
                doc_title: r.get(4),
                doc_type: r.get(5),
                snippet: r.get(6),
                rank: r.get(7),
            })
            .collect())
    }
    fn get_document(&self, id: &str) -> Result<Option<KnowledgeDocument>> {
        let client = self.client.lock();
        let row=self.runtime.block_on(client.query_opt("SELECT id,title,doc_type,content_md,version,status,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_documents WHERE id=$1", &[&id]))?;
        Ok(row.map(|r| KnowledgeDocument {
            id: r.get(0),
            title: r.get(1),
            doc_type: r.get(2),
            content: r.get(3),
            version: r.get(4),
            status: r.get(5),
            created_at: r.get(6),
        }))
    }
    fn list_documents(
        &self,
        doc_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<KnowledgeListRow>> {
        let client = self.client.lock();
        let rows = if let Some(doc_type) = doc_type {
            self.runtime.block_on(client.query("SELECT id,title,doc_type,version,status,to_char(updated_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_documents WHERE doc_type=$1 ORDER BY updated_at DESC LIMIT $2", &[&doc_type,&(limit as i64)]))?
        } else {
            self.runtime.block_on(client.query("SELECT id,title,doc_type,version,status,to_char(updated_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_documents ORDER BY updated_at DESC LIMIT $1", &[&(limit as i64)]))?
        };
        Ok(rows
            .into_iter()
            .map(|r| KnowledgeListRow {
                id: r.get(0),
                title: r.get(1),
                doc_type: r.get(2),
                version: r.get(3),
                status: r.get(4),
                updated_at: r.get(5),
            })
            .collect())
    }
}

#[cfg(feature = "trifecta")]
impl PostgresJobStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        let (client, connection) =
            runtime.block_on(tokio_postgres::connect(url, tokio_postgres::NoTls))?;
        runtime.spawn(async move {
            let _ = connection.await;
        });
        Ok(Arc::new(Self {
            runtime,
            client: Mutex::new(client),
        }))
    }
    fn state(value: &str, detail: Option<String>) -> JobState {
        match value {
            "Queued" => JobState::Queued,
            "WaitingForResource" => JobState::WaitingForResource,
            "Running" => JobState::Running,
            "Cancelling" => JobState::Cancelling,
            "Succeeded" => JobState::Succeeded,
            "Cancelled" => JobState::Cancelled,
            "Failed" => JobState::Failed(detail.unwrap_or_default()),
            _ => JobState::Queued,
        }
    }
    fn record(row: &tokio_postgres::Row) -> Result<JobRecord> {
        Ok(JobRecord {
            id: JobId(row.get::<_, String>(0).parse().unwrap_or_default()),
            tool: row.get(1),
            operation: row.get(2),
            state: Self::state(&row.get::<_, String>(3), row.get(4)),
            progress: match (
                row.get::<_, Option<String>>(4),
                row.get::<_, Option<f64>>(5),
            ) {
                (Some(message), Some(percent)) => Some(JobProgress { message, percent }),
                _ => None,
            },
            receipt_id: row.get(6),
            created_at: row
                .get::<_, String>(7)
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
            updated_at: row
                .get::<_, String>(8)
                .parse()
                .unwrap_or_else(|_| chrono::Utc::now()),
        })
    }
}

#[cfg(feature = "trifecta")]
impl ValkeyLeaseStore {
    pub fn connect(url: &str) -> Result<Arc<Self>> {
        let client = redis::Client::open(url)?;
        Ok(Arc::new(Self {
            connection: Mutex::new(client.get_connection()?),
        }))
    }
}

#[cfg(feature = "trifecta")]
impl DuckDbProjectionStore {
    pub fn open(path: &str) -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            connection: Mutex::new(duckdb::Connection::open(path)?),
        }))
    }
}

#[cfg(feature = "trifecta")]
#[cfg(feature = "trifecta")]
impl PostgresArtifactRepository {
    pub fn connect(base: &std::path::Path, url: &str) -> Result<Arc<Self>> {
        std::fs::create_dir_all(base)?;
        let (runtime, client) = connect_postgres(url)?;
        Ok(Arc::new(Self {
            base: base.to_owned(),
            runtime,
            client: Mutex::new(client),
        }))
    }

    fn kind_name(kind: &ArtifactKind) -> String {
        format!("{kind:?}").to_lowercase()
    }

    fn kind(value: &str) -> Result<ArtifactKind> {
        let kind = match value {
            "cimage" => ArtifactKind::Cimage,
            "kernelrecipe" => ArtifactKind::KernelRecipe,
            "hsaco" => ArtifactKind::Hsaco,
            "metallibrary" => ArtifactKind::MetalLibrary,
            "coremlbundle" => ArtifactKind::CoreMlBundle,
            "cpuobject" => ArtifactKind::CpuObject,
            "compilerir" => ArtifactKind::CompilerIr,
            "llvmir" => ArtifactKind::LlvmIr,
            "disassembly" => ArtifactKind::Disassembly,
            "benchmarktrace" => ArtifactKind::BenchmarkTrace,
            "validationcorpus" => ArtifactKind::ValidationCorpus,
            "buildlog" => ArtifactKind::BuildLog,
            "modelmanifest" => ArtifactKind::ModelManifest,
            "tensorinventory" => ArtifactKind::TensorInventory,
            "buildplan" => ArtifactKind::BuildPlan,
            "compilerdiagnostics" => ArtifactKind::CompilerDiagnostics,
            "kernelcandidateset" => ArtifactKind::KernelCandidateSet,
            "resourcereport" => ArtifactKind::ResourceReport,
            "admissionplan" => ArtifactKind::AdmissionPlan,
            "calibrationcorpus" => ArtifactKind::CalibrationCorpus,
            "validationreport" => ArtifactKind::ValidationReport,
            "benchmarkplan" => ArtifactKind::BenchmarkPlan,
            "benchmarksamples" => ArtifactKind::BenchmarkSamples,
            "benchmarkreport" => ArtifactKind::BenchmarkReport,
            "tracecapture" => ArtifactKind::TraceCapture,
            "tracesummary" => ArtifactKind::TraceSummary,
            "replaybundle" => ArtifactKind::ReplayBundle,
            "experimentspec" => ArtifactKind::ExperimentSpec,
            "experimentreport" => ArtifactKind::ExperimentReport,
            _ => anyhow::bail!("unknown persisted artifact kind: {value}"),
        };
        Ok(kind)
    }

    fn path(&self, id: &ArtifactId, kind: &ArtifactKind) -> std::path::PathBuf {
        let hex = id.hex();
        self.base
            .join(Self::kind_name(kind))
            .join(&hex[..2])
            .join(&hex[2..])
    }
}

#[cfg(feature = "trifecta")]
impl ArtifactRepository for PostgresArtifactRepository {
    fn put(
        &self,
        data: &[u8],
        kind: ArtifactKind,
        producer: &ToolInvocationId,
    ) -> Result<ArtifactId> {
        let id = ArtifactId::from_data(data);
        let path = self.path(&id, &kind);
        if !path.exists() {
            std::fs::create_dir_all(path.parent().expect("artifact path has parent"))?;
            let temp = path.with_extension("tmp");
            std::fs::write(&temp, data)?;
            std::fs::rename(temp, &path)?;
        }
        let client = self.client.lock();
        self.runtime.block_on(client.execute(
            "INSERT INTO prism_artifacts (id_hash,kind,byte_len,media_type,producer) VALUES ($1,$2,$3,$4,$5) ON CONFLICT (id_hash) DO NOTHING",
            &[&id.hex(), &Self::kind_name(&kind), &(data.len() as i64), &"application/octet-stream", &producer.0.to_string()],
        ))?;
        Ok(id)
    }

    fn list(&self, kind: Option<&ArtifactKind>) -> Result<Vec<ArtifactRecord>> {
        let client = self.client.lock();
        let rows = if let Some(kind) = kind {
            self.runtime.block_on(client.query("SELECT id_hash,kind,byte_len,media_type,producer,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_artifacts WHERE kind=$1 ORDER BY created_at DESC", &[&Self::kind_name(kind)]))?
        } else {
            self.runtime.block_on(client.query("SELECT id_hash,kind,byte_len,media_type,producer,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_artifacts ORDER BY created_at DESC", &[]))?
        };
        rows.into_iter()
            .map(|row| {
                let bytes = hex::decode(row.get::<_, String>(0))?;
                let mut digest = [0u8; 32];
                digest.copy_from_slice(&bytes);
                let kind_name = row.get::<_, String>(1);
                Ok(ArtifactRecord {
                    id: ArtifactId { digest },
                    kind: Self::kind(&kind_name)?,
                    byte_len: row.get::<_, i64>(2) as u64,
                    media_type: row.get(3),
                    producer: ToolInvocationId(row.get::<_, String>(4).parse()?),
                    target: None,
                    created_at: row.get::<_, String>(5).parse()?,
                })
            })
            .collect()
    }
}

/* DuckDB does not own artifact metadata in the trifecta profile. */
#[cfg(any())]
impl ArtifactRepository for DuckDbArtifactRepository {
    fn put(
        &self,
        data: &[u8],
        kind: ArtifactKind,
        producer: &ToolInvocationId,
    ) -> Result<ArtifactId> {
        let id = ArtifactId::from_data(data);
        let hex = id.hex();
        let directory = self.base.join(Self::kind_name(&kind)).join(&hex[..2]);
        let path = directory.join(&hex[2..]);
        if !path.exists() {
            std::fs::create_dir_all(&directory)?;
            std::fs::write(&path, data)?;
        }
        let connection = self.connection.lock();
        connection.execute("INSERT OR IGNORE INTO artifact_projection (id_hash,kind,byte_len,media_type,producer,target) VALUES (?,?,?,?,?,?)", duckdb::params![hex, Self::kind_name(&kind), data.len() as i64, "application/octet-stream", producer.0.to_string(), Option::<String>::None])?;
        Ok(id)
    }
    fn list(&self, kind: Option<&ArtifactKind>) -> Result<Vec<ArtifactRecord>> {
        let connection = self.connection.lock();
        let mut statement = if kind.is_some() {
            connection.prepare("SELECT id_hash,kind,byte_len,media_type,producer,target,created_at FROM artifact_projection WHERE kind=? ORDER BY created_at DESC")?
        } else {
            connection.prepare("SELECT id_hash,kind,byte_len,media_type,producer,target,created_at FROM artifact_projection ORDER BY created_at DESC")?
        };
        let rows = if let Some(kind) = kind {
            statement.query(duckdb::params![Self::kind_name(kind)])?
        } else {
            statement.query([])?
        };
        let mut rows = rows;
        let mut result = Vec::new();
        while let Some(row) = rows.next()? {
            let hex: String = row.get(0)?;
            let bytes = hex::decode(hex)?;
            let mut digest = [0u8; 32];
            digest.copy_from_slice(&bytes);
            result.push(ArtifactRecord {
                id: ArtifactId { digest },
                kind: Self::kind(&row.get::<_, String>(1)?),
                byte_len: row.get::<_, i64>(2)? as u64,
                media_type: row.get(3)?,
                producer: ToolInvocationId(row.get::<_, String>(4)?.parse().unwrap_or_default()),
                target: row.get(5)?,
                created_at: row
                    .get::<_, String>(6)?
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
            });
        }
        Ok(result)
    }
}

#[cfg(feature = "trifecta")]
impl prism_mcp_core::ProjectionStore for DuckDbProjectionStore {
    fn record_benchmark(
        &self,
        report_id: &str,
        plan_id: &str,
        elapsed_ms: f64,
        exit_code: i32,
        output: &str,
    ) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute("INSERT INTO benchmark_projection (report_id,plan_id,elapsed_ms,exit_code,output) VALUES (?,?,?,?,?)", duckdb::params![report_id, plan_id, elapsed_ms, exit_code, output])?;
        Ok(())
    }

    fn put_trace(&self, trace_id: &str, snapshot: &serde_json::Value) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute("INSERT INTO trace_projection (trace_id,event_index,operation,duration_ms,payload) VALUES (?, -1, 'trace_snapshot', NULL, ?)", duckdb::params![trace_id, snapshot.to_string()])?;
        Ok(())
    }

    fn get_trace(&self, trace_id: &str) -> Result<Option<serde_json::Value>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT payload FROM trace_projection WHERE trace_id=? AND event_index=-1 ORDER BY observed_at DESC LIMIT 1")?;
        let mut rows = statement.query(duckdb::params![trace_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(serde_json::from_str::<serde_json::Value>(
                &row.get::<_, String>(0)?,
            )?)),
            None => Ok(None),
        }
    }

    fn record_kernel(
        &self,
        name: &str,
        backend: &str,
        artifact_hash: &str,
        byte_len: u64,
        target: &str,
    ) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute("INSERT OR REPLACE INTO kernel_projection (name,backend,artifact_hash,byte_len,target) VALUES (?,?,?,?,?)", duckdb::params![name, backend, artifact_hash, byte_len as i64, target])?;
        Ok(())
    }

    fn put_replay(&self, replay_id: &str, status: &str, payload: &serde_json::Value) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute(
            "INSERT INTO replay_projection (replay_id,status,payload) VALUES (?,?,?)",
            duckdb::params![replay_id, status, payload.to_string()],
        )?;
        Ok(())
    }

    fn get_replay(&self, replay_id: &str) -> Result<Option<(String, serde_json::Value)>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT status,payload FROM replay_projection WHERE replay_id=? ORDER BY observed_at DESC LIMIT 1")?;
        let mut rows = statement.query(duckdb::params![replay_id])?;
        match rows.next()? {
            Some(row) => Ok(Some((
                row.get(0)?,
                serde_json::from_str::<serde_json::Value>(&row.get::<_, String>(1)?)?,
            ))),
            None => Ok(None),
        }
    }
}

#[cfg(feature = "trifecta")]
impl prism_mcp_core::ExperimentStore for DuckDbProjectionStore {
    fn put_experiment(&self, experiment_id: &str, document: &serde_json::Value) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute("INSERT INTO experiment_projection (experiment_id,document) VALUES (?,?) ON CONFLICT (experiment_id) DO UPDATE SET document=excluded.document, observed_at=CURRENT_TIMESTAMP", duckdb::params![experiment_id, document.to_string()])?;
        Ok(())
    }

    fn get_experiment(&self, experiment_id: &str) -> Result<Option<serde_json::Value>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT document FROM experiment_projection WHERE experiment_id=? ORDER BY observed_at DESC LIMIT 1")?;
        let mut rows = statement.query(duckdb::params![experiment_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(serde_json::from_str::<serde_json::Value>(
                &row.get::<_, String>(0)?,
            )?)),
            None => Ok(None),
        }
    }

    fn list_experiments(&self) -> Result<Vec<(String, serde_json::Value)>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare(
            "SELECT experiment_id,document FROM experiment_projection ORDER BY observed_at DESC",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.map(|row| {
            let (id, value) = row?;
            Ok((id, serde_json::from_str(&value)?))
        })
        .collect()
    }
}

#[cfg(feature = "trifecta")]
impl prism_mcp_core::BenchmarkStore for DuckDbProjectionStore {
    fn put_plan(&self, plan_id: &str, name: &str, spec: &serde_json::Value) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute(
            "INSERT OR REPLACE INTO benchmark_plan_projection(plan_id,name,spec) VALUES (?,?,?)",
            duckdb::params![plan_id, name, spec.to_string()],
        )?;
        Ok(())
    }
    fn get_plan(&self, plan_id: &str) -> Result<Option<serde_json::Value>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT spec FROM benchmark_plan_projection WHERE plan_id=? ORDER BY observed_at DESC LIMIT 1")?;
        let mut rows = statement.query(duckdb::params![plan_id])?;
        match rows.next()? {
            Some(row) => Ok(Some(serde_json::from_str::<serde_json::Value>(
                &row.get::<_, String>(0)?,
            )?)),
            None => Ok(None),
        }
    }
    fn get_report(&self, report_id: &str) -> Result<Option<(f64, i32)>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT elapsed_ms,exit_code FROM benchmark_projection WHERE report_id=? ORDER BY observed_at DESC LIMIT 1")?;
        let mut rows = statement.query(duckdb::params![report_id])?;
        match rows.next()? {
            Some(row) => Ok(Some((row.get(0)?, row.get(1)?))),
            None => Ok(None),
        }
    }
    fn get_baseline(&self, name: &str) -> Result<Option<String>> {
        let connection = self.connection.lock();
        let mut statement = connection.prepare("SELECT report_id FROM benchmark_baseline_projection WHERE baseline_name=? ORDER BY observed_at DESC LIMIT 1")?;
        let mut rows = statement.query(duckdb::params![name])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }
    fn put_baseline(&self, name: &str, report_id: &str) -> Result<()> {
        let connection = self.connection.lock();
        connection.execute("INSERT OR REPLACE INTO benchmark_baseline_projection(baseline_name,report_id) VALUES (?,?)", duckdb::params![name,report_id])?;
        Ok(())
    }
}

#[cfg(feature = "trifecta")]
impl prism_mcp_core::LeaseStore for ValkeyLeaseStore {
    fn acquire(&self, key: &str, owner: &str, ttl_seconds: u64) -> Result<bool> {
        let mut connection = self.connection.lock();
        let result: Option<String> = redis::cmd("SET")
            .arg(format!("prism:lease:{key}"))
            .arg(owner)
            .arg("NX")
            .arg("EX")
            .arg(ttl_seconds)
            .query(&mut *connection)?;
        Ok(result.is_some())
    }

    fn release(&self, key: &str, owner: &str) -> Result<()> {
        let mut connection = self.connection.lock();
        redis::Script::new("if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end")
            .key(format!("prism:lease:{key}")).arg(owner).invoke::<i64>(&mut *connection)?;
        Ok(())
    }
}

#[cfg(feature = "trifecta")]
impl JobStore for PostgresJobStore {
    fn create_job(&self, tool: &str, operation: &str) -> Result<JobId> {
        let id = JobId::new();
        let client = self.client.lock();
        self.runtime.block_on(client.execute(
            "INSERT INTO prism_jobs (id,tool,operation,state) VALUES ($1,$2,$3,'Queued')",
            &[&id.to_string(), &tool, &operation],
        ))?;
        Ok(id)
    }
    fn update_state(&self, id: &JobId, state: JobState) -> Result<()> {
        let state_name = state.as_str();
        let detail = match &state {
            JobState::Failed(message) => Some(message),
            _ => None,
        };
        let client = self.client.lock();
        self.runtime.block_on(client.execute("UPDATE prism_jobs SET state=$1, progress_message=COALESCE($2,progress_message), updated_at=now() WHERE id=$3", &[&state_name, &detail, &id.to_string()]))?;
        drop(client);
        self.push_event(id, "state_change", &format!("→ {state_name}"))
    }
    fn update_progress(&self, id: &JobId, progress: JobProgress) -> Result<()> {
        let client = self.client.lock();
        self.runtime.block_on(client.execute("UPDATE prism_jobs SET progress_message=$1, progress_percent=$2, updated_at=now() WHERE id=$3", &[&progress.message, &progress.percent, &id.to_string()]))?;
        Ok(())
    }
    fn get_job(&self, id: &JobId) -> Result<JobRecord> {
        let client = self.client.lock();
        let row = self.runtime.block_on(client.query_one("SELECT id,tool,operation,state,progress_message,progress_percent,receipt_id,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),to_char(updated_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_jobs WHERE id=$1", &[&id.to_string()]))?;
        Self::record(&row)
    }
    fn list_jobs(&self, tool: Option<&str>) -> Result<Vec<JobRecord>> {
        let client = self.client.lock();
        let rows = if let Some(tool) = tool {
            self.runtime.block_on(client.query("SELECT id,tool,operation,state,progress_message,progress_percent,receipt_id,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),to_char(updated_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_jobs WHERE tool=$1 ORDER BY created_at DESC", &[&tool]))?
        } else {
            self.runtime.block_on(client.query("SELECT id,tool,operation,state,progress_message,progress_percent,receipt_id,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"'),to_char(updated_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_jobs ORDER BY created_at DESC", &[]))?
        };
        rows.iter().map(Self::record).collect()
    }
    fn cancel_job(&self, id: &JobId) -> Result<()> {
        self.update_state(id, JobState::Cancelling)
    }
    fn push_event(&self, id: &JobId, event_type: &str, message: &str) -> Result<()> {
        let client = self.client.lock();
        self.runtime.block_on(client.execute(
            "INSERT INTO prism_job_events (job_id,event_type,message) VALUES ($1,$2,$3)",
            &[&id.to_string(), &event_type, &message],
        ))?;
        Ok(())
    }
    fn get_events(&self, id: &JobId) -> Result<Vec<JobEvent>> {
        let client = self.client.lock();
        let rows = self.runtime.block_on(client.query("SELECT event_type,message,to_char(created_at,'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"') FROM prism_job_events WHERE job_id=$1 ORDER BY id", &[&id.to_string()]))?;
        Ok(rows
            .into_iter()
            .map(|row| JobEvent {
                event_type: row.get(0),
                message: row.get(1),
                created_at: row
                    .get::<_, String>(2)
                    .parse()
                    .unwrap_or_else(|_| chrono::Utc::now()),
            })
            .collect())
    }
}

#[cfg(feature = "trifecta")]
impl EvidenceStore for PostgresEvidenceStore {
    fn record(&self, receipt: &EvidenceReceipt) -> Result<()> {
        let status = match &receipt.status {
            EvidenceStatus::Success => String::new(),
            EvidenceStatus::Failure(message) => message.clone(),
            EvidenceStatus::Partial(messages) => messages.join("; "),
        };
        let payload = serde_json::to_value(receipt)?;
        let client = self.client.lock();
        self.runtime.block_on(client.execute(
            "INSERT INTO prism_evidence_receipts (id, operation, status, payload) VALUES ($1,$2,$3,$4) ON CONFLICT (id) DO NOTHING",
            &[&receipt.invocation_id.0.to_string(), &receipt.operation, &status, &payload],
        ))?;
        Ok(())
    }

    fn query(
        &self,
        tool: &str,
        operation: Option<&str>,
        limit: usize,
    ) -> Result<Vec<EvidenceReceipt>> {
        let client = self.client.lock();
        let rows = if let Some(operation) = operation {
            self.runtime.block_on(client.query("SELECT payload FROM prism_evidence_receipts WHERE payload->>'tool'=$1 AND operation=$2 ORDER BY created_at DESC LIMIT $3", &[&tool, &operation, &(limit as i64)]))?
        } else {
            self.runtime.block_on(client.query("SELECT payload FROM prism_evidence_receipts WHERE payload->>'tool'=$1 ORDER BY created_at DESC LIMIT $2", &[&tool, &(limit as i64)]))?
        };
        rows.into_iter()
            .map(|row| Ok(serde_json::from_value(row.get(0))?))
            .collect()
    }
}
