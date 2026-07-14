use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use std::ops::{Deref, DerefMut};
use std::path::Path;

/// Thread-safe database manager with a serialized writer and a bounded reader pool.
///
/// - `writer`: single `Connection` behind a `Mutex` — all mutations go here.
/// - `reader_pool`: a fixed set of `Connection`s checked out via channel.
///   Readers use WAL mode and are returned to the pool on `ReaderGuard` drop.
pub struct DbManager {
    writer: Mutex<Connection>,
    reader_pool: Sender<Connection>,
    reader_checkout: Receiver<Connection>,
}

/// RAII guard that returns a reader connection to the pool on drop.
pub struct ReaderGuard<'a> {
    conn: Option<Connection>,
    return_tx: &'a Sender<Connection>,
}

impl Deref for ReaderGuard<'_> {
    type Target = Connection;
    fn deref(&self) -> &Connection {
        self.conn.as_ref().expect("ReaderGuard: connection missing")
    }
}

impl DerefMut for ReaderGuard<'_> {
    fn deref_mut(&mut self) -> &mut Connection {
        self.conn.as_mut().expect("ReaderGuard: connection missing")
    }
}

impl Drop for ReaderGuard<'_> {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            let _ = self.return_tx.send(conn);
        }
    }
}

impl DbManager {
    /// Open (or create) the database at `path`, run `migration` SQL, and
    /// set up the reader pool with `pool_size` connections.
    pub fn open(path: &Path, migration: &str, pool_size: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open the writer connection
        let writer = Connection::open(path)
            .with_context(|| format!("opening database at {}", path.display()))?;

        writer
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
            )
            .context("setting writer PRAGMAs")?;

        writer
            .execute_batch(migration)
            .context("running schema migration on writer")?;

        // Build the reader pool
        let (pool_tx, pool_rx) = crossbeam_channel::bounded(pool_size);
        for _ in 0..pool_size {
            let reader = Connection::open(path)
                .with_context(|| format!("opening reader connection to {}", path.display()))?;
            reader
                .execute_batch(
                    "PRAGMA journal_mode=WAL; PRAGMA query_only=1; PRAGMA busy_timeout=5000;",
                )
                .context("setting reader PRAGMAs")?;
            pool_tx
                .send(reader)
                .expect("reader pool channel should accept initial connections");
        }

        Ok(Self {
            writer: Mutex::new(writer),
            reader_pool: pool_tx,
            reader_checkout: pool_rx,
        })
    }

    /// Run a read operation against a pooled reader connection.
    /// The connection is automatically returned to the pool when `f` returns.
    pub fn with_reader<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let conn = self
            .reader_checkout
            .recv()
            .context("reader pool exhausted — this should not happen with correct sizing")?;
        // Reset any lingering transaction state on the reader
        let _ = conn.execute_batch("ROLLBACK;");
        let guard = ReaderGuard {
            conn: Some(conn),
            return_tx: &self.reader_pool,
        };
        f(&guard)
    }

    /// Run a write operation against the serialized writer connection.
    pub fn with_writer<T>(&self, f: impl FnOnce(&mut Connection) -> Result<T>) -> Result<T> {
        let mut conn = self.writer.lock();
        f(&mut conn)
    }
}

impl crate::storage::ProjectionStore for DbManager {
    fn record_benchmark(
        &self,
        report_id: &str,
        plan_id: &str,
        elapsed_ms: f64,
        exit_code: i32,
        output: &str,
    ) -> Result<()> {
        self.with_writer(|conn| {
            conn.execute(
                "INSERT INTO prism_bench_reports VALUES(?1,?2,?3,?4,?5)",
                rusqlite::params![report_id, plan_id, elapsed_ms, exit_code, output],
            )?;
            Ok(())
        })
    }

    fn put_trace(&self, trace_id: &str, snapshot: &serde_json::Value) -> Result<()> {
        self.with_writer(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS prism_traces(id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)")?;
            conn.execute("INSERT INTO prism_traces(id,snapshot_json) VALUES(?1,?2) ON CONFLICT(id) DO UPDATE SET snapshot_json=excluded.snapshot_json", rusqlite::params![trace_id, serde_json::to_string(snapshot)?])?;
            Ok(())
        })
    }

    fn get_trace(&self, trace_id: &str) -> Result<Option<serde_json::Value>> {
        self.with_reader(|conn| {
            let value = conn
                .query_row(
                    "SELECT snapshot_json FROM prism_traces WHERE id=?1",
                    [trace_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            value
                .map(|raw| serde_json::from_str(&raw))
                .transpose()
                .map_err(Into::into)
        })
    }

    fn record_kernel(
        &self,
        name: &str,
        backend: &str,
        artifact_hash: &str,
        byte_len: u64,
        target: &str,
    ) -> Result<()> {
        self.with_writer(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS kernel_registry (name TEXT PRIMARY KEY, backend TEXT NOT NULL, artifact_hash TEXT NOT NULL, byte_len INTEGER NOT NULL, target TEXT, registered_at TEXT NOT NULL DEFAULT(datetime('now')))")?;
            conn.execute("INSERT INTO kernel_registry(name,backend,artifact_hash,byte_len,target) VALUES(?1,?2,?3,?4,?5) ON CONFLICT(name) DO UPDATE SET backend=excluded.backend,artifact_hash=excluded.artifact_hash,byte_len=excluded.byte_len,target=excluded.target", rusqlite::params![name, backend, artifact_hash, byte_len as i64, target])?;
            Ok(())
        })
    }

    fn put_replay(&self, replay_id: &str, status: &str, payload: &serde_json::Value) -> Result<()> {
        self.with_writer(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS prism_replays(id TEXT PRIMARY KEY, payload TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT(datetime('now'))) ")?;
            conn.execute("INSERT INTO prism_replays(id,payload,status) VALUES(?1,?2,?3)", rusqlite::params![replay_id, serde_json::to_string(payload)?, status])?;
            Ok(())
        })
    }

    fn get_replay(&self, replay_id: &str) -> Result<Option<(String, serde_json::Value)>> {
        self.with_reader(|conn| {
            let value = conn
                .query_row(
                    "SELECT status,payload FROM prism_replays WHERE id=?1",
                    [replay_id],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()?;
            value
                .map(|(status, payload)| Ok((status, serde_json::from_str(&payload)?)))
                .transpose()
        })
    }
}

impl crate::storage::ExperimentStore for DbManager {
    fn put_experiment(&self, experiment_id: &str, document: &serde_json::Value) -> Result<()> {
        self.with_writer(|conn| {
            conn.execute_batch("CREATE TABLE IF NOT EXISTS experiment_projection(id TEXT PRIMARY KEY, document_json TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT(datetime('now'))) ")?;
            conn.execute("INSERT INTO experiment_projection(id,document_json) VALUES(?1,?2) ON CONFLICT(id) DO UPDATE SET document_json=excluded.document_json,updated_at=datetime('now')", rusqlite::params![experiment_id, serde_json::to_string(document)?])?;
            Ok(())
        })
    }

    fn get_experiment(&self, experiment_id: &str) -> Result<Option<serde_json::Value>> {
        self.with_reader(|conn| {
            let raw = conn
                .query_row(
                    "SELECT document_json FROM experiment_projection WHERE id=?1",
                    [experiment_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            raw.map(|value| serde_json::from_str(&value))
                .transpose()
                .map_err(Into::into)
        })
    }

    fn list_experiments(&self) -> Result<Vec<(String, serde_json::Value)>> {
        self.with_reader(|conn| {
            let mut statement = conn.prepare(
                "SELECT id,document_json FROM experiment_projection ORDER BY updated_at DESC",
            )?;
            let rows = statement.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.map(|row| {
                let (id, value) = row?;
                Ok((id, serde_json::from_str(&value)?))
            })
            .collect()
        })
    }
}

impl crate::storage::BenchmarkStore for DbManager {
    fn put_plan(&self, plan_id: &str, name: &str, spec: &serde_json::Value) -> Result<()> {
        self.with_writer(|conn| { conn.execute_batch("CREATE TABLE IF NOT EXISTS prism_bench_plans(id TEXT PRIMARY KEY,name TEXT NOT NULL,spec TEXT NOT NULL); CREATE TABLE IF NOT EXISTS prism_bench_reports(id TEXT PRIMARY KEY,plan_id TEXT NOT NULL,elapsed_ms REAL NOT NULL,exit_code INTEGER NOT NULL,output TEXT NOT NULL); CREATE TABLE IF NOT EXISTS prism_bench_baselines(name TEXT PRIMARY KEY,report_id TEXT NOT NULL)")?; conn.execute("INSERT INTO prism_bench_plans VALUES(?1,?2,?3)", rusqlite::params![plan_id, name, serde_json::to_string(spec)?])?; Ok(()) })
    }
    fn get_plan(&self, plan_id: &str) -> Result<Option<serde_json::Value>> {
        self.with_reader(|conn| {
            let raw = conn
                .query_row(
                    "SELECT spec FROM prism_bench_plans WHERE id=?1",
                    [plan_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            raw.map(|v| serde_json::from_str(&v))
                .transpose()
                .map_err(Into::into)
        })
    }
    fn get_report(&self, report_id: &str) -> Result<Option<(f64, i32)>> {
        self.with_reader(|conn| {
            Ok(conn
                .query_row(
                    "SELECT elapsed_ms,exit_code FROM prism_bench_reports WHERE id=?1",
                    [report_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()?)
        })
    }
    fn get_baseline(&self, name: &str) -> Result<Option<String>> {
        self.with_reader(|conn| {
            Ok(conn
                .query_row(
                    "SELECT report_id FROM prism_bench_baselines WHERE name=?1",
                    [name],
                    |row| row.get(0),
                )
                .optional()?)
        })
    }
    fn put_baseline(&self, name: &str, report_id: &str) -> Result<()> {
        self.with_writer(|conn| { conn.execute_batch("CREATE TABLE IF NOT EXISTS prism_bench_baselines(name TEXT PRIMARY KEY,report_id TEXT NOT NULL)")?; conn.execute("INSERT INTO prism_bench_baselines VALUES(?1,?2) ON CONFLICT(name) DO UPDATE SET report_id=excluded.report_id", rusqlite::params![name, report_id])?; Ok(()) })
    }
}

impl crate::storage::KnowledgeStore for DbManager {
    fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<crate::storage::KnowledgeSearchResult>> {
        self.with_reader(|conn| {
            let fts = query.split_whitespace().map(|word| format!("\"{}\"", word)).collect::<Vec<_>>().join(" AND ");
            let mut statement = conn.prepare("SELECT s.id,s.document_id,s.heading,s.word_count,d.title,d.doc_type,s.content,rank FROM sections_fts JOIN sections s ON sections_fts.rowid=s.rowid JOIN documents d ON s.document_id=d.id WHERE sections_fts MATCH ?1 ORDER BY rank LIMIT ?2")?;
            let rows = statement.query_map(rusqlite::params![fts, limit as i64], |row| Ok(crate::storage::KnowledgeSearchResult { section_id: row.get(0)?, document_id: row.get(1)?, heading: row.get(2)?, word_count: row.get(3)?, doc_title: row.get(4)?, doc_type: row.get(5)?, snippet: row.get(6)?, rank: row.get(7)? }))?;
            rows.collect::<std::result::Result<Vec<_>, _>>().map_err(Into::into)
        })
    }
    fn get_document(&self, id: &str) -> Result<Option<crate::storage::KnowledgeDocument>> {
        self.with_reader(|conn| Ok(conn.query_row("SELECT id,title,doc_type,content_md,version,status,created_at FROM documents WHERE id=?1", [id], |row| Ok(crate::storage::KnowledgeDocument { id: row.get(0)?, title: row.get(1)?, doc_type: row.get(2)?, content: row.get(3)?, version: row.get(4)?, status: row.get(5)?, created_at: row.get(6)? })).optional()?))
    }
    fn list_documents(
        &self,
        doc_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<crate::storage::KnowledgeListRow>> {
        self.with_reader(|conn| { let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(kind) = doc_type { ("SELECT id,title,doc_type,version,status,updated_at FROM documents WHERE doc_type=?1 ORDER BY updated_at DESC LIMIT ?2", vec![Box::new(kind.to_string()),Box::new(limit as i64)]) } else { ("SELECT id,title,doc_type,version,status,updated_at FROM documents ORDER BY updated_at DESC LIMIT ?1", vec![Box::new(limit as i64)]) }; let mut statement=conn.prepare(sql)?; let rows=statement.query_map(rusqlite::params_from_iter(params.iter()), |row| Ok(crate::storage::KnowledgeListRow{id:row.get(0)?,title:row.get(1)?,doc_type:row.get(2)?,version:row.get(3)?,status:row.get(4)?,updated_at:row.get(5)?}))?; rows.collect::<std::result::Result<Vec<_>,_>>().map_err(Into::into) })
    }
}
