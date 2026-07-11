use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender};
use parking_lot::Mutex;
use rusqlite::Connection;
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
