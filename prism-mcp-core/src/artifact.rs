use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::db::DbManager;
use crate::evidence::ToolInvocationId;
use std::sync::Arc;

/// A content-addressed identifier derived from a BLAKE3 hash of the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArtifactId {
    pub digest: [u8; 32],
}

impl ArtifactId {
    pub fn from_data(data: &[u8]) -> Self {
        Self {
            digest: *blake3::hash(data).as_bytes(),
        }
    }

    pub fn hex(&self) -> String {
        hex::encode(self.digest)
    }
}

/// Classification of stored artifacts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    Cimage,
    KernelRecipe,
    Hsaco,
    MetalLibrary,
    CoreMlBundle,
    CpuObject,
    CompilerIr,
    LlvmIr,
    Disassembly,
    BenchmarkTrace,
    ValidationCorpus,
    BuildLog,
    ModelManifest,
    TensorInventory,
    BuildPlan,
    CompilerDiagnostics,
    KernelCandidateSet,
    ResourceReport,
    AdmissionPlan,
    CalibrationCorpus,
    ValidationReport,
    BenchmarkPlan,
    BenchmarkSamples,
    BenchmarkReport,
    TraceCapture,
    TraceSummary,
    ReplayBundle,
    ExperimentSpec,
    ExperimentReport,
}

/// Metadata record for a stored artifact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: ArtifactId,
    pub kind: ArtifactKind,
    pub byte_len: u64,
    pub media_type: String,
    pub producer: ToolInvocationId,
    pub target: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// Content-addressed blob store.
///
/// Filesystem layout:
///   <base>/<kind>/<first-two-hex>/<remaining-hex>
///
/// SQLite stores metadata and relationships. The filesystem stores
/// immutable blobs keyed by content hash.
#[derive(Clone)]
pub struct ArtifactStore {
    base: PathBuf,
    db: Arc<DbManager>,
}

impl ArtifactStore {
    /// Open or create the artifact store rooted at `base`.
    /// The SQLite metadata database lives at `base/metadata.db`.
    pub fn open(base: &Path) -> Result<Self> {
        std::fs::create_dir_all(base)?;

        let db_path = base.join("metadata.db");
        let migration = concat!(
            "CREATE TABLE IF NOT EXISTS artifact_edges (
                parent_hash  BLOB NOT NULL REFERENCES artifacts(id_hash),
                child_hash   BLOB NOT NULL REFERENCES artifacts(id_hash),
                relation     TEXT NOT NULL,
                PRIMARY KEY (parent_hash, child_hash, relation)
            );
            CREATE TABLE IF NOT EXISTS artifacts (",
            "  id_hash       BLOB PRIMARY KEY,",
            "  kind          TEXT NOT NULL,",
            "  byte_len      INTEGER NOT NULL,",
            "  media_type    TEXT NOT NULL,",
            "  producer      TEXT NOT NULL,",
            "  target        TEXT,",
            "  created_at    TEXT NOT NULL DEFAULT (datetime('now'))",
            ");",
            "CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);",
        );

        let db = Arc::new(DbManager::open(&db_path, migration, 4)?);

        Ok(Self {
            base: base.to_owned(),
            db,
        })
    }

    /// Store a blob and return its content-addressed identifier.
    /// If the blob already exists, returns the existing id without rewriting.
    pub fn put(
        &self,
        data: &[u8],
        kind: ArtifactKind,
        producer: &ToolInvocationId,
    ) -> Result<ArtifactId> {
        let id = ArtifactId::from_data(data);
        let hex = id.hex();
        let subdir = &hex[0..2];
        let filename = &hex[2..];

        let kind_dir = self.base.join(format!("{:?}", kind).to_lowercase());
        let file_path = kind_dir.join(subdir).join(filename);

        if !file_path.exists() {
            std::fs::create_dir_all(file_path.parent().unwrap())?;
            std::fs::write(&file_path, data)?;
        }

        // Register metadata (ignore duplicate)
        let pid = producer.0.to_string();
        self.db.with_writer(|conn| {
            conn.execute(
                "INSERT OR IGNORE INTO artifacts (id_hash, kind, byte_len, media_type, producer) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    id.digest.as_slice(),
                    format!("{:?}", kind),
                    data.len() as i64,
                    "application/octet-stream",
                    pid,
                ],
            )?;
            Ok(())
        })?;

        Ok(id)
    }

    /// Retrieve a blob by its content id.
    pub fn get(&self, id: &ArtifactId) -> Result<Option<Vec<u8>>> {
        match self.get_path(id) {
            Some(path) => std::fs::read(&path).map(Some).map_err(Into::into),
            None => Ok(None),
        }
    }

    /// Get the filesystem path for an artifact, if it exists.
    pub fn get_path(&self, id: &ArtifactId) -> Option<PathBuf> {
        let hex = id.hex();
        let subdir = &hex[0..2];
        let filename = &hex[2..];

        // Search across all kind directories
        let kind_dirs = [
            "cimage",
            "kernelrecipe",
            "hsaco",
            "metallibrary",
            "coremlbundle",
            "cpuobject",
            "compilerir",
            "disassembly",
            "benchmarktrace",
            "validationcorpus",
            "buildlog",
            "modelmanifest",
        ];

        for kind in &kind_dirs {
            let path = self.base.join(kind).join(subdir).join(filename);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    /// List all artifact records, optionally filtered by kind.
    pub fn list(&self, kind: Option<&ArtifactKind>) -> Result<Vec<ArtifactRecord>> {
        let kind_filter = kind.map(|k| format!("{:?}", k).to_lowercase());
        self.db.with_reader(|conn| {
            let (sql, params): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(ref k) = kind_filter {
                ("SELECT id_hash, kind, byte_len, media_type, producer, target, created_at FROM artifacts WHERE kind = ?1".into(),
                 vec![Box::new(k.clone()) as Box<dyn rusqlite::types::ToSql>])
            } else {
                ("SELECT id_hash, kind, byte_len, media_type, producer, target, created_at FROM artifacts ORDER BY created_at DESC".into(),
                 vec![])
            };

            let mut stmt = conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
                let hash_bytes: Vec<u8> = row.get(0)?;
                let mut digest = [0u8; 32];
                if hash_bytes.len() == 32 {
                    digest.copy_from_slice(&hash_bytes);
                }
                Ok(ArtifactRecord {
                    id: ArtifactId { digest },
                    kind: match row.get::<_, String>(1)?.to_lowercase().as_str() {
                        "cimage" => ArtifactKind::Cimage,
                        "kernelrecipe" => ArtifactKind::KernelRecipe,
                        "hsaco" => ArtifactKind::Hsaco,
                        "metallibrary" => ArtifactKind::MetalLibrary,
                        "coremlbundle" => ArtifactKind::CoreMlBundle,
                        "cpuobject" => ArtifactKind::CpuObject,
                        "compilerir" => ArtifactKind::CompilerIr,
                        "disassembly" => ArtifactKind::Disassembly,
                        "benchmarktrace" => ArtifactKind::BenchmarkTrace,
                        "validationcorpus" => ArtifactKind::ValidationCorpus,
                        "buildlog" => ArtifactKind::BuildLog,
                        "modelmanifest" => ArtifactKind::ModelManifest,
                        _ => ArtifactKind::BuildLog,
                    },
                    byte_len: row.get::<_, i64>(2)? as u64,
                    media_type: row.get(3)?,
                    producer: crate::evidence::ToolInvocationId(row.get::<_, String>(4)?.parse().unwrap_or_default()),
                    target: row.get(5)?,
                    created_at: row.get::<_, String>(6)?.parse().unwrap_or_else(|_| Utc::now()),
                })
            })?;

            rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
        })
    }

    /// Link two artifacts with a typed relationship.
    pub fn link(&self, parent: &ArtifactId, child: &ArtifactId, relation: &str) -> Result<()> {
        self.db.with_writer(|conn| {
            conn.execute(
                "INSERT OR IGNORE INTO artifact_edges (parent_hash, child_hash, relation) VALUES (?1, ?2, ?3)",
                rusqlite::params![parent.digest.as_slice(), child.digest.as_slice(), relation],
            )?;
            Ok(())
        })
    }

    /// Returns all children of this artifact with their relationship labels.
    pub fn children(&self, id: &ArtifactId) -> Result<Vec<(ArtifactId, String)>> {
        self.db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT child_hash, relation FROM artifact_edges WHERE parent_hash = ?1",
            )?;
            let rows = stmt.query_map(rusqlite::params![id.digest.as_slice()], |row| {
                let hash_bytes: Vec<u8> = row.get(0)?;
                let mut digest = [0u8; 32];
                if hash_bytes.len() == 32 {
                    digest.copy_from_slice(&hash_bytes);
                }
                Ok((ArtifactId { digest }, row.get::<_, String>(1)?))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Into::into)
        })
    }

    /// Returns all parents of this artifact with their relationship labels.
    pub fn parents(&self, id: &ArtifactId) -> Result<Vec<(ArtifactId, String)>> {
        self.db.with_reader(|conn| {
            let mut stmt = conn.prepare(
                "SELECT parent_hash, relation FROM artifact_edges WHERE child_hash = ?1",
            )?;
            let rows = stmt.query_map(rusqlite::params![id.digest.as_slice()], |row| {
                let hash_bytes: Vec<u8> = row.get(0)?;
                let mut digest = [0u8; 32];
                if hash_bytes.len() == 32 {
                    digest.copy_from_slice(&hash_bytes);
                }
                Ok((ArtifactId { digest }, row.get::<_, String>(1)?))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Into::into)
        })
    }
}

impl crate::storage::ArtifactRepository for ArtifactStore {
    fn put(
        &self,
        data: &[u8],
        kind: ArtifactKind,
        producer: &ToolInvocationId,
    ) -> Result<ArtifactId> {
        self.put(data, kind, producer)
    }
    fn list(&self, kind: Option<&ArtifactKind>) -> Result<Vec<ArtifactRecord>> {
        self.list(kind)
    }
}
