//! Calibration frontier — disk-backed, append-only store for teacher/student
//! hidden states used by the distill-compiler.
//!
//! Each shard is a fixed-size microbatch of hidden states. The frontier
//! maintains a digest chain so any tampering with past stages is detectable
//! at verification time.
//!
//! Directory layout (under `compile-run/`):
//!
//! ```text
//! teacher-frontier/
//!   stage-000/shards.bin
//!   stage-001/shards.bin
//!   scheduler-state.json
//! student-frontier/
//!   stage-000/shards.bin
//!   scheduler-state.json
//! ```

use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};

// ── CalibrationFrontier ────────────────────────────────────────────────────

/// Disk-backed, append-only calibration frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationFrontier {
    pub base_path: PathBuf,
    pub namespace: FrontierNamespace,
    pub stages: Vec<FrontierStage>,
    /// Digest chain: each stage's digest incorporates the previous.
    pub digest_chain: Vec<[u8; 32]>,
}

// ── FrontierNamespace ──────────────────────────────────────────────────────

/// Whether this frontier belongs to the teacher or student model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrontierNamespace {
    Teacher,
    Student,
}

impl FrontierNamespace {
    /// Returns the sub-directory name for this namespace under `base_path`.
    pub fn as_dir_name(&self) -> &'static str {
        match self {
            Self::Teacher => "teacher-frontier",
            Self::Student => "student-frontier",
        }
    }
}

// ── FrontierStage ──────────────────────────────────────────────────────────

/// A single frontier stage — one fixed-size microbatch of hidden states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierStage {
    pub stage_index: u32,
    pub microbatch_count: u32,
    pub shard_path: PathBuf,
    pub metadata: FrontierMetadata,
    pub digest: [u8; 32],
}

// ── FrontierMetadata ───────────────────────────────────────────────────────

/// Metadata describing the tensor layout and provenance of a stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierMetadata {
    pub sequence_length: usize,
    pub hidden_dim: usize,
    pub microbatch_bytes: u64,
    pub created_at_ns: u64,
    pub attention_mask_digest: [u8; 32],
    pub positional_metadata_digest: [u8; 32],
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Canonical byte encoding of metadata for digest computation.
///
/// Uses JCS (JSON Canonicalization Scheme) so digests are architecture-
/// independent: same logical data always produces the same bytes regardless
/// of padding, endianness, or field ordering in the struct definition.
fn canonical_metadata_bytes(m: &FrontierMetadata) -> Vec<u8> {
    serde_json_canonicalizer::to_vec(m).unwrap_or_else(|_| {
        // Fallback: fixed-width LE encoding (not portable to BE hosts).
        let mut buf = Vec::with_capacity(8 + 8 + 8 + 8 + 32 + 32);
        buf.extend_from_slice(&(m.sequence_length as u64).to_le_bytes());
        buf.extend_from_slice(&(m.hidden_dim as u64).to_le_bytes());
        buf.extend_from_slice(&m.microbatch_bytes.to_le_bytes());
        buf.extend_from_slice(&m.created_at_ns.to_le_bytes());
        buf.extend_from_slice(&m.attention_mask_digest);
        buf.extend_from_slice(&m.positional_metadata_digest);
        buf
    })
}

// ── Implementation ─────────────────────────────────────────────────────────

impl CalibrationFrontier {
    /// Create a new empty frontier rooted at `base_path / namespace_dir /`.
    ///
    /// No directories or files are created until [`append_stage`] is called.
    pub fn new(base_path: PathBuf, namespace: FrontierNamespace) -> Self {
        Self {
            base_path,
            namespace,
            stages: Vec::new(),
            digest_chain: Vec::new(),
        }
    }

    /// Resolve the namespace sub-directory.
    fn namespace_dir(&self) -> PathBuf {
        self.base_path.join(self.namespace.as_dir_name())
    }

    /// Resolve the stage-`index` sub-directory.
    fn stage_dir(&self, index: u32) -> PathBuf {
        self.namespace_dir().join(format!("stage-{:03}", index))
    }

    /// Append a new stage containing a fixed-size microbatch.
    ///
    /// Writes `shard_data` to `stage-{index:03}/shards.bin`, computes the
    /// chained digest (incorporating the previous stage's digest, canonical
    /// metadata bytes, and shard data), and records the stage in memory.
    pub fn append_stage(
        &mut self,
        shard_data: &[u8],
        metadata: FrontierMetadata,
    ) -> io::Result<FrontierStage> {
        let stage_index = self.stages.len() as u32;
        let dir = self.stage_dir(stage_index);

        std::fs::create_dir_all(&dir)?;

        let shard_path = dir.join("shards.bin");
        std::fs::write(&shard_path, shard_data)?;

        // Chained digest: blake3(prev_digest || canonical_metadata || shard_data)
        let prev_digest = self.digest_chain.last().copied().unwrap_or([0u8; 32]);
        let mut hasher = blake3::Hasher::new();
        hasher.update(&prev_digest);
        hasher.update(&canonical_metadata_bytes(&metadata));
        hasher.update(shard_data);
        let digest: [u8; 32] = hasher.finalize().into();

        let stage = FrontierStage {
            stage_index,
            microbatch_count: 1,
            shard_path,
            metadata,
            digest,
        };

        self.stages.push(stage.clone());
        self.digest_chain.push(digest);

        Ok(stage)
    }

    /// Verify every stage's digest against the chained hash.
    ///
    /// Re-reads each shard from disk, recomputes the digest chain from
    /// genesis, and compares against both the stage's stored digest and
    /// the chain vector. Returns `true` if and only if the entire chain
    /// is intact.
    pub fn verify_chain(&self) -> bool {
        let mut prev = [0u8; 32];

        for (i, stage) in self.stages.iter().enumerate() {
            let shard = match std::fs::read(&stage.shard_path) {
                Ok(d) => d,
                Err(_) => return false,
            };

            let mut hasher = blake3::Hasher::new();
            hasher.update(&prev);
            hasher.update(&canonical_metadata_bytes(&stage.metadata));
            hasher.update(&shard);
            let computed: [u8; 32] = hasher.finalize().into();

            // Stage's own claimed digest must match
            if computed != stage.digest {
                return false;
            }

            // Digest chain entry must match
            if self.digest_chain.get(i).map_or(true, |&d| d != computed) {
                return false;
            }

            prev = computed;
        }

        true
    }

    /// Iterate over stages in order.
    pub fn stages(&self) -> impl Iterator<Item = &FrontierStage> {
        self.stages.iter()
    }

    /// Load frontier state from a `scheduler-state.json` path.
    ///
    /// The JSON must contain the full `CalibrationFrontier` serialized state
    /// (binary shard data stays on disk). After deserialization every stage's
    /// shard file is verified to exist.
    pub fn load_scheduler_state(path: &Path) -> io::Result<Self> {
        let json_bytes = std::fs::read_to_string(path)?;
        let frontier: Self = serde_json::from_str(&json_bytes)?;

        // Validate that every stage has an on-disk shard file.
        for stage in &frontier.stages {
            if !stage.shard_path.exists() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("stage shard not found: {}", stage.shard_path.display()),
                ));
            }
        }

        Ok(frontier)
    }

    /// Save frontier state to `scheduler-state.json` under the namespace directory.
    pub fn save_scheduler_state(&self) -> io::Result<()> {
        std::fs::create_dir_all(self.namespace_dir())?;
        let path = self.namespace_dir().join("scheduler-state.json");
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata(sequence_length: usize, hidden_dim: usize) -> FrontierMetadata {
        FrontierMetadata {
            sequence_length,
            hidden_dim,
            microbatch_bytes: (sequence_length * hidden_dim * 4) as u64,
            created_at_ns: 1_000_000_000,
            attention_mask_digest: [1u8; 32],
            positional_metadata_digest: [2u8; 32],
        }
    }

    #[test]
    fn append_and_verify_chain() {
        let dir = tempfile::tempdir().unwrap();
        let mut frontier = CalibrationFrontier::new(dir.path().to_path_buf(), FrontierNamespace::Teacher);

        let stage1 = frontier
            .append_stage(b"shard_one", test_metadata(128, 64))
            .unwrap();
        assert_eq!(stage1.stage_index, 0);
        assert_eq!(frontier.stages.len(), 1);
        assert_eq!(frontier.digest_chain.len(), 1);

        let stage2 = frontier
            .append_stage(b"shard_two", test_metadata(256, 128))
            .unwrap();
        assert_eq!(stage2.stage_index, 1);
        assert_eq!(frontier.stages.len(), 2);

        // Chain must verify after appends.
        assert!(frontier.verify_chain());
    }

    #[test]
    fn verify_chain_fails_on_tampered_shard() {
        let dir = tempfile::tempdir().unwrap();
        let mut frontier = CalibrationFrontier::new(dir.path().to_path_buf(), FrontierNamespace::Student);

        frontier
            .append_stage(b"shard_one", test_metadata(128, 64))
            .unwrap();
        frontier
            .append_stage(b"shard_two", test_metadata(256, 128))
            .unwrap();

        // Tamper with the first stage's shard on disk.
        let stage0_path = &frontier.stages[0].shard_path;
        std::fs::write(stage0_path, b"tampered_data").unwrap();

        assert!(!frontier.verify_chain());
    }

    #[test]
    fn save_and_load_scheduler_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut frontier = CalibrationFrontier::new(dir.path().to_path_buf(), FrontierNamespace::Teacher);

        frontier
            .append_stage(b"data_a", test_metadata(64, 32))
            .unwrap();
        frontier
            .append_stage(b"data_b", test_metadata(128, 64))
            .unwrap();

        // Save
        frontier.save_scheduler_state().unwrap();

        // Load from the expected scheduler-state.json path
        let state_path = dir
            .path()
            .join(FrontierNamespace::Teacher.as_dir_name())
            .join("scheduler-state.json");
        let loaded = CalibrationFrontier::load_scheduler_state(&state_path).unwrap();

        assert!(loaded.verify_chain());
        assert_eq!(loaded.stages.len(), 2);
        assert_eq!(loaded.namespace, FrontierNamespace::Teacher);
        assert_eq!(loaded.stages[0].digest, frontier.stages[0].digest);
        assert_eq!(loaded.stages[1].digest, frontier.stages[1].digest);
    }

    #[test]
    fn load_scheduler_state_missing_shard_fails() {
        let dir = tempfile::tempdir().unwrap();
        let mut frontier = CalibrationFrontier::new(dir.path().to_path_buf(), FrontierNamespace::Student);
        frontier
            .append_stage(b"data", test_metadata(64, 32))
            .unwrap();

        // Save and then delete the shard file.
        frontier.save_scheduler_state().unwrap();
        std::fs::remove_file(&frontier.stages[0].shard_path).unwrap();

        let state_path = dir
            .path()
            .join(FrontierNamespace::Student.as_dir_name())
            .join("scheduler-state.json");
        let result = CalibrationFrontier::load_scheduler_state(&state_path);
        assert!(result.is_err());
    }

    #[test]
    fn stages_iterator() {
        let dir = tempfile::tempdir().unwrap();
        let mut frontier = CalibrationFrontier::new(dir.path().to_path_buf(), FrontierNamespace::Teacher);

        frontier
            .append_stage(b"a", test_metadata(64, 32))
            .unwrap();
        frontier
            .append_stage(b"b", test_metadata(128, 64))
            .unwrap();

        let collected: Vec<_> = frontier.stages().collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].stage_index, 0);
        assert_eq!(collected[1].stage_index, 1);
    }
}
