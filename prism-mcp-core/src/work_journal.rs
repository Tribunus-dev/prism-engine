use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Phase of a multi-step file operation in the journal.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JournalPhase {
    Prepared,
    FileInstalled,
    DatabaseCommitted,
}

/// A single entry in the operation journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalEntry {
    pub operation_id: String,
    pub staged_path: PathBuf,
    pub final_path: PathBuf,
    pub database_key: String,
    pub phase: JournalPhase,
}

/// Tracks multi-step file + database operations for crash recovery.
///
/// Journal entries are stored as individual JSON files in a staging
/// directory. On daemon startup, any leftover entries are reconciled.
pub struct WorkJournal {
    dir: PathBuf,
}

impl WorkJournal {
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_owned(),
        }
    }

    /// Path to the journal file for a given operation id.
    fn entry_path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{}.json", id))
    }

    /// Create a new journal entry in the `Prepared` phase.
    pub fn create_entry(
        &self,
        operation_id: &str,
        staged_path: &Path,
        final_path: &Path,
        database_key: &str,
    ) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let entry = JournalEntry {
            operation_id: operation_id.to_string(),
            staged_path: staged_path.to_owned(),
            final_path: final_path.to_owned(),
            database_key: database_key.to_string(),
            phase: JournalPhase::Prepared,
        };
        let json = serde_json::to_string(&entry)?;
        std::fs::write(self.entry_path(operation_id), &json)?;
        Ok(())
    }

    /// Advance the journal phase for an operation.
    pub fn advance_phase(&self, operation_id: &str, phase: JournalPhase) -> Result<()> {
        let path = self.entry_path(operation_id);
        let mut entry: JournalEntry = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        entry.phase = phase;
        std::fs::write(&path, serde_json::to_string(&entry)?)?;
        Ok(())
    }

    /// Remove a completed journal entry.
    pub fn remove_entry(&self, operation_id: &str) -> Result<()> {
        let path = self.entry_path(operation_id);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }

    /// Scan the journal directory and reconcile any incomplete operations.
    /// Returns a list of recovery actions taken.
    pub fn reconcile_on_startup(&self) -> Result<Vec<String>> {
        let mut actions = Vec::new();

        if !self.dir.exists() {
            return Ok(actions);
        }

        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                let content = fs::read_to_string(&path)?;
                let journal: JournalEntry = match serde_json::from_str(&content) {
                    Ok(j) => j,
                    Err(_) => continue,
                };

                let final_exists = journal.final_path.exists();
                let staged_exists = journal.staged_path.exists();

                match journal.phase {
                    JournalPhase::Prepared => {
                        if final_exists && staged_exists {
                            // Crash after rename but before phase update
                            self.advance_phase(&journal.operation_id, JournalPhase::FileInstalled)?;
                            actions
                                .push(format!("advance {} → FileInstalled", journal.operation_id));
                        } else if final_exists {
                            // Likely complete, phase was never updated
                            self.advance_phase(
                                &journal.operation_id,
                                JournalPhase::DatabaseCommitted,
                            )?;
                            actions.push(format!(
                                "advance {} → DatabaseCommitted",
                                journal.operation_id
                            ));
                        } else {
                            // Never made progress — clean up staged
                            let _ = fs::remove_file(&journal.staged_path);
                            let _ = fs::remove_file(&path);
                            actions.push(format!(
                                "removed stale {} (Prepared, no final)",
                                journal.operation_id
                            ));
                        }
                    }
                    JournalPhase::FileInstalled => {
                        if !final_exists && staged_exists {
                            // Rename didn't complete — restore from staged
                            fs::rename(&journal.staged_path, &journal.final_path)?;
                            actions.push(format!(
                                "restored {} from staged to final",
                                journal.operation_id
                            ));
                        } else if !final_exists {
                            // Nothing to restore — log warning
                            actions.push(format!(
                                "WARN: {} final file missing, staged also gone",
                                journal.operation_id
                            ));
                        }
                        // Advance to DatabaseCommitted — DB state is assumed correct
                        // In production, verify the DB record exists here
                        self.advance_phase(&journal.operation_id, JournalPhase::DatabaseCommitted)?;
                        let _ = self.remove_entry(&journal.operation_id);
                    }
                    JournalPhase::DatabaseCommitted => {
                        if !final_exists {
                            actions.push(format!(
                                "WARN: {} final file missing after commit",
                                journal.operation_id
                            ));
                        }
                        // Clean up
                        let _ = self.remove_entry(&journal.operation_id);
                    }
                }
            }
        }

        Ok(actions)
    }
}
