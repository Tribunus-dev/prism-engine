use crate::persistence::{EventLogEntry, EventStore, Snapshot};
use crate::types::*;
use std::io::{BufWriter, Read, Write};
use std::path::Path;

/// A file-backed EventStore that persists events to an append-only log.
///
/// Format: one JSON line per EventLogEntry, terminated by '\n'.
/// Snapshots stored in a separate file (one JSON line per snapshot).
///
/// On construction, existing log files are scanned to rebuild
/// the in-memory index.
pub struct FsEventStore {
    events: Vec<EventLogEntry>,
    snapshots: Vec<Snapshot>,
    #[expect(dead_code, reason = "stored for potential recovery/reporting use")]
    log_path: String,
    snapshot_path: String,
    file: Option<std::fs::File>,
}

impl FsEventStore {
    /// Create or open an event store at the given log path.
    /// If the file exists, it is scanned to rebuild the in-memory index.
    pub fn open(
        log_path: impl AsRef<Path>,
        snapshot_path: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let log_path = log_path.as_ref().to_string_lossy().to_string();
        let snapshot_path = snapshot_path.as_ref().to_string_lossy().to_string();

        // Read existing events from file (if any)
        let events = if Path::new(&log_path).exists() {
            Self::read_events_from_file(&log_path)?
        } else {
            Vec::new()
        };

        // Read existing snapshots (if any)
        let snapshots = if Path::new(&snapshot_path).exists() {
            Self::read_snapshots_from_file(&snapshot_path)?
        } else {
            Vec::new()
        };

        // Open file for appending
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| format!("failed to open event log {}: {}", log_path, e))?;

        Ok(Self {
            events,
            snapshots,
            log_path,
            snapshot_path,
            file: Some(file),
        })
    }

    fn read_events_from_file(path: &str) -> Result<Vec<EventLogEntry>, String> {
        let mut file =
            std::fs::File::open(path).map_err(|e| format!("failed to open {}: {}", path, e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("failed to read {}: {}", path, e))?;

        let mut events = Vec::new();
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: EventLogEntry = serde_json::from_str(trimmed)
                .map_err(|e| format!("failed to parse event line in {}: {}", path, e))?;
            events.push(entry);
        }
        Ok(events)
    }

    fn read_snapshots_from_file(path: &str) -> Result<Vec<Snapshot>, String> {
        let mut file =
            std::fs::File::open(path).map_err(|e| format!("failed to open {}: {}", path, e))?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .map_err(|e| format!("failed to read {}: {}", path, e))?;

        let mut snapshots = Vec::new();
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let snap: Snapshot = serde_json::from_str(trimmed)
                .map_err(|e| format!("failed to parse snapshot line in {}: {}", path, e))?;
            snapshots.push(snap);
        }
        Ok(snapshots)
    }

    fn append_entry_to_file(file: &mut std::fs::File, entry: &EventLogEntry) -> Result<(), String> {
        let json = serde_json::to_string(entry)
            .map_err(|e| format!("failed to serialize event: {}", e))?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{}", json).map_err(|e| format!("failed to write event: {}", e))?;
        writer
            .flush()
            .map_err(|e| format!("failed to flush: {}", e))?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|e| format!("failed to fsync: {}", e))?;
        Ok(())
    }
}

impl EventStore for FsEventStore {
    fn append_events(
        &mut self,
        epoch: WorldEpoch,
        entries: &[EventLogEntry],
    ) -> Result<(), String> {
        for entry in entries {
            if entry.epoch != epoch {
                return Err(format!(
                    "epoch mismatch: entry {:?} != batch {:?}",
                    entry.epoch, epoch
                ));
            }
        }

        // Write to file first (durable before in-memory)
        if let Some(ref mut file) = self.file {
            for entry in entries {
                Self::append_entry_to_file(file, entry)?;
            }
        }

        // Now update in-memory index
        self.events.extend(entries.iter().cloned());
        Ok(())
    }

    fn get_events_from(&self, from_epoch: WorldEpoch) -> Vec<EventLogEntry> {
        self.events
            .iter()
            .filter(|e| e.epoch >= from_epoch)
            .cloned()
            .collect()
    }

    fn store_snapshot(&mut self, snapshot: Snapshot) -> Result<(), String> {
        if !self.snapshot_path.is_empty() {
            let json = serde_json::to_string(&snapshot)
                .map_err(|e| format!("failed to serialize snapshot: {}", e))?;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.snapshot_path)
                .map_err(|e| format!("failed to open snapshot file: {}", e))?;
            writeln!(file, "{}", json).map_err(|e| format!("failed to write snapshot: {}", e))?;
            file.sync_all()
                .map_err(|e| format!("failed to fsync snapshot: {}", e))?;
        }

        self.snapshots.push(snapshot);
        Ok(())
    }

    fn latest_snapshot(&self) -> Option<Snapshot> {
        self.snapshots.last().cloned()
    }

    fn event_count(&self) -> u64 {
        self.events.len() as u64
    }

    fn latest_epoch(&self) -> Option<WorldEpoch> {
        self.events.last().map(|e| e.epoch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::DomainEvent;
    use crate::persistence::EventLogEntry;

    fn make_event(epoch: u64, seq: u64) -> EventLogEntry {
        EventLogEntry {
            epoch: WorldEpoch(epoch),
            sequence: seq,
            event: DomainEvent {
                id: MessageId::compute(format!("ev-{}", seq).as_bytes()),
                kind: "test_event".to_string(),
                entity_id: None,
                payload: serde_json::json!({"seq": seq}),
            },
            world_digest: [seq as u8; 32],
        }
    }

    fn temp_path(suffix: &str) -> String {
        format!(
            "/tmp/prism-fs-eventstore-test-{}-{}",
            suffix,
            std::process::id()
        )
    }

    #[test]
    fn test_fs_event_store_open_create() {
        let log_path = temp_path("create");
        let snap_path = temp_path("create-snap");
        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);

        let store = FsEventStore::open(&log_path, &snap_path).expect("should create new store");
        assert_eq!(store.event_count(), 0);
        assert_eq!(store.latest_epoch(), None);

        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);
    }

    #[test]
    fn test_fs_event_store_write_and_read() {
        let log_path = temp_path("wr");
        let snap_path = temp_path("wr-snap");
        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);

        {
            let mut store = FsEventStore::open(&log_path, &snap_path).expect("open");
            let entry = make_event(1, 1);
            store
                .append_events(WorldEpoch(1), &[entry])
                .expect("append");

            assert_eq!(store.event_count(), 1);
            assert_eq!(store.latest_epoch(), Some(WorldEpoch(1)));
        }

        // Re-open and verify data persisted
        {
            let store = FsEventStore::open(&log_path, &snap_path).expect("reopen");
            assert_eq!(store.event_count(), 1);
            assert_eq!(store.latest_epoch(), Some(WorldEpoch(1)));

            let events = store.get_events_from(WorldEpoch(1));
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].sequence, 1);
            assert_eq!(events[0].epoch, WorldEpoch(1));
        }

        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);
    }

    #[test]
    fn test_fs_event_store_multiple_epochs() {
        let log_path = temp_path("multi");
        let snap_path = temp_path("multi-snap");
        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);

        let mut store = FsEventStore::open(&log_path, &snap_path).expect("open");

        store
            .append_events(WorldEpoch(1), &[make_event(1, 1)])
            .expect("epoch 1");
        store
            .append_events(WorldEpoch(2), &[make_event(2, 2)])
            .expect("epoch 2");
        store
            .append_events(WorldEpoch(3), &[make_event(3, 3), make_event(3, 4)])
            .expect("epoch 3");

        assert_eq!(store.event_count(), 4);

        let from_epoch_2 = store.get_events_from(WorldEpoch(2));
        assert_eq!(from_epoch_2.len(), 3);
        assert_eq!(from_epoch_2[0].epoch, WorldEpoch(2));

        // Re-open and verify
        drop(store);
        let store = FsEventStore::open(&log_path, &snap_path).expect("reopen");
        assert_eq!(store.event_count(), 4);
        assert_eq!(store.latest_epoch(), Some(WorldEpoch(3)));

        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);
    }

    #[test]
    fn test_fs_event_store_epoch_mismatch_rejected() {
        let log_path = temp_path("epoch");
        let snap_path = temp_path("epoch-snap");
        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);

        let mut store = FsEventStore::open(&log_path, &snap_path).expect("open");
        let entry = make_event(1, 1);
        let result = store.append_events(WorldEpoch(2), &[entry]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("epoch mismatch"));

        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);
    }

    #[test]
    fn test_fs_event_store_snapshot() {
        let log_path = temp_path("snap");
        let snap_path = temp_path("snap-snap");
        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);

        let mut store = FsEventStore::open(&log_path, &snap_path).expect("open");
        assert_eq!(store.latest_snapshot(), None);

        let snap = Snapshot {
            epoch: WorldEpoch(5),
            world_digest: [0xab; 32],
            entity_count: 10,
            component_count: 42,
            created_at: Timestamp(1_700_000_000_000_000_000),
        };
        store.store_snapshot(snap.clone()).expect("store snapshot");
        assert_eq!(store.latest_snapshot(), Some(snap));

        // Re-open and verify snapshot persisted
        drop(store);
        let store = FsEventStore::open(&log_path, &snap_path).expect("reopen");
        let loaded = store.latest_snapshot().expect("should have snapshot");
        assert_eq!(loaded.epoch, WorldEpoch(5));
        assert_eq!(loaded.world_digest, [0xab; 32]);

        let _ = std::fs::remove_file(&log_path);
        let _ = std::fs::remove_file(&snap_path);
    }
}
