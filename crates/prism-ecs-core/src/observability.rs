use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

static GLOBAL_CONTEXT: std::sync::OnceLock<TraceContext> = std::sync::OnceLock::new();
static GLOBAL_STREAM: std::sync::OnceLock<StateStream> = std::sync::OnceLock::new();

pub fn global_context() -> TraceContext {
    GLOBAL_CONTEXT.get_or_init(TraceContext::new).clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub request_id: String,
    pub parent_id: Option<String>,
}

impl TraceContext {
    pub fn new() -> Self {
        Self {
            trace_id: Uuid::new_v4().to_string(),
            request_id: Uuid::new_v4().to_string(),
            parent_id: None,
        }
    }
    pub fn child(&self) -> Self {
        Self {
            trace_id: self.trace_id.clone(),
            request_id: Uuid::new_v4().to_string(),
            parent_id: Some(self.request_id.clone()),
        }
    }
}

impl Default for TraceContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateRecord {
    pub sequence: u64,
    pub timestamp_ms: u128,
    pub context: TraceContext,
    pub entity: Option<String>,
    pub domain: String,
    pub phase: String,
    pub kind: String,
    pub status: String,
    pub state: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateSnapshot {
    pub trace_id: String,
    pub sequence: u64,
    pub status: String,
    pub domains: BTreeMap<String, String>,
    pub state: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Default)]
pub struct StateStream {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    next: u64,
    snapshot: StateSnapshot,
    subscribers: Vec<mpsc::Sender<StateRecord>>,
    writer: Option<std::fs::File>,
}

impl StateStream {
    pub fn global() -> Self {
        GLOBAL_STREAM
            .get_or_init(|| {
                let context = GLOBAL_CONTEXT.get_or_init(TraceContext::new);
                StateStream::new(context)
            })
            .clone()
    }

    pub fn new(context: &TraceContext) -> Self {
        let mut inner = Inner::default();
        inner.snapshot.trace_id = context.trace_id.clone();
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }
    pub fn with_jsonl(path: &Path, context: &TraceContext) -> Result<Self, String> {
        let stream = Self::new(context);
        let writer = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("open ECS state stream: {error}"))?;
        stream
            .inner
            .lock()
            .map_err(|_| "state stream lock poisoned".to_string())?
            .writer = Some(writer);
        Ok(stream)
    }
    pub fn subscribe(&self) -> mpsc::Receiver<StateRecord> {
        let (tx, rx) = mpsc::channel();
        if let Ok(mut inner) = self.inner.lock() {
            inner.subscribers.push(tx);
        }
        rx
    }
    pub fn publish(&self, mut record: StateRecord) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.next += 1;
            record.sequence = inner.next;
            inner.snapshot.sequence = record.sequence;
            inner.snapshot.status = record.status.clone();
            inner
                .snapshot
                .domains
                .insert(record.domain.clone(), record.phase.clone());
            inner.snapshot.state.extend(record.state.clone());
            if let Some(writer) = inner.writer.as_mut() {
                if let Ok(bytes) = serde_json::to_vec(&record) {
                    let _ = writer.write_all(&bytes);
                    let _ = writer.write_all(b"\n");
                    let _ = writer.flush();
                }
            }
            inner
                .subscribers
                .retain(|tx| tx.send(record.clone()).is_ok());
        }
    }
    pub fn snapshot(&self) -> StateSnapshot {
        self.inner
            .lock()
            .map(|inner| inner.snapshot.clone())
            .unwrap_or_default()
    }
    pub fn emit(
        &self,
        context: &TraceContext,
        domain: impl Into<String>,
        phase: impl Into<String>,
        kind: impl Into<String>,
        status: impl Into<String>,
        state: BTreeMap<String, serde_json::Value>,
    ) {
        self.publish(StateRecord {
            sequence: 0,
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or_default(),
            context: context.clone(),
            entity: None,
            domain: domain.into(),
            phase: phase.into(),
            kind: kind.into(),
            status: status.into(),
            state,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_preserves_correlation_and_snapshot() {
        let context = TraceContext::new();
        let stream = StateStream::new(&context);
        let receiver = stream.subscribe();
        stream.emit(
            &context,
            "runtime",
            "dispatch",
            "started",
            "running",
            BTreeMap::new(),
        );
        let record = receiver.recv().expect("state record");
        assert_eq!(record.context.trace_id, context.trace_id);
        assert_eq!(record.sequence, 1);
        assert_eq!(
            stream.snapshot().domains.get("runtime"),
            Some(&"dispatch".to_string())
        );
    }
}
