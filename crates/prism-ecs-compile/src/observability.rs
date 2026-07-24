use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{CompilationEvent, CompilationStage, EventKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcsCorrelation {
    pub trace_id: String,
    pub request_id: String,
    pub parent_id: Option<String>,
    pub entity_id: Option<String>,
}

impl EcsCorrelation {
    pub fn new() -> Self {
        let trace = Uuid::new_v4().to_string();
        Self {
            trace_id: trace.clone(),
            request_id: Uuid::new_v4().to_string(),
            parent_id: None,
            entity_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcsStateEvent {
    pub sequence: u64,
    pub timestamp: DateTime<Utc>,
    pub correlation: EcsCorrelation,
    pub kind: String,
    pub stage: Option<CompilationStage>,
    pub event: Option<CompilationEvent>,
    pub state: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EcsStateSnapshot {
    pub trace_id: String,
    pub last_sequence: u64,
    pub phase: Option<CompilationStage>,
    pub status: String,
    pub state: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone)]
pub struct EcsStateStream {
    inner: Arc<Mutex<StreamInner>>,
    core_context: prism_ecs_core::TraceContext,
    core_stream: prism_ecs_core::StateStream,
}

#[derive(Default)]
struct StreamInner {
    next_sequence: u64,
    snapshot: EcsStateSnapshot,
    subscribers: Vec<mpsc::Sender<EcsStateEvent>>,
    writer: Option<std::fs::File>,
}

impl EcsStateStream {
    pub fn new(correlation: &EcsCorrelation) -> Self {
        let mut inner = StreamInner::default();
        inner.snapshot.trace_id = correlation.trace_id.clone();
        let core_context = prism_ecs_core::global_context();
        Self {
            inner: Arc::new(Mutex::new(inner)),
            core_stream: prism_ecs_core::StateStream::global(),
            core_context,
        }
    }

    pub fn with_jsonl(path: &Path, correlation: &EcsCorrelation) -> Result<Self, String> {
        let stream = Self::new(correlation);
        let writer = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("open ECS observability stream: {e}"))?;
        stream
            .inner
            .lock()
            .map_err(|_| "observability lock poisoned".to_string())?
            .writer = Some(writer);
        Ok(stream)
    }

    pub fn subscribe(&self) -> mpsc::Receiver<EcsStateEvent> {
        let (sender, receiver) = mpsc::channel();
        if let Ok(mut inner) = self.inner.lock() {
            inner.subscribers.push(sender);
        }
        receiver
    }

    pub fn subscribe_core(&self) -> mpsc::Receiver<prism_ecs_core::StateRecord> {
        self.core_stream.subscribe()
    }

    pub fn publish(&self, mut event: EcsStateEvent) {
        let core_state = event.state.clone();
        self.core_stream.emit(
            &self.core_context,
            "compiler",
            event
                .stage
                .map(|stage| stage.to_string())
                .unwrap_or_else(|| "unknown".into()),
            &event.kind,
            event
                .event
                .as_ref()
                .map(|value| value.status.as_str())
                .unwrap_or("ok"),
            core_state,
        );
        if let Ok(mut inner) = self.inner.lock() {
            inner.next_sequence += 1;
            event.sequence = inner.next_sequence;
            inner.snapshot.last_sequence = event.sequence;
            inner.snapshot.phase = event.stage;
            if let Some(status) = event.state.get("status").and_then(|v| v.as_str()) {
                inner.snapshot.status = status.to_string();
            }
            inner.snapshot.state.extend(event.state.clone());
            if let Some(writer) = inner.writer.as_mut() {
                if let Ok(bytes) = serde_json::to_vec(&event) {
                    let _ = writer.write_all(&bytes);
                    let _ = writer.write_all(b"\n");
                    let _ = writer.flush();
                }
            }
            inner
                .subscribers
                .retain(|subscriber| subscriber.send(event.clone()).is_ok());
        }
    }

    pub fn stage(
        &self,
        correlation: &EcsCorrelation,
        stage: CompilationStage,
        kind: EventKind,
        detail: impl Into<String>,
        state: BTreeMap<String, serde_json::Value>,
    ) {
        self.publish(EcsStateEvent {
            sequence: 0,
            timestamp: Utc::now(),
            correlation: correlation.clone(),
            kind: kind.to_string(),
            stage: Some(stage),
            event: Some(CompilationEvent {
                sequence: 0,
                timestamp: Utc::now(),
                phase: stage,
                event_type: kind,
                entity_id: correlation.entity_id.clone(),
                duration_ms: 0,
                detail: detail.into(),
                inputs: vec![],
                outputs: vec![],
                digests: vec![],
                status: "ok".into(),
            }),
            state,
        });
    }

    pub fn snapshot(&self) -> EcsStateSnapshot {
        self.inner
            .lock()
            .map(|inner| inner.snapshot.clone())
            .unwrap_or_default()
    }
}
