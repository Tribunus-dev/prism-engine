use std::collections::BTreeMap;
use std::sync::OnceLock;

use prism_ecs_core::{global_context, StateRecord, StateStream};

static STREAM: OnceLock<StateStream> = OnceLock::new();

fn stream() -> &'static StateStream {
    STREAM.get_or_init(StateStream::global)
}

pub fn subscribe() -> std::sync::mpsc::Receiver<StateRecord> {
    stream().subscribe()
}

pub fn publish(domain: &str, phase: &str, kind: &str, status: &str, model_path: &str) {
    stream().emit(
        &global_context(),
        domain,
        phase,
        kind,
        status,
        BTreeMap::from([(String::from("model_path"), serde_json::json!(model_path))]),
    );
}
