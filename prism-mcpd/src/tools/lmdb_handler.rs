use anyhow::Result;
use parking_lot::Mutex;
use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

/// LMDB memory store MCP handler.
pub struct LmdbHandler {
    store: Arc<Mutex<prism_memory_store::LmdbMemoryStore>>,
}

impl LmdbHandler {
    pub fn new(path: PathBuf, map_size: usize) -> Result<Self> {
        let store = prism_memory_store::LmdbMemoryStore::open(path, map_size)
            .map_err(|e| anyhow::anyhow!("open LMDB: {e}"))?;
        Ok(Self {
            store: Arc::new(Mutex::new(store)),
        })
    }
}

impl McpHandler for LmdbHandler {
    fn name(&self) -> &'static str {
        "lmdb"
    }

    fn description(&self) -> &'static str {
        "LMDB-backed memory store. Commands: put, get, query, delete"
    }

    fn input_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["put", "get", "query", "delete"]
                },
                "key": { "type": "string", "description": "Storage key" },
                "value": { "type": "string", "description": "Value to store (for put)" },
                "prefix": { "type": "string", "description": "Key prefix to query" }
            },
            "required": ["command"]
        })
    }

    fn call(
        &self,
        request: ToolRequest<'_>,
        _context: &RequestContext,
        _state: &DaemonState,
    ) -> Result<ToolResult> {
        let command = request.args["command"].as_str().unwrap_or("");
        let store = self.store.lock();

        match command {
            "put" => {
                let key_str = request.args["key"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("key required"))?;
                let value_str = request.args["value"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("value required"))?;
                let key = key_str.as_bytes().to_vec();
                let value = value_str.as_bytes().to_vec();
                store
                    .put(&key, &value)
                    .map_err(|e| anyhow::anyhow!("put: {e}"))?;
                Ok(ToolResult::text(
                    json!({"ok": true, "key": key_str}).to_string(),
                ))
            }
            "get" => {
                let key_str = request.args["key"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("key required"))?;
                let key = key_str.as_bytes().to_vec();
                match store.get(&key) {
                    Ok(data) => {
                        let value = String::from_utf8_lossy(&data).to_string();
                        Ok(ToolResult::text(
                            json!({"ok": true, "key": key_str, "value": value}).to_string(),
                        ))
                    }
                    Err(prism_memory_store::MemoryStoreError::KeyNotFound(_)) => {
                        Ok(ToolResult::text(
                            json!({"ok": false, "key": key_str, "error": "not_found"}).to_string(),
                        ))
                    }
                    Err(e) => Err(anyhow::anyhow!("get: {e}")),
                }
            }
            "query" => {
                let prefix = request.args["prefix"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("prefix required"))?;
                let results = store
                    .query_prefix(prefix.as_bytes())
                    .map_err(|e| anyhow::anyhow!("query: {e}"))?;
                let entries: Vec<Value> = results
                    .into_iter()
                    .map(|(k, v)| json!({"key": k, "value": String::from_utf8_lossy(&v)}))
                    .collect();
                Ok(ToolResult::text(
                    json!({"ok": true, "entries": entries}).to_string(),
                ))
            }
            "delete" => {
                let key_str = request.args["key"]
                    .as_str()
                    .ok_or_else(|| anyhow::anyhow!("key required"))?;
                let key = key_str.as_bytes().to_vec();
                store
                    .delete(&key)
                    .map_err(|e| anyhow::anyhow!("delete: {e}"))?;
                Ok(ToolResult::text(
                    json!({"ok": true, "key": key_str}).to_string(),
                ))
            }
            _ => Err(anyhow::anyhow!(
                "Unknown command: {command}. Use put, get, query, or delete"
            )),
        }
    }
}
