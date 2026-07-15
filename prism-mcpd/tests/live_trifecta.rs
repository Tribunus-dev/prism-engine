#![cfg(feature = "live-trifecta")]

use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};
use tempfile::TempDir;

struct Gate {
    state: TempDir,
    artifacts: TempDir,
    duckdb: TempDir,
    postgres: String,
    valkey: String,
}

impl Gate {
    fn new() -> Self {
        let postgres = std::env::var("PRISM_MCPD_TEST_POSTGRES_URL")
            .expect("PRISM_MCPD_TEST_POSTGRES_URL is required for live-trifecta");
        let valkey = std::env::var("PRISM_MCPD_TEST_VALKEY_URL")
            .expect("PRISM_MCPD_TEST_VALKEY_URL is required for live-trifecta");
        assert!(
            std::env::var("PRISM_MCPD_TEST_DUCKDB_PATH").is_ok(),
            "PRISM_MCPD_TEST_DUCKDB_PATH is required for live-trifecta"
        );
        Self {
            state: tempfile::tempdir().unwrap(),
            artifacts: tempfile::tempdir().unwrap(),
            duckdb: tempfile::tempdir().unwrap(),
            postgres,
            valkey,
        }
    }

    fn call(&self, name: &str, arguments: Value) -> Value {
        let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
            .env("PRISM_MCPD_STORAGE", "trifecta")
            .env("PRISM_MCPD_TEST_ISOLATION", "1")
            .env("PRISM_MCPD_POSTGRES_URL", &self.postgres)
            .env("PRISM_MCPD_VALKEY_URL", &self.valkey)
            .env(
                "PRISM_MCPD_DUCKDB_PATH",
                self.duckdb.path().join("projection.duckdb"),
            )
            .env("PRISM_MCPD_STATE_DIR", self.state.path())
            .env("PRISM_MCPD_ARTIFACT_DIR", self.artifacts.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        writeln!(stdin, "{}", json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"live-trifecta","version":"1"}}})).unwrap();
        writeln!(
            stdin,
            "{}",
            json!({"jsonrpc":"2.0","method":"notifications/initialized"})
        )
        .unwrap();
        writeln!(stdin, "{}", json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":arguments}})).unwrap();
        drop(stdin);
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "MCP proxy failed: {}\n{}",
            String::from_utf8_lossy(&output.stderr),
            std::fs::read_to_string(self.state.path().join("daemon.log")).unwrap_or_default()
        );
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| serde_json::from_str::<Value>(line).ok())
            .find(|v| v["id"] == 2)
            .expect("tool response")
    }

    fn result(&self, name: &str, args: Value) -> Value {
        let response = self.call(name, args);
        assert_eq!(
            response["result"]["isError"], false,
            "tool failed: {response}"
        );
        response["result"]["structuredContent"]["result"].clone()
    }
}

#[test]
fn distributed_coordination_production_gate() {
    let gate = Gate::new();
    let suffix = uuid::Uuid::new_v4().to_string();
    let a = format!("gate-a-{suffix}");
    let b = format!("gate-b-{suffix}");
    let work = format!("gate-work-{suffix}");
    assert_eq!(
        gate.result(
            "agent_session_start",
            json!({"session_id":&a,"agent_id":"a"})
        )["status"],
        "active"
    );
    assert_eq!(
        gate.result(
            "agent_session_start",
            json!({"session_id":&b,"agent_id":"b"})
        )["status"],
        "active"
    );
    assert_eq!(
        gate.result(
            "agent_work_create",
            json!({"work_id":&work,"work_title":"production gate"})
        )["status"],
        "queued"
    );
    let claim = gate.result("agent_work_claim", json!({"work_id":&work,"session_id":&a}));
    assert_eq!(claim["claimed"], true);
    let conflict = gate.result("agent_work_claim", json!({"work_id":&work,"session_id":&b}));
    assert_eq!(conflict["claimed"], false);
    assert_eq!(conflict["conflict_session_id"], a);
    let path = format!("/tmp/prism-live-{suffix}");
    assert_eq!(
        gate.result(
            "agent_path_lock",
            json!({"session_id":&a,"path":&path,"lock_kind":"write","ttl_seconds":60})
        )["acquired"],
        true
    );
    assert_eq!(
        gate.result(
            "agent_path_lock",
            json!({"session_id":&b,"path":&path,"lock_kind":"write","ttl_seconds":60})
        )["acquired"],
        false
    );
    assert_eq!(
        gate.result(
            "agent_work_handoff",
            json!({"work_id":&work,"from_session":&a,"to_session":&b,"context":{"gate":true}})
        )["ok"],
        true
    );
    let event = gate.result(
        "agent_coordination_event",
        json!({"session_id":&a,"event_type":"production_gate","payload":{"ok":true}}),
    );
    assert!(event["sequence"].as_i64().unwrap() > 0);
    let pid = std::fs::read_to_string(gate.state.path().join("mcpd.pid")).unwrap();
    let _ = Command::new("kill").args(["-KILL", pid.trim()]).status();
    std::thread::sleep(std::time::Duration::from_millis(250));
    let status = gate.result("agent_coordination_status", json!({}));
    assert!(status["active_sessions"].as_i64().unwrap() >= 2);
    let expired_path = format!("/tmp/prism-expire-{suffix}");
    gate.result(
        "agent_path_lock",
        json!({"session_id":&a,"path":&expired_path,"lock_kind":"write","ttl_seconds":1}),
    );
    std::thread::sleep(std::time::Duration::from_secs(2));
    assert!(
        gate.result("agent_coordination_recover", json!({}))["expired_locks"]
            .as_i64()
            .unwrap()
            >= 1
    );
    let restarted = gate.result("agent_coordination_status", json!({}));
    assert!(restarted["active_sessions"].as_i64().unwrap() >= 2);
}
