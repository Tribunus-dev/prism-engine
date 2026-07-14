#![cfg(feature = "live-browser")]

use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

struct DaemonCleanup(std::path::PathBuf);

impl Drop for DaemonCleanup {
    fn drop(&mut self) {
        if let Ok(pid) = std::fs::read_to_string(self.0.join("mcpd.pid")) {
            let _ = Command::new("kill").args(["-TERM", pid.trim()]).status();
        }
    }
}

fn call(state: &std::path::Path, artifacts: &std::path::Path, name: &str, args: Value) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
        .env("PRISM_MCPD_STATE_DIR", state)
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts)
        .env("PRISM_BROWSER_ALLOWED_HOSTS", "example.com")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"live-browser","version":"1"}}})).unwrap();
    writeln!(
        input,
        "{}",
        json!({"jsonrpc":"2.0","method":"notifications/initialized"})
    )
    .unwrap();
    writeln!(input, "{}", json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":args}})).unwrap();
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "browser proxy failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .find(|v| v["id"] == 2)
        .unwrap();
    assert_eq!(
        response["result"]["isError"], false,
        "browser operation failed: {response}"
    );
    response["result"]["structuredContent"].clone()
}

fn call_sequence(
    state: &std::path::Path,
    artifacts: &std::path::Path,
    calls: &[(&str, Value)],
) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
        .env("PRISM_MCPD_STATE_DIR", state)
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts)
        .env("PRISM_BROWSER_ALLOWED_HOSTS", "example.com")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"live-dom","version":"1"}}})).unwrap();
    writeln!(
        input,
        "{}",
        json!({"jsonrpc":"2.0","method":"notifications/initialized"})
    )
    .unwrap();
    for (id, (name, args)) in calls.iter().enumerate() {
        writeln!(input, "{}", json!({"jsonrpc":"2.0","id":id + 2,"method":"tools/call","params":{"name":name,"arguments":args}})).unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "browser sequence failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut responses = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|v| v.get("id").and_then(Value::as_u64).unwrap_or_default() >= 2)
        .collect::<Vec<_>>();
    responses.sort_by_key(|v| v["id"].as_u64().unwrap_or_default());
    responses.into_iter().map(|v| v["result"].clone()).collect()
}

#[test]
fn safari_browser_production_gate() {
    assert!(std::process::Command::new("safaridriver")
        .arg("--version")
        .status()
        .unwrap()
        .success());
    let state = tempfile::tempdir().unwrap();
    let artifacts = tempfile::tempdir().unwrap();
    let _cleanup = DaemonCleanup(state.path().to_owned());
    let owner = "live-browser";
    let nav = call(
        state.path(),
        artifacts.path(),
        "browser_navigate",
        json!({"url":"https://example.com","session_owner":owner}),
    );
    assert_eq!(nav["status"], "navigated");
    assert_eq!(
        call(
            state.path(),
            artifacts.path(),
            "browser_validate_js",
            json!({"code":"const answer = 40 + 2;","session_owner":owner})
        )["valid"],
        true
    );
    let dom = call(
        state.path(),
        artifacts.path(),
        "browser_structured_extract",
        json!({"session_owner":owner}),
    );
    assert_eq!(dom["title"], "Example Domain");
    assert!(dom["text"]
        .as_str()
        .unwrap_or_default()
        .contains("documentation examples"));
    let element = call(
        state.path(),
        artifacts.path(),
        "browser_find_element",
        json!({"selector":"a","session_owner":owner}),
    );
    assert!(
        element.get("element-6066-11e4-a52e-4f735466cecf").is_some()
            || element.get("ELEMENT").is_some()
    );
    let tabs = call(
        state.path(),
        artifacts.path(),
        "browser_get_tabs",
        json!({"session_owner":owner}),
    );
    assert!(tabs.as_array().map(|v| !v.is_empty()).unwrap_or(false));
    let screenshot = call(
        state.path(),
        artifacts.path(),
        "browser_screenshot",
        json!({"session_owner":owner}),
    );
    assert!(screenshot["base64_png"].as_str().unwrap_or_default().len() > 100);
}

#[test]
fn safari_dom_revision_and_typed_handle_gate() {
    assert!(Command::new("safaridriver")
        .arg("--version")
        .status()
        .unwrap()
        .success());
    let state = tempfile::tempdir().unwrap();
    let artifacts = tempfile::tempdir().unwrap();
    let _cleanup = DaemonCleanup(state.path().to_owned());
    let owner = "live-dom";
    let node = json!({"id":{"tab":"current","revision":1,"ordinal":0},"tag":"a","role":"link","name":"More information...","text":"More information...","selector":"a","visible":true,"enabled":true,"x":0.0,"y":0.0,"width":100.0,"height":20.0});
    let results = call_sequence(
        state.path(),
        artifacts.path(),
        &[
            (
                "browser_navigate",
                json!({"url":"https://example.com","session_owner":owner}),
            ),
            ("dom_snapshot", json!({"session_owner":owner})),
            ("dom_query", json!({"role":"link","session_owner":owner})),
            ("dom_click", json!({"node":node,"session_owner":owner})),
            ("dom_click", json!({"node":node,"session_owner":owner})),
        ],
    );
    assert_eq!(results[0]["isError"], false);
    assert_eq!(results[1]["isError"], false);
    assert_eq!(results[2]["isError"], false);
    assert_eq!(results[2]["structuredContent"]["nodes"][0]["role"], "link");
    assert_eq!(results[3]["isError"], false);
    assert_eq!(results[4]["isError"], true);
}
