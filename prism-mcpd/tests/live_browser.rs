#![cfg(feature = "live-browser")]

use serde_json::{json, Value};
use std::io::Write;
use std::process::{Command, Stdio};

fn call(state: &std::path::Path, artifacts: &std::path::Path, name: &str, args: Value) -> Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
        .env("PRISM_MCPD_STATE_DIR", state)
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts)
        .env("PRISM_BROWSER_ALLOWED_HOSTS", "example.com")
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "{}", json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"live-browser","version":"1"}}})).unwrap();
    writeln!(input, "{}", json!({"jsonrpc":"2.0","method":"notifications/initialized"})).unwrap();
    writeln!(input, "{}", json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":name,"arguments":args}})).unwrap();
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "browser proxy failed: {}", String::from_utf8_lossy(&output.stderr));
    let response = String::from_utf8_lossy(&output.stdout).lines().filter_map(|line| serde_json::from_str::<Value>(line).ok()).find(|v| v["id"] == 2).unwrap();
    assert_eq!(response["result"]["isError"], false, "browser operation failed: {response}");
    response["result"]["structuredContent"].clone()
}

#[test]
fn safari_browser_production_gate() {
    assert!(std::process::Command::new("safaridriver").arg("--version").status().unwrap().success());
    let state = tempfile::tempdir().unwrap();
    let artifacts = tempfile::tempdir().unwrap();
    let owner = "live-browser";
    let nav = call(state.path(), artifacts.path(), "browser_navigate", json!({"url":"https://example.com","session_owner":owner}));
    assert_eq!(nav["status"], "navigated");
    let dom = call(state.path(), artifacts.path(), "browser_structured_extract", json!({"session_owner":owner}));
    assert_eq!(dom["title"], "Example Domain");
    assert!(dom["text"].as_str().unwrap_or_default().contains("documentation examples"));
    let element = call(state.path(), artifacts.path(), "browser_find_element", json!({"selector":"a","session_owner":owner}));
    assert!(element.get("element-6066-11e4-a52e-4f735466cecf").is_some() || element.get("ELEMENT").is_some());
    let tabs = call(state.path(), artifacts.path(), "browser_get_tabs", json!({"session_owner":owner}));
    assert!(tabs.as_array().map(|v| !v.is_empty()).unwrap_or(false));
    let screenshot = call(state.path(), artifacts.path(), "browser_screenshot", json!({"session_owner":owner}));
    assert!(screenshot["base64_png"].as_str().unwrap_or_default().len() > 100);
}
