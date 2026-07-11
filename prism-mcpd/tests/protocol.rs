use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn initializes_lists_tools_and_exits_after_stdin_closes() {
    let state = tempfile::tempdir().expect("state directory");
    let artifacts = tempfile::tempdir().expect("artifact directory");
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STATE_DIR", state.path())
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("start proxy");

    let mut stdin = child.stdin.take().expect("stdin");
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-03-26","capabilities":{{}},"clientInfo":{{"name":"test","version":"1"}}}}}}"#).unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
    )
    .unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{{}}}}"#
    )
    .unwrap();
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{{"name":"inspect_model","arguments":{{"source":"Cargo.toml"}}}}}}"#
    )
    .unwrap();
    let stdout = child.stdout.take().expect("stdout");
    let lines: Vec<serde_json::Value> = BufReader::new(stdout)
        .lines()
        .take(3)
        .map(|line| serde_json::from_str(&line.unwrap()).unwrap())
        .collect();
    drop(stdin);
    assert_eq!(lines[0]["result"]["serverInfo"]["name"], "prism-mcpd");
    assert!(lines[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "inspect_model"));
    assert!(lines[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .any(|tool| tool["name"] == "validate_model_assets"));
    let inspect = lines[1]["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "inspect_model")
        .unwrap();
    assert_eq!(
        inspect["outputSchema"]["properties"]["size_bytes"]["type"],
        "integer"
    );
    assert_eq!(lines[2]["result"]["isError"], false);
    assert!(lines[2]["result"]["structuredContent"].is_object());

    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    let _ = child.kill();
    panic!("proxy did not exit after stdin closed");
}
