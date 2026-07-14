use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn wait_for_exit(child: &mut std::process::Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    false
}

#[test]
fn production_profile_requires_trifecta_configuration() {
    let state = tempfile::tempdir().expect("state directory");
    let artifacts = tempfile::tempdir().expect("artifact directory");
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .arg("--daemon")
        .env_remove("PRISM_MCPD_STORAGE")
        .env_remove("PRISM_MCPD_POSTGRES_URL")
        .env_remove("PRISM_MCPD_VALKEY_URL")
        .env_remove("PRISM_MCPD_DUCKDB_PATH")
        .env("PRISM_MCPD_STATE_DIR", state.path())
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start production-profile daemon");
    assert!(wait_for_exit(&mut child, Duration::from_secs(3)));
    assert_ne!(child.wait().unwrap().code(), Some(0));
}

fn read_health(socket: &std::path::Path) -> serde_json::Value {
    let mut stream = UnixStream::connect(socket).expect("connect health probe");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    writeln!(
        stream,
        r#"{{"jsonrpc":"2.0","id":"health","method":"prism/health"}}"#
    )
    .unwrap();
    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line).unwrap();
    serde_json::from_str(&line).expect("health response JSON")
}

#[test]
fn initializes_lists_tools_and_exits_after_stdin_closes() {
    let state = tempfile::tempdir().expect("state directory");
    let artifacts = tempfile::tempdir().expect("artifact directory");
    let mut child = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
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
    let tool_names: std::collections::HashSet<&str> = lines[1]["result"]["tools"]
        .as_array().unwrap().iter().filter_map(|tool| tool["name"].as_str()).collect();
    for name in [
        "agent_session_start", "agent_session_heartbeat", "agent_session_close",
        "agent_work_create", "agent_work_list", "agent_work_claim", "agent_work_release",
        "agent_work_handoff", "agent_path_lock", "agent_path_unlock",
        "agent_coordination_event", "agent_coordination_status", "agent_coordination_recover",
    ] { assert!(tool_names.contains(name), "missing native coordination tool {name}"); }
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

    if wait_for_exit(&mut child, Duration::from_secs(3)) {
        return;
    }
    let _ = child.kill();
    panic!("proxy did not exit after stdin closed");
}

#[test]
fn daemon_removes_runtime_files_after_sigterm() {
    let state = tempfile::tempdir().expect("state directory");
    let artifacts = tempfile::tempdir().expect("artifact directory");
    let mut daemon = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
        .arg("--daemon")
        .env("PRISM_MCPD_STATE_DIR", state.path())
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start daemon");

    let socket = state.path().join("mcpd.sock");
    let pid = state.path().join("mcpd.pid");
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && (!socket.exists() || !pid.exists()) {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(socket.exists(), "daemon socket was not created");
    assert!(pid.exists(), "daemon PID file was not created");

    let health = read_health(&socket);
    assert_eq!(health["result"]["status"], "healthy");
    assert_eq!(health["result"]["database_ok"], true);
    assert_eq!(health["result"]["artifact_database_ok"], true);
    assert_eq!(health["result"]["artifacts_ok"], true);
    assert_eq!(health["result"]["scheduler_ok"], true);
    assert!(health["result"]["queue_depth"].is_number());
    assert!(health["result"]["connections"].is_number());

    for _ in 0..25 {
        let sample = read_health(&socket);
        assert_eq!(sample["result"]["status"], "healthy");
        assert_eq!(sample["result"]["scheduler_ok"], true);
    }

    unsafe {
        libc::kill(daemon.id() as i32, libc::SIGTERM);
    }
    assert!(
        wait_for_exit(&mut daemon, Duration::from_secs(3)),
        "daemon did not exit after SIGTERM"
    );
    assert!(!socket.exists(), "daemon socket survived graceful shutdown");
    assert!(!pid.exists(), "daemon PID file survived graceful shutdown");
}

#[test]
fn same_state_directory_allows_only_one_daemon_owner() {
    let state = tempfile::tempdir().expect("state directory");
    let artifacts = tempfile::tempdir().expect("artifact directory");
    let mut first = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
        .arg("--daemon")
        .env("PRISM_MCPD_STATE_DIR", state.path())
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start first daemon");
    let socket = state.path().join("mcpd.sock");
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && !socket.exists() {
        std::thread::sleep(Duration::from_millis(25));
    }
    assert!(socket.exists(), "first daemon did not create its socket");

    let mut second = Command::new(env!("CARGO_BIN_EXE_prism-mcpd"))
        .env("PRISM_MCPD_STORAGE", "sqlite")
        .arg("--daemon")
        .env("PRISM_MCPD_STATE_DIR", state.path())
        .env("PRISM_MCPD_ARTIFACT_DIR", artifacts.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start second daemon");
    assert!(wait_for_exit(&mut second, Duration::from_secs(3)));
    assert_ne!(second.wait().unwrap().code(), Some(0));

    unsafe {
        libc::kill(first.id() as i32, libc::SIGTERM);
    }
    assert!(wait_for_exit(&mut first, Duration::from_secs(3)));
}

#[test]
fn concurrent_proxies_converge_on_one_daemon() {
    let state = tempfile::tempdir().expect("state directory");
    let artifacts = tempfile::tempdir().expect("artifact directory");
    let binary = env!("CARGO_BIN_EXE_prism-mcpd").to_string();
    let state_path = state.path().to_owned();
    let artifact_path = artifacts.path().to_owned();

    let clients: Vec<_> = (0..8)
        .map(|_| {
            let binary = binary.clone();
            let state_path = state_path.clone();
            let artifact_path = artifact_path.clone();
            std::thread::spawn(move || {
                let mut child = Command::new(binary)
                    .env("PRISM_MCPD_STORAGE", "sqlite")
                    .env("PRISM_MCPD_STATE_DIR", state_path)
                    .env("PRISM_MCPD_ARTIFACT_DIR", artifact_path)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::inherit())
                    .spawn()
                    .expect("start concurrent proxy");
                let mut stdin = child.stdin.take().unwrap();
                writeln!(stdin, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-03-26"}}}}"#).unwrap();
                let mut line = String::new();
                BufReader::new(child.stdout.take().unwrap())
                    .read_line(&mut line)
                    .unwrap();
                drop(stdin);
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&line).unwrap()["result"]
                        ["serverInfo"]["name"],
                    "prism-mcpd"
                );
                assert!(wait_for_exit(&mut child, Duration::from_secs(3)));
            })
        })
        .collect();

    for client in clients {
        client.join().expect("concurrent proxy thread");
    }

    let pid_path = state.path().join("mcpd.pid");
    let pid: i32 = std::fs::read_to_string(&pid_path)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(
        read_health(&state.path().join("mcpd.sock"))["result"]["pid"],
        pid
    );
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
}
