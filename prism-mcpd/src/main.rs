mod backends;
mod daemon;
mod dashboard;
mod proxy;
mod tools;
mod trifecta_store;

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::time::{Duration, Instant};

const HEALTH_TIMEOUT: Duration = Duration::from_millis(750);
const START_TIMEOUT: Duration = Duration::from_secs(5);

fn default_state_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    format!("{}/.local/state/prism-mcpd", home)
}

fn effective_state_dir() -> String {
    if std::env::var("PRISM_MCPD_TEST_ISOLATION").as_deref() == Ok("1") {
        return std::env::var("PRISM_MCPD_STATE_DIR").unwrap_or_else(|_| default_state_dir());
    }
    default_state_dir()
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let is_daemon = args.iter().any(|a| a == "--daemon");

    // Production MCPD is deliberately a single persistent per-user daemon.
    // Arbitrary client state directories previously created one daemon per
    // agent and defeated shared supervision and caching. Integration tests
    // opt into isolated state explicitly.
    let state_dir = effective_state_dir();

    let artifact_dir = std::env::var("PRISM_MCPD_ARTIFACT_DIR")
        .unwrap_or_else(|_| format!("{}/artifacts", state_dir));

    if is_daemon {
        // Initialize logging
        prism_mcp_core::init_logging();
        eprintln!(
            "prism-mcpd: starting daemon (state={}, artifacts={})",
            state_dir, artifact_dir
        );
        daemon::run_daemon(&state_dir, &artifact_dir)?;
    } else {
        // Proxy mode
        let socket_path = format!("{}/mcpd.sock", state_dir);
        ensure_daemon(&state_dir, &artifact_dir, &socket_path)?;
        proxy::run_proxy(&socket_path)?;
    }

    Ok(())
}

fn ensure_daemon(state_dir: &str, artifact_dir: &str, socket_path: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(state_dir)?;
    let startup_lock_path = Path::new(state_dir).join("startup.lock");
    let startup_lock = prism_mcp_core::FileLock::new(&startup_lock_path);
    let _startup_guard = startup_lock.lock()?;

    // Another proxy may have repaired the daemon while this process waited.
    if daemon_is_healthy(socket_path) {
        return Ok(());
    }

    eprintln!("prism-mcpd: health probe failed; replacing daemon for {socket_path}");
    terminate_recorded_daemon(state_dir);
    let _ = std::fs::remove_file(socket_path);
    start_daemon(state_dir, artifact_dir)?;

    let deadline = Instant::now() + START_TIMEOUT;
    while Instant::now() < deadline {
        if daemon_is_healthy(socket_path) {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    anyhow::bail!("prism-mcpd did not become healthy within {START_TIMEOUT:?}")
}

fn start_daemon(state_dir: &str, artifact_dir: &str) -> anyhow::Result<()> {
    let current_exe = std::env::current_exe().ok();
    let installed_exe =
        std::env::var_os("HOME").map(|home| Path::new(&home).join(".local/bin/prism-mcpd"));
    let use_launchd = Path::new(state_dir).join("supervised").exists()
        && current_exe
            .as_ref()
            .zip(installed_exe.as_ref())
            .is_none_or(|(current, installed)| current == installed);
    if use_launchd {
        let domain = format!("gui/{}", unsafe { libc::getuid() });
        let service = format!("{domain}/com.prism.engine.mcpd");
        let status = std::process::Command::new("launchctl")
            .args(["kickstart", "-k", &service])
            .status()?;
        if !status.success() {
            anyhow::bail!("launchctl failed to start {service}");
        }
        return Ok(());
    }
    spawn_daemon(state_dir, artifact_dir)
}

fn daemon_is_healthy(socket_path: &str) -> bool {
    let Ok(mut stream) = UnixStream::connect(socket_path) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(HEALTH_TIMEOUT));
    let _ = stream.set_write_timeout(Some(HEALTH_TIMEOUT));
    if writeln!(
        stream,
        r#"{{"jsonrpc":"2.0","id":"health","method":"prism/health"}}"#
    )
    .and_then(|_| stream.flush())
    .is_err()
    {
        return false;
    }
    let mut response = String::new();
    if BufReader::new(stream).read_line(&mut response).is_err() {
        return false;
    }
    serde_json::from_str::<serde_json::Value>(&response)
        .ok()
        .is_some_and(|value| health_response_is_compatible(&value))
}

fn health_response_is_compatible(value: &serde_json::Value) -> bool {
    let protocol_is_compatible = value["id"] == "health"
        && value["result"]["protocol"] == 1
        && value["result"]["status"] == "healthy";
    if !protocol_is_compatible {
        return false;
    }

    // A proxy and the persistent daemon may legitimately come from different
    // build profiles or rolling deployments. Protocol compatibility is the
    // default contract; strict build identity remains available for diagnostics.
    std::env::var("PRISM_MCPD_REQUIRE_BUILD_ID").as_deref() != Ok("1")
        || value["result"]["build_id"] == env!("PRISM_MCPD_BUILD_ID")
}

fn terminate_recorded_daemon(state_dir: &str) {
    let pid_path = Path::new(state_dir).join("mcpd.pid");
    let Ok(pid) = std::fs::read_to_string(&pid_path) else {
        return;
    };
    let Ok(pid) = pid.trim().parse::<i32>() else {
        return;
    };
    if pid <= 1 || !recorded_process_is_prism_mcpd(pid) {
        return;
    }
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if unsafe { libc::kill(pid, 0) } != 0 {
            return;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    unsafe {
        libc::kill(pid, libc::SIGKILL);
    }
}

fn recorded_process_is_prism_mcpd(pid: i32) -> bool {
    std::process::Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "command="])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| String::from_utf8_lossy(&output.stdout).contains("prism-mcpd"))
}

fn spawn_daemon(state_dir: &str, artifact_dir: &str) -> anyhow::Result<()> {
    let exe = std::env::current_exe().unwrap_or_else(|_| "prism-mcpd".into());
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--daemon");
    cmd.env("PRISM_MCPD_STATE_DIR", state_dir);
    cmd.env("PRISM_MCPD_ARTIFACT_DIR", artifact_dir);
    if std::env::var("PRISM_MCPD_TEST_ISOLATION").as_deref() == Ok("1") {
        cmd.env("PRISM_MCPD_TEST_ISOLATION", "1");
    }
    std::fs::create_dir_all(state_dir)?;
    let log_path = Path::new(state_dir).join("daemon.log");
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)?;
    let log_err = log.try_clone()?;
    // Detach — don't wait, but retain durable diagnostics for startup/recovery.
    cmd.stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(log_err))
        .process_group(0)
        .spawn()?;
    Ok(())
}
