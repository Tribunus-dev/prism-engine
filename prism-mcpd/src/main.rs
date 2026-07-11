mod daemon;
mod proxy;
mod tools;

fn default_state_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    format!("{}/.local/state/prism-mcpd", home)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let is_daemon = args.iter().any(|a| a == "--daemon");

    let state_dir = std::env::var("PRISM_MCPD_STATE_DIR").unwrap_or_else(|_| default_state_dir());

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
        // If daemon not running, spawn one
        if std::path::Path::new(&socket_path).exists() {
            // Try connecting
            if let Err(_) = std::os::unix::net::UnixStream::connect(&socket_path) {
                // Stale socket — spawn daemon
                spawn_daemon(&state_dir, &artifact_dir);
            }
        } else {
            spawn_daemon(&state_dir, &artifact_dir);
        }
        // Wait briefly for daemon to be ready
        for _ in 0..20 {
            if std::os::unix::net::UnixStream::connect(&socket_path).is_ok() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        proxy::run_proxy(&socket_path)?;
    }

    Ok(())
}

fn spawn_daemon(state_dir: &str, artifact_dir: &str) {
    let exe = std::env::current_exe().unwrap_or_else(|_| "prism-mcpd".into());
    let mut cmd = std::process::Command::new(&exe);
    cmd.arg("--daemon");
    cmd.env("PRISM_MCPD_STATE_DIR", state_dir);
    cmd.env("PRISM_MCPD_ARTIFACT_DIR", artifact_dir);
    // Detach — don't wait
    let _ = cmd.spawn();
}
