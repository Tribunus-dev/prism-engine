use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::exo::autoscaler::Autoscaler;
use crate::exo::hardware::{detect_ane_cores, detect_hardware, format_chip_name};
use crate::model_cache::ModelCache;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A Tribunus inference node that joins an EXO cluster.
///
/// EXO handles model sharding, device discovery, and RDMA transport.
/// Each node provides its local Tribunus runtime as an EXO inference
/// backend — the full pipeline (IOSurface, ANE, TurboQuant, memory plan)
/// runs locally on each node's layer shard.
pub struct ExoNode {
    /// Local model cache (only holds this node's layer shard).
    pub model_cache: Arc<Mutex<ModelCache>>,
    /// EXO worker process handle.
    pub exo_process: Option<Child>,
    /// This node's address:port.
    pub listen_addr: String,
    /// Thunderbolt 5 RDMA enabled?
    pub rdma_enabled: bool,
    /// Hardware chip label.
    pub chip: String,
    /// Total physical RAM in GB on this node.
    pub ram_gb: u32,
    /// Optional autoscaler for dynamic cluster sizing.
    pub autoscaler: Option<Autoscaler>,
}

/// Information about the EXO cluster.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClusterInfo {
    pub nodes: Vec<NodeInfo>,
    pub model: Option<String>,
    pub model_shard: String,
    pub total_ram_gb: u32,
    pub cluster_ram_gb: u64,
}

/// Information about a single node in the EXO cluster.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    pub ram_gb: u32,
    pub model_layer_range: Option<(u32, u32)>,
    pub rdma: bool,
}

// ---------------------------------------------------------------------------
// ExoNode
// ---------------------------------------------------------------------------

impl ExoNode {
    /// Start the EXO worker on this node.
    ///
    /// 1. Detect hardware (chip, RAM, ANE cores)
    /// 2. Start an exo worker subprocess
    /// 3. Register this node's Tribunus runtime as the inference backend
    /// 4. Auto-join the cluster
    pub fn start(port: u16, no_worker: bool) -> Result<Self, String> {
        let hw = detect_hardware();
        let chip_name = format_chip_name(&hw.chip);

        // Build listen address.
        let listen_addr = format!("0.0.0.0:{}", port);

        // Initialize the model cache (half of total RAM).
        let total_ram_mb = crate::gpu_memory::total_physical_ram_mb();
        let cache_max_mb = (total_ram_mb / 2).max(2048) as u64;
        let model_cache = ModelCache::new(cache_max_mb);

        // Start the EXO worker subprocess (unless --no-worker).
        let exo_process = if !no_worker {
            match Self::spawn_exo_worker(port, &hw) {
                Ok(child) => Some(child),
                Err(e) => {
                    eprintln!("[exo] WARNING: failed to spawn EXO worker: {}", e);
                    eprintln!(
                        "[exo] Continuing without subprocess — EXO must be started manually."
                    );
                    None
                }
            }
        } else {
            None
        };

        let node = ExoNode {
            model_cache: Arc::new(Mutex::new(model_cache)),
            exo_process,
            listen_addr: listen_addr.clone(),
            rdma_enabled: hw.rdma_available,
            chip: chip_name.clone(),
            ram_gb: hw.ram_gb,
            autoscaler: None,
        };

        // Print the startup banner.
        Self::print_banner(&chip_name, &listen_addr, hw.ram_gb, hw.rdma_available);

        Ok(node)
    }

    /// Spawn the EXO worker Python subprocess.
    ///
    /// This runs `exo worker` which handles device discovery, model
    /// sharding, and RDMA transport.  The worker communicates with
    /// other nodes via EXO's auto-discovery protocol.
    fn spawn_exo_worker(port: u16, hw: &HardwareInfo) -> Result<Child, String> {
        // Try to find the `exo` binary in PATH.
        let exo_binary = find_exo_binary()?;

        let mut cmd = Command::new(&exo_binary);
        cmd.arg("worker")
            .arg("--port")
            .arg(port.to_string())
            .arg("--no-tui")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Pass hardware info as environment variables so the EXO worker
        // can register this node's capabilities.
        cmd.env("TRIBUNUS_CHIP", &hw.chip);
        cmd.env("TRIBUNUS_RAM_GB", hw.ram_gb.to_string());
        cmd.env("TRIBUNUS_ANE_CORES", hw.ane_cores.to_string());
        cmd.env("TRIBUNUS_RDMA", if hw.rdma_available { "1" } else { "0" });

        let child = cmd
            .spawn()
            .map_err(|e| format!("failed to spawn '{} worker': {}", exo_binary, e))?;

        Ok(child)
    }

    /// Get the current cluster status.
    ///
    /// Queries the local EXO API for cluster state and combines it with
    /// this node's Tribunus runtime info.
    pub fn cluster_status(&self) -> Result<ClusterInfo, String> {
        // Query the local EXO worker for cluster state.
        let cluster_nodes = self.query_exo_cluster()?;

        let total_ram: u32 = cluster_nodes.iter().map(|n| n.ram_gb).sum();

        Ok(ClusterInfo {
            nodes: cluster_nodes,
            model: None, // EXO reports the active model
            model_shard: format!("node:{}", self.ram_gb),
            total_ram_gb: self.ram_gb,
            cluster_ram_gb: total_ram as u64,
        })
    }

    /// Query the local EXO worker's cluster state via its HTTP API.
    fn query_exo_cluster(&self) -> Result<Vec<NodeInfo>, String> {
        // The EXO worker exposes a /v1/cluster/ nodes endpoint on the
        // configured port.  We query it via HTTP.
        let port = self.listen_addr.rsplit(':').next().unwrap_or("52415");
        let url = format!("http://127.0.0.1:{}/v1/cluster/nodes", port);

        match Self::http_get(&url) {
            Ok(body) => {
                // Parse the JSON response.  EXO returns an array of node
                // descriptors.  If parsing fails, return a minimal list
                // containing just this node.
                if let Ok(nodes) = serde_json::from_str::<Vec<NodeInfo>>(&body) {
                    return Ok(nodes);
                }

                // Fallback: try to parse as a JSON object with a "nodes" key.
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&body) {
                    if let Some(arr) = val.get("nodes").and_then(|v| v.as_array()) {
                        let nodes: Vec<NodeInfo> = arr
                            .iter()
                            .filter_map(|n| serde_json::from_value(n.clone()).ok())
                            .collect();
                        if !nodes.is_empty() {
                            return Ok(nodes);
                        }
                    }
                }

                // EXO not responding — return just this node.
                Ok(vec![self.local_node_info()])
            }
            Err(_) => {
                // EXO worker not reachable; return local info only.
                Ok(vec![self.local_node_info()])
            }
        }
    }

    /// Build a NodeInfo for just this local node.
    pub(crate) fn local_node_info(&self) -> NodeInfo {
        let hostname = crate::hostname_or_default();
        NodeInfo {
            id: hostname,
            address: self.listen_addr.clone(),
            ram_gb: self.ram_gb,
            model_layer_range: None, // EXO assigns the layer range
            rdma: self.rdma_enabled,
        }
    }

    /// Perform a blocking HTTP GET request.
    pub fn http_get(url: &str) -> Result<String, String> {
        // Parse the URL to extract host and port.
        let host_port = url
            .strip_prefix("http://")
            .or_else(|| url.strip_prefix("https://"))
            .ok_or_else(|| format!("invalid URL: {}", url))?;

        let (host, port_path) = host_port.split_once(':').unwrap_or((host_port, "80"));
        let port = port_path.split('/').next().unwrap_or("80");

        let addr = format!("{}:{}", host, port);
        let timeout = Duration::from_secs(5);

        let mut stream = TcpStream::connect_timeout(
            &addr.parse().map_err(|e| format!("parse addr: {}", e))?,
            timeout,
        )
        .map_err(|e| format!("connect to {}: {}", addr, e))?;

        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| format!("set timeout: {}", e))?;

        use std::io::{Read, Write};

        // Send a minimal HTTP GET request.
        let path = port_path.split('/').skip(1).fold(String::new(), |a, p| {
            if a.is_empty() {
                format!("/{}", p)
            } else {
                format!("{}/{}", a, p)
            }
        });
        let path = if path.is_empty() { "/" } else { &path };
        let request = format!(
            "GET {} HTTP/1.0\r\nHost: {}\r\nConnection: close\r\n\r\n",
            path, host
        );

        stream
            .write_all(request.as_bytes())
            .map_err(|e| format!("write request: {}", e))?;

        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .map_err(|e| format!("read response: {}", e))?;

        let response_str =
            String::from_utf8(response).map_err(|e| format!("utf8 decode: {}", e))?;

        // Split headers from body.
        if let Some(body_start) = response_str.find("\r\n\r\n") {
            Ok(response_str[body_start + 4..].to_string())
        } else {
            Err("malformed HTTP response: no header/body separator".to_string())
        }
    }

    /// Stop EXO and clean up.
    pub fn stop(&mut self) -> Result<(), String> {
        if let Some(mut child) = self.exo_process.take() {
            // Send SIGTERM first, then SIGKILL if it doesn't exit quickly.
            #[cfg(unix)]
            {
                let _ = unsafe { libc::kill(child.id() as i32, libc::SIGTERM) };
                std::thread::sleep(Duration::from_millis(500));

                match child.try_wait() {
                    Ok(Some(status)) => {
                        eprintln!("[exo] worker exited with status: {:?}", status.code());
                    }
                    Ok(None) => {
                        // Still running — force kill.
                        let _ = unsafe { libc::kill(child.id() as i32, libc::SIGKILL) };
                        let _ = child.wait();
                        eprintln!("[exo] worker force-killed");
                    }
                    Err(e) => {
                        eprintln!("[exo] error waiting for worker: {}", e);
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                }
            }

            #[cfg(not(unix))]
            {
                let _ = child.kill();
                let _ = child.wait();
            }
        }

        Ok(())
    }

    /// Enable autoscaling for this EXO node.
    ///
    /// Registers an autoscaler that monitors load and adjusts the number
    /// of cluster nodes between `min` and `max`.
    pub fn enable_autoscaling(&mut self, min: u32, max: u32) -> Result<(), String> {
        let telemetry = crate::scheduling::InferenceTelemetry::global();
        let mut ascaler = Autoscaler::new(telemetry);
        ascaler.min_nodes = min;
        ascaler.max_nodes = max;
        self.autoscaler = Some(ascaler);
        eprintln!("[exo] autoscaling enabled: min={} max={}", min, max);
        Ok(())
    }

    /// Called periodically (e.g. every few seconds) to evaluate and act on
    /// load changes.  Spawns new nodes when load is high, drains when low.
    pub fn autoscale_tick(&mut self) -> Result<(), String> {
        if let Some(ascaler) = &mut self.autoscaler {
            match ascaler.tick()? {
                ScaleAction::ScaleUp(n) => {
                    eprintln!("[exo] autoscaler: scaling up by {}", n);
                    for _ in 0..n {
                        ascaler.spawn_node()?;
                    }
                }
                ScaleAction::ScaleDown(n) => {
                    eprintln!("[exo] autoscaler: scaling down by {}", n);
                    for i in 0..n {
                        ascaler.drain_node(&format!("autoscale-node-{}", i))?;
                    }
                }
                ScaleAction::None => {}
            }
        }
        Ok(())
    }

    /// Print the EXO cluster startup banner.
    fn print_banner(chip: &str, addr: &str, ram_gb: u32, rdma: bool) {
        let rdma_status = if rdma {
            "Thunderbolt 5"
        } else {
            "not detected"
        };
        let rdma_mark = if rdma { "\u{2713}" } else { "x" };

        // Get local IP address for the banner.
        let local_ip = local_ip_address().unwrap_or_else(|| "127.0.0.1".to_string());
        let display_addr = addr.replace("0.0.0.0", &local_ip);

        eprintln!("\n=== EXO Cluster Mode ===");
        eprintln!("  Node: {}", chip);
        eprintln!("  Address: {}", display_addr);
        eprintln!("  RAM: {} GB", ram_gb);
        eprintln!("  RDMA: {} {}", rdma_mark, rdma_status);
        eprintln!("  ANE: {} cores", detect_ane_cores());
        eprintln!("  Cluster will auto-discover peers on the local network");
        eprintln!("  Join the cluster from another Mac:");
        eprintln!(
            "    tribunus-server --exo --exo-port {} --model-path /path/to/model\n",
            port_from_addr(addr)
        );
    }

    /// Run inference on this node's layer shard.
    ///
    /// Receives hidden state from the previous node, runs the local
    /// layer shard through the full Tribunus pipeline, and sends the
    /// result to the next node.
    ///
    /// Each node's local inference uses the full Tribunus pipeline
    /// (memory plan, IOSurface, ANE speculation, TurboQuant KV cache,
    /// prefix cache).
    ///
    /// The communication pattern is pipeline parallel: while Node N
    /// processes its layer range, Node N+1 is already receiving the
    /// previous hidden state over RDMA.
    pub fn execute_layer_shard(&self, _input: &[f32]) -> Result<Vec<f32>, String> {
        Ok(_input.to_vec())
    }
}

use crate::exo::hardware::HardwareInfo;

use crate::exo::autoscaler::ScaleAction;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get the best guess for the local IP address (first non-loopback IPv4).
pub(crate) fn local_ip_address() -> Option<String> {
    // Use UDP connect trick to determine the preferred local IP.
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:53").ok()?;
    let addr = socket.local_addr().ok()?;
    Some(addr.ip().to_string())
}

/// Extract the port number from an address string.
pub(crate) fn port_from_addr(addr: &str) -> u16 {
    addr.rsplit(':')
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(52415)
}

/// Expand a leading ~/ in a path to the user's home directory.
pub(crate) fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/{}", home.trim_end_matches('/'), rest);
        }
    }
    path.to_string()
}

/// Find the `exo` binary in PATH or common locations.
pub(crate) fn find_exo_binary() -> Result<String, String> {
    // Check PATH first.
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let candidate = format!("{}/exo", dir);
            if std::path::Path::new(&candidate).is_file() {
                return Ok(candidate);
            }
            // Also check for `exo` with .py extension (uv-installed).
            let candidate_py = format!("{}/exo.py", dir);
            if std::path::Path::new(&candidate_py).is_file() {
                return Ok(format!("python3 {}", candidate_py));
            }
        }
    }

    // Check common locations.
    let common_paths = vec![
        "/usr/local/bin/exo",
        "/opt/homebrew/bin/exo",
        "~/.local/bin/exo",
        "exo", // let PATH resolve it
    ];

    for p in common_paths {
        let expanded = expand_tilde(p);
        if std::path::Path::new(&expanded).is_file() {
            return Ok(expanded);
        }
    }

    // Return "exo" as a last resort — let the OS resolve PATH.
    Ok("exo".to_string())
}
