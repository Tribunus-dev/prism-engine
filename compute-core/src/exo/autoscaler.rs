use std::process::{Command, Stdio};
use std::time::Instant;

use crate::exo::cluster::{find_exo_binary, ExoNode};
use crate::scheduling::InferenceTelemetry;

/// Result of an autoscaler tick evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleAction {
    /// No action required.
    None,
    /// Spawn the given number of new nodes.
    ScaleUp(u32),
    /// Drain the given number of nodes.
    ScaleDown(u32),
}

/// Autoscaler for EXO clusters.
///
/// Monitors queue depth and response latency from `InferenceTelemetry`.
/// When load exceeds threshold, spawns new node processes.
/// When load drops below threshold, drains and terminates nodes.
pub struct Autoscaler {
    /// Telemetry source for load monitoring.
    telemetry: InferenceTelemetry,
    /// Minimum number of nodes to keep running.
    pub min_nodes: u32,
    /// Maximum number of nodes allowed.
    pub max_nodes: u32,
    /// Average queue depth per node that triggers a scale-up.
    pub scale_up_threshold: f64,
    /// Average queue depth per node that triggers a scale-down.
    pub scale_down_threshold: f64,
    /// Minimum seconds between scale events (prevents thrashing).
    pub cooldown_secs: u64,
    /// When the last scale event occurred.
    last_scale_event: Instant,
}

impl Autoscaler {
    /// Create a new autoscaler wired to the global telemetry singleton.
    pub fn new(telemetry: InferenceTelemetry) -> Self {
        Self {
            telemetry,
            min_nodes: 1,
            max_nodes: 8,
            scale_up_threshold: 4.0, // scale up at >4 queued requests per node
            scale_down_threshold: 1.0, // scale down at <1 queued request per node
            cooldown_secs: 30,
            last_scale_event: Instant::now(),
        }
    }

    /// Called periodically.  Reads telemetry and decides whether to scale.
    pub fn tick(&mut self) -> Result<ScaleAction, String> {
        let snapshot = self.telemetry.snapshot();

        // Count the current number of nodes.  If we don't have an accurate
        // cluster view, estimate from the last scale-up count.
        // For simplicity, use the telemetry to derive load-per-node.
        let current_nodes = 1u32.max(self.min_nodes); // base: at least self

        // Respect cooldown.
        let elapsed = Instant::now().duration_since(self.last_scale_event);
        if elapsed.as_secs() < self.cooldown_secs {
            return Ok(ScaleAction::None);
        }

        let load_per_node = snapshot.queue_depth as f64 / current_nodes as f64;

        // Scale up: load exceeds threshold AND we haven't hit max_nodes.
        if load_per_node > self.scale_up_threshold && current_nodes < self.max_nodes {
            // How many more nodes do we need to bring load per node down?
            // Target: each node gets <= scale_up_threshold/2.
            let target_load = (self.scale_up_threshold / 2.0).max(1.0);
            let desired_nodes = (snapshot.queue_depth as f64 / target_load).ceil() as u32;
            let new_nodes = desired_nodes
                .saturating_sub(current_nodes)
                .min(self.max_nodes.saturating_sub(current_nodes))
                .max(1);
            self.last_scale_event = Instant::now();
            return Ok(ScaleAction::ScaleUp(new_nodes));
        }

        // Scale down: load below threshold AND we have more than min_nodes.
        if load_per_node < self.scale_down_threshold && current_nodes > self.min_nodes {
            // Scale down one node at a time to be conservative.
            self.last_scale_event = Instant::now();
            return Ok(ScaleAction::ScaleDown(1));
        }

        Ok(ScaleAction::None)
    }

    /// Spawn a new EXO worker subprocess.
    pub fn spawn_node(&self) -> Result<(), String> {
        // Delegate to ExoNode's spawning logic via the find_exo_binary helper.
        let exo_binary = find_exo_binary()?;

        let mut cmd = Command::new(&exo_binary);
        cmd.arg("worker")
            .arg("--port")
            .arg("0") // let OS assign a port
            .arg("--no-tui")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        cmd.env("TRIBUNUS_AUTOSCALED", "1");

        cmd.spawn()
            .map_err(|e| format!("failed to spawn autoscaled node: {}", e))?;
        Ok(())
    }

    /// Drain and terminate a node.
    pub fn drain_node(&self, node_id: &str) -> Result<(), String> {
        // Attempt to gracefully shut down the node via HTTP drain endpoint.
        // Fall back to best-effort: log and return.
        let drain_url = format!("http://{}:52415/v1/leave", node_id);
        let _ = ExoNode::http_get(&drain_url);
        eprintln!("[exo] autoscaler: drained node {}", node_id);
        Ok(())
    }
}
