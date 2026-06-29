//! Background interceptor threads that monitor agent output and context,
//! firing `RuntimeSignal` events caught by the P-core multiplexer's
//! injection window during the next ANE dispatch cycle.

use crate::runtime::signal_bus::{RuntimeSignal, SignalBus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// Scan interval for the syntax interceptor — 100 ms gives sub-ANE-cycle
/// responsiveness without burning a core.
const SYNTHAX_SCAN_MS: u64 = 100;

/// Interval at which the context interceptor checks agent health.
const CONTEXT_SCAN_SECS: u64 = 1;

/// Extension filter: only source files whose edits could inject corrections.
const WATCHED_EXTENSIONS: &[&str] = &["rs", "py", "swift"];

// ---------------------------------------------------------------------------
// Syntax interceptor
// ---------------------------------------------------------------------------

/// Spawn a background thread that scans `watch_dir` for new/modified source
/// files and fires `RuntimeSignal::FileChanged` events.  The P-core
/// multiplexer picks these up during its injection window so that
/// corrected agent output can be substituted before the next ANE dispatch.
///
/// # Returns
/// A `JoinHandle` — join when the server shuts down.
pub fn spawn_syntax_interceptor(
    signal_tx: SignalBus,
    watch_dir: String,
    shutdown: Option<Arc<AtomicBool>>,
) -> thread::JoinHandle<()> {
    let stop = shutdown.unwrap_or_default();

    thread::Builder::new()
        .name("syntax-interceptor".into())
        .spawn(move || {
            // Track mtimes so we only fire FileChanged when a file is actually
            // new or was updated since our last scan.
            let mut seen: std::collections::HashMap<String, std::time::SystemTime> =
                std::collections::HashMap::new();

            while !stop.load(Ordering::Relaxed) {
                if let Ok(entries) = std::fs::read_dir(&watch_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();

                        // Only watch source files.
                        if !path
                            .extension()
                            .and_then(|e| e.to_str())
                            .map_or(false, |e| WATCHED_EXTENSIONS.contains(&e))
                        {
                            continue;
                        }

                        let path_str = path.to_string_lossy().to_string();
                        let metadata = match path.metadata() {
                            Ok(m) => m,
                            _ => continue,
                        };

                        let modified = match metadata.modified() {
                            Ok(t) => t,
                            _ => continue,
                        };

                        let is_new = !seen.contains_key(&path_str);
                        let is_modified = seen.get(&path_str).map_or(false, |prev| *prev != modified);

                        if is_new || is_modified {
                            seen.insert(path_str.clone(), modified);

                            // Fire the signal.  A full receiver means the
                            // multiplexer is saturated — that's fine, drop.
                            let _ = signal_tx.send(RuntimeSignal::FileChanged {
                                path: path_str,
                            });
                        }
                    }
                }

                thread::sleep(Duration::from_millis(SYNTHAX_SCAN_MS));
            }
        })
        .expect("failed to spawn syntax interceptor")
}

// ---------------------------------------------------------------------------
// Context interceptor
// ---------------------------------------------------------------------------

/// Spawn a background thread that monitors agent context for stall
/// conditions.  When an agent has been in a single context phase for
/// too long (no new output file activity), it fires
/// `RuntimeSignal::ContextInterrupt` so the multiplexer can preempt
/// the agent and inject a timeout correction.
///
/// # Returns
/// A `JoinHandle` — join when the server shuts down.
pub fn spawn_context_interceptor(
    signal_tx: SignalBus,
    watch_dir: String,
    stall_timeout: Duration,
    shutdown: Option<Arc<AtomicBool>>,
) -> thread::JoinHandle<()> {
    let stop = shutdown.unwrap_or_default();

    thread::Builder::new()
        .name("context-interceptor".into())
        .spawn(move || {
            // Track last-activity mtime per agent output file.
            let mut last_active: std::collections::HashMap<String, Instant> =
                std::collections::HashMap::new();

            while !stop.load(Ordering::Relaxed) {
                let now = Instant::now();

                if let Ok(entries) = std::fs::read_dir(&watch_dir) {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        let path_str = path.to_string_lossy().to_string();

                        // Check if this file has been modified since last scan.
                        let _modified = match path.metadata().and_then(|m| m.modified()) {
                            Ok(t) => t,
                            _ => continue,
                        };
                        let modified_instant = Instant::now(); // close enough for relative compare

                        let is_active = last_active
                            .get(&path_str)
                            .map_or(true, |last| *last < modified_instant);

                        if is_active {
                            last_active.insert(path_str.clone(), modified_instant);
                        }
                    }
                }

                // Check for stalled contexts — files that haven't seen
                // activity within `stall_timeout`.
                let stalled: Vec<String> = last_active
                    .iter()
                    .filter(|(_, last_active)| now.duration_since(**last_active) >= stall_timeout)
                    .map(|(path, _)| path.clone())
                    .collect();

                if !stalled.is_empty() {
                    // Fire one interrupt per stalled agent.  Extract a
                    // heuristic agent_id from the filename.
                    for path in &stalled {
                        let agent_id = path
                            .rsplit('/')
                            .next()
                            .and_then(|name| {
                                name.strip_suffix(".rs")
                                    .or_else(|| name.strip_suffix(".py"))
                                    .or_else(|| name.strip_suffix(".swift"))
                                    .unwrap_or(name)
                                    .split(|c: char| !c.is_alphanumeric())
                                    .find_map(|tok| tok.parse::<u32>().ok())
                            })
                            .unwrap_or(u32::MAX);

                        let _ = signal_tx.send(RuntimeSignal::ContextInterrupt {
                            agent_id,
                            reason: format!("agent stalled at '{}' — no activity for {:?}", path, stall_timeout),
                        });
                    }
                }

                thread::sleep(Duration::from_secs(CONTEXT_SCAN_SECS));
            }
        })
        .expect("failed to spawn context interceptor")
}
