use anyhow::Result;
use std::collections::HashMap;
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Run an external command with a timeout. Returns stdout as a string.
pub fn run_with_timeout(
    program: &str,
    args: &[&str],
    cwd: Option<&str>,
    timeout: Duration,
) -> Result<String> {
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn()?;
    let mut stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture {program} stdout"))?;
    let mut stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("failed to capture {program} stderr"))?;
    let stdout_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let start = Instant::now();

    loop {
        match child.try_wait()? {
            Some(status) => {
                let stdout = String::from_utf8_lossy(
                    &stdout_reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("{program} stdout reader panicked"))??,
                )
                .to_string();
                let stderr = String::from_utf8_lossy(
                    &stderr_reader
                        .join()
                        .map_err(|_| anyhow::anyhow!("{program} stderr reader panicked"))??,
                )
                .to_string();
                if status.success() {
                    return Ok(stdout);
                } else {
                    return Err(anyhow::anyhow!(
                        "{} exited with {}\nstdout:\n{}\nstderr:\n{}",
                        program,
                        status,
                        output_tail(&stdout),
                        output_tail(&stderr)
                    ));
                }
            }
            None => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    child.wait()?;
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    return Err(anyhow::anyhow!("{} timed out after {:?}", program, timeout));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn output_tail(value: &str) -> String {
    let lines: Vec<&str> = value.lines().collect();
    lines[lines.len().saturating_sub(100)..].join("\n")
}

/// Cache for warm subprocess outputs.
pub struct ProcessCache {
    cache: Mutex<HashMap<String, ProcessResult>>,
    max_entries: usize,
}

struct ProcessResult {
    stdout: String,
    stderr: String,
    exit_code: i32,
    cached_at: Instant,
}

impl ProcessCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            max_entries,
        }
    }

    pub fn run(&self, program: &str, args: &[&str], ttl: Duration) -> Result<String> {
        let key = format!("{} {}", program, args.join(" "));
        // Check cache
        {
            let cache = self.cache.lock().unwrap();
            if let Some(entry) = cache.get(&key) {
                if entry.cached_at.elapsed() < ttl {
                    if entry.exit_code == 0 {
                        return Ok(entry.stdout.clone());
                    }
                    return Err(anyhow::anyhow!("cached error: {}", entry.stderr));
                }
            }
        }
        // Run
        let output = Command::new(program).args(args).output()?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);
        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            if cache.len() >= self.max_entries {
                cache.clear();
            }
            cache.insert(
                key,
                ProcessResult {
                    stdout: stdout.clone(),
                    stderr,
                    exit_code,
                    cached_at: Instant::now(),
                },
            );
        }
        if exit_code == 0 {
            Ok(stdout)
        } else {
            Err(anyhow::anyhow!("{} exited with {}", program, exit_code))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_runner_captures_nonzero_diagnostics() {
        let error = run_with_timeout(
            "/bin/sh",
            &["-c", "echo stdout-marker; echo stderr-marker >&2; exit 7"],
            None,
            Duration::from_secs(2),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("stdout-marker"));
        assert!(error.contains("stderr-marker"));
        assert!(error.contains("status: 7"), "unexpected error: {error}");
    }

    #[test]
    fn timeout_runner_drains_output_while_process_runs() {
        let output = run_with_timeout(
            "/bin/sh",
            &[
                "-c",
                "i=0; while [ $i -lt 20000 ]; do echo line-$i; i=$((i+1)); done",
            ],
            None,
            Duration::from_secs(5),
        )
        .unwrap();
        assert!(output.contains("line-19999"));
    }
}
