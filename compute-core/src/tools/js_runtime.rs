//! Sandboxed JavaScript runtime via deno_core (V8) with Rust↔JS ops.
//!
//! Ops access sandbox configuration via thread-local storage (the
//! approach that works with deno_core v0.405's public API).

use deno_core::{extension, op2, JsRuntime, RuntimeOptions};
use std::cell::RefCell;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const _DEFAULT_TIMEOUT_MS: u64 = 30_000;

thread_local! {
    static SANDBOX_CFG: RefCell<Option<Arc<SandboxCfg>>> = const { RefCell::new(None) };
}

struct SandboxCfg {
    root: PathBuf,
    output: RefCell<String>,
}

// ── Web bridge (V8 ↔ WKWebView via shared channel) ───────────────────

/// Trait for driving a real web browser from V8 ops (implemented by
/// prism-bridge via UniFFI BrowserRuntimeDriver callback).
pub trait WebDriver: Send + Sync {
    fn navigate(&self, url: &str) -> Result<String, String>;
    fn snapshot(&self) -> Result<String, String>;
    fn interact(&self, id: u32, action: &str, value: Option<&str>) -> Result<String, String>;
    fn evaluate_js(&self, script: &str) -> Result<String, String>;
}

thread_local! {
    static WEB_DRIVER: RefCell<Option<Arc<dyn WebDriver>>> = const { RefCell::new(None) };
}

/// Set the web driver before running JS (called from prism-bridge).
pub fn set_web_driver(driver: Arc<dyn WebDriver>) {
    WEB_DRIVER.with(|cell| {
        *cell.borrow_mut() = Some(driver);
    });
}

pub enum WebRequest {
    Navigate { url: String },
    Snapshot,
    Interact { id: u32, action: String, value: Option<String> },
    EvaluateJs { script: String },
}

fn do_web_request(req: WebRequest) -> Result<String, io::Error> {
    WEB_DRIVER.with(|cell| {
        let guard = cell.borrow();
        let driver = guard.as_ref().ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotConnected, "web bridge not configured")
        })?;
        match req {
            WebRequest::Navigate { url } => driver.navigate(&url),
            WebRequest::Snapshot => driver.snapshot(),
            WebRequest::Interact { id, action, value } => driver.interact(id, &action, value.as_deref()),
            WebRequest::EvaluateJs { script } => driver.evaluate_js(&script),
        }.map_err(|e| io::Error::new(io::ErrorKind::Other, e))
    })
}

#[op2]
#[string]
fn op_web_navigate(#[string] url: String) -> Result<String, io::Error> {
    do_web_request(WebRequest::Navigate { url })
}

#[op2]
#[string]
fn op_web_snapshot() -> Result<String, io::Error> {
    do_web_request(WebRequest::Snapshot)
}

#[op2]
#[string]
fn op_web_interact(#[string] args_json: String) -> Result<String, io::Error> {
    let parsed: serde_json::Value = serde_json::from_str(&args_json)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("web_interact: {e}")))?;
    let id = parsed.get("id").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let action = parsed.get("action").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let value = parsed.get("value").and_then(|v| v.as_str()).map(|s| s.to_string());
    do_web_request(WebRequest::Interact { id, action, value })
}

#[op2]
#[string]
fn op_web_evaluate_js(#[string] script: String) -> Result<String, io::Error> {
    do_web_request(WebRequest::EvaluateJs { script })
}

// ── Ops ───────────────────────────────────────────────────────────────
// Error type is io::Error because it implements deno_error::JsErrorClass,
// which deno_core v0.405 requires for op return types.

#[op2]
#[string]
fn op_read_file(#[string] path: String) -> Result<String, io::Error> {
    use crate::tools::sandbox;
    let root = SANDBOX_CFG.with(|c| c.borrow().as_ref().unwrap().root.clone());
    let root = Path::new(&root);
    let canon = sandbox::resolve_sandbox_path(&path, root)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read {path}: {e}")))?;
    std::fs::read_to_string(&canon)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("read {path}: {e}")))
}

#[op2(fast)]
fn op_write_file(#[string] path: String, #[string] content: String) -> Result<(), io::Error> {
    use crate::tools::sandbox;
    let root = SANDBOX_CFG.with(|c| c.borrow().as_ref().unwrap().root.clone());
    let root = Path::new(&root);
    let canon = sandbox::resolve_sandbox_path_relaxed(&path, root)
        .map_err(|e| io::Error::new(io::ErrorKind::PermissionDenied, format!("write {path}: {e}")))?;
    if let Some(parent) = canon.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(&canon, &content)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("write {path}: {e}")))
}

#[op2]
#[string]
fn op_list_directory(#[string] path: String) -> Result<String, io::Error> {
    use crate::tools::sandbox;
    let root = SANDBOX_CFG.with(|c| c.borrow().as_ref().unwrap().root.clone());
    let root = Path::new(&root);
    let canon = sandbox::resolve_sandbox_path(&path, root)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("list {path}: {e}")))?;
    let mut entries: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(&canon)? {
        let e = entry?;
        let name = e.file_name().to_string_lossy().to_string();
        let kind = match e.file_type() {
            Ok(t) if t.is_dir() => "dir",
            Ok(t) if t.is_file() => "file",
            Ok(t) if t.is_symlink() => "symlink",
            _ => "other",
        };
        entries.push(format!("{kind} {name}"));
    }
    entries.sort();
    Ok(entries.join("\n"))
}

#[op2(fast)]
fn op_console_log(#[string] msg: String) {
    SANDBOX_CFG.with(|c| {
        if let Some(cfg) = c.borrow().as_ref() {
            let mut out = cfg.output.borrow_mut();
            const MAX: usize = 1_048_576;
            if out.len() < MAX {
                out.push_str(&msg);
                out.push('\n');
            }
        }
    });
}

// ── Extension ─────────────────────────────────────────────────────────

extension!(
    prism_sandbox,
    ops = [
        op_read_file, op_write_file, op_list_directory, op_console_log,
        op_web_navigate, op_web_snapshot, op_web_interact, op_web_evaluate_js,
    ],
);

// ── Result type ───────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct JsExecutionResult {
    pub ok: bool,
    pub output: String,
    pub error: Option<String>,
    pub duration_ms: u64,
}

// ── Public API ────────────────────────────────────────────────────────

/// Run JavaScript code in a sandboxed V8 isolate with Rust-backed file ops.
pub fn run_javascript(
    code: &str,
    root_path: Option<&Path>,
    _timeout_ms: Option<u64>,
) -> JsExecutionResult {
    let start = Instant::now();
    let root = root_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| {
            std::env::var("PRISM_SANDBOX_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default())
        });

    SANDBOX_CFG.with(|c| {
        *c.borrow_mut() = Some(Arc::new(SandboxCfg {
            root,
            output: RefCell::new(String::new()),
        }));
    });

    let mut runtime = JsRuntime::new(RuntimeOptions {
        extensions: vec![prism_sandbox::init()],
        ..Default::default()
    });

    let bootstrap = r#"
        globalThis.console = {
            log: (...args) => Deno.core.ops.op_console_log(args.map(String).join(' ')),
            error: (...args) => Deno.core.ops.op_console_log('ERROR: ' + args.map(String).join(' ')),
        };
        globalThis.readFile = (path) => Deno.core.ops.op_read_file(path);
        globalThis.writeFile = (path, content) => Deno.core.ops.op_write_file(path, content);
        globalThis.listDirectory = (path) => Deno.core.ops.op_list_directory(path);
        globalThis.webNavigate = (url) => Deno.core.ops.op_web_navigate(url);
        globalThis.webSnapshot = () => Deno.core.ops.op_web_snapshot();
        globalThis.webInteract = (args) => Deno.core.ops.op_web_interact(JSON.stringify(args));
        globalThis.webEvaluateJs = (script) => Deno.core.ops.op_web_evaluate_js(script);
    "#;

    if let Err(e) = runtime.execute_script("bootstrap", bootstrap) {
        return JsExecutionResult {
            ok: false,
            output: String::new(),
            error: Some(format!("bootstrap: {e}")),
            duration_ms: start.elapsed().as_millis() as u64,
        };
    }

    let wrapped = format!("(async () => {{ {} }})()", code);
    match runtime.execute_script("user_code", wrapped) {
        Ok(_) => {
            let output = SANDBOX_CFG.with(|c| {
                c.borrow().as_ref().map(|cfg| cfg.output.borrow().clone()).unwrap_or_default()
            });
            JsExecutionResult {
                ok: true,
                output,
                error: None,
                duration_ms: start.elapsed().as_millis() as u64,
            }
        }
        Err(e) => {
            let output = SANDBOX_CFG.with(|c| {
                c.borrow().as_ref().map(|cfg| cfg.output.borrow().clone()).unwrap_or_default()
            });
            JsExecutionResult {
                ok: false,
                output,
                error: Some(format!("{e}")),
                duration_ms: start.elapsed().as_millis() as u64,
            }
        }
    }
}
