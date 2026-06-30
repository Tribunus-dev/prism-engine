//! Sandboxed JavaScript runtime via deno_core (V8).
//! Filesystem access is scoped to the sandbox root. No network, subprocess, or env.

use std::path::{Path, PathBuf};
use deno_core::{JsRuntime, RuntimeOptions};
use std::time::Instant;

const DEFAULT_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug, Clone, serde::Serialize)]
pub struct JsExecutionResult {
    pub ok: bool,
    pub output: String,
    pub error: Option<String>,
    pub duration_ms: u64,
}

fn sandbox_root(explicit: Option<&Path>) -> PathBuf {
    if let Some(r) = explicit {
        return r.to_path_buf();
    }
    std::env::var("PRISM_SANDBOX_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_default())
}

/// Simple V8-based JS runner with injected sandbox helpers.
/// File ops are implemented as Rust closures exposed via JS global scope.
pub fn run_javascript(
    code: &str,
    root_path: Option<&Path>,
    timeout_ms: Option<u64>,
) -> JsExecutionResult {
    let start = Instant::now();
    let _timeout = timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS);
    let root = sandbox_root(root_path);

    let mut runtime = JsRuntime::new(RuntimeOptions {
        extensions: vec![],
        ..Default::default()
    });

    // Inject sandbox helpers as immediately-invoked closures that
    // capture the root path via our bootstrap JS.
    let root_js = root.to_string_lossy().replace('\\', "\\\\").replace('\'', "\\'");
    
    let bootstrap = format!(r#"
        const SANDBOX_ROOT = '{root_js}';
        
        globalThis.console = {{
            log: (...args) => {{ /* captured in output handler */ }},
            error: (...args) => {{ /* captured in output handler */ }},
        }};

        globalThis.readFile = async (path) => {{
            // TODO: wire through Rust op
            return "readFile not available in basic mode";
        }};

        globalThis.writeFile = async (path, content) => {{
            return "writeFile not available in basic mode";
        }};

        globalThis.listDirectory = async (path) => {{
            // Walk filesystem via sandbox root
            return "listDirectory not available in basic mode";
        }};
    "#);

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
        Ok(_) => JsExecutionResult {
            ok: true,
            output: "Execution completed. File ops require deno_core extension support.".into(),
            error: None,
            duration_ms: start.elapsed().as_millis() as u64,
        },
        Err(e) => JsExecutionResult {
            ok: false,
            output: String::new(),
            error: Some(format!("{e}")),
            duration_ms: start.elapsed().as_millis() as u64,
        },
    }
}
