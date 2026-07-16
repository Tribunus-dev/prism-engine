use anyhow::{bail, Result};
use deno_core::{JsRuntime, RuntimeOptions};

pub fn validate_script(source: &str, max_bytes: usize) -> Result<()> {
    if source.len() > max_bytes {
        bail!("script exceeds configured limit");
    }
    for forbidden in [
        "Deno.",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "import(",
        "while (true)",
        "while(true)",
    ] {
        if source.contains(forbidden) {
            bail!("script contains forbidden capability: {forbidden}");
        }
    }
    let mut runtime = JsRuntime::new(RuntimeOptions::default());
    runtime
        .execute_script("prism-browser-validation.js", source.to_owned())
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("JavaScript validation failed: {error}"))
}
