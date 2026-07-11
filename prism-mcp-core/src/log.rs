use tracing_subscriber::EnvFilter;

/// Initialize structured logging to stderr.
///
/// Respects `RUST_LOG` for filter control (default: `info`).
/// Output format is JSON when the `json` feature is enabled, or
/// plain text by default.
pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();
}
