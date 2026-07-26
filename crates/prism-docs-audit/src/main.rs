//! `prism-docs-audit` — the A-list axiom runner.
//!
//! Usage:
//!
//! ```text
//! prism-docs-audit --site <dir> [--out <report.md>]
//! ```
//!
//! The runner takes a directory of SSG output
//! (`docs/`), runs every axiom check, and prints a
//! 22-row table to stdout. Optionally writes the same
//! data as Markdown to `--out`. Exits 0 if every
//! blocking check passes, 1 otherwise.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use prism_docs_audit::context::SiteSource;
use prism_docs_audit::runner::run_audit;

#[derive(Parser, Debug)]
#[command(name = "prism-docs-audit", about = "A-list axiom runner for Prism Observatory v1", version)]
struct Cli {
    /// Path to the site to audit (a directory containing
    /// the SSG output: index.html, site.css, the data
    /// layer, the manuscript, the schemas).
    #[arg(long, default_value = "docs")]
    site: PathBuf,

    /// Optional path to write a Markdown report. The
    /// table is always printed to stdout.
    #[arg(long)]
    out: Option<PathBuf>,

    /// Print the full evidence for every check, not
    /// just the one-line summary.
    #[arg(long, default_value_t = false)]
    verbose: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let source = SiteSource::LocalDir(cli.site.clone());
    let report = match run_audit(&source) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("prism-docs-audit: error: {e}");
            return ExitCode::from(1);
        }
    };

    // Print the table to stdout.
    println!("{}", report.to_markdown());

    if cli.verbose {
        println!("\n## Detail\n");
        for r in &report.results {
            println!("### {} {} — {}", r.id, r.verdict, r.name);
            if !r.spec_ref.is_empty() {
                println!("Spec: {}", r.spec_ref);
            }
            println!("Severity: {:?}", r.severity);
            println!("Evidence: {}", r.evidence);
            if !r.detail.is_empty() {
                println!("Detail: {}", r.detail);
            }
            println!();
        }
    }

    if let Some(out) = cli.out {
        if let Err(e) = std::fs::write(&out, report.to_markdown()) {
            eprintln!("prism-docs-audit: failed to write report: {e}");
            return ExitCode::from(1);
        }
        eprintln!("prism-docs-audit: wrote report to {}", out.display());
    }

    ExitCode::from(report.exit_code() as u8)
}
