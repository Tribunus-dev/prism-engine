use std::fs;
use std::path::PathBuf;
use std::process;

use prism_dialect_gen::{emit_rust, parse_document, resolve_document};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: prism-dialect-gen <input.td> [output.rs]");
        process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let source = match fs::read_to_string(&input_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {e}", input_path.display());
            process::exit(1);
        }
    };

    // Parse
    let doc = match parse_document(&source) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Parse error: {e}");
            process::exit(1);
        }
    };

    // Resolve
    let resolved = match resolve_document(&doc) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Resolution error: {e}");
            process::exit(1);
        }
    };

    // Emit
    let output = match emit_rust(&resolved) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Emit error: {e}");
            process::exit(1);
        }
    };

    if let Some(out_path) = args.get(2) {
        fs::write(out_path, &output).unwrap_or_else(|e| {
            eprintln!("Error writing {out_path}: {e}");
            process::exit(1);
        });
    } else {
        println!("{output}");
    }
}
