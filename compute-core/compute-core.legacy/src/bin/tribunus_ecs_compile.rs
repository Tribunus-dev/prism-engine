use std::path::PathBuf;

use tribunus_compute_core::ecs::compile_session::CompileSession;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut source: Option<String> = None;
    let mut output: PathBuf = PathBuf::from("output.cimage");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                i += 1;
                source = Some(args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("error: --source requires a path argument");
                    std::process::exit(1);
                }));
            }
            "--output" => {
                i += 1;
                output = PathBuf::from(args.get(i).cloned().unwrap_or_else(|| {
                    eprintln!("error: --output requires a path argument");
                    std::process::exit(1);
                }));
            }
            other => {
                eprintln!("error: unknown flag '{other}'");
                eprintln!("usage: tribunus-ecs-compile --source <path> [--output <path>]");
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let source = source.unwrap_or_else(|| {
        eprintln!("error: --source is required");
        eprintln!("usage: tribunus-ecs-compile --source <path> [--output <path>]");
        std::process::exit(1);
    });

    let mut session = CompileSession::new();
    session.set_output_path(output.clone());
    session.register_builtin_systems();

    if let Err(e) = session.load_model(&source) {
        eprintln!("ECS compile error: {e}");
        std::process::exit(1);
    }

    match session.compile() {
        Ok(Some(path)) => {
            println!("ECS compile completed: {path}");
        }
        Ok(None) => {
            println!("ECS compile completed (no output path configured)");
        }
        Err(e) => {
            eprintln!("ECS compile error: {e}");
            std::process::exit(1);
        }
    }
}
