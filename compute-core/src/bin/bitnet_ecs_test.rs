//! BitNet ECS integration test — load and run inference through the
//! canonical ECS pipeline.
//!
//! Usage:
//!   cargo run --package tribunus-compute-core --features prism-backend \
//!     --bin bitnet-ecs-test \
//!     -- --cimage artifacts/bitnet-b1.58-2B-4T.cimage --tokens 1000

#![cfg(any(feature = "mlx-backend", feature = "prism-backend"))]

use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    cimage: PathBuf,
    #[arg(long, default_value = "1000")]
    tokens: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("=== BitNet ECS Integration Test ===");
    println!("CImage: {}", args.cimage.display());
    println!("Tokens: {}", args.tokens);

    // 1. Create world and register all systems
    let mut session = tribunus_compute_core::ecs::compile_session::CompileSession::new();
    session.register_builtin_systems();
    session.register_execution_systems();

    // 2. Run the full compilation pipeline to set up inference
    //    Phase A: Model loading
    println!("Phase A: Model loading...");
    session
        .world
        .run_phase(tribunus_compute_core::ecs::SchedulePhase::ModelLoading)?;
    let entity_count = session.world.entity_count();
    println!("  {} entities in world", entity_count);

    // 3. Run execution phase (inference)
    println!("\nPhase I: Execution ({} tokens)...", args.tokens);
    let start = Instant::now();

    for step in 0..args.tokens {
        session
            .world
            .run_phase(tribunus_compute_core::ecs::SchedulePhase::Execution)?;
        if (step + 1) % 100 == 0 {
            let elapsed = start.elapsed();
            let rate = (step + 1) as f64 / elapsed.as_secs_f64();
            print!(
                "\r  Step {}/{} ({:.1} steps/sec)  ",
                step + 1,
                args.tokens,
                rate
            );
            std::io::stdout().flush()?;
        }
        // Check for completion
        if step >= args.tokens - 1 {
            break;
        }
    }

    let elapsed = start.elapsed();
    let rate = args.tokens as f64 / elapsed.as_secs_f64();

    println!("\n\n=== Results ===");
    println!("Tokens: {} / {}", args.tokens, args.tokens);
    println!("Elapsed: {:.2}s", elapsed.as_secs_f64());
    println!("Throughput: {:.1} tok/s", rate);
    println!("Final entities: {}", session.world.entity_count());

    Ok(())
}
