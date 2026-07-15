//! BitNet ECS inference test — load a compiled CImage and drive 1000 decode
//! steps through the canonical ECS pipeline.
//!
//! Usage:
//!   cargo run --package tribunus-compute-core --features prism-backend \
//!     --bin bitnet-ecs-test \
//!     -- --cimage artifacts/bitnet-b1.58-2B-4T.cimage --tokens 1000

#![cfg(any(feature = "mlx-backend", feature = "prism-backend"))]

use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use tribunus_compute_core::ecs::component::engine::InFlightDecode;
use tribunus_compute_core::ecs::system::engine_systems::{
    CimageGenerateSystem, CimageLoadRequest, CimageLoadSystem, EngineInitSystem,
};
use tribunus_compute_core::ecs::system::metal_init::MetalInitSystem;
use tribunus_compute_core::ecs::WorldSystemsExt;
use tribunus_compute_core::ecs::{EntityKind, SchedulePhase, World};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    cimage: PathBuf,
    #[arg(long, default_value = "1000")]
    tokens: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    println!("=== BitNet ECS Inference Test ===");
    println!("CImage: {}", args.cimage.display());
    println!("Tokens: {}", args.tokens);

    // ── 1. Build the ECS world with the systems we need ────────────────
    let mut world = World::new();

    // Phase A: Model loading — engine singleton + CImage loading
    world.add_system(Box::new(EngineInitSystem));
    #[cfg(target_os = "macos")]
    world.add_system(Box::new(MetalInitSystem));
    world.add_system(Box::new(CimageLoadSystem));

    // Phase I: Execution — decode loop driver
    world.add_system(Box::new(CimageGenerateSystem));

    // ── 2. Read the CImage file ────────────────────────────────────────
    println!("Reading CImage...");
    let cimage_bytes =
        std::fs::read(&args.cimage).map_err(|e| format!("Cannot read CImage: {e}"))?;
    println!(
        "  {} bytes ({} MiB)",
        cimage_bytes.len(),
        cimage_bytes.len() / 1_048_576
    );

    // ── 3. Load the CImage via Phase A ─────────────────────────────────
    let model_entity = world.spawn(
        EntityKind::Model,
        Some(args.cimage.to_string_lossy().to_string()),
    )?;
    let (load_tx, load_rx) = mpsc::channel();
    let _ = world.add_component(
        model_entity,
        CimageLoadRequest {
            cimage_bytes,
            result_tx: Some(load_tx),
        },
    );

    println!("Phase A: ModelLoading...");
    world.run_phase(SchedulePhase::ModelLoading)?;
    println!("  Entity count: {}", world.entity_count());

    // Check for errors from the CImage load system
    match load_rx.try_recv() {
        Ok(Ok(())) => println!("  CImage load: OK"),
        Ok(Err(msg)) => {
            eprintln!("  CImage load FAILED: {msg}");
            std::process::exit(1);
        }
        Err(mpsc::TryRecvError::Empty) => {
            eprintln!("  CImage load: no result received (system may not have run)");
            std::process::exit(1);
        }
        Err(mpsc::TryRecvError::Disconnected) => {
            eprintln!("  CImage load: result channel disconnected");
            std::process::exit(1);
        }
    }

    // ── 4. Add InFlightDecode to start inference tracking ──────────────
    let _ = world.add_component(
        model_entity,
        InFlightDecode {
            token_count: 0,
            kv_block_index: 0,
            eos: false,
        },
    );

    // ── 5. Run decode loop via Phase I ────────────────────────────────
    println!("\nPhase I: Execution ({} tokens)...", args.tokens);
    let start = Instant::now();

    for step in 0..args.tokens {
        world.run_phase(SchedulePhase::Execution)?;

        if (step + 1) % 100 == 0 {
            let elapsed = start.elapsed();
            let rate = (step + 1) as f64 / elapsed.as_secs_f64();
            print!(
                "\r  Token {}/{} ({:.1} tok/s)  ",
                step + 1,
                args.tokens,
                rate
            );
            std::io::stdout().flush()?;
        }
    }

    let elapsed = start.elapsed();
    let rate = args.tokens as f64 / elapsed.as_secs_f64();

    // ── 6. Verify decode progress ─────────────────────────────────────
    let decode = world
        .get_component::<InFlightDecode>(model_entity)
        .ok_or_else(|| "InFlightDecode component missing after execution")?;
    let final_token_count = decode.token_count;

    println!();
    println!("\n=== Results ===");
    println!("Tokens requested: {}", args.tokens);
    println!("Tokens decoded:   {}", final_token_count);
    println!("Elapsed:         {:.2}s", elapsed.as_secs_f64());
    println!("Throughput:      {:.1} tok/s", rate);
    println!("Final entities:  {}", world.entity_count());

    assert!(
        final_token_count > 0,
        "Decode should have produced at least one token"
    );
    println!("\nPASS: ECS inference pipeline verified ✓");

    Ok(())
}
