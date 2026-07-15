//! prism_compile — CLI entry point for the CimageDeploymentCompiler.
//!
//! Wires the full compilation pipeline as explicit sequential steps:
//!   checkpoint inspection → prism compile → assembly → seal → promote
//!
//! Usage:
//!   cargo run --bin prism_compile -- \
//!     --model ./gemma-4-12b-unified \
//!     --output ~/.prism/models/gemma4/latest.cimage \
//!     --target apple-m1 \
//!     --precision nf4

use std::path::PathBuf;
use std::process;

use tribunus_compute_core::ecs::canonical::CompileRequest;
use tribunus_compute_core::ecs::compiler::deployment_compiler::{
    CimageDeploymentCompiler, DeploymentRequest,
};
use tribunus_compute_core::ecs::compute_image::model_family::gemma4_inspect::inspect_gemma4_checkpoint;

fn get_opt(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if has_flag(&args, "--help")
        || has_flag(&args, "-h")
        || !args.iter().any(|a| a.starts_with("--model"))
    {
        eprintln!("Usage: prism_compile --model <path> [--output <path>] [--target <target>] [--precision <prec>] [--no-mtp]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --model <path>       Path to Gemma 4 unified checkpoint directory (required)");
        eprintln!("  --output <path>      Output .cimage path (default: <model-name>.cimage)");
        eprintln!("  --target <target>    Hardware target (default: apple-m1)");
        eprintln!("  --precision <prec>   Precision policy (default: nf4)");
        eprintln!("  --no-mtp             Disable MTP speculative decoding");
        eprintln!("  --help, -h           Show this help");
        process::exit(if has_flag(&args, "--help") { 0 } else { 1 });
    }

    let model_path = get_opt(&args, "--model").expect("--model is required");
    let output_path = get_opt(&args, "--output").map(PathBuf::from);
    let target = get_opt(&args, "--target").unwrap_or_else(|| "apple-m1".into());
    let precision = get_opt(&args, "--precision").unwrap_or_else(|| "nf4".into());
    let mtp = !has_flag(&args, "--no-mtp");

    // ── Early guard: evolutionary-ternary not production-ready ──────
    if precision == "evolutionary-ternary" {
        eprintln!(
            "[prism] evolutionary-ternary not yet production-ready — \
             requires sensitivity analysis + truthful calibration (Phases 3-4). \
             Use --precision nf4 or --precision int8 for now."
        );
        process::exit(1);
    }

    let request = DeploymentRequest {
        model_path: PathBuf::from(&model_path),
        output_path,
        target: target.clone(),
        precision: precision.clone(),
        mtp,
        ..Default::default()
    };

    // Determine the resolved output path before compilation steps
    let resolved_output = request.output_path.clone().unwrap_or_else(|| {
        let stem = request
            .model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        PathBuf::from(format!("{}.cimage", stem))
    });

    // ── Step 1: Inspect checkpoint ──────────────────────────────────
    eprintln!(
        "[prism] step 1/5: inspecting checkpoint at {}...",
        request.model_path.display()
    );
    let inspection = match inspect_gemma4_checkpoint(&request.model_path) {
        Ok(ins) => ins,
        Err(e) => {
            eprintln!("[prism] checkpoint inspection failed: {e}");
            process::exit(1);
        }
    };
    eprintln!(
        "[prism]   {} tensors, {} layers, vocab={}, mtp={}",
        inspection.inventory.total_tensors,
        inspection.config.num_layers,
        inspection.config.vocab_size,
        inspection.config.mtp_depth.is_some()
    );

    // ── Step 2: Prism compilation ──────────────────────────────────
    eprintln!("[prism] step 2/5: compiling model...");

    let mut compiler = CimageDeploymentCompiler::new();

    let compile_req = CompileRequest {
        source_path: request.model_path.to_string_lossy().to_string(),
        output_path: Some(resolved_output.to_string_lossy().to_string()),
        target_lanes: Vec::new(),
        target_hardware: Some(target),
        policy_path: None,
        quant_mode: Some(precision),
        source_type: Some("safetensors".into()),
        authority: Some("SealedComputeImage".into()),
        draft_path: if mtp && inspection.config.mtp_depth.unwrap_or(0) > 0 {
            Some(request.model_path.to_string_lossy().to_string())
        } else {
            None
        },
        ..Default::default()
    };

    let compile_outcome = match compiler.prism_compiler.compile(compile_req) {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("[prism] compilation failed: {e}");
            process::exit(1);
        }
    };
    eprintln!(
        "[prism]   compiled: {} tensor payloads, {} kernel artifacts",
        compile_outcome.build_input.tensor_payloads.len(),
        compile_outcome.compiled_kernels.len(),
    );

    // ── Step 3: Build CimageAssembly ────────────────────────────────
    eprintln!("[prism] step 3/5: assembling deployable cimage...");
    let assembly =
        compiler.build_deployable_cimage(&compile_outcome, &request, &inspection, &resolved_output);
    eprintln!(
        "[prism]   assembly complete: {} segments, {} kernel artifacts, mtp={}",
        assembly.segments.len(),
        assembly.kernel_artifacts.len(),
        assembly.serving_profile.mtp_enabled,
    );

    // ── Step 4: Seal and validate ───────────────────────────────────
    eprintln!("[prism] step 4/5: sealing and validating assembly...");
    let digest = assembly.compute_digest();
    eprintln!("[prism]   assembly digest: {}", &digest[..16]);

    let promotable = match compiler.seal_and_validate(assembly) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[prism] seal/validation failed: {e}");
            process::exit(1);
        }
    };
    eprintln!(
        "[prism]   sealed: validated={}, digest={}",
        promotable.validated,
        &promotable.digest[..16]
    );

    // ── Step 5: Promote through lifecycle ───────────────────────────
    eprintln!("[prism] step 5/5: promoting through lifecycle...");
    match compiler.promote_cimage(promotable) {
        Ok(result) => {
            eprintln!(
                "[prism] deployment complete: gen_id={}, output={}, mtp={}",
                result.generation_id.0,
                result.cimage_path.display(),
                result.mtp_enabled
            );
        }
        Err(e) => {
            eprintln!("[prism] promotion failed: {e}");
            process::exit(1);
        }
    }
}
