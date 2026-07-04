use std::ffi::CString;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;

use tribunus_compute_core::coreai_pipeline::build_nf4_tile640_stateless_region;
use tribunus_compute_core::ffi::prism_compile_and_pack;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    metallib: PathBuf,
    #[arg(long)]
    main_mlmodelc: Option<PathBuf>,
    #[arg(long)]
    mtp_mlmodelc: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    generate_placeholder_modelc: bool,
}

fn ensure_modelc(
    requested: Option<PathBuf>,
    output_root: &Path,
    region_id: &str,
) -> Result<PathBuf, String> {
    if let Some(path) = requested {
        return Ok(path);
    }

    let modelc_root = output_root.join("generated_modelc");
    fs::create_dir_all(&modelc_root)
        .map_err(|e| format!("create {}: {e}", modelc_root.display()))?;
    let receipt =
        build_nf4_tile640_stateless_region("activations", &[1, 640], 64, &modelc_root, region_id)?;
    Ok(PathBuf::from(receipt.compiled_modelc_path))
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("create {}: {e}", dst.display()))?;
    for entry in fs::read_dir(src).map_err(|e| format!("read_dir {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("dir entry {}: {e}", src.display()))?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        let ty = entry
            .file_type()
            .map_err(|e| format!("file_type {}: {e}", src_path.display()))?;
        if ty.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).map_err(|e| {
                format!("copy {} -> {}: {e}", src_path.display(), dst_path.display())
            })?;
        }
    }
    Ok(())
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if !args.source.is_dir() {
        return Err(format!("source dir not found: {}", args.source.display()));
    }
    if !args.metallib.is_file() {
        return Err(format!("metallib not found: {}", args.metallib.display()));
    }

    let resource_root = args
        .output
        .parent()
        .unwrap_or(Path::new("."))
        .join("nf4tile640_resources");
    if resource_root.exists() {
        fs::remove_dir_all(&resource_root)
            .map_err(|e| format!("remove {}: {e}", resource_root.display()))?;
    }
    fs::create_dir_all(&resource_root)
        .map_err(|e| format!("create {}: {e}", resource_root.display()))?;

    let main_mlmodelc = if args.generate_placeholder_modelc || args.main_mlmodelc.is_none() {
        ensure_modelc(args.main_mlmodelc.clone(), &resource_root, "main_12b")?
    } else {
        args.main_mlmodelc.clone().unwrap()
    };
    let mtp_mlmodelc = if args.generate_placeholder_modelc || args.mtp_mlmodelc.is_none() {
        ensure_modelc(args.mtp_mlmodelc.clone(), &resource_root, "mtp_1b")?
    } else {
        args.mtp_mlmodelc.clone().unwrap()
    };

    fs::copy(&args.metallib, resource_root.join("default.metallib"))
        .map_err(|e| format!("copy metallib: {e}"))?;
    copy_dir_recursive(&main_mlmodelc, &resource_root.join("main_12b.mlmodelc"))?;
    copy_dir_recursive(&mtp_mlmodelc, &resource_root.join("mtp_1b.mlmodelc"))?;

    let source_c = CString::new(args.source.to_string_lossy().as_bytes())
        .map_err(|e| format!("source cstring: {e}"))?;
    let output_c = CString::new(args.output.to_string_lossy().as_bytes())
        .map_err(|e| format!("output cstring: {e}"))?;
    let resources_c = CString::new(resource_root.to_string_lossy().as_bytes())
        .map_err(|e| format!("resources cstring: {e}"))?;

    let rc = unsafe {
        prism_compile_and_pack(source_c.as_ptr(), output_c.as_ptr(), resources_c.as_ptr())
    };
    if rc != 0 {
        return Err(format!("prism_compile_and_pack failed with code {rc}"));
    }

    println!("{}", args.output.display());
    Ok(())
}
