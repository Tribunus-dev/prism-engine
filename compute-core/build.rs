#![allow(unexpected_cfgs)]

fn forward(name: &str) {
    let value = std::env::var(name).unwrap_or_else(|_| format!("{name}_MISSING"));
    println!("cargo:rustc-env=TRIBUNUS_{name}={value}");
}

fn main() {
    // ── Metal kernel compilation ────────────────────────────────────────
    // Compile palettized shaders into a loadable .metallib.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let template_dir = std::path::Path::new(&manifest_dir)
        .join("src")
        .join("compute_image")
        .join("templates");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");

    let metal_sources = &[
        "palettized_gemv.metal",
        "palettized_gemv_swiglu.metal",
        "palettized_gemm.metal",
        "fused_gate_up.metal",
    ];
    for src in metal_sources {
        println!(
            "cargo:rerun-if-changed={}",
            template_dir.join(src).display()
        );
    }

    // Step 1: compile each .metal → .air
    let mut air_files = Vec::new();
    for src in metal_sources {
        let src_path = template_dir.join(src);
        let air_file = std::path::Path::new(&out_dir)
            .join(src)
            .with_extension("air");
        let status = std::process::Command::new("xcrun")
            .args(["-sdk", "macosx", "metal", "-c"])
            .arg(&src_path)
            .arg("-o")
            .arg(&air_file)
            .status()
            .expect("Failed to execute xcrun metal");
        assert!(status.success(), "xcrun metal failed for {src}");
        air_files.push(air_file);
    }

    // Step 2: link all .air → .metallib
    let metallib_path = std::path::Path::new(&out_dir).join("palettized_kernels.metallib");
    let mut link_cmd = std::process::Command::new("xcrun");
    link_cmd.args(["-sdk", "macosx", "metallib", "-o"]);
    link_cmd.arg(&metallib_path);
    for air in &air_files {
        link_cmd.arg(air);
    }
    let status = link_cmd.status().expect("Failed to execute xcrun metallib");
    assert!(status.success(), "xcrun metallib failed");

    println!(
        "cargo:rustc-env=TRIBUNUS_METALLIB={}",
        metallib_path.display()
    );

    // Forward git SHA and branch for artifact provenance.
    if std::env::var("VERGEN_GIT_SHA").is_err() {
        if let Ok(out) = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
        {
            let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !sha.is_empty() {
                println!("cargo:rustc-env=VERGEN_GIT_SHA={}", sha);
            }
        }
    }
    if std::env::var("VERGEN_GIT_BRANCH").is_err() {
        if let Ok(out) = std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
        {
            let branch = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !branch.is_empty() {
                println!("cargo:rustc-env=VERGEN_GIT_BRANCH={}", branch);
            }
        }
    }

    forward("PROFILE");
    forward("OPT_LEVEL");
    forward("TARGET");
    forward("DEBUG");

    // Record RUSTFLAGS
    if let Ok(flags) = std::env::var("RUSTFLAGS") {
        println!("cargo:rustc-env=TRIBUNUS_RUSTFLAGS={}", flags);
    }

    // Record linker if set
    if let Ok(ld) = std::env::var("RUSTC_LINKER") {
        println!("cargo:rustc-env=TRIBUNUS_LINKER={}", ld);
    }

    // Record host info
    println!("cargo:rustc-env=TRIBUNUS_HOST_OS={}", std::env::consts::OS);
    println!(
        "cargo:rustc-env=TRIBUNUS_HOST_ARCH={}",
        std::env::consts::ARCH
    );

    // Guard: on non-macOS targets, a CPU backend feature must be explicit.

    // Compile the ObjC++ Core ML / IOSurface bridge.
    #[cfg(all(
        target_os = "macos",
        any(feature = "mlx-backend", feature = "prism-backend", feature = "ffi")
    ))]
    {
        cc::Build::new()
            .file("src/bridge/coreml_arena.mm")
            .flag("-fobjc-arc")
            .flag("-std=c++17")
            .compile("coreml_arena");
        cc::Build::new()
            .file("src/bridge/coreml_exec.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("coreml_exec");
        cc::Build::new()
            .file("src/bridge/coreml_state.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("coreml_state");
        cc::Build::new()
            .file("src/bridge/ane_private.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("ane_private");
        // Framework dependencies for the Core ML / IOSurface bridge.
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
    }
    eprintln!("build.rs: END of main() — all link directives emitted");
}
