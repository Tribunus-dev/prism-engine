#![allow(unexpected_cfgs)]

fn forward(name: &str) {
    let value = std::env::var(name).unwrap_or_else(|_| format!("{name}_MISSING"));
    println!("cargo:rustc-env=TRIBUNUS_{name}={value}");
}

fn metal_sdk_for_target(target: &str) -> &'static str {
    if target.contains("apple-ios") {
        "iphoneos"
    } else {
        "macosx"
    }
}

fn main() {
    let host_target = std::env::var("TARGET").unwrap_or_default();

    // ── Metal kernel compilation ────────────────────────────────────────
    // Only compile Metal shaders when the metal-dispatch feature is active.
    if cfg!(feature = "metal-dispatch") {
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
            "ternary_tile640_gemv.metal",
            // Batched FP32 GEMV for compile-time operator validation on GPU.
            "batched_gemv_fp32.metal",
            // NF4 teacher forward GEMV — without this the kernel is never
            // compiled into the metallib and KernelRegistry cannot find
            // `fused_gemv_nf4_tile640_fp32`, so the teacher cimage can't execute.
            "nf4_tile640_gemv.metal",
            "int8_tile640_gemv.metal",
            "nf4_tile640_scaled_reduction_gemv.metal",
            "fused_teacher_student_gemv.metal",
        ];
        for src in metal_sources {
            println!(
                "cargo:rerun-if-changed={}",
                template_dir.join(src).display()
            );
        }
        // Track nf4tile640 compute kernel (compiled at runtime via include_str! in metal.rs).
        println!(
            "cargo:rerun-if-changed={}",
            std::path::Path::new(&manifest_dir)
                .join("shaders")
                .join("nf4tile640.metal")
                .display()
        );
        // Track tts_codec compute kernel (compiled at runtime via include_str! in metal.rs).
        println!(
            "cargo:rerun-if-changed={}",
            std::path::Path::new(&manifest_dir)
                .join("shaders")
                .join("tts_codec.metal")
                .display()
        );

        // Step 1: compile each .metal -> .air
        let mut air_files = Vec::new();
        for src in metal_sources {
            let sdk = metal_sdk_for_target(&host_target);
            let src_path = template_dir.join(src);
            let air_file = std::path::Path::new(&out_dir)
                .join(src)
                .with_extension("air");
            let status = std::process::Command::new("xcrun")
                .args(["-sdk", &sdk, "metal", "-c"])
                .arg(&src_path)
                .arg("-o")
                .arg(&air_file)
                .status()
                .expect("Failed to execute xcrun metal");
            assert!(status.success(), "xcrun metal failed for {src}");
            air_files.push(air_file);
        }

        // Step 2: link all .air → .metallib
        let sdk = metal_sdk_for_target(&host_target);
        let metallib_path = std::path::Path::new(&out_dir).join("palettized_kernels.metallib");
        let mut link_cmd = std::process::Command::new("xcrun");
        link_cmd.args(["-sdk", &sdk, "metallib", "-o"]);
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
    }

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
    // Uses a runtime TARGET check because build.rs cfg reflects the HOST platform,
    // not the cross-compilation target. IOSurface is macOS-only on iOS.
    let is_macos_target =
        host_target == "aarch64-apple-darwin" || host_target == "x86_64-apple-darwin";

    if is_macos_target
        && (cfg!(feature = "mlx-backend")
            || cfg!(feature = "prism-backend")
            || cfg!(feature = "ffi"))
    {
        let _out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
        if !cfg!(feature = "coreai-backend") {
            cc::Build::new()
                .file("src/bridge/coreai_arena.mm")
                .flag("-fobjc-arc")
                .flag("-std=c++17")
                .compile("coreai_arena");
            // ObjC++ .mm files need C++ standard library for personality v0.
            println!("cargo:rustc-link-lib=c++");
            cc::Build::new()
                .file("src/bridge/coreai_exec.mm")
                .flag("-fobjc-arc")
                .flag("-fblocks")
                .flag("-std=c++17")
                .compile("coreai_exec");
            cc::Build::new()
                .file("src/bridge/coreai_state.mm")
                .flag("-fobjc-arc")
                .flag("-fblocks")
                .flag("-std=c++17")
                .compile("coreai_state");
        }
        cc::Build::new()
            .file("src/bridge/ane_private.mm")
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .compile("ane_private");

        // Framework dependencies.
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // Swift @C bridge prototype. Replaces coreai_exec.mm + coreai_state.mm
        // when the `coreai-backend` feature is enabled. Core AI's types are
        // Swift structs — not bridgeable from ObjC++.
        if cfg!(feature = "coreai-backend") {
            let swift_out = format!("{}/libcoreai_bridge.o", _out_dir);
            let swift_src = "src/bridge/coreai_bridge.swift";
            let status = std::process::Command::new("swiftc")
                .args(["-c", "-emit-object", "-module-name", "CoreAiBridge"])
                .arg(swift_src)
                .args(["-o", &swift_out])
                .args(["-framework", "CoreAI", "-framework", "Foundation"])
                .arg("-v")
                .status()
                .expect("swiftc failed");

            assert!(status.success(), "swiftc returned non-zero");
            cc::Build::new().object(&swift_out).compile("coreai_bridge");
            println!("cargo:rustc-link-lib=framework=CoreAI");
            println!("cargo:rustc-link-lib=framework=CoreML");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
    }
    eprintln!("build.rs: END of main() — all link directives emitted");
}
