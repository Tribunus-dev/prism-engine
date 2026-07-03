fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");

    // ── Metal kernel compilation ────────────────────────────────────────
    let template_dir = std::path::Path::new(&manifest_dir).join("templates");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");

    let metal_sources = &[
        "attention_decode.metal",
        "rms_norm.metal",
        "rope_fp16.metal",
        "softmax_fp16.metal",
        "palettized_gemv.metal",
        "palettized_gemv_swiglu.metal",
        "palettized_gemm.metal",
        "fused_gate_up.metal",
        "silu_vec.metal",
    ];
    for src in metal_sources {
        println!(
            "cargo:rerun-if-changed={}",
            template_dir.join(src).display()
        );
    }

    // Metal kernels can only be compiled on macOS via `xcrun`. Off macOS — or
    // when PRISM_MOCK_BUILD is set for a fast, non-GPU dev/CI build — we emit a
    // placeholder metallib so the crate still links; the GPU path is never used
    // on those targets. On macOS WITHOUT the mock flag, a missing or failing
    // toolchain is a HARD ERROR: we must never silently ship an empty kernel
    // library that would then fail (or worse, misbehave) at runtime.
    let mock_requested = std::env::var("PRISM_MOCK_BUILD").is_ok();
    let is_macos = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos");
    let use_mock = mock_requested || !is_macos;
    if mock_requested && is_macos {
        println!(
            "cargo:warning=PRISM_MOCK_BUILD is set — embedding a MOCK Metal kernel \
             library. This build cannot run GPU inference; do not ship it."
        );
    }

    let mut air_files = Vec::new();
    for src in metal_sources {
        let src_path = template_dir.join(src);
        let air_file = std::path::Path::new(&out_dir)
            .join(src)
            .with_extension("air");
        if use_mock {
            std::fs::write(&air_file, "").unwrap();
        } else {
            let status = std::process::Command::new("xcrun")
                .args(["-sdk", "macosx", "metal", "-c"])
                .arg(&src_path)
                .arg("-o")
                .arg(&air_file)
                .status()
                .unwrap_or_else(|e| {
                    panic!(
                        "xcrun not found while compiling Metal kernel {src} on macOS: {e}. \
                         Install the Xcode command-line tools, or set PRISM_MOCK_BUILD=1 \
                         for a non-GPU dev build."
                    )
                });
            assert!(status.success(), "xcrun metal failed for {src}");
        }
        air_files.push(air_file);
    }

    // Link all .air → .metallib
    let metallib_path = std::path::Path::new(&out_dir).join("palettized_kernels.metallib");
    if use_mock {
        std::fs::write(&metallib_path, b"mock_metallib").unwrap();
    } else {
        let mut link_cmd = std::process::Command::new("xcrun");
        link_cmd.args(["-sdk", "macosx", "metallib", "-o"]);
        link_cmd.arg(&metallib_path);
        for air in &air_files {
            link_cmd.arg(air);
        }
        let status = link_cmd.status().unwrap_or_else(|e| {
            panic!(
                "xcrun not found while linking Metal kernels on macOS: {e}. \
                 Install the Xcode command-line tools, or set PRISM_MOCK_BUILD=1."
            )
        });
        assert!(status.success(), "xcrun metallib failed");
    }

    // Generate embedded_metallib.rs with the kernel bytes baked into the binary.
    let metallib_bytes = std::fs::read(&metallib_path).expect("read metallib");
    let rs_path = std::path::Path::new(&out_dir).join("embedded_metallib.rs");
    std::fs::write(
        &rs_path,
        format!(
            "/// Auto-generated: embedded Metal kernel library ({} bytes)\n\
         pub const KERNEL_BYTES: &[u8] = &{:?};\n",
            metallib_bytes.len(),
            metallib_bytes
        ),
    )
    .expect("write embedded metallib");

    // Also keep the env var for fallback
    println!(
        "cargo:rustc-env=TRIBUNUS_METALLIB={}",
        metallib_path.display()
    );

    // ── ANE ObjC bridge ────────────────────────────────────────────────
    #[cfg(all(target_os = "macos", feature = "ane"))]
    {
        let bridge_dir = std::path::Path::new(&manifest_dir)
            .join("src")
            .join("bridge");
        let mut build = cc::Build::new();
        build
            .flag("-fobjc-arc")
            .flag("-fblocks")
            .flag("-std=c++17")
            .flag("-O2")
            .include(&bridge_dir);

        build.file(bridge_dir.join("coreml_exec.mm"));
        build.file(bridge_dir.join("coreml_state.mm"));
        build.file(bridge_dir.join("coreml_arena.mm"));
        build.file(bridge_dir.join("ane_private.mm"));

        build.compile("prism_ane_bridge");

        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=Cocoa");
    }
}
