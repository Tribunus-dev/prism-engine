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
    ];
    for src in metal_sources {
        println!("cargo:rerun-if-changed={}", template_dir.join(src).display());
    }

    let mut air_files = Vec::new();
    for src in metal_sources {
        let src_path = template_dir.join(src);
        let air_file = std::path::Path::new(&out_dir).join(src).with_extension("air");
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

    // Link all .air → .metallib
    let metallib_path = std::path::Path::new(&out_dir).join("palettized_kernels.metallib");
    let mut link_cmd = std::process::Command::new("xcrun");
    link_cmd.args(["-sdk", "macosx", "metallib", "-o"]);
    link_cmd.arg(&metallib_path);
    for air in &air_files {
        link_cmd.arg(air);
    }
    let status = link_cmd.status().expect("Failed to execute xcrun metallib");
    assert!(status.success(), "xcrun metallib failed");

    // Generate embedded_metallib.rs with the kernel bytes baked into the binary.
    let metallib_bytes = std::fs::read(&metallib_path).expect("read metallib");
    let rs_path = std::path::Path::new(&out_dir).join("embedded_metallib.rs");
    std::fs::write(&rs_path, format!(
        "/// Auto-generated: embedded Metal kernel library ({} bytes)\n\
         pub const KERNEL_BYTES: &[u8] = &{:?};\n",
        metallib_bytes.len(), metallib_bytes
    )).expect("write embedded metallib");

    // Also keep the env var for fallback
    println!("cargo:rustc-env=TRIBUNUS_METALLIB={}", metallib_path.display());

    // ── ANE ObjC bridge ────────────────────────────────────────────────
    #[cfg(all(target_os = "macos", feature = "ane"))]
    {
        let bridge_dir = std::path::Path::new(&manifest_dir).join("src").join("bridge");
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
