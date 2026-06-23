fn main() {
    // ── Metal kernel compilation ────────────────────────────────────────
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let template_dir = std::path::Path::new(&manifest_dir).join("templates");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");

    let metal_sources = &[
        "palettized_gemv.metal",
        "palettized_gemv_swiglu.metal",
        "palettized_gemm.metal",
        "fused_gate_up.metal",
    ];
    for src in metal_sources {
        println!("cargo:rerun-if-changed={}", template_dir.join(src).display());
    }

    // Step 1: compile each .metal → .air
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

    println!("cargo:rustc-env=TRIBUNUS_METALLIB={}", metallib_path.display());
}
