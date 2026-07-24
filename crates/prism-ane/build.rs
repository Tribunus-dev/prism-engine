fn main() {
    println!("cargo:rerun-if-changed=src/coreml_bridge.mm");
    println!("cargo:rerun-if-changed=src/arena_info.h");
    #[cfg(target_os = "macos")]
    {
        cc::Build::new()
            .cpp(true)
            .file("src/coreml_bridge.mm")
            .flag_if_supported("-fobjc-arc")
            .flag("-framework")
            .flag("Foundation")
            .flag("-framework")
            .flag("CoreML")
            .compile("prism_coreml_bridge");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreML");
    }
}
