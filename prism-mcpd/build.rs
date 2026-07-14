use std::hash::{DefaultHasher, Hash, Hasher};

fn main() {
    let inputs = [
        "Cargo.toml",
        "src/main.rs",
        "src/daemon.rs",
        "src/proxy.rs",
        "src/tools/mod.rs",
    ];
    let mut hasher = DefaultHasher::new();
    for input in inputs {
        println!("cargo:rerun-if-changed={input}");
        input.hash(&mut hasher);
        std::fs::read(input).unwrap_or_default().hash(&mut hasher);
    }
    println!(
        "cargo:rustc-env=PRISM_MCPD_BUILD_ID={:016x}",
        hasher.finish()
    );
}
