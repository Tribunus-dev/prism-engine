#[cfg(feature = "evidence")]
fn main() {
    use mlx_rs::backend::MlxBackendCapabilities;
    let caps = MlxBackendCapabilities::detect();
    match serde_json::to_string_pretty(&caps) {
        Ok(s) => println!("{}", s),
        Err(e) => {
            eprintln!("Failed to serialize capabilities: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "evidence"))]
fn main() {
    eprintln!("This example requires the 'evidence' feature");
    std::process::exit(1);
}
