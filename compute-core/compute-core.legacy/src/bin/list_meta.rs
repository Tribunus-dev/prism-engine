fn main() {
    let path = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let (metadata, tensors) = tribunus_compute_core::gguf::parse_gguf_header(&path).unwrap();
    // Print ALL metadata keys
    for (k, v) in &metadata {
        let v_short: String = v.chars().take(80).collect();
        println!("{k}: {v_short}");
    }
    // Find K projection tensor to compute head_dim
    for t in &tensors {
        if t.name.contains("attn_k.weight") && t.shape.len() == 2 {
            println!("\n{} shape: {}×{}", t.name, t.shape[0], t.shape[1]);
        }
    }
    // Find Q projection tensor
    for t in &tensors {
        if t.name.contains("attn_q.weight") && t.shape.len() == 2 {
            println!("{} shape: {}×{}", t.name, t.shape[0], t.shape[1]);
            break;
        }
    }
}
