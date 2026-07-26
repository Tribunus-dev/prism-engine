fn main() {
    let path = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let (metadata, tensors) = tribunus_compute_core::gguf::parse_gguf_header(&path).unwrap();
    for (k, v) in &metadata {
        let vs: String = v.chars().take(80).collect();
        println!("{k}: {vs}");
    }
    for t in &tensors {
        if t.name.contains("attn_k.weight") && t.shape.len() == 2 {
            println!("\n{}: [{}×{}]", t.name, t.shape[0], t.shape[1]);
            break;
        }
    }
    for t in &tensors {
        if t.name.contains("attn_q.weight") && t.shape.len() == 2 {
            println!("{}: [{}×{}]", t.name, t.shape[0], t.shape[1]);
            break;
        }
    }
}
