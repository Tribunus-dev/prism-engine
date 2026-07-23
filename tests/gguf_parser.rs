use prism_gguf::parse_gguf_header;
use std::path::Path;

#[test]
fn test_gguf_parser() {
    let gguf_path = Path::new("tests/data/sample.gguf");
    let (_metadata, tensors) = parse_gguf_header(gguf_path).unwrap();
    assert_eq!(tensors.len(), 10);
}
