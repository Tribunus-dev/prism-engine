use std::collections::HashMap;
use std::sync::LazyLock;

/// Map PyTorch aten op names to Prism IR op names.
pub fn aten_to_prism(aten_op: &str) -> Option<&'static str> {
    OP_MAP.get(aten_op).copied()
}

static OP_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("aten::matmul", "linalg.matmul");
    m.insert("aten::linear", "linalg.matmul");
    m.insert("aten::add", "arith.addf");
    m.insert("aten::mul", "arith.mulf");
    m.insert("aten::sub", "arith.subf");
    m.insert("aten::div", "arith.divf");
    m.insert("aten::relu", "activation.relu");
    m.insert("aten::gelu", "activation.gelu");
    m.insert("aten::silu", "activation.silu");
    m.insert("aten::softmax", "activation.softmax");
    m.insert("aten::layer_norm", "normalization.layer_norm");
    m.insert("aten::rms_norm", "normalization.rms_norm");
    m.insert("aten::embedding", "lookup.embedding");
    m.insert("aten::scaled_dot_product_attention", "attention.sdpa");
    m.insert("aten::conv2d", "convolution.conv2d");
    m.insert("aten::reshape", "tensor.reshape");
    m.insert("aten::transpose", "tensor.transpose");
    m.insert("aten::cat", "tensor.cat");
    m
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_mapping() {
        assert_eq!(aten_to_prism("aten::matmul"), Some("linalg.matmul"));
    }

    #[test]
    fn test_unknown_op() {
        assert_eq!(aten_to_prism("aten::unknown_op"), None);
    }
}
