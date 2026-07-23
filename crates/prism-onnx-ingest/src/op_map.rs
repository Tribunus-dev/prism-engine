use std::collections::HashMap;
use std::sync::LazyLock;

/// Map ONNX op types to prism_ir op names.
pub fn onnx_to_prism(onnx_op: &str) -> Option<&'static str> {
    OP_MAP.get(onnx_op).copied()
}

static OP_MAP: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("MatMul", "linalg.matmul");
    m.insert("Gemm", "linalg.matmul");
    m.insert("Conv", "convolution.conv2d");
    m.insert("Relu", "activation.relu");
    m.insert("Softmax", "activation.softmax");
    m.insert("Add", "arith.addf");
    m.insert("Mul", "arith.mulf");
    m.insert("Sub", "arith.subf");
    m.insert("Div", "arith.divf");
    m.insert("BatchNormalization", "normalization.batch_norm");
    m.insert("Reshape", "tensor.reshape");
    m.insert("Transpose", "tensor.transpose");
    m.insert("Concat", "tensor.cat");
    m.insert("Constant", "constant");
    m.insert("Sigmoid", "activation.sigmoid");
    m.insert("Tanh", "activation.tanh");
    m
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_mapping() {
        assert_eq!(onnx_to_prism("MatMul"), Some("linalg.matmul"));
    }

    #[test]
    fn test_unknown_op() {
        assert_eq!(onnx_to_prism("Unknown"), None);
    }
}
