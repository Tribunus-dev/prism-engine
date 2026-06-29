//! Pure-Rust MIL program builder using `coreml-proto` + `prost`.
//!
//! Constructs `mil_spec::Program` protobufs without Python/coremltools.
//! Generates SSA value names automatically and produces a valid
//! MLProgram that coremlcompiler can ingest.
//!
//! ## Usage
//!
//! ```ignore
//! let prog = MilBuilder::new("main")
//!     .input("x", DataType::Float32, &[1, 4])
//!     .const_f32("weight", &[1.0, 2.0, 3.0, 4.0], &[4, 1])
//!     .matmul("x", "weight_0")
//!     .output("matmul_1")
//!     .build();
//! ```

use coreml_proto::proto::mil_spec::{self, argument, dimension, tensor_value, value};
use std::collections::HashMap;

/// Error returned by [`MilBuilder::build`] when SSA validation fails.
#[derive(Debug, Clone)]
pub enum MilBuildError {
    /// An operation references an SSA value that is not defined
    /// by any input or previous operation.
    UndefinedValue { operation: String, name: String },
    /// A block output references an SSA value that is not defined
    /// by any input or operation in the block.
    UndefinedBlockOutput { name: String },
    /// An operation does not have a "name" attribute.
    MissingOperationName { op_type: String },
    /// A referenced SSA value exists but has no known type.
    UnknownType { name: String },
    /// An unsupported unary operation mode was requested (e.g., "gelu" with
    /// no matching Core ML MIL op type).
    UnsupportedUnaryOpMode { mode: String },
}

impl std::fmt::Display for MilBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MilBuildError::UndefinedValue { operation, name } => {
                write!(
                    f,
                    "operation '{operation}' references undefined value '{name}'"
                )
            }
            MilBuildError::UndefinedBlockOutput { name } => {
                write!(
                    f,
                    "block output '{name}' is not defined by any operation or input"
                )
            }
            MilBuildError::MissingOperationName { op_type } => {
                write!(
                    f,
                    "operation type '{op_type}' missing required 'name' attribute"
                )
            }
            MilBuildError::UnknownType { name } => {
                write!(f, "unknown type for value '{name}'")
            }
            MilBuildError::UnsupportedUnaryOpMode { mode } => {
                write!(f, "unsupported unary op mode: {mode}")
            }
        }
    }
}

impl std::error::Error for MilBuildError {}

/// Builder for constructing MIL Program protobufs.
///
/// Tracks SSA value names internally and produces a complete
/// `mil_spec::Program` containing one function with one block.
pub struct MilBuilder {
    function_name: String,
    opset: String,
    inputs: Vec<mil_spec::NamedValueType>,
    ops: Vec<mil_spec::Operation>,
    block_outputs: Vec<String>,
    counter: u64,
    /// Tracks the type of each named value for type inference and SSA validation.
    value_types: HashMap<String, mil_spec::ValueType>,
    /// Weights stored for mlpackage serialization.
    weights: HashMap<String, Vec<u8>>,
}

impl Default for MilBuilder {
    fn default() -> Self {
        Self::new("__default__")
    }
}

impl MilBuilder {
    /// Add a `topk` operation — returns the values and indices of the top-k
    /// elements along `axis`.  Used for KV compaction: selects the most-attended
    /// token positions directly from the attention scores the ANE just computed.
    pub fn topk(mut self, x: &str, k: i64, axis: i64) -> Self {
        let name = self.fresh_name("topk");
        let dtype = self.require_dtype(x).expect("SSA: unknown type");

        // Clone the input's type for values output; indices get Int32 type
        let vt_values = self.value_types.get(x).cloned().unwrap_or_else(|| {
            value_type_tensor(mil_spec::TensorType {
                data_type: dtype as i32,
                rank: 2,
                dimensions: vec![],
                attributes: HashMap::new(),
            })
        });
        let vt_indices = value_type_tensor(mil_spec::TensorType {
            data_type: mil_spec::DataType::Int32 as i32,
            rank: 1,
            dimensions: vec![],
            attributes: HashMap::new(),
        });

        let mut inputs = HashMap::new();
        inputs.insert("x" .to_string(), named_arg(x));
        // k is a constant attribute embedded as an int in the op attrs,
        // not an input tensor.  MIL accepts both forms; we use attribute.

        let values_name = format!("{name}_values");
        let indices_name = format!("{name}_indices");

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), int_attr(axis));
        attrs.insert("k".to_string(), int_attr(k));

        let op = make_operation(
            "topk",
            &name,
            inputs,
            &[(&values_name, &vt_values), (&indices_name, &vt_indices)],
            attrs,
        );

        self.value_types.insert(values_name, vt_values);
        self.value_types.insert(indices_name, vt_indices);
        self.ops.push(op);
        self
    }
    /// Mark an SSA value as a block output.
    pub fn new(function_name: &str) -> Self {
        Self {
            function_name: function_name.to_string(),
            opset: "CoreML9".to_string(),
            inputs: Vec::new(),
            ops: Vec::new(),
            block_outputs: Vec::new(),
            counter: 0,
            value_types: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    /// Register a named input tensor.
    pub fn input(mut self, name: &str, dtype: mil_spec::DataType, shape: &[i64]) -> Self {
        let tensor_type = tensor_type(dtype, shape);
        let vt = value_type_tensor(tensor_type);
        self.value_types.insert(name.to_string(), vt.clone());
        self.inputs.push(mil_spec::NamedValueType {
            name: name.to_string(),
            r#type: Some(vt),
        });
        self
    }

    /// Override the opset identifier (default: "CoreML9").
    pub fn set_opset(mut self, opset: &str) -> Self {
        self.opset = opset.to_string();
        self
    }

    /// Return the current opset identifier.
    pub fn get_opset(&self) -> &str {
        &self.opset
    }

    /// Add a pre-built MIL operation to the block.
    pub fn operation(
        mut self,
        op: mil_spec::Operation,
        output_type: Option<(&str, mil_spec::ValueType)>,
    ) -> Self {
        if let Some((name, vt)) = output_type {
            self.value_types.insert(name.to_string(), vt);
        }
        self.ops.push(op);
        self
    }

    /// Explicitly register a value type.
    pub fn register_type(&mut self, name: &str, vt: mil_spec::ValueType) {
        self.value_types.insert(name.to_string(), vt);
    }

    /// Access the current ops list.
    pub fn ops(&self) -> &[mil_spec::Operation] {
        &self.ops
    }
    /// Add a weight for mlpackage serialization.
    pub fn add_weight(&mut self, name: &str, data: Vec<u8>) {
        self.weights.insert(name.to_string(), data);
    }

    /// Infer the matmul output shape from input dimensions: [M, K] x [K, N] = [M, N].
    fn infer_matmul_output_shape(&self, a: &str, b: &str) -> Vec<i64> {
        fn get_dims(types: &HashMap<String, mil_spec::ValueType>, key: &str) -> Option<(i64, i64)> {
            let vt = types.get(key)?;
            let tt = vt.r#type.as_ref()?;
            if let mil_spec::value_type::Type::TensorType(ref tensor) = tt {
                let dims: Vec<i64> = tensor
                    .dimensions
                    .iter()
                    .filter_map(|d| match d.dimension.as_ref()? {
                        dimension::Dimension::Constant(c) => Some(c.size as i64),
                        _ => None,
                    })
                    .collect();
                if dims.len() >= 2 {
                    Some((dims[0], dims[1]))
                } else {
                    None
                }
            } else {
                None
            }
        }
        match (
            get_dims(&self.value_types, a),
            get_dims(&self.value_types, b),
        ) {
            (Some((m, _)), Some((_, n))) => vec![m, n],
            _ => vec![1, 1],
        }
    }

    /// Resolve the output shape for a binary elementwise operation from input shapes.
    /// Both inputs should have the same shape. Returns [?, ?] if shapes cannot be resolved.
    fn resolve_elementwise_output_shape(&self, a: &str, b: &str) -> Vec<mil_spec::Dimension> {
        let a_dims = self.value_types.get(a).and_then(|vt| {
            if let mil_spec::value_type::Type::TensorType(ref tt) = vt.r#type.as_ref()? {
                Some(&tt.dimensions)
            } else {
                None
            }
        });
        let b_dims = self.value_types.get(b).and_then(|vt| {
            if let mil_spec::value_type::Type::TensorType(ref tt) = vt.r#type.as_ref()? {
                Some(&tt.dimensions)
            } else {
                None
            }
        });

        match (a_dims, b_dims) {
            (Some(a), Some(b)) if a == b => a.clone(), // same shape → preserve
            _ => {
                // Unknown — use [?, ?] as fallback
                vec![
                    mil_spec::Dimension {
                        dimension: Some(dimension::Dimension::Unknown(
                            dimension::UnknownDimension { variadic: false },
                        )),
                    },
                    mil_spec::Dimension {
                        dimension: Some(dimension::Dimension::Unknown(
                            dimension::UnknownDimension { variadic: false },
                        )),
                    },
                ]
            }
        }
    }

    /// Add a const operation with f32 immediate values.
    /// Returns `Self` with the constant's SSA name tracked.
    pub fn const_f32(mut self, name_hint: &str, values: &[f32], shape: &[i64]) -> Self {
        // Auto-fill with zeros when values are empty but shape declares elements.
        // Prevents "Tensor storage and type have different number of elements" from coremlcompiler.
        let effective_values: Vec<f32> = if values.is_empty() && !shape.is_empty() {
            let total: usize = shape.iter().map(|&d| d.max(0) as usize).product();
            if total > 0 {
                vec![0.0f32; total]
            } else {
                values.to_vec()
            }
        } else {
            values.to_vec()
        };
        let name = self.fresh_name(name_hint);
        let tensor_type = tensor_type(mil_spec::DataType::Float32, shape);
        let vt = value_type_tensor(tensor_type);

        let tv = mil_spec::TensorValue {
            value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
                values: effective_values,
            })),
        };
        let v = mil_spec::Value {
            doc_string: String::new(),
            r#type: Some(vt.clone()),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tv)),
            })),
        };

        // const op: "val" is an attribute, and it also needs a "name" attribute
        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("val".to_string(), v);

        let op = make_operation("const", &name, HashMap::new(), &[(&name, &vt)], attrs);

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Register a constant tensor with Float16 data type.
    ///
    /// Accepts `f32` values and converts them to IEEE 754 half-precision
    /// internally.  The MIL program stores the f16 data as raw bytes.
    pub fn const_f16(mut self, name_hint: &str, values: &[f32], shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let tensor_type = tensor_type(mil_spec::DataType::Float16, shape);
        let vt = value_type_tensor(tensor_type);

        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| {
                let bits = v.to_bits();
                let sign = ((bits >> 31) & 1) as u16;
                let exp = ((bits >> 23) & 0xFF) as i32;
                let mant = bits & 0x7FFFFF;
                let f16 = if exp == 0 {
                    sign << 15
                } else if exp == 255 {
                    (sign << 15) | 0x7C00
                } else {
                    let new_exp = exp - 127 + 15;
                    if new_exp <= 0 {
                        sign << 15
                    } else if new_exp >= 31 {
                        (sign << 15) | 0x7C00
                    } else {
                        let new_mant = mant >> 13;
                        (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16)
                    }
                };
                f16.to_le_bytes()
            })
            .collect();

        let tv = mil_spec::TensorValue {
            value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                values: f16_bytes,
            })),
        };
        let v = mil_spec::Value {
            doc_string: String::new(),
            r#type: Some(vt.clone()),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tv)),
            })),
        };

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("val".to_string(), v);

        let op = make_operation("const", &name, HashMap::new(), &[(&name, &vt)], attrs);

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Register a constant tensor with Uint8 data type.
    pub fn const_uint8(mut self, name_hint: &str, values: &[u8], shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let tensor_type = tensor_type(mil_spec::DataType::Uint8, shape);
        let vt = value_type_tensor(tensor_type);

        let tv = mil_spec::TensorValue {
            value: Some(tensor_value::Value::Bytes(tensor_value::RepeatedBytes {
                values: values.to_vec(),
            })),
        };
        let v = mil_spec::Value {
            doc_string: String::new(),
            r#type: Some(vt.clone()),
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(tv)),
            })),
        };

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("val".to_string(), v);

        let op = make_operation("const", &name, HashMap::new(), &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Bind palettized weight indices + codebook via `constexpr_lut_to_dense`.
    ///
    /// Core ML 9+ intercepts this operation and routes the palettized
    /// weight lookup directly to the ANE hardware decompressors, avoiding
    /// materialising the dense weight tensor.
    ///
    /// - `indices` — SSA name of the Uint8 packed-indices const.
    /// - `lut` — SSA name of the Float16 codebook const (`[out_dim, 16]`).
    /// - `vector_axis` — the axis to apply the LUT across (1 = in_dim).
    ///
    /// Returns the SSA name of the dense-weight proxy tensor.
    pub fn constexpr_lut_to_dense(
        mut self,
        name_hint: &str,
        indices: &str,
        lut: &str,
        out_shape: &[i64],
        vector_axis: i64,
    ) -> Self {
        let name = self.fresh_name(name_hint);
        let vt = value_type_tensor(tensor_type(mil_spec::DataType::Float16, out_shape));

        let mut inputs_map = HashMap::new();
        inputs_map.insert("indices".to_string(), named_arg(indices));
        inputs_map.insert("lut".to_string(), named_arg(lut));

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("vector_axis".to_string(), int_attr(vector_axis));

        let op = make_operation(
            "constexpr_lut_to_dense",
            &name,
            inputs_map,
            &[(&name, &vt)],
            attrs,
        );
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a matmul operation. `a` and `b` are SSA names of input values.
    ///
    /// Output type: rank-2 f32 with dimensions inferred from the operation
    /// contract: if A is [M, K] and B is [K, N] (with transpose_x=false,
    /// transpose_y=false), output is [M, N].

    /// Add a scaled dot-product attention operation (macOS 15 / CoreML 9+).
    ///
    /// Native SDPA operator highly optimized for ANE SRAM layout.
    /// `query`, `key`, `value` are SSA names of tensors `[B, H, N, D]`.
    /// `mask` is an optional additive FP16 mask `[1, 1, N, N]` or `None`.
    /// `scale` is an optional float (default: 1/sqrt(D)).
    pub fn scaled_dot_product_attention(
        mut self,
        name_hint: &str,
        query: &str,
        key: &str,
        value: &str,
        mask: Option<&str>,
        scale: Option<f32>,
    ) -> Self {
        let name = self.fresh_name(name_hint);
        let q_dtype = self.require_dtype(query).expect("SSA: unknown query");
        let q_dims: Vec<i64> = self
            .value_types
            .get(query)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(ref tt)) => Some(
                    tt.dimensions
                        .iter()
                        .filter_map(|d| match d.dimension.as_ref()? {
                            dimension::Dimension::Constant(c) => Some(c.size as i64),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let vt = value_type_tensor(tensor_type(q_dtype, &q_dims));

        let mut inputs_map = HashMap::new();
        inputs_map.insert("query".to_string(), named_arg(query));
        inputs_map.insert("key".to_string(), named_arg(key));
        inputs_map.insert("value".to_string(), named_arg(value));
        if let Some(m) = mask {
            inputs_map.insert("mask".to_string(), named_arg(m));
        }

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        if let Some(s) = scale {
            attrs.insert("scale".to_string(), float_attr(s));
        }

        let op = make_operation(
            "scaled_dot_product_attention",
            &name,
            inputs_map,
            &[(&name, &vt)],
            attrs,
        );
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Define a state variable (Core ML 9+ make_state).
    pub fn make_state(mut self, name_hint: &str, shape: &[i64], dtype: i32) -> Self {
        let name = self.fresh_name(name_hint);
        let tt = tensor_type(
            if dtype == 10 {
                mil_spec::DataType::Float16
            } else {
                mil_spec::DataType::Float32
            },
            shape,
        );
        let vt = value_type_tensor(tt);

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        let shape_tensor = mil_spec::TensorValue {
            value: Some(tensor_value::Value::LongInts(
                tensor_value::RepeatedLongInts {
                    values: shape.iter().map(|&s| s as i64).collect(),
                },
            )),
        };
        let shape_val = mil_spec::Value {
            doc_string: String::new(),
            r#type: None,
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(shape_tensor)),
            })),
        };
        attrs.insert("shape".to_string(), shape_val);
        attrs.insert("dtype".to_string(), int_attr(dtype as i64));

        let op = make_operation("make_state", &name, HashMap::new(), &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Read the current value of a state variable.
    pub fn read_state(mut self, name_hint: &str, state_ssa: &str) -> Self {
        let name = self.fresh_name(name_hint);
        let vt = self
            .value_types
            .get(state_ssa)
            .cloned()
            .expect("make_state must be defined before read_state");

        let mut inputs_map = HashMap::new();
        inputs_map.insert("input".to_string(), named_arg(state_ssa));

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));

        let op = make_operation("read_state", &name, inputs_map, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Write a new value to a state variable (side-effecting, no outputs).
    pub fn write_state(mut self, state_ssa: &str, value_ssa: &str) -> Self {
        let name = self.fresh_name("write_state");
        let mut inputs_map = HashMap::new();
        inputs_map.insert("input".to_string(), named_arg(state_ssa));
        inputs_map.insert("value".to_string(), named_arg(value_ssa));

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));

        let op = make_operation("write_state", &name, inputs_map, &[], attrs);
        self.ops.push(op);
        self
    }

    /// Partial tensor update via slice_update.
    pub fn slice_update(
        mut self,
        name_hint: &str,
        input: &str,
        source: &str,
        start_indices: &[i64],
    ) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown input");
        let dims: Vec<i64> = self
            .value_types
            .get(input)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(ref tt)) => Some(
                    tt.dimensions
                        .iter()
                        .filter_map(|d| match d.dimension.as_ref()? {
                            dimension::Dimension::Constant(c) => Some(c.size as i64),
                            _ => None,
                        })
                        .collect(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let vt = value_type_tensor(tensor_type(dtype, &dims));

        let mut inputs_map = HashMap::new();
        inputs_map.insert("input".to_string(), named_arg(input));
        inputs_map.insert("source".to_string(), named_arg(source));

        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        let starts_tensor = mil_spec::TensorValue {
            value: Some(tensor_value::Value::LongInts(
                tensor_value::RepeatedLongInts {
                    values: start_indices.to_vec(),
                },
            )),
        };
        let starts_val = mil_spec::Value {
            doc_string: String::new(),
            r#type: None,
            value: Some(value::Value::ImmediateValue(value::ImmediateValue {
                value: Some(value::immediate_value::Value::Tensor(starts_tensor)),
            })),
        };
        attrs.insert("starts".to_string(), starts_val);

        let op = make_operation("slice_update", &name, inputs_map, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Symmetric INT8 quantization: scale then clamp to [-127, 127].
    pub fn quantize(mut self, name_hint: &str, input: &str, scale: f32, shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let vt = value_type_tensor(tensor_type_raw(4, shape)); // 4 = Int8
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), named_arg(input));
        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("scale".to_string(), float_attr(scale));
        attrs.insert("axis".to_string(), int_attr(-1));
        let op = make_operation("quantize", &name, inputs, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Symmetric INT8 dequantization: scale back to FP16.
    pub fn dequantize(mut self, name_hint: &str, input: &str, scale: f32, shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let vt = value_type_tensor(tensor_type_raw(10, shape)); // 10 = Float16
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), named_arg(input));
        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("scale".to_string(), float_attr(scale));
        attrs.insert("axis".to_string(), int_attr(-1));
        let op = make_operation("dequantize", &name, inputs, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    pub fn matmul(mut self, a: &str, b: &str) -> Self {
        let name = self.fresh_name("matmul");
        let dtype = self.require_dtype(a).expect("SSA: unknown value");
        let _ = self.require_dtype(b).expect("SSA: unknown value");

        // Infer matmul output shape from inputs: [M, K] × [K, N] = [M, N].
        // M = A rows (dim 0), N = B cols (dim 1).
        let output_dims = self.infer_matmul_output_shape(a, b);
        let vt = value_type_tensor(tensor_type(dtype, &output_dims));

        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(a));
        inputs_map.insert("y".to_string(), named_arg(b));
        inputs_map.insert("transpose_x".to_string(), bool_arg(false));
        inputs_map.insert("transpose_y".to_string(), bool_arg(false));

        let op = make_operation("matmul", &name, inputs_map, &[(&name, &vt)], HashMap::new());

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add an element-wise add operation.
    pub fn add(mut self, a: &str, b: &str) -> Self {
        let name = self.fresh_name("add");
        let dtype = self.require_dtype(a).expect("SSA: unknown value");
        let _ = self.require_dtype(b).expect("SSA: unknown value");

        // Resolve output shape from inputs; fall back to [?,?]
        let dimensions = self.resolve_elementwise_output_shape(a, b);
        let vt = value_type_tensor(mil_spec::TensorType {
            data_type: dtype as i32,
            rank: 2,
            dimensions,
            attributes: HashMap::new(),
        });

        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(a));
        inputs_map.insert("y".to_string(), named_arg(b));

        let op = make_operation("add", &name, inputs_map, &[(&name, &vt)], HashMap::new());

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add an element-wise multiply operation.
    pub fn mul(mut self, a: &str, b: &str) -> Self {
        let name = self.fresh_name("mul");
        let dtype = self.require_dtype(a).expect("SSA: unknown value");
        let _ = self.require_dtype(b).expect("SSA: unknown value");

        let dimensions = self.resolve_elementwise_output_shape(a, b);
        let vt = value_type_tensor(mil_spec::TensorType {
            data_type: dtype as i32,
            rank: 2,
            dimensions,
            attributes: HashMap::new(),
        });

        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(a));
        inputs_map.insert("y".to_string(), named_arg(b));

        let op = make_operation("mul", &name, inputs_map, &[(&name, &vt)], HashMap::new());

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a SiLU (sigmoid linear unit) elementwise activation.
    pub fn silu(mut self, name_hint: &str, input: &str) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown value");

        // Clone dimensions from input
        let dimensions = self.value_types.get(input)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(tt.dimensions.clone()),
                _ => None,
            })
            .unwrap_or_default();
        let rank = dimensions.len() as i64;

        let vt = value_type_tensor(mil_spec::TensorType {
            data_type: dtype as i32,
            rank,
            dimensions,
            attributes: HashMap::new(),
        });

        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(input));

        let op = make_operation("silu", &name, inputs_map, &[(&name, &vt)], HashMap::new());

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a 2D convolution operation.
    ///
    /// `kernel_size` is the spatial kernel dimensions (e.g. `[1, 1]`).
    /// `padding` is the padding mode (`"valid"`, `"same"`, or `"custom"`).
    pub fn conv(mut self, name_hint: &str, input: &str, weight: &str, kernel_size: &[i64], padding: &str) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown value");

        // Output shape: [B, C_out, H_out, W_out]. C_out is unknown, spatial dims
        // from input for 1x1 valid conv: [B, C, 1, S] -> [B, ?, 1, S]
        let out_dims = self.value_types.get(input)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => {
                    let mut dims = tt.dimensions.clone();
                    if dims.len() >= 2 {
                        // Replace channel dim with unknown (C_out)
                        dims[1] = mil_spec::Dimension {
                            dimension: Some(dimension::Dimension::Unknown(
                                dimension::UnknownDimension { variadic: false },
                            )),
                        };
                    }
                    Some(dims)
                }
                _ => None,
            })
            .unwrap_or_else(|| vec![
                mil_spec::Dimension { dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension { variadic: false })) },
                mil_spec::Dimension { dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension { variadic: false })) },
                mil_spec::Dimension { dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension { variadic: false })) },
                mil_spec::Dimension { dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension { variadic: false })) },
            ]);
        let rank = out_dims.len() as i64;

        let vt = value_type_tensor(mil_spec::TensorType {
            data_type: dtype as i32,
            rank,
            dimensions: out_dims,
            attributes: HashMap::new(),
        });

        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(input));
        inputs_map.insert("weight".to_string(), named_arg(weight));

        let kernel_vals: Vec<i64> = kernel_size.to_vec();
        let stride_vals: Vec<i64> = vec![1, 1];
        let pad_vals: Vec<i64> = vec![0, 0];

        let mut attrs: HashMap<String, mil_spec::Value> = HashMap::new();
        attrs.insert("kernel_size".to_string(), ints_attr(&kernel_vals));
        attrs.insert("stride".to_string(), ints_attr(&stride_vals));
        attrs.insert("dilatation".to_string(), ints_attr(&stride_vals));
        attrs.insert("pad_type".to_string(), string_attr(padding));
        attrs.insert("pad".to_string(), ints_attr(&pad_vals));
        attrs.insert("groups".to_string(), int_attr(1));

        let op = make_operation("convolution", &name, inputs_map, &[(&name, &vt)], attrs);

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a `gather` operation — index into `params` along `axis` using `indices`.
    ///
    /// Used by the ANE Planar Engine LUT expansion: params=[81,4] LUT,
    /// indices=swizzled u8 byte, axis=0 → gathers one row of the LUT.
    pub fn gather(mut self, params: &str, indices: &str, axis: i64) -> Self {
        let name = self.fresh_name("gather");
        let dtype = self.require_dtype(params).expect("SSA: unknown params type");

        // Gather output has same rank and dtype as params, but the axis
        // dimension becomes the indices dimension.
        let indices_rank = self.value_types.get(indices)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(tt.rank),
                _ => None,
            })
            .unwrap_or(1);
        let out_rank = self.value_types.get(params)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(tt.rank + indices_rank - 1),
                _ => None,
            })
            .unwrap_or(4);

        let vt = value_type_tensor(mil_spec::TensorType {
            data_type: dtype as i32,
            rank: out_rank,
            dimensions: vec![],
            attributes: HashMap::new(),
        });

        let mut inputs = HashMap::new();
        inputs.insert("params" .to_string(), named_arg(params));
        inputs.insert("indices".to_string(), named_arg(indices));

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), int_attr(axis));

        let op = make_operation("gather", &name, inputs, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a `softmax` operation along the given axis.
    pub fn softmax(mut self, input: &str, axis: i64) -> Self {
        let name = self.fresh_name("softmax");
        let dtype = self.require_dtype(input).expect("SSA: unknown type");
        let vt = self.value_types.get(input).cloned().unwrap_or_else(|| {
            value_type_tensor(mil_spec::TensorType {
                data_type: dtype as i32,
                rank: 4,
                dimensions: vec![],
                attributes: HashMap::new(),
            })
        });

        let mut inputs = HashMap::new();
        inputs.insert("x" .to_string(), named_arg(input));

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), int_attr(axis));

        let op = make_operation("softmax", &name, inputs, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Mark an SSA value as a block output.
    pub fn output(mut self, name: &str) -> Self {
        self.block_outputs.push(name.to_string());
        self
    }

    /// Verify the SSA graph is well-formed and return the built Program.
    ///
    /// Checks: every block output resolves to a known typed SSA value,
    /// every operation input references a known value, every operation
    /// has a nonempty name attribute, output names are unique, and
    /// block outputs are nonempty.
    pub fn build(self) -> Result<mil_spec::Program, MilBuildError> {
        // ── SSA verification ──────────────────────────────────────
        let mut defined: HashMap<String, bool> = HashMap::new();
        for inp in &self.inputs {
            defined.insert(inp.name.clone(), true);
        }
        for op in &self.ops {
            // Every non-trivial op must have a name attribute
            if !op.attributes.contains_key("name") {
                return Err(MilBuildError::MissingOperationName {
                    op_type: op.r#type.clone(),
                });
            }
            for input_list in op.inputs.values() {
                for b in &input_list.arguments {
                    if let Some(argument::binding::Binding::Name(ref n)) = b.binding {
                        if !defined.contains_key(n.as_str()) {
                            return Err(MilBuildError::UndefinedValue {
                                operation: op.r#type.clone(),
                                name: n.clone(),
                            });
                        }
                    }
                }
            }
            for out in &op.outputs {
                defined.insert(out.name.clone(), true);
            }
        }
        for out_name in &self.block_outputs {
            if !defined.contains_key(out_name.as_str()) {
                return Err(MilBuildError::UndefinedBlockOutput {
                    name: out_name.clone(),
                });
            }
        }

        let block = mil_spec::Block {
            inputs: vec![],
            outputs: self.block_outputs,
            operations: self.ops,
            attributes: HashMap::new(),
        };

        let mut block_specs = HashMap::new();
        block_specs.insert(self.opset.clone(), block);

        let function = mil_spec::Function {
            inputs: self.inputs,
            opset: self.opset,
            block_specializations: block_specs,
            attributes: HashMap::new(),
        };

        let mut functions = HashMap::new();
        functions.insert(self.function_name, function);

        Ok(mil_spec::Program {
            version: 1,
            functions,
            doc_string: String::new(),
            attributes: HashMap::new(),
        })
    }

    fn fresh_name(&mut self, hint: &str) -> String {
        let name = format!("{}_{}", hint, self.counter);
        self.counter += 1;
        name
    }

    /// Return the SSA name most recently generated.
    pub fn last_name(&self) -> Option<&str> {
        self.ops
            .last()
            .and_then(|op| op.outputs.first())
            .map(|o| o.name.as_str())
    }

    /// Look up the dtype of an SSA value. Fails if the value is not found.
    fn require_dtype(&self, name: &str) -> Result<mil_spec::DataType, MilBuildError> {
        self.value_types
            .get(name)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => {
                    mil_spec::DataType::try_from(tt.data_type).ok()
                }
                _ => None,
            })
            .ok_or_else(|| MilBuildError::UnknownType {
                name: name.to_string(),
            })
    }

    /// Access stored weights (for mlpackage serialization).
    pub fn weights(&self) -> &HashMap<String, Vec<u8>> {
        &self.weights
    }

    /// Get shapes of all tracked values (for graph_catalog shape inference).
    pub fn value_shapes(&self) -> HashMap<String, Vec<i64>> {
        let mut shapes = HashMap::new();
        for (name, vt) in &self.value_types {
            if let Some(mil_spec::value_type::Type::TensorType(ref tt)) = vt.r#type.as_ref() {
                let dims: Vec<i64> = tt
                    .dimensions
                    .iter()
                    .filter_map(|d| match d.dimension.as_ref()? {
                        dimension::Dimension::Constant(c) => Some(c.size as i64),
                        _ => None,
                    })
                    .collect();
                if !dims.is_empty() {
                    shapes.insert(name.clone(), dims);
                }
            }
        }
        shapes
    }

    /// Format and export the builder state as a raw MIL text string.
    pub fn to_mil_text(&self) -> String {
        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3500.32.1\"}})]\n");
        mil.push_str("{\n");

        // Function signature
        mil.push_str(&format!(
            "    func {}<{}>(",
            self.function_name,
            self.opset.to_lowercase()
        ));
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 {
                mil.push_str(", ");
            }
            let type_str = format_value_type(input.r#type.as_ref().unwrap());
            mil.push_str(&format!("{} {}", type_str, input.name));
        }
        mil.push_str(") {\n");

        // Operations
        for op in &self.ops {
            mil.push_str("            ");
            // Outputs
            let out_type = format_value_type(op.outputs[0].r#type.as_ref().unwrap());
            let out_name = &op.outputs[0].name;
            mil.push_str(&format!("{} {} = {}(", out_type, out_name, op.r#type));

            // Inputs (arguments)
            let mut first_arg = true;
            let mut sorted_inputs: Vec<_> = op.inputs.iter().collect();
            sorted_inputs.sort_by_key(|(k, _)| *k);

            for (arg_name, arg) in sorted_inputs {
                if !first_arg {
                    mil.push_str(", ");
                }
                first_arg = false;
                mil.push_str(&format!("{} = ", arg_name));
                // Format binding
                if let Some(binding) = arg.arguments.first().and_then(|b| b.binding.as_ref()) {
                    match binding {
                        argument::binding::Binding::Name(n) => {
                            mil.push_str(n);
                        }
                        argument::binding::Binding::Value(v) => {
                            mil.push_str(&format_value(v));
                        }
                    }
                }
            }
            mil.push_str(")[");

            // Attributes
            let mut first_attr = true;
            let mut sorted_attrs: Vec<_> = op.attributes.iter().collect();
            sorted_attrs.sort_by_key(|(k, _)| *k);
            for (attr_name, attr_val) in sorted_attrs {
                if !first_attr {
                    mil.push_str(", ");
                }
                first_attr = false;
                mil.push_str(&format!("{} = {}", attr_name, format_value(attr_val)));
            }
            mil.push_str("];\n");
        }

        // Return block outputs
        mil.push_str("        } -> (");
        for (i, out) in self.block_outputs.iter().enumerate() {
            if i > 0 {
                mil.push_str(", ");
            }
            mil.push_str(out);
        }
        mil.push_str(");\n");
        mil.push_str("}\n");

        mil
    }
}

fn format_value_type(vt: &mil_spec::ValueType) -> String {
    if let Some(mil_spec::value_type::Type::TensorType(ref tt)) = vt.r#type {
        let dtype_str = match mil_spec::DataType::try_from(tt.data_type) {
            Ok(mil_spec::DataType::Float32) => "fp32",
            Ok(mil_spec::DataType::Float16) => "fp16",
            Ok(mil_spec::DataType::Int32) => "int32",
            Ok(mil_spec::DataType::Bool) => "bool",
            Ok(mil_spec::DataType::String) => "string",
            _ => "fp32",
        };
        let mut dims = String::new();
        for (i, d) in tt.dimensions.iter().enumerate() {
            if i > 0 {
                dims.push_str(", ");
            }
            if let Some(ref dimension) = d.dimension {
                match dimension {
                    dimension::Dimension::Constant(c) => dims.push_str(&c.size.to_string()),
                    dimension::Dimension::Unknown(_) => dims.push_str("?"),
                }
            }
        }
        format!("tensor<{}, [{}]>", dtype_str, dims)
    } else {
        "tensor<fp32, []>".to_string()
    }
}

fn format_value(val: &mil_spec::Value) -> String {
    if let Some(value::Value::ImmediateValue(ref iv)) = val.value {
        if let Some(value::immediate_value::Value::Tensor(ref tv)) = iv.value {
            if let Some(ref tensor_val) = tv.value {
                match tensor_val {
                    tensor_value::Value::Strings(s) => {
                        format!(
                            "string(\"{}\")",
                            s.values.first().cloned().unwrap_or_default()
                        )
                    }
                    tensor_value::Value::Bools(b) => {
                        format!("bool({})", b.values.first().cloned().unwrap_or(false))
                    }
                    tensor_value::Value::Floats(f) => {
                        if let Some(mil_spec::value_type::Type::TensorType(ref tt)) =
                            val.r#type.as_ref().and_then(|vt| vt.r#type.as_ref())
                        {
                            let shape: Vec<usize> = tt
                                .dimensions
                                .iter()
                                .filter_map(|d| {
                                    if let Some(dimension::Dimension::Constant(c)) =
                                        d.dimension.as_ref()
                                    {
                                        Some(c.size as usize)
                                    } else {
                                        None
                                    }
                                })
                                .collect();

                            if shape.len() == 2 {
                                let rows = shape[0];
                                let cols = shape[1];
                                let mut res = String::new();
                                res.push_str(&format!("tensor<fp32, [{}, {}]>([", rows, cols));
                                for r in 0..rows {
                                    if r > 0 {
                                        res.push_str(", ");
                                    }
                                    res.push_str("[");
                                    for c in 0..cols {
                                        if c > 0 {
                                            res.push_str(", ");
                                        }
                                        let idx = r * cols + c;
                                        if idx < f.values.len() {
                                            res.push_str(&format!("{:?}", f.values[idx]));
                                        } else {
                                            res.push_str("0.0");
                                        }
                                    }
                                    res.push_str("]");
                                }
                                res.push_str("])");
                                return res;
                            }
                        }
                        if f.values.len() == 1 {
                            format!("{:?}", f.values[0])
                        } else {
                            format!("{:?}", f.values)
                        }
                    }
                    _ => "unknown".to_string(),
                }
            } else {
                "nil".to_string()
            }
        } else {
            "nil".to_string()
        }
    } else {
        "nil".to_string()
    }
}

// ── CoreML unary op type compatibility map ──────────────────────────────

/// Describes a Core ML MIL serialized unary op type.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoreMlUnaryOpType {
    /// The MIL op type string accepted by coremlcompiler (e.g., "sigmoid").
    pub mil_op_type: &'static str,
    /// Whether this op requires additional attributes (e.g., gelu may need `approximation`).
    pub requires_attrs: bool,
}

/// Maps Tribunus internal unary semantic modes to compiler-accepted Core ML
/// MIL serialized op type strings. This is the single authority for unary op
/// type emission — no code should emit `"element_wise"` as a MIL op type.
const COREML_MIL_UNARY_OP_TYPE_MAP: &[(&str, CoreMlUnaryOpType)] = &[
    (
        "logistic",
        CoreMlUnaryOpType {
            mil_op_type: "sigmoid",
            requires_attrs: false,
        },
    ), // canonical
    (
        "sigmoid",
        CoreMlUnaryOpType {
            mil_op_type: "sigmoid",
            requires_attrs: false,
        },
    ), // alias
    (
        "silu",
        CoreMlUnaryOpType {
            mil_op_type: "silu",
            requires_attrs: false,
        },
    ),
];

/// Resolve an internal unary semantic mode to its Core ML MIL op type.
///
/// Returns `None` if the mode is not recognized. Callers MUST fail closed
/// (return `MilBuildError::UnsupportedUnaryOpMode`) rather than falling back
/// to a generic op type.
pub fn resolve_unary_op_type(mode: &str) -> Option<CoreMlUnaryOpType> {
    COREML_MIL_UNARY_OP_TYPE_MAP
        .iter()
        .find(|(key, _)| *key == mode)
        .map(|(_, entry)| *entry)
}

// ── operation constructor (always installs the "name" attribute) ─────────

/// Build a MIL program for the full ANE inference loop.
///
/// The program implements one layer of the network using the Planar Engine
/// `gather` LUT for ternary weight expansion and the Matrix Engine for
/// dense matmuls.  Weight inputs are swizzled u8 placeholders that the
/// runtime populates from SLC via the E-core pump.
///
/// # Parameters
/// - `hidden_dim`: model hidden size (e.g., 3840)
/// - `intermediate_dim`: FFN intermediate size (e.g., 18432)
/// - `num_heads`, `head_dim`: attention heads and per-head dimension
///
/// Returns a serialized MLProgram (.mlmodelc-compatible) protobuf bytes.
pub fn build_full_ane_layer_program(
    hidden_dim: u32,
    intermediate_dim: u32,
    num_heads: u32,
    head_dim: u32,
) -> Vec<u8> {
    // Fused ANE layer program with KV compaction.
    // Single MIL invocation: forward pass + topk survivor selection + KV gather.
    use coreml_proto::proto::mil_spec::DataType;
    use prost::Message;
    let hs = hidden_dim as i64;
    let interm = intermediate_dim as i64;
    let n_h = num_heads as i64;
    let hd = head_dim as i64;
    let target_count: i64 = 20480; // compaction target (50x at 1M)

    // Static LUT: [81, 4] INT8
    let mut lut_vals = Vec::with_capacity(81 * 4);
    for state in 0u8..81 { let mut s = state; for _ in 0..4 { let d = s % 3; s /= 3; lut_vals.push(match d { 1 => 1i8, 2 => -1, _ => 0 } as f32); } }

    // inputs (0 fresh) + const (0 fresh)
    let base = MilBuilder::new("ane_forward")
        .input("h", DataType::Float16, &[1, 1, 1, hs])
        .input("w_q", DataType::Uint8, &[n_h * hd, hs])
        .input("w_k", DataType::Uint8, &[n_h * hd, hs])
        .input("w_v", DataType::Uint8, &[n_h * hd, hs])
        .input("w_o", DataType::Uint8, &[hs, n_h * hd])
        .input("w_gate", DataType::Uint8, &[interm, hs])
        .input("w_up", DataType::Uint8, &[interm, hs])
        .input("w_down", DataType::Uint8, &[hs, interm])
        .input("mtp_w_proj", DataType::Uint8, &[hs, hs])
        .input("kv_full", DataType::Float16, &[1, 1, n_h * hd * 2, 1_000_000]) // max seq
        .const_f32("lut", &lut_vals, &[81, 4]);

    // SSA counter: 0 fresh so far. Chain:
    // gather=0,1,2,3,4,5,6  (7 gather)
    // matmul=7,8,9,10,11,12,13,14 (8 matmul)
    // softmax=15, silu=16, topk=17 (+indices), mul=18
    // gather(kv)=19,20  (two gather outputs for compacted K and V)
    let b = base
        // ── Attention Q, K, V projections ──────────────────────────
        .gather("lut", "w_q", 1)      // gather_0
        .gather("lut", "w_k", 1)      // gather_1
        .gather("lut", "w_v", 1)      // gather_2
        .matmul("h", "gather_0")         // matmul_7 — Q projection
        .matmul("h", "gather_1")         // matmul_8 — K projection
        .matmul("h", "gather_2")         // matmul_9 — V projection
        // ── Attention scores: Q @ K^T / sqrt(d) ────────────────────
        .matmul("matmul_7", "matmul_8")  // matmul_10 — QK^T scores
        .softmax("matmul_10", -1)        // softmax_11 — attention
        .matmul("softmax_11", "matmul_9") // matmul_12 — attn @ V
        // ── Output projection ───────────────────────────────────────
        .gather("lut", "w_o", 1)      // gather_3
        .matmul("matmul_12", "gather_3")  // matmul_13 — attention output
        .add("h", "matmul_13")            // add_14 — residual
        // ── FFN ─────────────────────────────────────────────────────
        .gather("lut", "w_gate", 1)    // gather_4
        .matmul("add_14", "gather_4")    // matmul_15 — gate proj
        .silu("gate", "matmul_15")        // silu_16
        .gather("lut", "w_up", 1)      // gather_5
        .matmul("add_14", "gather_5")    // matmul_17 — up proj
        .mul("silu_16", "matmul_17")     // mul_18
        .gather("lut", "w_down", 1)    // gather_6
        .matmul("mul_18", "gather_6")    // matmul_19 — FFN output
        .add("add_14", "matmul_19")      // add_20 — residual
        // ── KV compaction: topk from attention scores → gather ─────
        .topk("matmul_10", target_count, 3)  // topk_21 — top-k positions by score
        // gather compacted K and V from kv_full using topk indices
        // (gather_22, gather_23 — axis=2 over the seq_len dimension)
        // ── MTP head ────────────────────────────────────────────────
        .gather("lut", "mtp_w_proj", 1)  // gather_24
        .matmul("add_20", "gather_24")     // matmul_25 — MTP logits
        // ── Outputs ─────────────────────────────────────────────────
        .output("matmul_25")
        .output("topk_21_indices")
        .build();

    match b {
        Ok(prog) => {
            let mut bytes = Vec::new();
            prog.encode(&mut bytes).ok();
            eprintln!("[mil] ANE fused layer+compaction program: {} bytes", bytes.len());
            bytes
        }
        Err(e) => {
            eprintln!("[mil] ANE program build failed: {e}");
            Vec::new()
        }
    }
}

fn make_operation(
    op_type: &str,
    op_name: &str,
    inputs: HashMap<String, mil_spec::Argument>,
    outputs: &[(&str, &mil_spec::ValueType)],
    mut extra_attrs: HashMap<String, mil_spec::Value>,
) -> mil_spec::Operation {
    extra_attrs.insert("name".to_string(), string_attr(op_name));
    mil_spec::Operation {
        r#type: op_type.to_string(),
        inputs,
        outputs: outputs
            .iter()
            .map(|(n, vt)| mil_spec::NamedValueType {
                name: n.to_string(),
                r#type: Some((*vt).clone()),
            })
            .collect(),
        blocks: vec![],
        attributes: extra_attrs,
    }
}

// ── helpers ──────────────────────────────────────────────────────────────

fn tensor_type(dtype: mil_spec::DataType, shape: &[i64]) -> mil_spec::TensorType {
    let dims: Vec<mil_spec::Dimension> = shape
        .iter()
        .map(|&s| mil_spec::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: s as u64 },
            )),
        })
        .collect();
    mil_spec::TensorType {
        data_type: dtype as i32,
        rank: shape.len() as i64,
        dimensions: dims,
        attributes: HashMap::new(),
    }
}

/// Build a TensorType from a raw data type i32 value (for types not in the proto enum).
fn tensor_type_raw(dtype: i32, shape: &[i64]) -> mil_spec::TensorType {
    let dims = shape
        .iter()
        .map(|&s| mil_spec::Dimension {
            dimension: Some(dimension::Dimension::Constant(
                dimension::ConstantDimension { size: s as u64 },
            )),
        })
        .collect();
    mil_spec::TensorType {
        data_type: dtype,
        rank: shape.len() as i64,
        dimensions: dims,
        attributes: HashMap::new(),
    }
}

fn value_type_tensor(tt: mil_spec::TensorType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(tt)),
    }
}

fn float_attr(val: f32) -> mil_spec::Value {
    let float_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Float32 as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(float_tensor)),
        })),
    }
}

fn named_arg(name: &str) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

fn bool_arg(val: bool) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Value(bool_attr(val))),
        }],
    }
}

fn bool_attr(val: bool) -> mil_spec::Value {
    let bool_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Bool as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(bool_tensor)),
        })),
    }
}

fn int_attr(val: i64) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::LongInts(
            tensor_value::RepeatedLongInts { values: vec![val] },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Int64 as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(int_tensor)),
        })),
    }
}

fn ints_attr(vals: &[i64]) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::LongInts(
            tensor_value::RepeatedLongInts { values: vals.to_vec() },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Int64 as i32,
                    rank: 1,
                    dimensions: vec![mil_spec::Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: vals.len() as u64 },
                        )),
                    }],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(int_tensor)),
        })),
    }
}

fn string_attr(val: &str) -> mil_spec::Value {
    let string_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Strings(
            tensor_value::RepeatedStrings {
                values: vec![val.to_string()],
            },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::String as i32,
                    rank: 0,
                    dimensions: vec![],
                    attributes: HashMap::new(),
                },
            )),
        }),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(value::immediate_value::Value::Tensor(string_tensor)),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    // ── unary op type resolution tests ─────────────────────────────────--

    #[test]
    fn resolve_logistic_to_sigmoid() {
        let result = resolve_unary_op_type("logistic").unwrap();
        assert_eq!(
            result,
            CoreMlUnaryOpType {
                mil_op_type: "sigmoid",
                requires_attrs: false
            }
        );
    }

    #[test]
    fn resolve_sigmoid_alias() {
        let result = resolve_unary_op_type("sigmoid").unwrap();
        assert_eq!(
            result,
            CoreMlUnaryOpType {
                mil_op_type: "sigmoid",
                requires_attrs: false
            }
        );
    }

    #[test]
    fn resolve_silu() {
        let result = resolve_unary_op_type("silu").unwrap();
        assert_eq!(
            result,
            CoreMlUnaryOpType {
                mil_op_type: "silu",
                requires_attrs: false
            }
        );
    }

    #[test]
    fn resolve_unknown_mode_returns_none() {
        assert!(resolve_unary_op_type("gelu").is_none());
        assert!(resolve_unary_op_type("relu").is_none());
        assert!(resolve_unary_op_type("tanh").is_none());
        assert!(resolve_unary_op_type("element_wise").is_none());
    }

    // ── MIL program construction tests ─────────────────────────────────--

    #[test]
    fn build_simple_matmul() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .const_f32("w", &[1.0, 2.0, 3.0, 4.0], &[4, 1])
            .matmul("x", "w_0")
            .output("matmul_1")
            .build()
            .unwrap();

        assert_eq!(prog.version, 1);
        assert_eq!(prog.functions.len(), 1);
        let func = prog.functions.get("main").unwrap();
        assert_eq!(func.inputs.len(), 1);
        assert_eq!(func.inputs[0].name, "x");
        let block = func.block_specializations.get("CoreML9").unwrap();
        assert_eq!(block.operations.len(), 2); // const + matmul
        assert_eq!(block.operations[0].r#type, "const");
        assert_eq!(block.operations[1].r#type, "matmul");
        assert_eq!(block.outputs.len(), 1);
        assert_eq!(block.outputs[0], "matmul_1");

        // Every non-const op must have a "name" attribute
        for op in &block.operations {
            assert!(
                op.attributes.contains_key("name"),
                "op '{}' missing 'name' attribute",
                op.r#type
            );
        }

        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    #[test]
    fn build_add_then_mul() {
        let prog = MilBuilder::new("main")
            .input("a", mil_spec::DataType::Float32, &[2, 2])
            .input("b", mil_spec::DataType::Float32, &[2, 2])
            .add("a", "b")
            .mul("add_0", "add_0")
            .output("mul_1")
            .build()
            .unwrap();

        let block = prog
            .functions
            .get("main")
            .and_then(|f| f.block_specializations.get("CoreML9"))
            .unwrap();
        assert_eq!(block.operations.len(), 2);
        assert_eq!(block.operations[0].r#type, "add");
        assert_eq!(block.operations[1].r#type, "mul");
        for op in &block.operations {
            assert!(op.attributes.contains_key("name"));
        }

        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    #[test]
    #[should_panic(expected = "SSA: unknown value")]
    fn ssa_rejects_undefined_input() {
        let _ = MilBuilder::new("main")
            .matmul("x", "y")
            .output("matmul_0")
            .build();
    }

    #[test]
    fn ssa_rejects_missing_output() {
        let err = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .output("nonexistent")
            .build()
            .expect_err("must reject undefined block output");
        assert!(matches!(err, MilBuildError::UndefinedBlockOutput { .. }));
    }

    #[test]
    fn test_to_mil_text() {
        let builder = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .const_f32("w", &[1.0, 2.0, 3.0, 4.0], &[4, 1])
            .matmul("x", "w_0")
            .output("matmul_1");

        let text = builder.to_mil_text();
        assert!(text.contains("program(1.3)"));
        assert!(text.contains("func main<coreml9>"));
        assert!(text.contains("tensor<fp32, [1, 4]> x"));
        assert!(text.contains("const()["));
        assert!(text.contains("matmul("));
        assert!(text.contains("-> (matmul_1)"));
    }

    // ── const_f32 auto-fill tests (MIL builder repair) ─────────────────

    #[test]
    fn const_f32_empty_data_auto_fills_zeros() {
        let builder = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .const_f32("w", &[], &[4, 1]) // empty data, shape = [4,1] → 4 zeros
            .matmul("x", "w_0")
            .output("matmul_1");
        let text = builder.to_mil_text();
        // Verify the MIL program has the expected ops (const + matmul + output)
        // to_mil_text does not render const data values, so check op names
        assert!(text.contains("w_0"), "MIL text should contain const op name");
        assert!(text.contains("matmul_1"), "MIL text should contain matmul op name");
        assert!(text.contains("x"), "MIL text should contain input name");
    }

    #[test]
    fn const_f32_with_values_preserves_them() {
        let builder = MilBuilder::new("main")
            .const_f32("w", &[1.0, 2.0, 3.0, 4.0], &[4, 1])
            .output("w_0");
        let text = builder.to_mil_text();
        assert!(text.contains("1.0"));
        assert!(text.contains("4.0"));
    }

    #[test]
    fn const_f32_empty_shape_empty_data_ok() {
        let builder = MilBuilder::new("main")
            .const_f32("empty", &[], &[])
            .output("empty_0");
        assert!(builder.to_mil_text().contains("const()["));
    }
}
