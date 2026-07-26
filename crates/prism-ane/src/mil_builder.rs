//! Pure-Rust MIL program builder using `coreml-proto` + `prost`.
//!
//! Constructs `mil_spec::Program` protobufs without Python/coremltools.
//! Generates SSA value names automatically and produces a valid
//! MLProgram that coremlcompiler can ingest.

use coreml_proto::proto::mil_spec::{self, argument, dimension, tensor_value, value};
use std::collections::HashMap;

/// Error returned by [`MilBuilder::build`] when SSA validation fails,
/// or by the high-level program constructors in
/// [`crate::mil_layer_programs`] when protobuf encoding fails.
#[derive(Debug, Clone)]
pub enum MilBuildError {
    UndefinedValue { operation: String, name: String },
    UndefinedBlockOutput { name: String },
    MissingOperationName { op_type: String },
    UnknownType { name: String },
    UnsupportedUnaryOpMode { mode: String },
    /// The constructed program could not be serialised to protobuf.
    /// Returned by the high-level program constructors
    /// ([`crate::mil_layer_programs::build_full_ane_layer_program`]
    /// and [`crate::mil_layer_programs::build_batched_matmul_program`]).
    ProgramEncodeFailed(String),
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
            MilBuildError::ProgramEncodeFailed(msg) => {
                write!(f, "failed to encode MIL program to protobuf: {msg}")
            }
        }
    }
}

impl std::error::Error for MilBuildError {}

/// Builder for constructing MIL Program protobufs.
pub struct MilBuilder {
    function_name: String,
    opset: String,
    inputs: Vec<mil_spec::NamedValueType>,
    ops: Vec<mil_spec::Operation>,
    block_outputs: Vec<String>,
    counter: u64,
    value_types: HashMap<String, mil_spec::ValueType>,
    weights: HashMap<String, Vec<u8>>,
    /// Batch size for fused MIL programs. When `> 1`, matmul broadcasts the
    /// weight across the batch dimension, processing all items in a single
    /// ANE invocation. Must be a power of 2 (1, 2, 4) for the ANE path.
    pub batch_size: u32,
}

impl Default for MilBuilder {
    fn default() -> Self {
        Self::new("__default__")
    }
}

impl MilBuilder {
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
            batch_size: 1,
        }
    }

    /// Set the batch size for this MIL program.
    ///
    /// When > 1 (power of 2: 1, 2, 4), the first dimension of inputs is
    /// scaled accordingly and matmul broadcasts the weight across the
    /// batch dimension, processing all items in a single ANE invocation.
    /// Default is 1 (no batching).
    pub fn batch_size(mut self, n: u32) -> Self {
        self.batch_size = n;
        self
    }

    pub fn input(mut self, name: &str, dtype: mil_spec::DataType, shape: &[i64]) -> Self {
        let tt = tensor_type(dtype, shape);
        let vt = value_type_tensor(tt);
        self.value_types.insert(name.to_string(), vt.clone());
        self.inputs.push(mil_spec::NamedValueType {
            name: name.to_string(),
            r#type: Some(vt),
        });
        self
    }

    pub fn set_opset(mut self, opset: &str) -> Self {
        self.opset = opset.to_string();
        self
    }

    pub fn get_opset(&self) -> &str {
        &self.opset
    }

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

    pub fn register_type(&mut self, name: &str, vt: mil_spec::ValueType) {
        self.value_types.insert(name.to_string(), vt);
    }

    pub fn ops(&self) -> &[mil_spec::Operation] {
        &self.ops
    }

    pub fn add_weight(&mut self, name: &str, data: Vec<u8>) {
        self.weights.insert(name.to_string(), data);
    }

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
            (Some(a), Some(b)) if a == b => a.clone(),
            _ => vec![
                mil_spec::Dimension {
                    dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension {
                        variadic: false,
                    })),
                },
                mil_spec::Dimension {
                    dimension: Some(dimension::Dimension::Unknown(dimension::UnknownDimension {
                        variadic: false,
                    })),
                },
            ],
        }
    }

    pub fn const_f32(mut self, name_hint: &str, values: &[f32], shape: &[i64]) -> Self {
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
        let tt = tensor_type(mil_spec::DataType::Float32, shape);
        let vt = value_type_tensor(tt);
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
        let mut attrs = HashMap::new();
        attrs.insert("name".to_string(), string_attr(&name));
        attrs.insert("val".to_string(), v);
        let op = make_operation("const", &name, HashMap::new(), &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    pub fn const_f16(mut self, name_hint: &str, values: &[f32], shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let tt = tensor_type(mil_spec::DataType::Float16, shape);
        let vt = value_type_tensor(tt);
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
                        (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
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

    pub fn const_uint8(mut self, name_hint: &str, values: &[u8], shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let tt = tensor_type(mil_spec::DataType::Uint8, shape);
        let vt = value_type_tensor(tt);
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

    pub fn quantize(mut self, name_hint: &str, input: &str, scale: f32, shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let vt = value_type_tensor(tensor_type_raw(4, shape));
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

    pub fn dequantize(mut self, name_hint: &str, input: &str, scale: f32, shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let vt = value_type_tensor(tensor_type_raw(10, shape));
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

    /// Add a `gather` operation — index into `params` along `axis` using `indices`.
    ///
    /// Used by the ANE Planar Engine LUT expansion: `params=[81,4]` LUT,
    /// `indices=swizzled u8 byte`, `axis=0` → gathers one row of the LUT.
    /// The output shape is the params prefix dims, then the indices dims,
    /// then the params suffix dims after `axis`. This is the canonical
    /// MIL gather output shape.
    pub fn gather(mut self, params: &str, indices: &str, axis: i64) -> Self {
        let name = self.fresh_name("gather");
        let dtype = self
            .require_dtype(params)
            .expect("SSA: unknown params type");

        // Resolve the param and index dim lists to plain i64 vectors so the
        // output shape calculation is straightforward.
        let params_dims: Vec<i64> = self
            .value_types
            .get(params)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(
                    tt.dimensions
                        .iter()
                        .filter_map(|d| match d.dimension.as_ref()? {
                            dimension::Dimension::Constant(c) => Some(c.size as i64),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let indices_dims: Vec<i64> = self
            .value_types
            .get(indices)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(
                    tt.dimensions
                        .iter()
                        .filter_map(|d| match d.dimension.as_ref()? {
                            dimension::Dimension::Constant(c) => Some(c.size as i64),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();

        // Negative axis counts from the end of `params_dims`.
        let axis_index = if axis < 0 {
            (params_dims.len() as i64 + axis).max(0) as usize
        } else {
            axis as usize
        };
        let mut out_dims = Vec::new();
        if axis_index <= params_dims.len() {
            out_dims.extend_from_slice(&params_dims[..axis_index.min(params_dims.len())]);
            out_dims.extend_from_slice(&indices_dims);
            if axis_index < params_dims.len() {
                out_dims.extend_from_slice(&params_dims[axis_index + 1..]);
            }
        }
        let vt = value_type_tensor(tensor_type(dtype, &out_dims));

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), named_arg(params));
        inputs.insert("indices".to_string(), named_arg(indices));
        inputs.insert("axis".to_string(), int32_arg(axis as i32));
        inputs.insert("validate_indices".to_string(), bool_arg(false));

        let op = make_operation("gather", &name, inputs, &[(&name, &vt)], HashMap::new());
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    pub fn reduce_sum(mut self, x: &str) -> Self {
        let name = self.fresh_name("reduce_sum");
        let dtype = self.require_dtype(x).expect("SSA: unknown value");
        let output_dims = vec![1, 1];
        let vt = value_type_tensor(tensor_type(dtype, &output_dims));
        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(x));
        let op = make_operation(
            "reduce_sum",
            &name,
            inputs_map,
            &[(&name, &vt)],
            HashMap::new(),
        );
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    pub fn matmul(mut self, a: &str, b: &str) -> Self {
        let name = self.fresh_name("matmul");
        let dtype = self.require_dtype(a).expect("SSA: unknown value");
        let _ = self.require_dtype(b).expect("SSA: unknown value");
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

    pub fn add(mut self, a: &str, b: &str) -> Self {
        let name = self.fresh_name("add");
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
        let op = make_operation("add", &name, inputs_map, &[(&name, &vt)], HashMap::new());
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

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

    /// Add a `topk` operation — returns the values and indices of the top-k
    /// elements along `axis`.  Used for KV compaction: selects the
    /// most-attended token positions directly from the attention scores
    /// the ANE just computed.
    ///
    /// Two outputs are produced:
    /// - `<name>_values`: the same shape as `x`
    /// - `<name>_indices`: int32 indices along the chosen axis
    pub fn topk(mut self, x: &str, k: i64, axis: i64) -> Self {
        let name = self.fresh_name("topk");
        let dtype = self.require_dtype(x).expect("SSA: unknown type");

        // The values output has the same type as `x`; the indices output is
        // a 1-D int32 tensor (the rank is the input's rank; coremlcompiler
        // decides the exact shape).
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
        inputs.insert("x".to_string(), named_arg(x));
        // `k` and `axis` are constant attributes embedded in the op, not
        // input tensors. MIL accepts both forms; we use attributes.

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

    /// Add a SiLU (sigmoid linear unit) element-wise activation.
    ///
    /// SiLU is a primitive MIL op; this wraps the composite
    /// `op_composite_silu` helper into a single self-returning call.
    pub fn silu(mut self, name_hint: &str, input: &str) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown value");

        // Clone dimensions from the input — silu is element-wise.
        let dimensions = self
            .value_types
            .get(input)
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

    /// Add a `softmax` operation along the given axis.
    pub fn softmax(mut self, input: &str, axis: i64) -> Self {
        let name = self.fresh_name("softmax");
        let dtype = self.require_dtype(input).expect("SSA: unknown type");
        // Softmax is element-wise along `axis`; the output has the same
        // shape as the input.
        let vt = self.value_types.get(input).cloned().unwrap_or_else(|| {
            value_type_tensor(mil_spec::TensorType {
                data_type: dtype as i32,
                rank: 4,
                dimensions: vec![],
                attributes: HashMap::new(),
            })
        });

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), named_arg(input));

        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), int_attr(axis));

        let op = make_operation("softmax", &name, inputs, &[(&name, &vt)], attrs);
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a matmul with `transpose_y=true`.
    /// A is `[M, K]`, B is `[N, K]` (transposed to `[K, N]` internally).
    /// Output is `[M, N]`.
    pub fn matmul_transpose_y(mut self, a: &str, b: &str) -> Self {
        let name = self.fresh_name("matmul");
        let dtype = self.require_dtype(a).expect("SSA: unknown value");
        let _ = self.require_dtype(b).expect("SSA: unknown value");

        // With `transpose_y=true`: A[M,K] × B^T[K,N] where K = B.cols, N = B.rows.
        // B is declared as [N, K], so B.rows = N, B.cols = K.
        // Output dims are [A.rows, B.rows] = [M, N].
        let get_dims = |types: &HashMap<String, mil_spec::ValueType>, key: &str| {
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
        };
        let output_dims = match (get_dims(&self.value_types, a), get_dims(&self.value_types, b)) {
            (Some((m, _)), Some((n, _))) => vec![m, n],
            _ => vec![1, 1],
        };
        let vt = value_type_tensor(tensor_type(dtype, &output_dims));

        let mut inputs_map = HashMap::new();
        inputs_map.insert("x".to_string(), named_arg(a));
        inputs_map.insert("y".to_string(), named_arg(b));
        inputs_map.insert("transpose_x".to_string(), bool_arg(false));
        inputs_map.insert("transpose_y".to_string(), bool_arg(true));

        let op = make_operation("matmul", &name, inputs_map, &[(&name, &vt)], HashMap::new());

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Concatenate tensors along a given axis.
    ///
    /// `inputs` — list of SSA value names to concatenate.
    /// `axis` — axis along which to concatenate (0-based).
    /// `_use_sequence_length` — placeholder for MIL spec compatibility (unused).
    pub fn concat(
        mut self,
        name_hint: &str,
        inputs: &[&str],
        axis: i64,
        _use_sequence_length: bool,
    ) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self
            .require_dtype(inputs[0])
            .expect("SSA: unknown type");

        // Infer the output shape from the first input; zero the concat axis
        // because the actual sum is computed at compile time by
        // coremlcompiler.
        let dimensions = self
            .value_types
            .get(inputs[0])
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => {
                    let mut dims = tt.dimensions.clone();
                    if let Some(d) = dims.get_mut(axis as usize) {
                        d.dimension = Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension { size: 0 },
                        ));
                    }
                    Some(dims)
                }
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
        inputs_map.insert("values".to_string(), multi_named_arg(inputs));

        let mut extra = HashMap::new();
        extra.insert("axis".to_string(), int_attr(axis));

        let op = make_operation("concat", &name, inputs_map, &[(&name, &vt)], extra);

        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a 2D convolution operation.
    ///
    /// `kernel_size` is the spatial kernel dimensions (e.g. `[1, 1]`).
    /// `padding` is the padding mode (`"valid"`, `"same"`, or `"custom"`).
    pub fn conv(
        mut self,
        name_hint: &str,
        input: &str,
        weight: &str,
        kernel_size: &[i64],
        padding: &str,
    ) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown value");

        // Output shape: [B, C_out, H_out, W_out]. C_out is unknown; spatial
        // dims come from the input. For a 1x1 valid conv: [B, C, 1, S] →
        // [B, ?, 1, S].
        let out_dims = self
            .value_types
            .get(input)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => {
                    let mut dims = tt.dimensions.clone();
                    if dims.len() >= 2 {
                        // Replace the channel dim with an unknown (C_out).
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
            .unwrap_or_else(|| {
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
            });
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

    /// Reshape a tensor to a new static shape.
    pub fn reshape(mut self, name_hint: &str, input: &str, shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown type");
        let vt = value_type_tensor(tensor_type(dtype, shape));

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), named_arg(input));
        inputs.insert("shape".to_string(), ints32_arg(shape));

        let op = make_operation("reshape", &name, inputs, &[(&name, &vt)], HashMap::new());
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Permute tensor dimensions using a static axis order.
    pub fn transpose(mut self, name_hint: &str, input: &str, perm: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let dtype = self.require_dtype(input).expect("SSA: unknown type");
        let input_dims: Vec<i64> = self
            .value_types
            .get(input)
            .and_then(|vt| match &vt.r#type {
                Some(mil_spec::value_type::Type::TensorType(tt)) => Some(
                    tt.dimensions
                        .iter()
                        .filter_map(|d| match d.dimension.as_ref()? {
                            dimension::Dimension::Constant(c) => Some(c.size as i64),
                            _ => None,
                        })
                        .collect::<Vec<_>>(),
                ),
                _ => None,
            })
            .unwrap_or_default();
        let mut output_shape = Vec::with_capacity(perm.len());
        for &axis in perm {
            let axis_usize = axis as usize;
            output_shape.push(*input_dims.get(axis_usize).unwrap_or(&1));
        }
        let vt = value_type_tensor(tensor_type(dtype, &output_shape));

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), named_arg(input));
        inputs.insert("perm".to_string(), ints32_arg(perm));

        let op = make_operation("transpose", &name, inputs, &[(&name, &vt)], HashMap::new());
        self.value_types.insert(name.clone(), vt);
        self.ops.push(op);
        self
    }

    /// Add a const operation with `i32` immediate values.
    pub fn const_i32(mut self, name_hint: &str, values: &[i32], shape: &[i64]) -> Self {
        let name = self.fresh_name(name_hint);
        let tensor_type = tensor_type(mil_spec::DataType::Int32, shape);
        let vt = value_type_tensor(tensor_type);

        let tv = mil_spec::TensorValue {
            value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
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

    /// Bump the SSA counter by `count` without creating any operation.
    /// Used when `input()` replaces `const_f32()` (stateless weights) but
    /// downstream SSA names must stay stable.
    pub fn reserve_names(mut self, count: u64) -> Self {
        self.counter += count;
        self
    }

    pub fn output(mut self, name: &str) -> Self {
        self.block_outputs.push(name.to_string());
        self
    }

    pub fn build(self) -> Result<mil_spec::Program, MilBuildError> {
        let mut defined: HashMap<String, bool> = HashMap::new();
        for inp in &self.inputs {
            defined.insert(inp.name.clone(), true);
        }
        for op in &self.ops {
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

    pub fn last_name(&self) -> Option<&str> {
        self.ops
            .last()
            .and_then(|op| op.outputs.first())
            .map(|o| o.name.as_str())
    }

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

    pub fn weights(&self) -> &HashMap<String, Vec<u8>> {
        &self.weights
    }

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

    pub fn to_mil_text(&self) -> String {
        let mut mil = String::new();
        mil.push_str("program(1.3)\n");
        mil.push_str("[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3500.32.1\"}})]\n");
        mil.push_str("{\n");
        mil.push_str(&format!(
            "    func {}<{}>(",
            self.function_name,
            self.opset.to_lowercase()
        ));
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 {
                mil.push_str(", ");
            }
            mil.push_str(&format!(
                "{} {}",
                format_value_type(input.r#type.as_ref().unwrap()),
                input.name
            ));
        }
        mil.push_str(") {\n");
        for op in &self.ops {
            mil.push_str("            ");
            mil.push_str(&format!(
                "{} {} = {}(",
                format_value_type(op.outputs[0].r#type.as_ref().unwrap()),
                op.outputs[0].name,
                op.r#type
            ));
            let mut first_arg = true;
            let mut sorted_inputs: Vec<_> = op.inputs.iter().collect();
            sorted_inputs.sort_by_key(|(k, _)| *k);
            for (arg_name, arg) in sorted_inputs {
                if !first_arg {
                    mil.push_str(", ");
                }
                first_arg = false;
                mil.push_str(&format!("{} = ", arg_name));
                if let Some(binding) = arg.arguments.first().and_then(|b| b.binding.as_ref()) {
                    match binding {
                        argument::binding::Binding::Name(n) => mil.push_str(n),
                        argument::binding::Binding::Value(v) => mil.push_str(&format_value(v)),
                    }
                }
            }
            mil.push_str(")[");
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
        mil.push_str("        } -> (");
        for (i, out) in self.block_outputs.iter().enumerate() {
            if i > 0 {
                mil.push_str(", ");
            }
            mil.push_str(out);
        }
        mil.push_str(");\n}\n");
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
                    tensor_value::Value::Strings(s) => format!(
                        "string(\"{}\")",
                        s.values.first().cloned().unwrap_or_default()
                    ),
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
                                let (rows, cols) = (shape[0], shape[1]);
                                let mut res = format!("tensor<fp32, [{rows}, {cols}]>([");
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

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CoreMlUnaryOpType {
    pub mil_op_type: &'static str,
    pub requires_attrs: bool,
}

const COREML_MIL_UNARY_OP_TYPE_MAP: &[(&str, CoreMlUnaryOpType)] = &[
    (
        "logistic",
        CoreMlUnaryOpType {
            mil_op_type: "sigmoid",
            requires_attrs: false,
        },
    ),
    (
        "sigmoid",
        CoreMlUnaryOpType {
            mil_op_type: "sigmoid",
            requires_attrs: false,
        },
    ),
    (
        "silu",
        CoreMlUnaryOpType {
            mil_op_type: "silu",
            requires_attrs: false,
        },
    ),
];

pub fn resolve_unary_op_type(mode: &str) -> Option<CoreMlUnaryOpType> {
    COREML_MIL_UNARY_OP_TYPE_MAP
        .iter()
        .find(|(key, _)| *key == mode)
        .map(|(_, entry)| *entry)
}

// ── operation constructor ──────────────────────────────────────────────

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

/// Build an [`Argument`] that carries a literal `i32` value (no SSA name).
fn int32_arg(val: i32) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Value(int32_attr(val))),
        }],
    }
}

/// Build an [`Argument`] that carries a literal `[i64]` shape, encoded as
/// `i32` (the MIL spec's default shape encoding). Used for static shape
/// inputs to ops like `reshape` and `transpose`.
fn ints32_arg(vals: &[i64]) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Value(ints32_attr(vals))),
        }],
    }
}

/// Build an [`Argument`] that references multiple SSA names (e.g. for
/// `concat`). Each name becomes a separate `Binding::Name` entry.
fn multi_named_arg(names: &[&str]) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: names
            .iter()
            .map(|n| argument::Binding {
                binding: Some(argument::binding::Binding::Name((*n).to_string())),
            })
            .collect(),
    }
}

/// `Int32` attribute with a single value. The MIL spec's default integer
/// encoding for op attributes.
fn int32_attr(val: i32) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Int32 as i32,
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

/// `Int64` array attribute — used by ops like `conv` whose integer-array
/// attributes (e.g. `kernel_size`, `stride`) expect 64-bit integers.
fn ints_attr(vals: &[i64]) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::LongInts(
            tensor_value::RepeatedLongInts {
                values: vals.to_vec(),
            },
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
                            dimension::ConstantDimension {
                                size: vals.len() as u64,
                            },
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

/// `Int32` array attribute — the truncated form of [`ints_attr`]. The MIL
/// spec's `kernel_size`, `stride`, and `pad` attributes accept either
/// 32- or 64-bit integers; we use 32-bit because that is what coremlcompiler
/// emits when targeting the ANE.
fn ints32_attr(vals: &[i64]) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
            values: vals.iter().map(|&v| v as i32).collect(),
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(mil_spec::ValueType {
            r#type: Some(mil_spec::value_type::Type::TensorType(
                mil_spec::TensorType {
                    data_type: mil_spec::DataType::Int32 as i32,
                    rank: 1,
                    dimensions: vec![mil_spec::Dimension {
                        dimension: Some(dimension::Dimension::Constant(
                            dimension::ConstantDimension {
                                size: vals.len() as u64,
                            },
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

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

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
    }

    #[test]
    fn build_simple_matmul() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .const_f32("w", &[1.0, 2.0, 3.0, 4.0], &[4, 1])
            .matmul("x", "w_0")
            .output("matmul_1")
            .build()
            .unwrap();
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        assert_eq!(block.operations.len(), 2);
        let _bytes = prog.encode_to_vec();
        assert!(!_bytes.is_empty());
    }

    #[test]
    fn ssa_rejects_missing_output() {
        let err = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .output("nonexistent")
            .build()
            .expect_err("must reject");
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
        assert!(text.contains("func main<coreml9>"));
        assert!(text.contains("-> (matmul_1)"));
    }

    // ── Tests for the methods absorbed from the engine's mil_builder ────

    #[test]
    fn default_batch_size_is_one() {
        let b = MilBuilder::new("main");
        assert_eq!(b.batch_size, 1);
    }

    #[test]
    fn batch_size_setter_updates_field() {
        let b = MilBuilder::new("main").batch_size(4);
        assert_eq!(b.batch_size, 4);
    }

    #[test]
    fn gather_with_axis_infers_output_shape() {
        // params is [81, 4], indices is [N], axis=0 → output is [N, 4].
        let prog = MilBuilder::new("main")
            .input("params", mil_spec::DataType::Float32, &[81, 4])
            .input("indices", mil_spec::DataType::Int32, &[5])
            .gather("params", "indices", 0)
            .output("gather_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        // The gather op's output type should have the inferred shape.
        let gather_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "gather")
            .expect("gather op present");
        let gather_output = &gather_op.outputs[0];
        let output_type = gather_output.r#type.as_ref().unwrap();
        let tt = match &output_type.r#type {
            Some(mil_spec::value_type::Type::TensorType(t)) => t,
            _ => panic!("expected tensor type"),
        };
        let dims: Vec<i64> = tt
            .dimensions
            .iter()
            .filter_map(|d| match d.dimension.as_ref()? {
                dimension::Dimension::Constant(c) => Some(c.size as i64),
                _ => None,
            })
            .collect();
        // [N, 4] = [5, 4].
        assert_eq!(dims, vec![5, 4]);
    }

    #[test]
    fn gather_with_negative_axis() {
        // Negative axis counts from the end. params is [10, 20, 30], axis=-1
        // (which is 2) → output is [10, 20, N].
        let prog = MilBuilder::new("main")
            .input("params", mil_spec::DataType::Float32, &[10, 20, 30])
            .input("indices", mil_spec::DataType::Int32, &[7])
            .gather("params", "indices", -1)
            .output("gather_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let gather_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "gather")
            .expect("gather op present");
        let tt = match &gather_op.outputs[0].r#type.as_ref().unwrap().r#type {
            Some(mil_spec::value_type::Type::TensorType(t)) => t,
            _ => panic!("expected tensor type"),
        };
        let dims: Vec<i64> = tt
            .dimensions
            .iter()
            .filter_map(|d| match d.dimension.as_ref()? {
                dimension::Dimension::Constant(c) => Some(c.size as i64),
                _ => None,
            })
            .collect();
        assert_eq!(dims, vec![10, 20, 7]);
    }

    #[test]
    fn topk_produces_values_and_indices_outputs() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[32, 64])
            .topk("x", 4, -1)
            .output("topk_0_values")
            .output("topk_0_indices")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let topk_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "topk")
            .expect("topk op present");
        assert_eq!(topk_op.outputs.len(), 2);
        assert_eq!(topk_op.outputs[0].name, "topk_0_values");
        assert_eq!(topk_op.outputs[1].name, "topk_0_indices");
        // The k and axis attributes are present.
        assert!(topk_op.attributes.contains_key("k"));
        assert!(topk_op.attributes.contains_key("axis"));
    }

    #[test]
    fn silu_is_a_primitive_op() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .silu("y", "x")
            .output("y_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let silu_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "silu")
            .expect("silu op present");
        assert_eq!(silu_op.outputs[0].name, "y_0");
    }

    #[test]
    fn softmax_carries_axis_attribute() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4])
            .softmax("x", -1)
            .output("softmax_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let softmax_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "softmax")
            .expect("softmax op present");
        assert!(softmax_op.attributes.contains_key("axis"));
    }

    #[test]
    fn reshape_sets_static_shape() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 16])
            .reshape("y", "x", &[4, 4])
            .output("y_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let reshape_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "reshape")
            .expect("reshape op present");
        let tt = match &reshape_op.outputs[0].r#type.as_ref().unwrap().r#type {
            Some(mil_spec::value_type::Type::TensorType(t)) => t,
            _ => panic!("expected tensor type"),
        };
        let dims: Vec<i64> = tt
            .dimensions
            .iter()
            .filter_map(|d| match d.dimension.as_ref()? {
                dimension::Dimension::Constant(c) => Some(c.size as i64),
                _ => None,
            })
            .collect();
        assert_eq!(dims, vec![4, 4]);
    }

    #[test]
    fn transpose_permutes_dims() {
        // [2, 3, 4] transposed by [2, 0, 1] → [4, 2, 3].
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[2, 3, 4])
            .transpose("y", "x", &[2, 0, 1])
            .output("y_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let transpose_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "transpose")
            .expect("transpose op present");
        let tt = match &transpose_op.outputs[0].r#type.as_ref().unwrap().r#type {
            Some(mil_spec::value_type::Type::TensorType(t)) => t,
            _ => panic!("expected tensor type"),
        };
        let dims: Vec<i64> = tt
            .dimensions
            .iter()
            .filter_map(|d| match d.dimension.as_ref()? {
                dimension::Dimension::Constant(c) => Some(c.size as i64),
                _ => None,
            })
            .collect();
        assert_eq!(dims, vec![4, 2, 3]);
    }

    #[test]
    fn const_i32_emits_const_op() {
        let prog = MilBuilder::new("main")
            .const_i32("idx", &[1, 2, 3, 4], &[4])
            .output("idx_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let const_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "const")
            .expect("const op present");
        let tt = match &const_op.outputs[0].r#type.as_ref().unwrap().r#type {
            Some(mil_spec::value_type::Type::TensorType(t)) => t,
            _ => panic!("expected tensor type"),
        };
        assert_eq!(tt.data_type, mil_spec::DataType::Int32 as i32);
    }

    #[test]
    fn matmul_transpose_y_sets_transpose_y_true() {
        let prog = MilBuilder::new("main")
            .input("a", mil_spec::DataType::Float32, &[2, 4])
            .input("b", mil_spec::DataType::Float32, &[8, 4]) // B = [N, K]
            .matmul_transpose_y("a", "b")
            .output("matmul_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let matmul_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "matmul")
            .expect("matmul op present");
        // The `y` argument is an SSA name binding (the input tensor).
        let y_arg = matmul_op.inputs.get("y").expect("y arg present");
        match &y_arg.arguments[0].binding {
            Some(argument::binding::Binding::Name(n)) => assert_eq!(n, "b"),
            other => panic!("expected Name binding for y, got {other:?}"),
        }
        // The `transpose_y` argument is a literal bool true.
        let transpose_y_arg = matmul_op.inputs.get("transpose_y").expect("transpose_y arg");
        let transpose_y_value = match &transpose_y_arg.arguments[0].binding {
            Some(argument::binding::Binding::Value(v)) => v,
            _ => panic!("expected Value binding for transpose_y"),
        };
        // The bool value is wrapped in a Bool tensor.
        if let Some(mil_spec::value::Value::ImmediateValue(iv)) = &transpose_y_value.value {
            if let Some(mil_spec::value::immediate_value::Value::Tensor(tv)) = &iv.value {
                if let Some(tensor_value::Value::Bools(b)) = &tv.value {
                    assert_eq!(b.values, vec![true]);
                    return;
                }
            }
        }
        panic!("transpose_y was not a bool true");
    }

    #[test]
    fn concat_takes_multiple_inputs() {
        let prog = MilBuilder::new("main")
            .input("a", mil_spec::DataType::Float32, &[2, 4])
            .input("b", mil_spec::DataType::Float32, &[2, 4])
            .concat("c", &["a", "b"], 0, false)
            .output("c_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let concat_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "concat")
            .expect("concat op present");
        // The `values` argument references both `a` and `b` as SSA names.
        let values_arg = concat_op.inputs.get("values").expect("values arg present");
        assert_eq!(values_arg.arguments.len(), 2);
    }

    #[test]
    fn conv_emits_convolution_op_with_attrs() {
        let prog = MilBuilder::new("main")
            .input("x", mil_spec::DataType::Float32, &[1, 4, 1, 8])
            .input("w", mil_spec::DataType::Float32, &[8, 4, 1, 1])
            .conv("y", "x", "w", &[1, 1], "valid")
            .output("y_0")
            .build()
            .expect("build");
        let block = prog
            .functions
            .get("main")
            .unwrap()
            .block_specializations
            .get("CoreML9")
            .unwrap();
        let conv_op = block
            .operations
            .iter()
            .find(|op| op.r#type == "convolution")
            .expect("conv op present");
        assert!(conv_op.attributes.contains_key("kernel_size"));
        assert!(conv_op.attributes.contains_key("stride"));
        assert!(conv_op.attributes.contains_key("pad_type"));
    }

    #[test]
    fn reserve_names_bumps_counter() {
        // After `reserve_names(5)`, the next fresh_name should yield `_5`.
        let mut b = MilBuilder::new("main").reserve_names(5);
        assert_eq!(b.fresh_name("x"), "x_5");
        assert_eq!(b.fresh_name("y"), "y_6");
    }
}
