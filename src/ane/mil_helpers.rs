//! MIL protobuf construction helpers for ops not directly supported by MilBuilder.
//! Ported from Tribunus Compute's compute-core/src/compute_image/subgraph_mil.rs.

use crate::ane::mil_builder::MilBuilder;
use coreml_proto::proto::mil_spec;
use coreml_proto::proto::mil_spec::{
    argument, dimension, tensor_value, value, value::immediate_value,
};
use std::collections::HashMap;

pub fn named_arg(name: &str) -> mil_spec::Argument {
    mil_spec::Argument {
        arguments: vec![argument::Binding {
            binding: Some(argument::binding::Binding::Name(name.to_string())),
        }],
    }
}

pub fn float_attr(val: f32) -> mil_spec::Value {
    let float_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Floats(tensor_value::RepeatedFloats {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Float32)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(immediate_value::Value::Tensor(float_tensor)),
        })),
    }
}

pub fn bool_attr(val: bool) -> mil_spec::Value {
    let bool_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Bools(tensor_value::RepeatedBools {
            values: vec![val],
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Bool)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(immediate_value::Value::Tensor(bool_tensor)),
        })),
    }
}

pub fn int32s_attr(vals: &[i32]) -> mil_spec::Value {
    let int_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Ints(tensor_value::RepeatedInts {
            values: vals.to_vec(),
        })),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::Int32)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(immediate_value::Value::Tensor(int_tensor)),
        })),
    }
}

pub fn string_attr(val: &str) -> mil_spec::Value {
    let string_tensor = mil_spec::TensorValue {
        value: Some(tensor_value::Value::Strings(
            tensor_value::RepeatedStrings {
                values: vec![val.to_string()],
            },
        )),
    };
    mil_spec::Value {
        doc_string: String::new(),
        r#type: Some(scalar_value_type(mil_spec::DataType::String)),
        value: Some(value::Value::ImmediateValue(value::ImmediateValue {
            value: Some(immediate_value::Value::Tensor(string_tensor)),
        })),
    }
}

pub fn scalar_value_type(dtype: mil_spec::DataType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(
            mil_spec::TensorType {
                data_type: dtype as i32,
                rank: 0,
                dimensions: vec![],
                attributes: HashMap::new(),
            },
        )),
    }
}

pub fn tensor_type(dtype: mil_spec::DataType, shape: &[i64]) -> mil_spec::TensorType {
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

pub fn value_type_tensor(tt: mil_spec::TensorType) -> mil_spec::ValueType {
    mil_spec::ValueType {
        r#type: Some(mil_spec::value_type::Type::TensorType(tt)),
    }
}

pub fn float32_tensor_type_2d(rows: i64, cols: i64) -> mil_spec::TensorType {
    tensor_type(mil_spec::DataType::Float32, &[rows, cols])
}

pub fn float32_value_type_2d(rows: i64, cols: i64) -> mil_spec::ValueType {
    value_type_tensor(float32_tensor_type_2d(rows, cols))
}

pub fn make_operation(
    op_type: &str,
    out_name: &str,
    inputs: HashMap<String, mil_spec::Argument>,
    out_vt: &mil_spec::ValueType,
    extra_attrs: HashMap<String, mil_spec::Value>,
) -> mil_spec::Operation {
    let mut attrs = extra_attrs;
    attrs.insert("name".to_string(), string_attr(out_name));
    mil_spec::Operation {
        r#type: op_type.to_string(),
        inputs,
        outputs: vec![mil_spec::NamedValueType {
            name: out_name.to_string(),
            r#type: Some(out_vt.clone()),
        }],
        blocks: vec![],
        attributes: attrs,
    }
}

pub fn resolve_shape(builder: &MilBuilder, name: &str) -> Vec<i64> {
    builder
        .value_shapes()
        .get(name)
        .cloned()
        .unwrap_or_else(|| vec![1, 1])
}

pub fn op_pow(builder: MilBuilder, input: &str, alpha: f32) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_name = format!("pow_{}", builder.ops().len());
    let vt = value_type_tensor(float32_tensor_type_2d(shape[0], shape[1]));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let mut attrs = HashMap::new();
    attrs.insert("alpha".to_string(), float_attr(alpha));
    let op = make_operation("pow", &out_name, inputs, &vt, attrs);
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

pub fn op_reduce_sum(builder: MilBuilder, input: &str, axis: i32) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_rows = if axis == 1 || axis == -1 { shape[0] } else { 1 };
    let out_name = format!("reduce_sum_{}", builder.ops().len());
    let vt = value_type_tensor(float32_tensor_type_2d(out_rows, 1));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let mut attrs = HashMap::new();
    attrs.insert("axes".to_string(), int32s_attr(&[axis]));
    attrs.insert("keep_dims".to_string(), bool_attr(true));
    let op = make_operation("reduce_sum", &out_name, inputs, &vt, attrs);
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

pub fn op_rsqrt(builder: MilBuilder, input: &str) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let out_name = format!("rsqrt_{}", builder.ops().len());
    let vt = value_type_tensor(float32_tensor_type_2d(shape[0], shape[1]));
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let op = make_operation("rsqrt", &out_name, inputs, &vt, HashMap::new());
    let builder = builder.operation(op, Some((out_name.as_str(), vt)));
    (builder, out_name)
}

pub fn op_composite_silu(builder: MilBuilder, input: &str) -> (MilBuilder, String) {
    let shape = resolve_shape(&builder, input);
    let vt = float32_value_type_2d(shape[0], shape[1]);
    // sigmoid(x)
    let sig_name = format!("sig_{}", builder.ops().len());
    let mut inputs = HashMap::new();
    inputs.insert("x".to_string(), named_arg(input));
    let sig_op = make_operation("sigmoid", &sig_name, inputs, &vt, HashMap::new());
    let builder = builder.operation(sig_op, Some((sig_name.as_str(), vt.clone())));
    // mul(x, sigmoid(x)) = SiLU
    let mul_name = format!("mul_{}", builder.ops().len());
    let mut mul_inputs = HashMap::new();
    mul_inputs.insert("x".to_string(), named_arg(input));
    mul_inputs.insert("y".to_string(), named_arg(&sig_name));
    let mul_op = make_operation("mul", &mul_name, mul_inputs, &vt, HashMap::new());
    let builder = builder.operation(mul_op, Some((mul_name.as_str(), vt)));
    (builder, mul_name)
}
