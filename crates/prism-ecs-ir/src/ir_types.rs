//! Type system for the ECS-native IR.
//!
//! Types are value objects (not entities) — they are compared by structural
//! equality and serialized inline. This mirrors MLIR's uniqued type system
//! but without a global uniquing table (structural equality is sufficient
//! for verification and rewriting).

use serde::{Deserialize, Serialize};

// ── TypeKind ────────────────────────────────────────────────────────────────

/// All builtin type kinds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Type {
    Integer(IntegerType),
    Float(FloatType),
    Index,
    NoneType,
    Function(FunctionType),
    Tensor(TensorType),
    Vector(VectorType),
    Complex(ComplexType),
}

// ── IntegerType ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signedness {
    Signed,
    Unsigned,
    Signless,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegerType {
    pub width: u32,
    pub signedness: Signedness,
}

impl IntegerType {
    pub fn new(width: u32, signedness: Signedness) -> Self {
        Self { width, signedness }
    }
}

// ── FloatType ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FloatKind {
    F16,
    BF16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FloatType {
    pub kind: FloatKind,
}

impl FloatType {
    pub fn new(kind: FloatKind) -> Self {
        Self { kind }
    }
}

// ── FunctionType ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionType {
    pub inputs: Vec<Type>,
    pub results: Vec<Type>,
}

impl FunctionType {
    pub fn new(inputs: Vec<Type>, results: Vec<Type>) -> Self {
        Self { inputs, results }
    }
}

// ── TensorType ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Vec<u64>,
    pub element_type: Box<Type>,
}

impl TensorType {
    pub fn new(shape: Vec<u64>, element_type: Type) -> Self {
        Self {
            shape,
            element_type: Box::new(element_type),
        }
    }
}

// ── VectorType ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorType {
    pub shape: Vec<u64>,
    pub element_type: Box<Type>,
}

impl VectorType {
    pub fn new(shape: Vec<u64>, element_type: Type) -> Self {
        Self {
            shape,
            element_type: Box::new(element_type),
        }
    }
}

// ── ComplexType ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComplexType {
    pub element_type: Box<Type>,
}

impl ComplexType {
    pub fn new(element_type: Type) -> Self {
        Self {
            element_type: Box::new(element_type),
        }
    }
}

// ── Type helpers ────────────────────────────────────────────────────────────

impl Type {
    /// Integer type shorthand.
    pub fn integer(width: u32, signedness: Signedness) -> Self {
        Type::Integer(IntegerType::new(width, signedness))
    }

    /// Float type shorthand.
    pub fn float(kind: FloatKind) -> Self {
        Type::Float(FloatType::new(kind))
    }

    /// f32 shorthand.
    pub fn f32() -> Self {
        Type::float(FloatKind::F32)
    }

    /// f16 shorthand.
    pub fn f16() -> Self {
        Type::float(FloatKind::F16)
    }

    /// bf16 shorthand.
    pub fn bf16() -> Self {
        Type::float(FloatKind::BF16)
    }

    /// i32 shorthand.
    pub fn i32() -> Self {
        Type::integer(32, Signedness::Signless)
    }

    /// i64 shorthand.
    pub fn i64() -> Self {
        Type::integer(64, Signedness::Signless)
    }

    /// Index type.
    pub fn index() -> Self {
        Type::Index
    }

    /// None type.
    pub fn none() -> Self {
        Type::NoneType
    }
}

// ── Formatting (deterministic) ──────────────────────────────────────────────

use std::fmt;

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Integer(int_ty) => {
                let sign = match int_ty.signedness {
                    Signedness::Signed => "si",
                    Signedness::Unsigned => "ui",
                    Signedness::Signless => "i",
                };
                write!(f, "{}{}", sign, int_ty.width)
            }
            Type::Float(flt) => match flt.kind {
                FloatKind::F16 => write!(f, "f16"),
                FloatKind::BF16 => write!(f, "bf16"),
                FloatKind::F32 => write!(f, "f32"),
                FloatKind::F64 => write!(f, "f64"),
                FloatKind::F8E4M3 => write!(f, "f8E4M3"),
                FloatKind::F8E5M2 => write!(f, "f8E5M2"),
            },
            Type::Index => write!(f, "index"),
            Type::NoneType => write!(f, "none"),
            Type::Function(func) => {
                write!(f, "(")?;
                for (i, input) in func.inputs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", input)?;
                }
                write!(f, ") -> ")?;
                if func.results.len() == 1 {
                    write!(f, "{}", func.results[0])?;
                } else {
                    write!(f, "(")?;
                    for (i, result) in func.results.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", result)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Type::Tensor(tensor) => {
                write!(f, "tensor<")?;
                if tensor.shape.is_empty() {
                    write!(f, "{}", tensor.element_type)?;
                } else {
                    for (i, dim) in tensor.shape.iter().enumerate() {
                        if i > 0 {
                            write!(f, "x")?;
                        }
                        if *dim == 0 {
                            write!(f, "?")?;
                        } else {
                            write!(f, "{}", dim)?;
                        }
                    }
                    write!(f, "x{}", tensor.element_type)?;
                }
                write!(f, ">")
            }
            Type::Vector(vec) => {
                write!(f, "vector<")?;
                for (i, dim) in vec.shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, "x")?;
                    }
                    write!(f, "{}", dim)?;
                }
                write!(f, "x{}>", vec.element_type)
            }
            Type::Complex(cplx) => {
                write!(f, "complex<{}>", cplx.element_type)
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_type_equality() {
        let a = Type::integer(32, Signedness::Signed);
        let b = Type::integer(32, Signedness::Signed);
        let c = Type::integer(64, Signedness::Signed);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn float_type_display() {
        assert_eq!(Type::f32().to_string(), "f32");
        assert_eq!(Type::f16().to_string(), "f16");
        assert_eq!(Type::bf16().to_string(), "bf16");
    }

    #[test]
    fn tensor_type_creation() {
        let t = Type::Tensor(TensorType::new(vec![4, 128], Type::f16()));
        assert_eq!(t.to_string(), "tensor<4x128xf16>");
    }

    #[test]
    fn dynamic_tensor_type() {
        let t = Type::Tensor(TensorType::new(vec![0, 128], Type::f32()));
        assert_eq!(t.to_string(), "tensor<?x128xf32>");
    }

    #[test]
    fn function_type_roundtrip() {
        let ft = Type::Function(FunctionType::new(
            vec![Type::i32(), Type::f32()],
            vec![Type::f32()],
        ));
        let s = ft.to_string();
        assert_eq!(s, "(i32, f32) -> f32");
    }

    #[test]
    fn type_serialization_roundtrip() {
        let ty = Type::Tensor(TensorType::new(vec![4, 128], Type::bf16()));
        let json = serde_json::to_string(&ty).unwrap();
        let back: Type = serde_json::from_str(&json).unwrap();
        assert_eq!(ty, back);
    }
}
