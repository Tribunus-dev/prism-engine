//! Attribute system for the ECS-native IR.
//!
//! Attributes are value objects (not entities) — they are compared by
//! structural equality and serialized inline.
//!
//! Minimal stub for Wave 13, sub-wave 1. Full attribute system (DenseElements,
//! SparseElements, Dictionary, etc.) is implemented in sub-wave 2.

use serde::{Deserialize, Serialize};

/// An IR attribute — carried by operations as metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Bool(bool),
    Integer(i64, crate::ir_types::Type),
    Float(f64, crate::ir_types::Type),
    String(String),
    Array(Vec<Attribute>),
    Dictionary(Vec<(String, Attribute)>),
}

impl Attribute {
    pub fn bool(value: bool) -> Self {
        Attribute::Bool(value)
    }

    pub fn integer(value: i64, ty: crate::ir_types::Type) -> Self {
        Attribute::Integer(value, ty)
    }

    pub fn float(value: f64, ty: crate::ir_types::Type) -> Self {
        Attribute::Float(value, ty)
    }

    pub fn string(value: impl Into<String>) -> Self {
        Attribute::String(value.into())
    }

    pub fn array(values: Vec<Attribute>) -> Self {
        Attribute::Array(values)
    }

    pub fn dictionary(entries: Vec<(String, Attribute)>) -> Self {
        Attribute::Dictionary(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_types::Type;

    #[test]
    fn attribute_equality() {
        let a = Attribute::bool(true);
        let b = Attribute::bool(true);
        let c = Attribute::bool(false);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn attribute_serialization() {
        let attr = Attribute::integer(42, Type::i32());
        let json = serde_json::to_string(&attr).unwrap();
        let back: Attribute = serde_json::from_str(&json).unwrap();
        assert_eq!(attr, back);
    }
}
