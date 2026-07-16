use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Supported column types (DuckDB-compatible subset).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DuckType {
    Boolean,
    TinyInt,
    SmallInt,
    Integer,
    BigInt,
    Float,
    Double,
    Varchar,
    Timestamp,
    Date,
    Decimal(u32, u32),
}

/// A runtime value stored in a column cell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DuckValue {
    Bool(bool),
    Tiny(i8),
    Small(i16),
    Int(i32),
    Big(i64),
    Float(f32),
    Double(f64),
    Varchar(String),
    Timestamp(i64),
    Null,
}

/// Trait for scalar types that can be stored in DuckDB-like columns.
pub trait DuckScalar: Debug + Clone + Default + Send + Sync + 'static {
    fn duck_type() -> DuckType;
}

impl DuckScalar for i32 {
    fn duck_type() -> DuckType {
        DuckType::Integer
    }
}

impl DuckScalar for f64 {
    fn duck_type() -> DuckType {
        DuckType::Double
    }
}

impl DuckScalar for String {
    fn duck_type() -> DuckType {
        DuckType::Varchar
    }
}

impl DuckScalar for i64 {
    fn duck_type() -> DuckType {
        DuckType::BigInt
    }
}
