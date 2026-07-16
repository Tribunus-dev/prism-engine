use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;

use crate::types::{DuckScalar, DuckType, DuckValue};

/// Type-erased interface for column operations.
pub trait AnyColumn: Debug + Send + Sync {
    fn name(&self) -> &str;
    fn dtype(&self) -> DuckType;
    fn len(&self) -> usize;
    fn append(&mut self, value: &DuckValue);
    fn as_any(&self) -> &dyn Any;
    /// Clone into a new boxed trait object.
    fn clone_box(&self) -> Box<dyn AnyColumn>;
}

/// Typed column storage with null tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column<T: DuckScalar> {
    pub name: String,
    pub data: Vec<T>,
    pub nulls: Vec<bool>,
}

impl<T: DuckScalar> Column<T> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            data: Vec::new(),
            nulls: Vec::new(),
        }
    }
}

impl<T: DuckScalar + 'static> AnyColumn for Column<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DuckType {
        T::duck_type()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn append(&mut self, value: &DuckValue) {
        match value {
            DuckValue::Null => {
                // Push a default value and mark null
                self.data.push(unsafe { std::mem::zeroed() });
                self.nulls.push(true);
            }
            other => {
                if let Some(v) = try_downcast::<T>(other) {
                    self.data.push(v);
                    self.nulls.push(false);
                } else {
                    // Type mismatch: push null
                    self.data.push(unsafe { std::mem::zeroed() });
                    self.nulls.push(true);
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn AnyColumn> {
        Box::new(self.clone())
    }
}

fn try_downcast<T: DuckScalar + 'static>(value: &DuckValue) -> Option<T> {
    let dummy: Option<T> = match value {
        DuckValue::Bool(v) => {
            // No Bool DuckScalar, skip
            None
        }
        DuckValue::Tiny(v) => {
            let v = *v as i64;
            extract_int::<T>(v)
        }
        DuckValue::Small(v) => {
            let v = *v as i64;
            extract_int::<T>(v)
        }
        DuckValue::Int(v) => {
            let v = *v as i64;
            extract_int::<T>(v)
        }
        DuckValue::Big(v) => extract_int::<T>(*v),
        DuckValue::Float(v) => extract_float::<T>(*v as f64),
        DuckValue::Double(v) => extract_float::<T>(*v),
        DuckValue::Varchar(v) => {
            // Only String can receive Varchar
            let type_id = std::any::TypeId::of::<T>();
            if type_id == std::any::TypeId::of::<String>() {
                let s: String = v.clone();
                // transmute through unsafe since we can't construct generic T
                unsafe { Some(std::mem::transmute_copy::<String, T>(&s)) }
            } else {
                None
            }
        }
        DuckValue::Timestamp(_) => None,
        DuckValue::Null => None,
    };
    dummy
}

fn extract_int<T: 'static>(v: i64) -> Option<T> {
    let type_id = std::any::TypeId::of::<T>();
    if type_id == std::any::TypeId::of::<i32>() {
        Some(unsafe { std::mem::transmute_copy::<i32, T>(&(v as i32)) })
    } else if type_id == std::any::TypeId::of::<i64>() {
        Some(unsafe { std::mem::transmute_copy::<i64, T>(&v) })
    } else {
        None
    }
}

fn extract_float<T: 'static>(v: f64) -> Option<T> {
    let type_id = std::any::TypeId::of::<T>();
    if type_id == std::any::TypeId::of::<f64>() {
        Some(unsafe { std::mem::transmute_copy::<f64, T>(&v) })
    } else {
        None
    }
}

/// Schema definition for a single column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub dtype: DuckType,
}

/// A materialized columnar table: schema + typed columns + row count.
#[derive(Debug)]
pub struct ColumnarTable {
    pub name: String,
    pub schema: Vec<ColumnDef>,
    pub columns: Vec<Box<dyn AnyColumn>>,
    pub row_count: u64,
}

impl Clone for ColumnarTable {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            schema: self.schema.clone(),
            columns: self.columns.iter().map(|c| c.clone_box()).collect(),
            row_count: self.row_count,
        }
    }
}

/// Create an empty table from a column descriptor slice.
pub fn create_table(columns: &[(&str, DuckType)]) -> ColumnarTable {
    let schema: Vec<ColumnDef> = columns
        .iter()
        .map(|(name, dtype)| ColumnDef {
            name: name.to_string(),
            dtype: *dtype,
        })
        .collect();

    let mut cols: Vec<Box<dyn AnyColumn>> = Vec::new();
    for (name, dtype) in columns {
        match dtype {
            DuckType::Integer => cols.push(Box::new(Column::<i32>::new(name))),
            DuckType::BigInt => cols.push(Box::new(Column::<i64>::new(name))),
            DuckType::Double => cols.push(Box::new(Column::<f64>::new(name))),
            DuckType::Varchar => cols.push(Box::new(Column::<String>::new(name))),
            _ => cols.push(Box::new(Column::<i64>::new(name))),
        }
    }

    ColumnarTable {
        name: "unnamed".to_string(),
        schema,
        columns: cols,
        row_count: 0,
    }
}

/// Append a row of values to a table.
pub fn append_row(table: &mut ColumnarTable, values: &[DuckValue]) {
    for (i, col) in table.columns.iter_mut().enumerate() {
        let value = values.get(i).unwrap_or(&DuckValue::Null);
        col.append(value);
    }
    table.row_count += 1;
}
