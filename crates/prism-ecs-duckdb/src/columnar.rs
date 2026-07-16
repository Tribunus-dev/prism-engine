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
                self.data.push(T::default());
                self.nulls.push(true);
            }
            other => {
                if let Some(v) = try_downcast::<T>(other) {
                    self.data.push(v);
                    self.nulls.push(false);
                } else {
                    self.data.push(T::default());
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
    match value {
        DuckValue::Tiny(v) => extract_int::<T>(*v as i64),
        DuckValue::Small(v) => extract_int::<T>(*v as i64),
        DuckValue::Int(v) => extract_int::<T>(*v as i64),
        DuckValue::Big(v) => extract_int::<T>(*v),
        DuckValue::Float(v) => extract_float::<T>(*v as f64),
        DuckValue::Double(v) => extract_float::<T>(*v),
        DuckValue::Varchar(v) => {
            let type_id = std::any::TypeId::of::<T>();
            if type_id == std::any::TypeId::of::<String>() {
                // SAFETY: TypeId check above guarantees T == String.
                // Clone the inner String, then read the bytes out as T.
                // forget() the clone so the returned T retains ownership
                // of the heap buffer (no double-free).
                let s = v.clone();
                let result: T = unsafe { std::ptr::read(&s as *const String as *const T) };
                std::mem::forget(s);
                Some(result)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_int<T: 'static>(v: i64) -> Option<T> {
    let type_id = std::any::TypeId::of::<T>();
    if type_id == std::any::TypeId::of::<i32>() {
        // SAFETY: TypeId check guarantees T == i32; both are Copy, same size.
        let tmp = v as i32;
        Some(unsafe { std::ptr::read(&tmp as *const i32 as *const T) })
    } else if type_id == std::any::TypeId::of::<i64>() {
        // SAFETY: TypeId check guarantees T == i64; both are Copy, same size.
        Some(unsafe { std::ptr::read(&v as *const i64 as *const T) })
    } else {
        None
    }
}

fn extract_float<T: 'static>(v: f64) -> Option<T> {
    let type_id = std::any::TypeId::of::<T>();
    if type_id == std::any::TypeId::of::<f64>() {
        // SAFETY: TypeId check guarantees T == f64; both are Copy, same size.
        Some(unsafe { std::ptr::read(&v as *const f64 as *const T) })
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
#[derive(Debug, Serialize, Deserialize)]
pub struct ColumnarTable {
    pub name: String,
    pub schema: Vec<ColumnDef>,
    #[serde(skip)]
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
