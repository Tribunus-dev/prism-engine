use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::columnar::{Column, ColumnarTable};
use crate::types::{DuckScalar, DuckType, DuckValue};

/// Filter expression for row selection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterExpr {
    Eq(usize, DuckValue),
    Gt(usize, DuckValue),
    Lt(usize, DuckValue),
    And(Box<FilterExpr>, Box<FilterExpr>),
}

/// Evaluate a filter expression against a row, returning true if the row matches.
fn eval_filter(table: &ColumnarTable, filter: &FilterExpr, row: usize) -> bool {
    match filter {
        FilterExpr::Eq(col, val) => compare_eq(table, *col, row, val),
        FilterExpr::Gt(col, val) => compare_gt(table, *col, row, val),
        FilterExpr::Lt(col, val) => compare_lt(table, *col, row, val),
        FilterExpr::And(a, b) => eval_filter(table, a, row) && eval_filter(table, b, row),
    }
}

fn get_scalar_value(table: &ColumnarTable, col: usize, row: usize) -> Option<f64> {
    if row >= table.columns[col].len() {
        return None;
    }
    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Integer => any.downcast_ref::<Column<i32>>().and_then(|c| {
            if row < c.data.len() && !c.nulls[row] {
                Some(c.data[row] as f64)
            } else {
                None
            }
        }),
        DuckType::BigInt => any.downcast_ref::<Column<i64>>().and_then(|c| {
            if row < c.data.len() && !c.nulls[row] {
                Some(c.data[row] as f64)
            } else {
                None
            }
        }),
        DuckType::Double => any.downcast_ref::<Column<f64>>().and_then(|c| {
            if row < c.data.len() && !c.nulls[row] {
                Some(c.data[row])
            } else {
                None
            }
        }),
        _ => None,
    }
}

fn compare_eq(table: &ColumnarTable, col: usize, row: usize, val: &DuckValue) -> bool {
    let any = table.columns[col].as_any();
    match (table.columns[col].dtype(), val) {
        (DuckType::Integer, DuckValue::Int(v)) => {
            any.downcast_ref::<Column<i32>>().map_or(false, |c| {
                row < c.data.len() && !c.nulls[row] && c.data[row] == *v
            })
        }
        (DuckType::BigInt, DuckValue::Big(v)) => {
            any.downcast_ref::<Column<i64>>().map_or(false, |c| {
                row < c.data.len() && !c.nulls[row] && c.data[row] == *v
            })
        }
        (DuckType::Double, DuckValue::Double(v)) => {
            any.downcast_ref::<Column<f64>>().map_or(false, |c| {
                row < c.data.len() && !c.nulls[row] && (c.data[row] - v).abs() < f64::EPSILON
            })
        }
        (DuckType::Varchar, DuckValue::Varchar(v)) => {
            any.downcast_ref::<Column<String>>().map_or(false, |c| {
                row < c.data.len() && !c.nulls[row] && c.data[row] == *v
            })
        }
        _ => false,
    }
}

fn compare_gt(table: &ColumnarTable, col: usize, row: usize, val: &DuckValue) -> bool {
    match val {
        DuckValue::Int(v) => get_scalar_value(table, col, row).map_or(false, |x| x > *v as f64),
        DuckValue::Big(v) => get_scalar_value(table, col, row).map_or(false, |x| x > *v as f64),
        DuckValue::Double(v) => get_scalar_value(table, col, row).map_or(false, |x| x > *v),
        _ => false,
    }
}

fn compare_lt(table: &ColumnarTable, col: usize, row: usize, val: &DuckValue) -> bool {
    match val {
        DuckValue::Int(v) => get_scalar_value(table, col, row).map_or(false, |x| x < *v as f64),
        DuckValue::Big(v) => get_scalar_value(table, col, row).map_or(false, |x| x < *v as f64),
        DuckValue::Double(v) => get_scalar_value(table, col, row).map_or(false, |x| x < *v),
        _ => false,
    }
}

/// Return the indices of all rows matching a filter expression.
pub fn filtered_rows(table: &ColumnarTable, filter: &FilterExpr) -> Vec<u64> {
    let n = table.row_count as usize;
    (0..n)
        .filter(|&r| eval_filter(table, filter, r))
        .map(|r| r as u64)
        .collect()
}

/// Count rows, optionally filtered.
pub fn count(table: &ColumnarTable, filter: Option<&FilterExpr>) -> u64 {
    match filter {
        Some(f) => filtered_rows(table, f).len() as u64,
        None => table.row_count,
    }
}

/// Compute sum of a numeric column.
pub fn sum(table: &ColumnarTable, col: usize) -> Result<DuckValue, String> {
    if col >= table.columns.len() {
        return Err("column index out of range".to_string());
    }
    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Integer => {
            let c = any.downcast_ref::<Column<i32>>().ok_or("type mismatch")?;
            let s: i64 = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v as i64)
                .sum();
            Ok(DuckValue::Big(s))
        }
        DuckType::BigInt => {
            let c = any.downcast_ref::<Column<i64>>().ok_or("type mismatch")?;
            let s: i64 = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .sum();
            Ok(DuckValue::Big(s))
        }
        DuckType::Double => {
            let c = any.downcast_ref::<Column<f64>>().ok_or("type mismatch")?;
            let s: f64 = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .sum();
            Ok(DuckValue::Double(s))
        }
        _ => Err("sum not supported for this type".to_string()),
    }
}

/// Compute average of a numeric column.
pub fn avg(table: &ColumnarTable, col: usize) -> Result<f64, String> {
    let s = sum(table, col)?;
    let total = count(table, None);
    if total == 0 {
        return Err("empty column".to_string());
    }
    match s {
        DuckValue::Big(v) => Ok(v as f64 / total as f64),
        DuckValue::Double(v) => Ok(v / total as f64),
        _ => Err("unexpected sum type".to_string()),
    }
}

/// Compute minimum of a numeric column.
pub fn min(table: &ColumnarTable, col: usize) -> Result<DuckValue, String> {
    if col >= table.columns.len() {
        return Err("column index out of range".to_string());
    }
    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Integer => {
            let c = any.downcast_ref::<Column<i32>>().ok_or("type mismatch")?;
            c.data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .min()
                .map(|v| DuckValue::Int(v))
                .ok_or_else(|| "empty column".to_string())
        }
        DuckType::BigInt => {
            let c = any.downcast_ref::<Column<i64>>().ok_or("type mismatch")?;
            c.data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .min()
                .map(|v| DuckValue::Big(v))
                .ok_or_else(|| "empty column".to_string())
        }
        DuckType::Double => {
            let c = any.downcast_ref::<Column<f64>>().ok_or("type mismatch")?;
            c.data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|v| DuckValue::Double(v))
                .ok_or_else(|| "empty column".to_string())
        }
        _ => Err("min not supported for this type".to_string()),
    }
}

/// Compute maximum of a numeric column.
pub fn max(table: &ColumnarTable, col: usize) -> Result<DuckValue, String> {
    if col >= table.columns.len() {
        return Err("column index out of range".to_string());
    }
    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Integer => {
            let c = any.downcast_ref::<Column<i32>>().ok_or("type mismatch")?;
            c.data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .max()
                .map(|v| DuckValue::Int(v))
                .ok_or_else(|| "empty column".to_string())
        }
        DuckType::BigInt => {
            let c = any.downcast_ref::<Column<i64>>().ok_or("type mismatch")?;
            c.data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .max()
                .map(|v| DuckValue::Big(v))
                .ok_or_else(|| "empty column".to_string())
        }
        DuckType::Double => {
            let c = any.downcast_ref::<Column<f64>>().ok_or("type mismatch")?;
            c.data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|v| DuckValue::Double(v))
                .ok_or_else(|| "empty column".to_string())
        }
        _ => Err("max not supported for this type".to_string()),
    }
}

/// Compute quantile using linear interpolation (DuckDB-compatible).
/// p*(n-1) index with linear interpolation between adjacent sorted values.
pub fn quantile(table: &ColumnarTable, col: usize, p: f64) -> Result<DuckValue, String> {
    if col >= table.columns.len() {
        return Err("column index out of range".to_string());
    }
    if !(0.0..=1.0).contains(&p) {
        return Err("quantile p must be in [0,1]".to_string());
    }

    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Double => {
            let c = any.downcast_ref::<Column<f64>>().ok_or("type mismatch")?;
            let mut values: Vec<f64> = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .collect();
            if values.is_empty() {
                return Err("empty column".to_string());
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = p * (values.len() - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                Ok(DuckValue::Double(values[lo]))
            } else {
                let frac = idx - lo as f64;
                let result = values[lo] + frac * (values[hi] - values[lo]);
                Ok(DuckValue::Double(result))
            }
        }
        DuckType::Integer => {
            let c = any.downcast_ref::<Column<i32>>().ok_or("type mismatch")?;
            let mut values: Vec<f64> = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v as f64)
                .collect();
            if values.is_empty() {
                return Err("empty column".to_string());
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = p * (values.len() - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                Ok(DuckValue::Double(values[lo]))
            } else {
                let frac = idx - lo as f64;
                let result = values[lo] + frac * (values[hi] - values[lo]);
                Ok(DuckValue::Double(result))
            }
        }
        DuckType::BigInt => {
            let c = any.downcast_ref::<Column<i64>>().ok_or("type mismatch")?;
            let mut values: Vec<f64> = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v as f64)
                .collect();
            if values.is_empty() {
                return Err("empty column".to_string());
            }
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = p * (values.len() - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            if lo == hi {
                Ok(DuckValue::Double(values[lo]))
            } else {
                let frac = idx - lo as f64;
                let result = values[lo] + frac * (values[hi] - values[lo]);
                Ok(DuckValue::Double(result))
            }
        }
        _ => Err("quantile not supported for this type".to_string()),
    }
}

/// Compute histogram with equal-width buckets.
pub fn histogram(table: &ColumnarTable, col: usize, buckets: u32) -> Result<Vec<u64>, String> {
    if col >= table.columns.len() || buckets == 0 {
        return Err("invalid parameters".to_string());
    }

    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Double => {
            let c = any.downcast_ref::<Column<f64>>().ok_or("type mismatch")?;
            let values: Vec<f64> = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v)
                .collect();
            build_histogram_f64(&values, buckets)
        }
        DuckType::Integer => {
            let c = any.downcast_ref::<Column<i32>>().ok_or("type mismatch")?;
            let values: Vec<f64> = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v as f64)
                .collect();
            build_histogram_f64(&values, buckets)
        }
        DuckType::BigInt => {
            let c = any.downcast_ref::<Column<i64>>().ok_or("type mismatch")?;
            let values: Vec<f64> = c
                .data
                .iter()
                .zip(c.nulls.iter())
                .filter(|(_, &n)| !n)
                .map(|(v, _)| *v as f64)
                .collect();
            build_histogram_f64(&values, buckets)
        }
        _ => Err("histogram not supported for this type".to_string()),
    }
}

fn build_histogram_f64(values: &[f64], buckets: u32) -> Result<Vec<u64>, String> {
    if values.is_empty() {
        return Ok(vec![0u64; buckets as usize]);
    }
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let b = buckets as usize;
    let range = max_val - min_val;
    let mut hist = vec![0u64; b];

    for &v in values {
        if (range).abs() < f64::EPSILON {
            // All values identical — put everything in the first bucket
            hist[0] += 1;
        } else {
            let normalized = (v - min_val) / range;
            let mut idx = (normalized * b as f64).floor() as usize;
            if idx >= b {
                idx = b - 1;
            }
            hist[idx] += 1;
        }
    }
    Ok(hist)
}
