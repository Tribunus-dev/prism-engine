use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TensorLayout { RowMajor, ColumnMajor, Tile { rows: usize, cols: usize } }
pub fn element_count(shape: &[usize]) -> Option<usize> { shape.iter().try_fold(1usize, |n,&d| n.checked_mul(d)) }
pub fn validate_shape(shape: &[usize]) -> Result<(), String> { if shape.is_empty() || shape.iter().any(|&d| d==0) { Err("tensor shape must contain only non-zero dimensions".into()) } else { Ok(()) } }
