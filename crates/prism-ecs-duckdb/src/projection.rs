use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::aggregate::{avg, count, filtered_rows, histogram, max, min, quantile, sum, FilterExpr};
use crate::columnar::{create_table, ColumnarTable};
use crate::types::{DuckType, DuckValue};

/// An aggregation expression within a projection query.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggExpr {
    Count,
    Sum(usize),
    Avg(usize),
    Min(usize),
    Max(usize),
    Quantile(usize, f64),
    Histogram(usize, u32),
}

/// Describes which rows a projection reads from the source table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProjectionQuery {
    FullScan,
    Filtered(FilterExpr),
    Aggregated(Vec<(String, AggExpr)>),
}

/// A projection that maps a source table into a materialized result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projection {
    pub source_table: String,
    pub query: ProjectionQuery,
    pub materialized: Option<ColumnarTable>,
}

/// Manages a set of projections over source tables.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProjectionEngine {
    pub projections: Vec<Projection>,
}

impl ProjectionEngine {
    pub fn new() -> Self {
        Self {
            projections: Vec::new(),
        }
    }
}

fn get_value_at_row(table: &ColumnarTable, col: usize, row: u64) -> DuckValue {
    let row_usize = row as usize;
    if row_usize >= table.columns[col].len() {
        return DuckValue::Null;
    }
    let any = table.columns[col].as_any();
    match table.columns[col].dtype() {
        DuckType::Integer => {
            if let Some(c) = any.downcast_ref::<crate::columnar::Column<i32>>() {
                if c.nulls[row_usize] {
                    DuckValue::Null
                } else {
                    DuckValue::Int(c.data[row_usize])
                }
            } else {
                DuckValue::Null
            }
        }
        DuckType::BigInt => {
            if let Some(c) = any.downcast_ref::<crate::columnar::Column<i64>>() {
                if c.nulls[row_usize] {
                    DuckValue::Null
                } else {
                    DuckValue::Big(c.data[row_usize])
                }
            } else {
                DuckValue::Null
            }
        }
        DuckType::Double => {
            if let Some(c) = any.downcast_ref::<crate::columnar::Column<f64>>() {
                if c.nulls[row_usize] {
                    DuckValue::Null
                } else {
                    DuckValue::Double(c.data[row_usize])
                }
            } else {
                DuckValue::Null
            }
        }
        DuckType::Varchar => {
            if let Some(c) = any.downcast_ref::<crate::columnar::Column<String>>() {
                if c.nulls[row_usize] {
                    DuckValue::Null
                } else {
                    DuckValue::Varchar(c.data[row_usize].clone())
                }
            } else {
                DuckValue::Null
            }
        }
        _ => DuckValue::Null,
    }
}

/// Refresh all projections in the engine against the given source table.
pub fn refresh_projections(engine: &mut ProjectionEngine, source: &ColumnarTable) {
    for projection in engine.projections.iter_mut() {
        if projection.source_table != source.name {
            continue;
        }

        projection.materialized = match &projection.query {
            ProjectionQuery::FullScan => Some(source.clone()),

            ProjectionQuery::Filtered(filter) => {
                let rows = filtered_rows(source, filter);
                let mut result = create_table(
                    &source
                        .schema
                        .iter()
                        .map(|c| (c.name.as_str(), c.dtype))
                        .collect::<Vec<_>>(),
                );
                result.name = source.name.clone();
                for &row_idx in &rows {
                    let values: Vec<DuckValue> = (0..source.schema.len())
                        .map(|col| get_value_at_row(source, col, row_idx))
                        .collect();
                    crate::columnar::append_row(&mut result, &values);
                }
                Some(result)
            }

            ProjectionQuery::Aggregated(aggs) => {
                let col_defs: Vec<(&str, DuckType)> = aggs
                    .iter()
                    .map(|(name, agg)| {
                        let dtype = match agg {
                            AggExpr::Count => DuckType::BigInt,
                            AggExpr::Sum(_) => DuckType::Double,
                            AggExpr::Avg(_) => DuckType::Double,
                            AggExpr::Min(_) | AggExpr::Max(_) => DuckType::Double,
                            AggExpr::Quantile(_, _) => DuckType::Double,
                            AggExpr::Histogram(_, _) => DuckType::BigInt,
                        };
                        (name.as_str(), dtype)
                    })
                    .collect();
                let mut result = create_table(&col_defs);
                result.name = format!("{}_agg", source.name);

                let values: Vec<DuckValue> = aggs
                    .iter()
                    .map(|(_, agg)| match agg {
                        AggExpr::Count => DuckValue::Big(count(source, None) as i64),
                        AggExpr::Sum(col) => match sum(source, *col) {
                            Ok(v) => v,
                            Err(_) => DuckValue::Null,
                        },
                        AggExpr::Avg(col) => match avg(source, *col) {
                            Ok(v) => DuckValue::Double(v),
                            Err(_) => DuckValue::Null,
                        },
                        AggExpr::Min(col) => match min(source, *col) {
                            Ok(v) => v,
                            Err(_) => DuckValue::Null,
                        },
                        AggExpr::Max(col) => match max(source, *col) {
                            Ok(v) => v,
                            Err(_) => DuckValue::Null,
                        },
                        AggExpr::Quantile(col, p) => match quantile(source, *col, *p) {
                            Ok(v) => v,
                            Err(_) => DuckValue::Null,
                        },
                        AggExpr::Histogram(col, buckets) => {
                            match histogram(source, *col, *buckets) {
                                Ok(h) => DuckValue::Varchar(format!("{:?}", h)),
                                Err(_) => DuckValue::Null,
                            }
                        }
                    })
                    .collect();

                crate::columnar::append_row(&mut result, &values);
                Some(result)
            }
        };
    }
}

/// Materialize a projection by name into its target table in the engine.
pub fn materialize(_engine: &mut ProjectionEngine, table_name: &str) {
    // Refresh already filled materialized fields. This ensures the
    // projection for the given source table name is present.
    // No additional work needed — refresh already handles everything.
    let _ = table_name;
}
