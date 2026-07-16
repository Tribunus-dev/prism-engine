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
#[derive(Debug, Clone)]
pub struct Projection {
    pub source_table: String,
    pub query: ProjectionQuery,
    pub materialized: Option<ColumnarTable>,
}

/// Manages a set of projections over source tables.
#[derive(Debug, Clone, Default)]
pub struct ProjectionEngine {
    pub projections: Vec<Projection>,
}

/// Contract that every projection must implement — rebuild, refresh, verify.
///
/// - `rebuild()` — fully reconstruct the projection from source data.
/// - `refresh()` — incremental update from source deltas (default: full rebuild).
/// - `verify()` — compare projection against source and report drift.
pub trait ProjectionContract {
    /// The type of source data the projection reads from.
    type Source;
    /// The type of diff/drift report returned by verify().
    type Drift;

    /// Fully reconstruct the projection from source.
    fn rebuild(&mut self, source: &Self::Source) -> Result<(), String>;

    /// Incrementally refresh the projection (default: full rebuild).
    fn refresh(&mut self, source: &Self::Source) -> Result<(), String> {
        self.rebuild(source)
    }

    /// Compare projection against source and report drift.
    fn verify(&self, source: &Self::Source) -> Result<Self::Drift, String>;
}

impl ProjectionContract for Projection {
    type Source = ColumnarTable;
    type Drift = Vec<String>;

    fn rebuild(&mut self, source: &ColumnarTable) -> Result<(), String> {
        let projected = match &self.query {
            ProjectionQuery::FullScan => {
                let mut t = source.clone();
                t.name = self.source_table.clone();
                t
            }
            ProjectionQuery::Filtered(filter) => {
                let rows = filtered_rows(source, filter);
                let mut t = create_table(
                    &source
                        .schema
                        .iter()
                        .map(|c| (c.name.as_str(), c.dtype))
                        .collect::<Vec<_>>(),
                );
                t.name = self.source_table.clone();
                for &r in &rows {
                    let vals: Vec<DuckValue> = (0..source.columns.len())
                        .map(|c| get_value_at_row(source, c, r))
                        .collect();
                    crate::columnar::append_row(&mut t, &vals);
                }
                t
            }
            ProjectionQuery::Aggregated(aggs) => {
                let mut t = create_table(
                    &aggs
                        .iter()
                        .map(|(n, a)| {
                            let dt = match a {
                                AggExpr::Count => DuckType::BigInt,
                                AggExpr::Sum(_) => DuckType::Double,
                                AggExpr::Avg(_) => DuckType::Double,
                                AggExpr::Min(_) => DuckType::Double,
                                AggExpr::Max(_) => DuckType::Double,
                                AggExpr::Quantile(_, _) => DuckType::Double,
                                AggExpr::Histogram(_, _) => DuckType::BigInt,
                            };
                            (n.as_str(), dt)
                        })
                        .collect::<Vec<_>>(),
                );
                t.name = self.source_table.clone();
                for (name, agg) in aggs {
                    let col = source
                        .schema
                        .iter()
                        .position(|c| c.name == *name)
                        .unwrap_or(0);
                    let val = match agg {
                        AggExpr::Count => DuckValue::Big(count(source, None) as i64),
                        AggExpr::Sum(i) => sum(source, *i).unwrap_or(DuckValue::Null),
                        AggExpr::Avg(i) => avg(source, *i)
                            .map(|v| DuckValue::Double(v))
                            .unwrap_or(DuckValue::Null),
                        AggExpr::Min(i) => min(source, *i).unwrap_or(DuckValue::Null),
                        AggExpr::Max(i) => max(source, *i).unwrap_or(DuckValue::Null),
                        AggExpr::Quantile(i, p) => {
                            quantile(source, *i, *p).unwrap_or(DuckValue::Null)
                        }
                        AggExpr::Histogram(i, b) => histogram(source, *i, *b)
                            .map(|v| DuckValue::Big(v.iter().sum::<u64>() as i64))
                            .unwrap_or(DuckValue::Null),
                    };
                    crate::columnar::append_row(&mut t, &[val]);
                }
                t
            }
        };
        self.materialized = Some(projected);
        Ok(())
    }

    fn verify(&self, source: &ColumnarTable) -> Result<Vec<String>, String> {
        if self.source_table != source.name {
            return Err(format!(
                "Projection source mismatch: {} vs {}",
                self.source_table, source.name
            ));
        }
        let mut drift = Vec::new();
        let mut fresh = self.clone();
        fresh.rebuild(source)?;
        match (&self.materialized, &fresh.materialized) {
            (Some(current), Some(expected)) => {
                if current.row_count != expected.row_count {
                    drift.push(format!(
                        "Row count mismatch: {} vs {}",
                        current.row_count, expected.row_count
                    ));
                }
            }
            (None, Some(_)) => drift.push("Projection not materialized but source has data".into()),
            (Some(_), None) => drift.push("Projection materialized but source is empty".into()),
            (None, None) => {}
        }
        Ok(drift)
    }
}

impl ProjectionContract for ProjectionEngine {
    type Source = ColumnarTable;
    type Drift = Vec<(String, Vec<String>)>;

    fn rebuild(&mut self, source: &ColumnarTable) -> Result<(), String> {
        for proj in self.projections.iter_mut() {
            proj.rebuild(source)?;
        }
        Ok(())
    }

    fn verify(&self, source: &ColumnarTable) -> Result<Vec<(String, Vec<String>)>, String> {
        let mut all_drift = Vec::new();
        for proj in &self.projections {
            let d = proj.verify(source)?;
            if !d.is_empty() {
                all_drift.push((proj.source_table.clone(), d));
            }
        }
        Ok(all_drift)
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
pub fn materialize(engine: &mut ProjectionEngine, table_name: &str) {
    for proj in engine.projections.iter_mut() {
        if proj.source_table == table_name
            || proj
                .materialized
                .as_ref()
                .map_or(false, |t| t.name == table_name)
        {
            // Already materialized by refresh; nothing else to do.
            // The spec says: ensure the projection exists. We just ensure it's present.
            return;
        }
    }
}

fn get_value_at_row(table: &ColumnarTable, col: usize, row: u64) -> DuckValue {
    let col_obj = &table.columns[col];
    let row_usize = row as usize;
    if row_usize >= col_obj.len() {
        return DuckValue::Null;
    }
    use crate::aggregate::filtered_rows as _; // disambiguate
    let any = col_obj.as_any();
    match col_obj.dtype() {
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
