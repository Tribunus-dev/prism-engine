//! prism-ecs-duckdb — ECS-native columnar query engine.
//!
//! Pure Rust, zero C++ dependencies. DuckDB-compatible subset for columnar
//! storage, query aggregation, and event-stream projection.

pub mod aggregate;
pub mod columnar;
pub mod projection;
pub mod types;

pub use aggregate::{avg, count, filtered_rows, histogram, max, min, quantile, sum, FilterExpr};
pub use columnar::{append_row, create_table, Column, ColumnDef, ColumnarTable, AnyColumn};
pub use projection::{
    materialize, refresh_projections, AggExpr, Projection, ProjectionEngine, ProjectionQuery,
};
pub use types::{DuckScalar, DuckType, DuckValue};
