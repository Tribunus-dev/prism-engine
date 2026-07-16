# Wave 26: DuckDB Absorption — ECS-Native Columnar Query Engine

**Status:** Draft (pre-implementation)
**Dependency:** Wave 9 persistence infrastructure, Wave 25 MLIR core parity
**Owner:** kernel

## 1. Scope

Deliver a Rust ECS-native columnar query engine for immutable event stream projections — replacing DuckDB as a C++ dependency with a native alternative that speaks DuckDB-compatible SQL semantics through ECS systems.

Not a full DuckDB port (~200K lines C++). We port the projection surface that ADR-005 needs: columnar storage, aggregate queries (count, quantile, histogram), and materialized views over immutable event streams.

## 2. Crate structure — `prism-ecs-duckdb`

New workspace member `crates/prism-ecs-duckdb/`.

### File map

| File | Contents |
|---|---|
| `Cargo.toml` | depends on `prism-ecs-core`, `serde`, `chrono` |
| `src/lib.rs` | Crate root — re-exports |
| `src/columnar.rs` | `Column<T>` storage, `ColumnarTable` resource, batch append |
| `src/types.rs` | `DuckType` enum mapping to SQL types (Integer, Float, String, Timestamp, etc.) |
| `src/query.rs` | `QueryPlan`, filter/aggregate/projection expressions as value types |
| `src/aggregate.rs` | `count`, `sum`, `avg`, `min`, `max`, `quantile`, `histogram` as ECS systems |
| `src/projection.rs` | `Projection` resource — materialized views from event streams |
| `src/stream.rs` | `EventStream` resource — appends events to columnar storage |

## 3. Design

### ColumnarTable resource

```rust
pub struct ColumnarTable {
    pub name: String,
    pub schema: Vec<ColumnDef>,
    pub columns: Vec<Box<dyn AnyColumn>>,
    pub row_count: u64,
}

pub struct ColumnDef {
    pub name: String,
    pub dtype: DuckType,
}

pub trait AnyColumn: std::fmt::Debug + Send + Sync {}

pub struct Column<T: DuckScalar> {
    pub data: Vec<T>,
    pub nulls: Vec<bool>,  // null bitmap
}
```

### EventStream resource

```rust
pub struct EventStream {
    pub name: String,
    pub schema: Vec<ColumnDef>,
    pub append_count: u64,
}
```

Stream appends are World transactions — each append adds rows to the corresponding ColumnarTable.

### Query expressions as value types (not SQL strings)

```rust
pub enum AggExpr {
    Count,
    Sum(usize),       // column index
    Avg(usize),
    Min(usize),
    Max(usize),
    Quantile(usize, f64),  // column, probability
    Histogram(usize, u32), // column, buckets
}

pub enum FilterExpr {
    Eq(usize, DuckValue),
    Neq(usize, DuckValue),
    Gt(usize, DuckValue),
    Lt(usize, DuckValue),
    And(Box<FilterExpr>, Box<FilterExpr>),
    Or(Box<FilterExpr>, Box<FilterExpr>),
}
```

### DuckDB wire compatibility

The columnar format matches DuckDB's internal representation for the types it supports. A future bridge could export `.duckdb` files for external tooling, but the query engine is pure Rust.

## 4. Gate

- Create a ColumnarTable, append 1000 rows, verify row_count
- Run `Quantile(latency, 0.5)` over a column with known values, verify median matches
- Create an EventStream, append events, verify projection materializes correctly
- `cargo check -p prism-ecs-duckdb` passes with 0 errors
- Zero C++ dependencies
