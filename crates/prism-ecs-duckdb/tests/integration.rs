use prism_ecs_duckdb::aggregate::{
    avg, count, filtered_rows, histogram, max, min, quantile, sum, FilterExpr,
};
use prism_ecs_duckdb::columnar::{append_row, create_table};
use prism_ecs_duckdb::projection::{
    materialize, refresh_projections, AggExpr, Projection, ProjectionEngine, ProjectionQuery,
};
use prism_ecs_duckdb::types::{DuckType, DuckValue};

/// Test: create table, append 5 rows, verify row_count == 5
#[test]
fn test_create_and_append() {
    let mut table = create_table(&[("x", DuckType::Integer), ("y", DuckType::Double)]);
    table.name = "test".to_string();

    append_row(&mut table, &[DuckValue::Int(1), DuckValue::Double(1.5)]);
    append_row(&mut table, &[DuckValue::Int(2), DuckValue::Double(2.5)]);
    append_row(&mut table, &[DuckValue::Int(3), DuckValue::Double(3.5)]);
    append_row(&mut table, &[DuckValue::Int(4), DuckValue::Double(4.5)]);
    append_row(&mut table, &[DuckValue::Int(5), DuckValue::Double(5.5)]);

    assert_eq!(table.row_count, 5);
    assert_eq!(table.columns[0].len(), 5);
    assert_eq!(table.columns[1].len(), 5);
}

/// Test: quantile on [1,2,3,4,5] at p=0.5 returns 3
#[test]
fn test_quantile_median() {
    let mut table = create_table(&[("x", DuckType::Integer)]);
    table.name = "q".to_string();

    for i in 1..=5 {
        append_row(&mut table, &[DuckValue::Int(i)]);
    }

    let result = quantile(&table, 0, 0.5).unwrap();
    match result {
        DuckValue::Double(v) => assert!((v - 3.0).abs() < 1e-10),
        _ => panic!("expected Double"),
    }
}

/// Test: quantile p=0.0 returns min
#[test]
fn test_quantile_min() {
    let mut table = create_table(&[("x", DuckType::Integer)]);
    table.name = "q".to_string();
    for i in 1..=5 {
        append_row(&mut table, &[DuckValue::Int(i)]);
    }
    let result = quantile(&table, 0, 0.0).unwrap();
    match result {
        DuckValue::Double(v) => assert!((v - 1.0).abs() < 1e-10),
        _ => panic!("expected Double"),
    }
}

/// Test: quantile p=1.0 returns max
#[test]
fn test_quantile_max() {
    let mut table = create_table(&[("x", DuckType::Integer)]);
    table.name = "q".to_string();
    for i in 1..=5 {
        append_row(&mut table, &[DuckValue::Int(i)]);
    }
    let result = quantile(&table, 0, 1.0).unwrap();
    match result {
        DuckValue::Double(v) => assert!((v - 5.0).abs() < 1e-10),
        _ => panic!("expected Double"),
    }
}

/// Test: histogram on 1..100 with 10 buckets returns counts of ~10 each
#[test]
fn test_histogram_even() {
    let mut table = create_table(&[("x", DuckType::Double)]);
    table.name = "h".to_string();

    for i in 1..=100 {
        append_row(&mut table, &[DuckValue::Double(i as f64)]);
    }

    let hist = histogram(&table, 0, 10).unwrap();
    assert_eq!(hist.len(), 10);
    for (i, &c) in hist.iter().enumerate() {
        assert!(
            c == 10 || c == 9 || c == 11,
            "bucket {i}: expected ~10, got {c}"
        );
    }
}

/// Test: projection + refresh + materialize round-trip
#[test]
fn test_projection_roundtrip() {
    let mut source = create_table(&[("name", DuckType::Varchar), ("score", DuckType::Integer)]);
    source.name = "students".to_string();

    append_row(
        &mut source,
        &[DuckValue::Varchar("alice".into()), DuckValue::Int(85)],
    );
    append_row(
        &mut source,
        &[DuckValue::Varchar("bob".into()), DuckValue::Int(92)],
    );
    append_row(
        &mut source,
        &[DuckValue::Varchar("carol".into()), DuckValue::Int(78)],
    );

    let mut engine = ProjectionEngine::new();

    // Full scan projection
    engine.projections.push(Projection {
        source_table: "students".to_string(),
        query: ProjectionQuery::FullScan,
        materialized: None,
    });

    // Filtered projection (score > 80)
    engine.projections.push(Projection {
        source_table: "students".to_string(),
        query: ProjectionQuery::Filtered(FilterExpr::Gt(1, DuckValue::Int(80))),
        materialized: None,
    });

    // Aggregated projection
    engine.projections.push(Projection {
        source_table: "students".to_string(),
        query: ProjectionQuery::Aggregated(vec![
            ("cnt".to_string(), AggExpr::Count),
            ("avg_score".to_string(), AggExpr::Avg(1)),
        ]),
        materialized: None,
    });

    // Refresh
    refresh_projections(&mut engine, &source);

    // Full scan: all 3 rows
    let full = &engine.projections[0].materialized;
    assert!(full.is_some());
    assert_eq!(full.as_ref().unwrap().row_count, 3);

    // Filtered: alice (85) and bob (92) pass score > 80
    let filtered = &engine.projections[1].materialized;
    assert!(filtered.is_some());
    assert_eq!(filtered.as_ref().unwrap().row_count, 2);

    // Aggregated: 1 row
    let agg = &engine.projections[2].materialized;
    assert!(agg.is_some());
    let agg_table = agg.as_ref().unwrap();
    assert_eq!(agg_table.row_count, 1);

    // Materialize (no-op, already refreshed)
    materialize(&mut engine, "students");
}

/// Test: sum/avg/min/max on known data
#[test]
fn test_aggregation_basics() {
    let mut table = create_table(&[("v", DuckType::Double)]);
    table.name = "agg".to_string();

    for i in 1..=10 {
        append_row(&mut table, &[DuckValue::Double(i as f64)]);
    }

    let s = sum(&table, 0).unwrap();
    match s {
        DuckValue::Double(v) => assert!((v - 55.0).abs() < 1e-10),
        _ => panic!("expected Double"),
    }

    let a = avg(&table, 0).unwrap();
    assert!((a - 5.5).abs() < 1e-10);

    let mn = min(&table, 0).unwrap();
    match mn {
        DuckValue::Double(v) => assert!((v - 1.0).abs() < 1e-10),
        _ => panic!("expected Double"),
    }

    let mx = max(&table, 0).unwrap();
    match mx {
        DuckValue::Double(v) => assert!((v - 10.0).abs() < 1e-10),
        _ => panic!("expected Double"),
    }
}

/// Test: filtered_rows with equality filter
#[test]
fn test_filtered_rows_eq() {
    let mut table = create_table(&[("x", DuckType::Integer)]);
    table.name = "f".to_string();
    for i in 0..10 {
        append_row(&mut table, &[DuckValue::Int(i)]);
    }

    let rows = filtered_rows(&table, &FilterExpr::Eq(0, DuckValue::Int(5)));
    assert_eq!(rows, vec![5]);
}

/// Test: filtered_rows with AND filter
#[test]
fn test_filtered_rows_and() {
    let mut table = create_table(&[("x", DuckType::Integer)]);
    table.name = "f".to_string();
    for i in 0..10 {
        append_row(&mut table, &[DuckValue::Int(i)]);
    }

    let rows = filtered_rows(
        &table,
        &FilterExpr::And(
            Box::new(FilterExpr::Gt(0, DuckValue::Int(2))),
            Box::new(FilterExpr::Lt(0, DuckValue::Int(7))),
        ),
    );
    assert_eq!(rows, vec![3, 4, 5, 6]);
}
