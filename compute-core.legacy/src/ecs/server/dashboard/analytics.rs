//! Analytical queries and DuckDB materialized views for the evidence dashboard.
//!
//! Provides:
//! - `create_duckdb_views` — creates analytical views in DuckDB backed by PostgreSQL
//! - `get_scatter_data` — admission scatterplot data via sqlx/PgPool
//! - `get_sweep_contour` — sweep contour data via DuckDB

use duckdb::{Connection, Result as DuckResult};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

/// One row in the admission scatterplot, joining `admission_receipts` with
/// `cimage_tensors` to surface the tensor class.
///
/// NUMERIC columns are cast to `DOUBLE PRECISION` in SQL because sqlx 0.8
/// does not map PostgreSQL NUMERIC → f64 without the `bigdecimal` feature.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct DashboardAdmissionSummary {
    pub receipt_id: Uuid,
    pub artifact_digest: String,
    pub tensor_key: String,
    pub codec: String,
    pub group_size: i32,
    pub effective_bpw: Option<f64>,
    pub zero_fraction: Option<f64>,
    pub neg_fraction: Option<f64>,
    pub pos_fraction: Option<f64>,
    pub scale_mean: Option<f64>,
    pub scale_std: Option<f64>,
    pub operator_nrmse: Option<f64>,
    pub output_cosine: Option<f64>,
    pub activation_shift_l2: Option<f64>,
    pub deadzone_collapse: bool,
    pub rescue_required: bool,
    pub rescue_codec: Option<String>,
    pub promotion_status: String,
    pub raw_json: serde_json::Value,
    pub created_at: String,
    pub tensor_class: String,
}

/// A single contour point from a sweep analysis, grouped by group-size and
/// calibration-steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContourPoint {
    pub group_size: i32,
    pub calibration_steps: i32,
    pub avg_nrmse: f64,
    pub passed_count: i64,
}

/// Environment variable used to obtain the PostgreSQL connection string for
/// the DuckDB `ATTACH DATABASE` statement.
const PG_URL_ENV: &str = "DASHBOARD_PG_URL";

/// Default PostgreSQL connection string used when the env var is not set.
/// This connects via the DuckDB postgres-scanner to a local PG instance.
const DEFAULT_PG_URL: &str = "host=localhost port=5432 dbname=tribunus user=tribunus";

/// Create the analytical views inside DuckDB.
///
/// The connection must already be open (in-memory or file-backed). This
/// function:
/// 1. ATTACHes the PostgreSQL database via the postgres-scanner (URL from
///    `DASHBOARD_PG_URL` env var, or a localhost default).
/// 2. Creates `admission_scatter_view` — scatterplot data joining
///    `admission_receipts` → `cimage_tensors` → `cimage_artifacts`.
/// 3. Creates `sweep_contour_view` — aggregated contour data from sweeps.
///
/// Both views are created with `CREATE OR REPLACE VIEW` and live in the
/// DuckDB default schema (not the attached `pg` schema).
pub fn create_duckdb_views(conn: &Connection) -> DuckResult<()> {
    let pg_url = std::env::var(PG_URL_ENV).unwrap_or_else(|_| DEFAULT_PG_URL.to_string());

    conn.execute_batch(&format!(
        r#"
        ATTACH DATABASE '{pg_url}' AS pg (TYPE postgres);

        CREATE OR REPLACE VIEW admission_scatter_view AS
        SELECT
            ar.receipt_id,
            ar.tensor_key,
            ar.codec,
            ar.group_size,
            ar.effective_bpw,
            ar.operator_nrmse,
            ar.output_cosine,
            ar.zero_fraction,
            ar.activation_shift_l2,
            ar.deadzone_collapse,
            ar.promotion_status,
            ct.tensor_class,
            ca.artifact_digest
        FROM pg.admission_receipts ar
        JOIN pg.cimage_tensors ct
            ON ct.artifact_digest = ar.artifact_digest
            AND ct.tensor_key = ar.tensor_key
        JOIN pg.cimage_artifacts ca
            ON ca.digest = ar.artifact_digest;

        CREATE OR REPLACE VIEW sweep_contour_view AS
        SELECT
            sc.group_size,
            sc.calibration_steps,
            AVG(sc.nrmse) AS avg_nrmse,
            COUNT(*) AS candidates,
            SUM(CASE WHEN sc.passed THEN 1 ELSE 0 END) AS passed,
            s.tensor_key,
            s.artifact_digest
        FROM pg.sweep_candidates sc
        JOIN pg.sweeps s ON s.sweep_id = sc.sweep_id
        GROUP BY sc.group_size, sc.calibration_steps, s.tensor_key, s.artifact_digest;
        "#,
    ))?;

    Ok(())
}

/// Fetch admission-scatter data for a single artifact digest from PostgreSQL.
///
/// Joins `admission_receipts` with `cimage_tensors` to include the tensor
/// class, ordered by `effective_bpw` (ascending). NUMERIC columns are cast to
/// `DOUBLE PRECISION` for direct f64 deserialization.
pub async fn get_scatter_data(
    pool: &PgPool,
    artifact_digest: &str,
) -> Result<Vec<DashboardAdmissionSummary>, sqlx::Error> {
    sqlx::query_as::<_, DashboardAdmissionSummary>(
        r#"
        SELECT
            ar.receipt_id               AS "receipt_id",
            ar.artifact_digest           AS "artifact_digest",
            ar.tensor_key                AS "tensor_key",
            ar.codec                     AS "codec",
            ar.group_size                AS "group_size",
            ar.effective_bpw::DOUBLE PRECISION    AS "effective_bpw",
            ar.zero_fraction::DOUBLE PRECISION    AS "zero_fraction",
            ar.neg_fraction::DOUBLE PRECISION     AS "neg_fraction",
            ar.pos_fraction::DOUBLE PRECISION     AS "pos_fraction",
            ar.scale_mean::DOUBLE PRECISION       AS "scale_mean",
            ar.scale_std::DOUBLE PRECISION        AS "scale_std",
            ar.operator_nrmse::DOUBLE PRECISION   AS "operator_nrmse",
            ar.output_cosine::DOUBLE PRECISION    AS "output_cosine",
            ar.activation_shift_l2::DOUBLE PRECISION AS "activation_shift_l2",
            ar.deadzone_collapse         AS "deadzone_collapse",
            ar.rescue_required           AS "rescue_required",
            ar.rescue_codec              AS "rescue_codec",
            ar.promotion_status          AS "promotion_status",
            ar.raw_json                  AS "raw_json",
ar.created_at::TEXT          AS "created_at",
            ct.tensor_class              AS "tensor_class"
        FROM admission_receipts ar
        JOIN cimage_tensors ct
            ON ct.artifact_digest = ar.artifact_digest
            AND ct.tensor_key = ar.tensor_key
        WHERE ar.artifact_digest = $1
        ORDER BY ar.effective_bpw
        "#,
    )
    .bind(artifact_digest)
    .fetch_all(pool)
    .await
}

/// Query the DuckDB `sweep_contour_view` for a specific artifact and tensor.
///
/// Returns aggregated contour points (`group_size`, `calibration_steps`,
/// `avg_nrmse`, `passed_count`) suitable for a contour/heatmap chart.
///
/// # Panics
/// If the view has not been created via `create_duckdb_views`.
pub fn get_sweep_contour(
    duck: &Connection,
    artifact_digest: &str,
    tensor_key: &str,
) -> DuckResult<Vec<ContourPoint>> {
    let mut stmt = duck.prepare(
        r#"
        SELECT
            group_size,
            calibration_steps,
            avg_nrmse,
            passed
        FROM sweep_contour_view
        WHERE artifact_digest = ?1
          AND tensor_key = ?2
        ORDER BY group_size, calibration_steps
        "#,
    )?;

    let rows = stmt.query_map([artifact_digest, tensor_key], |row| {
        Ok(ContourPoint {
            group_size: row.get(0)?,
            calibration_steps: row.get(1)?,
            avg_nrmse: row.get(2)?,
            passed_count: row.get(3)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}
