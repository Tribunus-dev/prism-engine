//! Hardware-backed fitness evaluator with DB projections.
//!
//! The MeasuredEvaluator replaces synthetic scoring for top Pareto-frontier
//! candidates with real hardware measurements. It:
//!
//! 1. Checks DuckDB projections first (persistent cache across restarts).
//! 2. On cache miss, dispatches a real matmul kernel for the candidate's
//!    format via Metal (when `metal-dispatch` is enabled) or CPU fallback
//!    (Accelerate / per-format dequant + BLAS).
//! 3. Records wall-clock throughput in tokens/second.
//! 4. Writes the result into both an in-memory cache and DuckDB.

use parking_lot::Mutex;
use prism_ecs_ir::evolution::evaluate::EvaluationStrategy;
use prism_ecs_ir::evolution::foundation::{CandidateGenome, FitnessScore, RepresentationAxis};
use prism_ecs_ir::evolution::mutation_table::TensorFormat;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

/// A measured evaluator that checks DuckDB projections first,
/// then compiles + runs on hardware for cache misses.
pub struct MeasuredEvaluator {
    /// Cache of (dim_m, dim_n, format) -> avg_tok_sec (interior mutability
    /// so [`store`] works from `EvaluationStrategy::evaluate(&self, ...)`).
    cache: Mutex<HashMap<(u32, u32, String), f64>>,
    /// Minimum synthetic fitness to bother measuring.
    min_fitness: f64,
    /// Database connection for persistent projections.
    #[cfg(feature = "duckdb-projections")]
    duckdb: Mutex<Option<duckdb::Connection>>,
}

impl MeasuredEvaluator {
    pub fn new(min_fitness: f64) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            min_fitness,
            #[cfg(feature = "duckdb-projections")]
            duckdb: Mutex::new(None),
        }
    }

    #[cfg(feature = "duckdb-projections")]
    pub fn with_duckdb(self, connection: duckdb::Connection) -> Self {
        *self.duckdb.lock() = Some(connection);
        self
    }

    /// Derive tensor dimensions from the evaluation context.
    ///
    /// Context is serialized as `"dim_m,dim_n"`. Defaults to `(4096, 4096)`
    /// (typical attention Q/K projection size) when parsing fails.
    fn derive_tensor_shape(&self, context: &[u8]) -> (u32, u32) {
        if let Ok(ctx_str) = std::str::from_utf8(context) {
            if let Some((m, n)) = ctx_str.split_once(',') {
                if let (Ok(m), Ok(n)) = (m.trim().parse(), n.trim().parse()) {
                    return (m, n);
                }
            }
        }
        (4096, 4096)
    }

    /// Human-readable format label.
    fn format_label(repr: &RepresentationAxis) -> &'static str {
        match repr {
            RepresentationAxis::Fp16 => "FP16",
            RepresentationAxis::Bf16 => "BF16",
            RepresentationAxis::Int8 => "Int8",
            RepresentationAxis::Int4 => "Int4",
            RepresentationAxis::Nf4 => "NF4",
            RepresentationAxis::Nf8 => "NF8",
            RepresentationAxis::Ternary158 => "Ternary158",
            RepresentationAxis::Binary1 => "Binary1",
        }
    }

    /// Look up a cached benchmark result.
    fn lookup(&self, dim_m: u32, dim_n: u32, format: &str) -> Option<f64> {
        let key = (dim_m, dim_n, format.to_string());
        if let Some(&tok_sec) = self.cache.lock().get(&key) {
            return Some(tok_sec);
        }
        // Try DuckDB query
        #[cfg(feature = "duckdb-projections")]
        if let Some(conn) = &*self.duckdb.lock() {
            let sql = format!(
                "SELECT avg_tok_sec FROM tensor_benchmarks WHERE dim_m = ? AND dim_n = ? AND format = ?"
            );
            if let Ok(mut stmt) = conn.prepare(&sql) {
                if let Ok(row) = stmt.query_row(duckdb::params![dim_m, dim_n, format], |row| {
                    row.get::<_, f64>(0)
                }) {
                    return Some(row);
                }
            }
        }
        None
    }

    /// Store a benchmark result into cache and DuckDB.
    fn store(&self, dim_m: u32, dim_n: u32, format: &str, tok_sec: f64) {
        let key = (dim_m, dim_n, format.to_string());
        self.cache.lock().insert(key, tok_sec);

        #[cfg(feature = "duckdb-projections")]
        if let Some(conn) = &*self.duckdb.lock() {
            let sql = r#"
                INSERT INTO tensor_benchmarks (dim_m, dim_n, format, avg_tok_sec)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (dim_m, dim_n, format)
                DO UPDATE SET avg_tok_sec = (avg_tok_sec + ?) / 2.0,
                              sample_count = sample_count + 1
            "#;
            let _ = conn.execute(sql, duckdb::params![dim_m, dim_n, format, tok_sec, tok_sec]);
        }
    }

    /// Measure a candidate format on real hardware and return throughput
    /// (tokens/second, higher = better).
    ///
    /// Strategy:
    /// 1. Generate random weight data for the given (dim_m, dim_n, format).
    /// 2. Try Metal dispatch (compile + run a format-aware GEMV kernel).
    /// 3. If Metal isn't available or returns error, fall back to CPU
    ///    dispatch (per-format dequant + BLAS/Accelerate GEMV).
    /// 4. Measure wall-clock time and convert to tok/sec.
    fn measure_on_hardware(&self, dim_m: u32, dim_n: u32, format: &str) -> Result<f64, String> {
        let tensor_format = format_from_label(format)?;
        let weight_bytes = estimate_weight_bytes(dim_m, dim_n, format)?;
        let mut weight_data = vec![0u8; weight_bytes];
        let mut rng = rand::thread_rng();
        rng.fill(&mut weight_data[..]);

        // Input vector: random FP32 activations of length dim_n
        let n_usize = dim_n as usize;
        let mut input = vec![0.0f32; n_usize];
        rng.fill(&mut input[..]);

        // Warm-up run (discard)
        let _ = dispatch_benchmark_run(dim_m, dim_n, &input, &weight_data, &tensor_format);

        // Timed runs — take the minimum of 3 for consistent measurement
        let mut times = Vec::with_capacity(3);
        for _ in 0..3 {
            let start = Instant::now();
            dispatch_benchmark_run(dim_m, dim_n, &input, &weight_data, &tensor_format)?;
            times.push(start.elapsed());
        }

        let min_duration = times.into_iter().min().unwrap();
        let secs = min_duration.as_secs_f64();

        if secs < 1e-12 {
            return Err("measurement duration too short".to_string());
        }

        // Throughput: tokens/second. One token = one GEMV for dim_m output rows.
        let tok_sec = 1.0 / secs;
        Ok(tok_sec)
    }
}

/// Dispatch one benchmark GEMV via Metal or CPU.
fn dispatch_benchmark_run(
    dim_m: u32,
    dim_n: u32,
    input: &[f32],
    weight_data: &[u8],
    format: &TensorFormat,
) -> Result<Vec<f32>, String> {
    // Try Metal first (feature-gated)
    #[cfg(feature = "metal-dispatch")]
    {
        let result = crate::metal::dispatch_matmul(
            "_benchmark_tensor",
            input,
            weight_data,
            dim_m,
            dim_n,
            format,
        );
        if result.is_ok() {
            return result;
        }
    }

    // Fall back to CPU dispatch (Accelerate / per-format dequant)
    crate::cpu::matmul(input, weight_data, dim_m, dim_n, format)
}

/// Parse a human-readable format label back into `TensorFormat`.
fn format_from_label(label: &str) -> Result<TensorFormat, String> {
    match label {
        "FP16" => Ok(TensorFormat::Fp16),
        "BF16" => Ok(TensorFormat::Bf16),
        "Int8" => Ok(TensorFormat::Int8),
        "Int4" => Ok(TensorFormat::Int4),
        "NF4" => Ok(TensorFormat::Nf4),
        "NF8" => Ok(TensorFormat::Nf8),
        "Ternary158" => Ok(TensorFormat::Ternary158),
        "Binary1" => Ok(TensorFormat::Binary1),
        other => Err(format!("unknown format label: {other}")),
    }
}

/// Calculate the byte size of weight data for the given shape and format.
fn estimate_weight_bytes(dim_m: u32, dim_n: u32, format: &str) -> Result<usize, String> {
    let bpp = match format {
        "FP16" => 16u32,
        "BF16" => 16,
        "Int8" => 8,
        "Int4" => 4,
        "NF4" => 4,
        "NF8" => 8,
        "Ternary158" => 2,
        "Binary1" => 1,
        other => return Err(format!("unknown format for size estimation: {other}")),
    };
    let total_entries = dim_m as u64 * dim_n as u64;
    let bits = total_entries * bpp as u64;
    Ok(((bits + 7) / 8) as usize)
}

/// Bandwidth-based throughput estimate for fallback when hardware dispatch
/// is unavailable.
fn estimate_tok_sec(dim_m: u32, dim_n: u32, format: &str) -> f64 {
    let bpp = match format {
        "FP16" => 16.0,
        "BF16" => 16.0,
        "Int8" => 8.0,
        "Int4" => 4.0,
        "NF4" => 4.0,
        "NF8" => 8.0,
        "Ternary158" => 1.58,
        "Binary1" => 1.0,
        _ => 16.0,
    };
    let byte_count = (dim_m as f64 * dim_n as f64 * bpp) / 8.0;
    let throughput = 100.0e9 / byte_count; // 100 GB/s memory bandwidth ÷ bytes per token
    throughput.min(1e7)
}

impl EvaluationStrategy for MeasuredEvaluator {
    fn evaluate(&self, genome: &CandidateGenome, context: &[u8]) -> FitnessScore {
        let (dim_m, dim_n) = self.derive_tensor_shape(context);
        let format = Self::format_label(&genome.representation);

        match self.lookup(dim_m, dim_n, format) {
            Some(tok_sec) => {
                // Cache hit — normalize to 0..1 range; cap at 10M tok/s = 1.0
                FitnessScore::new((tok_sec / 10_000_000.0).min(1.0))
            }
            None => {
                // Gate hardware measurement behind min_fitness threshold.
                // Derive a cheap proxy: lower bit width ≈ lower memory cost ≈ higher fitness.
                let synthetic = estimate_tok_sec(dim_m, dim_n, format) / 10_000_000.0;
                if synthetic < self.min_fitness {
                    return FitnessScore::new(synthetic);
                }

                // First time seeing this format+shape — measure on hardware.
                match self.measure_on_hardware(dim_m, dim_n, format) {
                    Ok(tok_sec) => {
                        self.store(dim_m, dim_n, format, tok_sec);
                        FitnessScore::new((tok_sec / 10_000_000.0).min(1.0))
                    }
                    Err(_e) => {
                        // Hardware dispatch failed — fall back to bandwidth estimate.
                        FitnessScore::new(synthetic)
                    }
                }
            }
        }
    }

    fn name(&self) -> &str {
        "measured-evaluator"
    }
}
