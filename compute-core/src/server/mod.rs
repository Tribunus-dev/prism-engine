pub mod auth;
pub mod benchmark;
pub mod cpu;
#[cfg(feature = "server-dashboard")]
pub mod dashboard;
pub mod models;
pub mod rate_limiter;

#[cfg(feature = "mlx-backend")]
pub mod admin;
#[cfg(feature = "prism-backend")]
pub mod distill_worker;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod engine;
#[cfg(feature = "prism-backend")]
pub mod idle_detector;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod routes;
#[cfg(feature = "prism-backend")]
pub mod state;
