pub mod auth;
pub mod benchmark;
pub mod cpu;
pub mod models;
pub mod rate_limiter;

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod admin;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod engine;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod routes;
