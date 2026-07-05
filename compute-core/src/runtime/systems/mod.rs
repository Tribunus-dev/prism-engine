//! ECS systems for the Prism Engine runtime (Slice 2).

#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod audio;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod inference;
pub mod inference_step;
pub mod npu;
#[cfg(feature = "mlx-backend")] // research surface: MLX executor/model stack
pub mod vision;
pub mod worker;
