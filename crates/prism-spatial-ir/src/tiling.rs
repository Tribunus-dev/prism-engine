//! Target-aware tile and threadgroup geometry validation.

use crate::graph::TileGeometry;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TilingBackend {
    Metal,
    AnEngine,
    Cpu,
    Xdna,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TilingConfiguration {
    pub geometry: TileGeometry,
    pub backend: TilingBackend,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error, Serialize, Deserialize)]
pub enum TilingValidationError {
    #[error("tile dimensions must be non-zero (got {width}x{height})")]
    ZeroDimension { width: usize, height: usize },
    #[error(
        "{backend:?} tile dimensions {width}x{height} exceed the maximum {max_width}x{max_height}"
    )]
    DimensionsTooLarge {
        backend: TilingBackend,
        width: usize,
        height: usize,
        max_width: usize,
        max_height: usize,
    },
    #[error("{backend:?} tile area {area} exceeds the maximum of {max_area}")]
    AreaTooLarge {
        backend: TilingBackend,
        area: usize,
        max_area: usize,
    },
}

pub fn validate_tiling_geometry(
    geometry: TileGeometry,
    backend: TilingBackend,
) -> Result<usize, TilingValidationError> {
    if geometry.width == 0 || geometry.height == 0 {
        return Err(TilingValidationError::ZeroDimension {
            width: geometry.width,
            height: geometry.height,
        });
    }
    let (max_width, max_height, max_area) = match backend {
        TilingBackend::Metal => (32, 32, 1024),
        TilingBackend::AnEngine => (256, 256, 65_536),
        TilingBackend::Cpu => (usize::MAX, usize::MAX, usize::MAX),
        TilingBackend::Xdna => (64, 64, 4096),
    };
    if geometry.width > max_width || geometry.height > max_height {
        return Err(TilingValidationError::DimensionsTooLarge {
            backend,
            width: geometry.width,
            height: geometry.height,
            max_width,
            max_height,
        });
    }
    let area = geometry
        .width
        .checked_mul(geometry.height)
        .unwrap_or(usize::MAX);
    if area > max_area {
        return Err(TilingValidationError::AreaTooLarge {
            backend,
            area,
            max_area,
        });
    }
    Ok(area)
}

pub fn validate_joint_tiling_geometry(
    geometry: TileGeometry,
) -> Result<usize, TilingValidationError> {
    validate_tiling_geometry(geometry, TilingBackend::Metal)
}
