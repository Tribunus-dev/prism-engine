//! nf4tile640 format constants.
//!
//! These constants define the canonical Tile640 quantization format used
//! by the Metal and ANE backends. They are pure integer constants with no
//! native dependencies, extracted here so both `tribunus-compute-core` and
//! `prism-ecs-backend` can reference them without creating a dependency cycle.

/// Number of elements in a quantization group.
pub const GROUP_SIZE: usize = 128;

/// Number of groups per tile (640 / 128).
pub const GROUPS_PER_TILE: usize = 5;

/// Total elements per tile — the tuning target for GPU workgroup convergence.
pub const TILE_ELEMENTS: usize = 640;

/// Bytes of packed codes per tile (640 / 2).
pub const PACKED_BYTES_PER_TILE: usize = TILE_ELEMENTS / 2; // 320

/// Number of f32 scale values per tile (one per group).
pub const SCALES_F32_PER_TILE: usize = GROUPS_PER_TILE; // 5
