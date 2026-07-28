#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
//! Shape-specialized program variant.

<<<<<<<< HEAD:compute-core/src/ecs/compute_image/legacy_compute_image_runtime/executable/variant.rs
use crate::ecs::compute_image::legacy_compute_image_runtime::program::phase_program::SerializedPhaseProgram;
|||||||| e64c7d94:compute-core/src/ecs/compute_image/executable/variant.rs
use crate::ecs::compute_image::program::phase_program::SerializedPhaseProgram;
========
use crate::ecs::legacy_compute_image_core::program::phase_program::SerializedPhaseProgram;
>>>>>>>> migrate/ci-core:compute-core/src/ecs/legacy_compute_image_core/executable/variant.rs
use crate::integration::ContentHash;
use serde::{Deserialize, Serialize};

pub type ShapeSpecializedVariantId = String;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeSpecializedProgram {
    pub variant_id: ShapeSpecializedVariantId,
    pub shape_profile: ShapeProfile,
    pub phase_program: SerializedPhaseProgram,
    pub program_hash: ContentHash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeProfile {
    pub max_batch: u32,
    pub max_tokens: u32,
    pub label: String,
}
