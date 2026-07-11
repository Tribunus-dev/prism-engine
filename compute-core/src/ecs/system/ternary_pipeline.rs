//! ECS-native ternary cimage format handling — wraps `compute_image::compile::ternary`.
//!
//! Reads weight segments from tensor entities, writes a cimage header via
//! `write_cimage_header_le`, and stores the resulting binary as
//! `CimageBinaryComp`.

use crate::ecs::component::model_source::CimageBinaryComp;
use crate::ecs::compute_image::compile::ternary::{
    write_cimage_header_le, CimageHeader, SegmentEntry, SegmentKind, CIMAGE_SEGMENT_CAPACITY,
};
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};
use sha2::{Digest, Sha256};

/// Build a sealed cimage binary from the ECS world state.
///
/// Collects segment metadata from entities carrying component data,
/// writes the canonical LE header, and appends segment payloads.
/// The result is stored as `CimageBinaryComp` on the model entity.
pub struct TertiaryPipelineSystem {
    /// The quantization schema tag to embed in the header.
    pub quant_schema: u32,
}

impl TertiaryPipelineSystem {
    pub fn new() -> Self {
        Self { quant_schema: 0 } // TernaryTile640
    }

    pub fn with_schema(quant_schema: u32) -> Self {
        Self { quant_schema }
    }
}

impl Default for TertiaryPipelineSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl CompilerSystem for TertiaryPipelineSystem {
    fn name(&self) -> &str {
        "TertiaryPipelineSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Packaging
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let model_entities = world.entities_of_kind(EntityKind::Model);
        let model_entity = *model_entities
            .first()
            .ok_or_else(|| anyhow::anyhow!("no model entity found for cimage assembly"))?;

        let _model_name = world.name(model_entity).unwrap_or("model").to_string();

        // Collect segment metadata from tensor entities.
        let mut segments = [SegmentEntry {
            kind: 0,
            offset: 0,
            length: 0,
        }; CIMAGE_SEGMENT_CAPACITY];
        let mut seg_count = 0u32;

        let tensor_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);
        for &_entity in &tensor_entities {
            if seg_count >= CIMAGE_SEGMENT_CAPACITY as u32 {
                break;
            }
            let seg_kind = SegmentKind::Nf4Tile640Weights as u32;
            segments[seg_count as usize] = SegmentEntry {
                kind: seg_kind,
                offset: 0, // computed by final packer
                length: 0, // computed by final packer
            };
            seg_count += 1;
        }

        let header = CimageHeader {
            magic: crate::ecs::compute_image::compile::ternary::PRISM_MAGIC,
            version: 2,
            segment_count: seg_count,
            payload_hash: [0u8; 32],
            num_layers: 0,
            num_heads: 0,
            head_dim: 0,
            hidden_dim: 0,
            intermediate_dim: 0,
            vocab_size: 0,
            quantization_schema: self.quant_schema,
            draft_num_layers: 0,
            segments,
            _pad: [0u8; 8],
        };

        // Serialise header to a buffer → we'll embed it in the payload below.
        let mut header_buf = Vec::with_capacity(4096);
        write_cimage_header_le(&mut header_buf, &header)
            .map_err(|e| anyhow::anyhow!("write cimage header: {e}"))?;

        // Compute payload hash over header.
        let hash = Sha256::digest(&header_buf);

        // Update header with hash and re-serialise.
        let mut final_header = header;
        final_header.payload_hash.copy_from_slice(&hash);
        let mut buf = Vec::with_capacity(16384);
        write_cimage_header_le(&mut buf, &final_header)
            .map_err(|e| anyhow::anyhow!("write final cimage header: {e}"))?;

        // Store on model entity.
        world.add_component(model_entity, CimageBinaryComp(buf));

        Ok(())
    }
}
