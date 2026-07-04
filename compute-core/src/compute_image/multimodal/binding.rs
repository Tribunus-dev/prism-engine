//! Runtime binding for multimodal projector artifacts sealed into a `.cimage`.

use crate::compute_image::cimage_loader::CimageDeployment;
use crate::compute_image::compile::execution_graph::{ExecutionGraphDescriptor, NodeKind};
use crate::compute_image::compile::ternary::{verify_cimage, SegmentEntry, SegmentKind};
use crate::compute_image::multimodal::{
    MultimodalArtifactSummary, MultimodalCapabilities, MultimodalInputDescriptorV1,
    ProjectionPrecision, ProjectionRole, ProjectionTensorRecord,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SealedSegmentBinding {
    pub segment_index: u16,
    pub kind: SegmentKind,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct ProjectionTensorBinding {
    pub role: ProjectionRole,
    pub record: ProjectionTensorRecord,
    pub weights: SealedSegmentBinding,
    pub scales: Option<SealedSegmentBinding>,
}

#[derive(Debug, Clone)]
pub struct SealedMultimodalBindings {
    pub descriptor: MultimodalInputDescriptorV1,
    pub summary: MultimodalArtifactSummary,
    pub capabilities: MultimodalCapabilities,
    pub projection_precision: ProjectionPrecision,
    pub weight_segment: SealedSegmentBinding,
    pub scale_segment: Option<SealedSegmentBinding>,
    pub position_segment: Option<SealedSegmentBinding>,
    pub auxiliary_segment: Option<SealedSegmentBinding>,
    pub image_projection_bindings: Vec<ProjectionTensorBinding>,
    pub audio_projection_bindings: Vec<ProjectionTensorBinding>,
}

impl SealedMultimodalBindings {
    pub fn from_deployment(deployment: &CimageDeployment) -> Result<Self, String> {
        let desc = deployment
            .multimodal_descriptor()
            .ok_or_else(|| "missing multimodal descriptor".to_string())?;
        let records = deployment.multimodal_projection_records();
        let summary = deployment.multimodal_artifact_summary();
        let capabilities = deployment.multimodal_capabilities();
        let precision = summary.projection_precision;
        Self::from_parts(
            &deployment.mmap_data,
            desc,
            summary,
            capabilities,
            precision,
            &records,
        )
    }

    pub fn supports_image(&self) -> bool {
        self.capabilities.image
    }

    pub fn supports_audio(&self) -> bool {
        self.capabilities.audio
    }

    pub fn image_patch_embedding(&self) -> Option<&ProjectionTensorBinding> {
        self.image_projection_bindings
            .iter()
            .find(|binding| binding.role == ProjectionRole::ImagePatchEmbedding)
    }

    pub fn image_projection(&self) -> Option<&ProjectionTensorBinding> {
        self.image_projection_bindings
            .iter()
            .find(|binding| binding.role == ProjectionRole::ImageProjection)
    }

    pub fn image_position_embedding(&self) -> Option<&ProjectionTensorBinding> {
        self.image_projection_bindings
            .iter()
            .find(|binding| binding.role == ProjectionRole::ImagePositionEmbedding)
    }

    pub fn image_pooling(&self) -> Option<&ProjectionTensorBinding> {
        self.image_projection_bindings
            .iter()
            .find(|binding| binding.role == ProjectionRole::ImagePooling)
    }

    pub fn audio_projection(&self) -> Option<&ProjectionTensorBinding> {
        self.audio_projection_bindings
            .iter()
            .find(|binding| binding.role == ProjectionRole::AudioProjection)
    }

    pub fn binding_for_node_kind(&self, node_kind: NodeKind) -> Option<&ProjectionTensorBinding> {
        match node_kind {
            NodeKind::VisionPatchEmbed => self.image_patch_embedding(),
            NodeKind::VisionFinalProjection => self.image_projection(),
            NodeKind::AudioFrameEmbed => self
                .audio_projection_bindings
                .iter()
                .find(|binding| binding.role == ProjectionRole::AudioFrameEmbedding),
            NodeKind::AudioProjection => self.audio_projection(),
            _ => None,
        }
    }

    pub fn validate_node_binding(
        &self,
        node: &crate::compute_image::compile::execution_graph::LayerExecutionNode,
    ) -> Result<&ProjectionTensorBinding, String> {
        let node_kind = match node.node_kind {
            x if x == NodeKind::VisionPatchEmbed as u8 => NodeKind::VisionPatchEmbed,
            x if x == NodeKind::VisionFinalProjection as u8 => NodeKind::VisionFinalProjection,
            x if x == NodeKind::AudioFrameEmbed as u8 => NodeKind::AudioFrameEmbed,
            x if x == NodeKind::AudioProjection as u8 => NodeKind::AudioProjection,
            other => {
                return Err(format!(
                    "node kind {} is not a multimodal projection node",
                    other
                ));
            }
        };
        let binding = self
            .binding_for_node_kind(node_kind)
            .ok_or_else(|| format!("no sealed multimodal binding for {:?}", node_kind))?;
        if binding.record.output_width != node.hidden_dim {
            return Err(format!(
                "multimodal node hidden_dim mismatch for {:?}: graph={}, binding={}",
                node_kind, node.hidden_dim, binding.record.output_width
            ));
        }
        if node.weight_length != 0 && binding.record.weight_length != node.weight_length {
            return Err(format!(
                "multimodal node weight_length mismatch for {:?}: graph={}, binding={}",
                node_kind, node.weight_length, binding.record.weight_length
            ));
        }
        if node.weight_offset != binding.record.weight_offset {
            return Err(format!(
                "multimodal node weight_offset mismatch for {:?}: graph={}, binding={}",
                node_kind, node.weight_offset, binding.record.weight_offset
            ));
        }
        if node.scale_offset != binding.record.scale_offset {
            return Err(format!(
                "multimodal node scale_offset mismatch for {:?}: graph={}, binding={}",
                node_kind, node.scale_offset, binding.record.scale_offset
            ));
        }
        Ok(binding)
    }

    pub fn validate_graph_multimodal_prefix(
        &self,
        graph: &ExecutionGraphDescriptor,
    ) -> Result<usize, String> {
        for (index, node) in graph.layers.iter().enumerate() {
            match node.node_kind {
                x if x == NodeKind::VisionPatchEmbed as u8
                    || x == NodeKind::VisionFinalProjection as u8
                    || x == NodeKind::AudioFrameEmbed as u8
                    || x == NodeKind::AudioProjection as u8 =>
                {
                    self.validate_node_binding(node)?;
                }
                x if x == NodeKind::EmbeddingAssembly as u8 => {}
                _ => return Ok(index),
            }
        }
        Ok(graph.layers.len())
    }

    pub fn ready_for_direct_image_projection(&self) -> bool {
        self.supports_image()
            && self.image_patch_embedding().is_some()
            && self.image_projection().is_some()
            && self.position_segment.is_some()
    }

    fn from_parts(
        mmap_data: &[u8],
        descriptor: MultimodalInputDescriptorV1,
        summary: MultimodalArtifactSummary,
        capabilities: MultimodalCapabilities,
        projection_precision: ProjectionPrecision,
        records: &[ProjectionTensorRecord],
    ) -> Result<Self, String> {
        let (header, _) = verify_cimage(mmap_data)?;
        let weight_segment = resolve_segment(
            &header.segments,
            header.segment_count as usize,
            descriptor.projection_weight_segment_index,
            SegmentKind::MultimodalProjectionWeights,
        )?;
        let scale_segment = resolve_optional_segment(
            &header.segments,
            header.segment_count as usize,
            descriptor.projection_scale_segment_index,
            SegmentKind::MultimodalProjectionScales,
        )?;
        let position_segment = resolve_optional_segment(
            &header.segments,
            header.segment_count as usize,
            descriptor.position_embedding_segment_index,
            SegmentKind::MultimodalPositionEmbeddings,
        )?;
        let auxiliary_segment = resolve_optional_segment(
            &header.segments,
            header.segment_count as usize,
            descriptor.auxiliary_weight_segment_index,
            SegmentKind::MultimodalAuxiliaryWeights,
        )?;

        let mut image_projection_bindings = Vec::new();
        let mut audio_projection_bindings = Vec::new();
        for record in records {
            let Some(role) = projection_role_from_raw(record.role) else {
                continue;
            };
            let binding = ProjectionTensorBinding {
                role,
                record: *record,
                weights: weight_segment,
                scales: scale_segment,
            };
            if is_image_role(role) {
                image_projection_bindings.push(binding);
            } else if is_audio_role(role) {
                audio_projection_bindings.push(binding);
            }
        }

        Ok(Self {
            descriptor,
            summary,
            capabilities,
            projection_precision,
            weight_segment,
            scale_segment,
            position_segment,
            auxiliary_segment,
            image_projection_bindings,
            audio_projection_bindings,
        })
    }

    #[cfg(test)]
    fn from_verified_parts(
        segments: &[SegmentEntry],
        segment_count: usize,
        descriptor: MultimodalInputDescriptorV1,
        summary: MultimodalArtifactSummary,
        capabilities: MultimodalCapabilities,
        projection_precision: ProjectionPrecision,
        records: &[ProjectionTensorRecord],
    ) -> Result<Self, String> {
        let weight_segment = resolve_segment(
            segments,
            segment_count,
            descriptor.projection_weight_segment_index,
            SegmentKind::MultimodalProjectionWeights,
        )?;
        let scale_segment = resolve_optional_segment(
            segments,
            segment_count,
            descriptor.projection_scale_segment_index,
            SegmentKind::MultimodalProjectionScales,
        )?;
        let position_segment = resolve_optional_segment(
            segments,
            segment_count,
            descriptor.position_embedding_segment_index,
            SegmentKind::MultimodalPositionEmbeddings,
        )?;
        let auxiliary_segment = resolve_optional_segment(
            segments,
            segment_count,
            descriptor.auxiliary_weight_segment_index,
            SegmentKind::MultimodalAuxiliaryWeights,
        )?;

        let mut image_projection_bindings = Vec::new();
        let mut audio_projection_bindings = Vec::new();
        for record in records {
            let Some(role) = projection_role_from_raw(record.role) else {
                continue;
            };
            let binding = ProjectionTensorBinding {
                role,
                record: *record,
                weights: weight_segment,
                scales: scale_segment,
            };
            if is_image_role(role) {
                image_projection_bindings.push(binding);
            } else if is_audio_role(role) {
                audio_projection_bindings.push(binding);
            }
        }

        Ok(Self {
            descriptor,
            summary,
            capabilities,
            projection_precision,
            weight_segment,
            scale_segment,
            position_segment,
            auxiliary_segment,
            image_projection_bindings,
            audio_projection_bindings,
        })
    }
}

fn resolve_segment(
    segments: &[SegmentEntry],
    segment_count: usize,
    segment_index: u16,
    expected_kind: SegmentKind,
) -> Result<SealedSegmentBinding, String> {
    let entry = segments
        .get(segment_index as usize)
        .copied()
        .filter(|_| (segment_index as usize) < segment_count)
        .ok_or_else(|| format!("segment index {} out of range", segment_index))?;
    if entry.kind != expected_kind as u32 {
        return Err(format!(
            "segment index {} kind mismatch: expected {:?}, found {}",
            segment_index, expected_kind, entry.kind
        ));
    }
    Ok(SealedSegmentBinding {
        segment_index,
        kind: expected_kind,
        offset: entry.offset,
        length: entry.length,
    })
}

fn resolve_optional_segment(
    segments: &[SegmentEntry],
    segment_count: usize,
    segment_index: u16,
    expected_kind: SegmentKind,
) -> Result<Option<SealedSegmentBinding>, String> {
    if (segment_index as usize) >= segment_count {
        return Ok(None);
    }
    let entry = segments[segment_index as usize];
    if entry.length == 0 {
        return Ok(None);
    }
    if entry.kind != expected_kind as u32 {
        return Err(format!(
            "segment index {} kind mismatch: expected {:?}, found {}",
            segment_index, expected_kind, entry.kind
        ));
    }
    Ok(Some(SealedSegmentBinding {
        segment_index,
        kind: expected_kind,
        offset: entry.offset,
        length: entry.length,
    }))
}

fn projection_role_from_raw(raw: u16) -> Option<ProjectionRole> {
    match raw {
        x if x == ProjectionRole::ImagePatchEmbedding as u16 => {
            Some(ProjectionRole::ImagePatchEmbedding)
        }
        x if x == ProjectionRole::ImageProjection as u16 => Some(ProjectionRole::ImageProjection),
        x if x == ProjectionRole::ImagePositionEmbedding as u16 => {
            Some(ProjectionRole::ImagePositionEmbedding)
        }
        x if x == ProjectionRole::ImagePooling as u16 => Some(ProjectionRole::ImagePooling),
        x if x == ProjectionRole::AudioFrameEmbedding as u16 => {
            Some(ProjectionRole::AudioFrameEmbedding)
        }
        x if x == ProjectionRole::AudioProjection as u16 => Some(ProjectionRole::AudioProjection),
        x if x == ProjectionRole::AudioPositionEmbedding as u16 => {
            Some(ProjectionRole::AudioPositionEmbedding)
        }
        _ => None,
    }
}

fn is_image_role(role: ProjectionRole) -> bool {
    matches!(
        role,
        ProjectionRole::ImagePatchEmbedding
            | ProjectionRole::ImageProjection
            | ProjectionRole::ImagePositionEmbedding
            | ProjectionRole::ImagePooling
    )
}

fn is_audio_role(role: ProjectionRole) -> bool {
    matches!(
        role,
        ProjectionRole::AudioFrameEmbedding
            | ProjectionRole::AudioProjection
            | ProjectionRole::AudioPositionEmbedding
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_image::compile::execution_graph::{
        DeviceCapability, ExecutionGraphDescriptor, LayerExecutionNode, NodeKind,
    };
    use crate::compute_image::compile::ternary::CIMAGE_SEGMENT_CAPACITY;
    use crate::compute_image::multimodal::MULTIMODAL_DESCRIPTOR_MAGIC;

    fn make_descriptor() -> MultimodalInputDescriptorV1 {
        let mut desc = MultimodalInputDescriptorV1::default();
        desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
        desc.version = 1;
        desc.modality_mask = 0b0110;
        desc.image_projection_count = 3;
        desc.audio_projection_count = 1;
        desc.projection_weight_segment_index = 0;
        desc.projection_scale_segment_index = 1;
        desc.position_embedding_segment_index = 2;
        desc.auxiliary_weight_segment_index = 3;
        desc
    }

    fn make_segments() -> [SegmentEntry; CIMAGE_SEGMENT_CAPACITY] {
        let mut segments = [SegmentEntry {
            kind: 0,
            offset: 0,
            length: 0,
        }; CIMAGE_SEGMENT_CAPACITY];
        segments[0] = SegmentEntry {
            kind: SegmentKind::MultimodalProjectionWeights as u32,
            offset: 1024,
            length: 2048,
        };
        segments[1] = SegmentEntry {
            kind: SegmentKind::MultimodalProjectionScales as u32,
            offset: 4096,
            length: 512,
        };
        segments[2] = SegmentEntry {
            kind: SegmentKind::MultimodalPositionEmbeddings as u32,
            offset: 8192,
            length: 4096,
        };
        segments[3] = SegmentEntry {
            kind: SegmentKind::MultimodalAuxiliaryWeights as u32,
            offset: 12288,
            length: 1024,
        };
        segments
    }

    fn record(role: ProjectionRole) -> ProjectionTensorRecord {
        ProjectionTensorRecord {
            role: role as u16,
            weight_length: 128,
            scale_length: 32,
            ..ProjectionTensorRecord::default()
        }
    }

    #[test]
    fn binding_reports_direct_image_projection_readiness() {
        let summary = MultimodalArtifactSummary {
            modalities: 0b0110,
            image_soft_token_default: 280,
            image_soft_token_max: 1024,
            projection_precision: ProjectionPrecision::Hybrid,
            processor_contract_digest: [0; 32],
            tensor_layout_digest: [0; 32],
        };
        let bindings = SealedMultimodalBindings::from_verified_parts(
            &make_segments(),
            4,
            make_descriptor(),
            summary,
            MultimodalCapabilities {
                text: true,
                image: true,
                audio: true,
                image_projection_backend:
                    crate::compute_image::multimodal::ProjectionBackend::Metal,
                audio_projection_backend:
                    crate::compute_image::multimodal::ProjectionBackend::Metal,
                max_images_per_prompt: 1,
                max_soft_tokens_per_image: 1024,
                supports_mixed_embedding_prefill: true,
            },
            ProjectionPrecision::Hybrid,
            &[
                record(ProjectionRole::ImagePatchEmbedding),
                record(ProjectionRole::ImageProjection),
                record(ProjectionRole::ImagePositionEmbedding),
                record(ProjectionRole::AudioProjection),
            ],
        )
        .expect("bindings");
        assert!(bindings.ready_for_direct_image_projection());
        assert!(bindings.image_patch_embedding().is_some());
        assert!(bindings.audio_projection().is_some());
    }

    #[test]
    fn helper_filters_projection_roles() {
        assert!(is_image_role(ProjectionRole::ImagePatchEmbedding));
        assert!(is_audio_role(ProjectionRole::AudioProjection));
        assert!(!is_audio_role(ProjectionRole::ImageProjection));
    }

    #[test]
    fn optional_segment_absence_is_tolerated() {
        let segments = make_segments();
        let binding =
            resolve_optional_segment(&segments, 2, 3, SegmentKind::MultimodalAuxiliaryWeights)
                .expect("optional resolution");
        assert!(binding.is_none());
    }

    #[test]
    fn segment_resolution_validates_kind() {
        let segments = make_segments();
        let err = resolve_segment(&segments, 4, 0, SegmentKind::MultimodalPositionEmbeddings)
            .unwrap_err();
        assert!(err.contains("kind mismatch"));
    }

    #[test]
    fn validates_multimodal_prefix_and_stops_at_decoder() {
        let summary = MultimodalArtifactSummary {
            modalities: 0b0110,
            image_soft_token_default: 280,
            image_soft_token_max: 1024,
            projection_precision: ProjectionPrecision::Nf4Tile640,
            processor_contract_digest: [0; 32],
            tensor_layout_digest: [0; 32],
        };
        let bindings = SealedMultimodalBindings::from_verified_parts(
            &make_segments(),
            4,
            make_descriptor(),
            summary,
            MultimodalCapabilities {
                text: true,
                image: true,
                audio: true,
                image_projection_backend:
                    crate::compute_image::multimodal::ProjectionBackend::Metal,
                audio_projection_backend:
                    crate::compute_image::multimodal::ProjectionBackend::Metal,
                max_images_per_prompt: 1,
                max_soft_tokens_per_image: 1024,
                supports_mixed_embedding_prefill: true,
            },
            ProjectionPrecision::Nf4Tile640,
            &[
                ProjectionTensorRecord {
                    role: ProjectionRole::ImagePatchEmbedding as u16,
                    weight_offset: 128,
                    weight_length: 1280,
                    scale_offset: 64,
                    scale_length: 40,
                    output_width: 1152,
                    layout: ProjectionTensorRecord::LAYOUT_NF4_TILE640,
                    quantization_kind: ProjectionTensorRecord::QUANTIZATION_NF4_TILE640,
                    ..ProjectionTensorRecord::default()
                },
                ProjectionTensorRecord {
                    role: ProjectionRole::AudioProjection as u16,
                    weight_offset: 2048,
                    weight_length: 4096,
                    scale_offset: 512,
                    scale_length: 160,
                    output_width: 3840,
                    layout: ProjectionTensorRecord::LAYOUT_NF4_TILE640,
                    quantization_kind: ProjectionTensorRecord::QUANTIZATION_NF4_TILE640,
                    ..ProjectionTensorRecord::default()
                },
            ],
        )
        .expect("bindings");

        let graph = ExecutionGraphDescriptor {
            magic: crate::compute_image::compile::execution_graph::EXECUTION_GRAPH_MAGIC,
            version: 1,
            num_layers: 1,
            num_draft_layers: 0,
            num_compaction_epochs: 0,
            node_count: 3,
            _pad: [0; 2],
            layers: vec![
                LayerExecutionNode {
                    node_kind: NodeKind::VisionPatchEmbed as u8,
                    attention_kind: 2,
                    device_capability: DeviceCapability::Gpu as u8,
                    compaction_epoch: 0xFF,
                    layer_index: 0,
                    head_dim: 14,
                    num_heads: 16,
                    hidden_dim: 1152,
                    weight_offset: 128,
                    weight_length: 1280,
                    scale_offset: 64,
                    _reserved: [0; 8],
                },
                LayerExecutionNode {
                    node_kind: NodeKind::AudioProjection as u8,
                    attention_kind: 2,
                    device_capability: DeviceCapability::Gpu as u8,
                    compaction_epoch: 0xFF,
                    layer_index: 0,
                    head_dim: 256,
                    num_heads: 16,
                    hidden_dim: 3840,
                    weight_offset: 2048,
                    weight_length: 4096,
                    scale_offset: 512,
                    _reserved: [0; 8],
                },
                LayerExecutionNode {
                    node_kind: NodeKind::DecoderLayer as u8,
                    attention_kind: 1,
                    device_capability: DeviceCapability::Both as u8,
                    compaction_epoch: 0xFF,
                    layer_index: 7,
                    head_dim: 256,
                    num_heads: 16,
                    hidden_dim: 3840,
                    weight_offset: 0,
                    weight_length: 0,
                    scale_offset: 0,
                    _reserved: [0; 8],
                },
            ],
            compaction_epochs: vec![],
            draft_sub_graph: None,
        };

        let next = bindings
            .validate_graph_multimodal_prefix(&graph)
            .expect("validate prefix");
        assert_eq!(next, 2);
    }
}
