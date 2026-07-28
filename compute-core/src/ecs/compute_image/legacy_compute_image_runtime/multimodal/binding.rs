//! Runtime binding for multimodal projector artifacts sealed into a `.cimage`.

#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]

use crate::ecs::compute_image::cimage_loader::CimageDeployment;
use crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::{ExecutionGraphDescriptor, NodeKind};
use crate::ecs::compute_image::legacy_compute_image_compile::ternary::{verify_cimage, SegmentEntry, SegmentKind};
use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::{
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
    /// The bias segment, present only when the record carries
    /// `FLAG_HAS_BIAS` AND the artifact seals a
    /// `MultimodalProjectionBiases` segment. Per the parallel-layout
    /// contract the record's `scale_offset`/`scale_length` address this
    /// segment — see [`Self::bias_view_geometry`].
    pub biases: Option<SealedSegmentBinding>,
}

impl ProjectionTensorBinding {
    /// (offset, length) of this record's bias bytes **within the bias
    /// segment** — by the parallel-layout contract these are exactly the
    /// record's scale geometry. Returns `None` when the record has no
    /// resident biases (v1-compat zero-bias path).
    pub fn bias_view_geometry(&self) -> Option<(u64, u64)> {
        self.biases?;
        if !self.record.has_bias() {
            return None;
        }
        Some((self.record.scale_offset, self.record.scale_length))
    }
}

#[derive(Debug, Clone)]
pub struct SealedMultimodalBindings {
    pub descriptor: MultimodalInputDescriptorV1,
    pub summary: MultimodalArtifactSummary,
    pub capabilities: MultimodalCapabilities,
    pub projection_precision: ProjectionPrecision,
    pub weight_segment: SealedSegmentBinding,
    pub scale_segment: Option<SealedSegmentBinding>,
    pub bias_segment: Option<SealedSegmentBinding>,
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
        node: &crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::LayerExecutionNode,
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
        Self::from_verified_parts(
            &header.segments,
            header.segment_count as usize,
            descriptor,
            summary,
            capabilities,
            projection_precision,
            records,
        )
    }

    /// Resolution core shared by [`Self::from_parts`] (post-`verify_cimage`)
    /// and the synthetic-parts tests — the single home of segment resolution
    /// and the bias parallel-view load checks.
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
        let bias_segment = resolve_optional_segment(
            segments,
            segment_count,
            descriptor.projection_bias_segment_index,
            SegmentKind::MultimodalProjectionBiases,
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
            // Load-time parallel check (the derive-style triplet validation
            // of the main arena ABI, mirrored): a flagged record must resolve
            // a bias segment large enough for its scale-parallel view.
            if record.has_bias() {
                let seg = bias_segment.ok_or_else(|| {
                    format!(
                        "projection record {:#x} sets FLAG_HAS_BIAS but the artifact \
                         seals no MultimodalProjectionBiases segment",
                        record.logical_name_hash
                    )
                })?;
                let end = record.scale_offset.saturating_add(record.scale_length);
                if end > seg.length {
                    return Err(format!(
                        "projection record {:#x}: bias view [{}, {}) exceeds bias \
                         segment length {} — parallel-layout contract violated",
                        record.logical_name_hash, record.scale_offset, end, seg.length
                    ));
                }
            }
            let binding = ProjectionTensorBinding {
                role,
                record: *record,
                weights: weight_segment,
                scales: scale_segment,
                biases: if record.has_bias() {
                    bias_segment
                } else {
                    None
                },
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
            bias_segment,
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
        return Ok(None);
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
    use crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::{
        DeviceCapability, ExecutionGraphDescriptor, LayerExecutionNode, NodeKind,
    };
    use crate::ecs::compute_image::legacy_compute_image_compile::ternary::CIMAGE_SEGMENT_CAPACITY;
    use crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::MULTIMODAL_DESCRIPTOR_MAGIC;

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

    // ── Multimodal NF4 bias ABI validation gates ────────────────────────────
    // (kernels/MULTIMODAL_NF4_BIAS_ABI.md — implemented in the hardening pass)

    fn make_segments_with_biases(bias_len: u64) -> [SegmentEntry; CIMAGE_SEGMENT_CAPACITY] {
        let mut segments = make_segments();
        segments[4] = SegmentEntry {
            kind: SegmentKind::MultimodalProjectionBiases as u32,
            offset: 16384,
            length: bias_len,
        };
        segments
    }

    fn flagged_record(role: ProjectionRole, scale_offset: u64) -> ProjectionTensorRecord {
        ProjectionTensorRecord {
            role: role as u16,
            weight_length: 128,
            scale_offset,
            scale_length: 32,
            flags: ProjectionTensorRecord::FLAG_HAS_BIAS,
            ..ProjectionTensorRecord::default()
        }
    }

    fn summary() -> MultimodalArtifactSummary {
        MultimodalArtifactSummary {
            modalities: 0b0110,
            image_soft_token_default: 280,
            image_soft_token_max: 1024,
            projection_precision: ProjectionPrecision::Nf4Tile640,
            processor_contract_digest: [0; 32],
            tensor_layout_digest: [0; 32],
        }
    }

    fn caps() -> MultimodalCapabilities {
        MultimodalCapabilities {
            text: true,
            image: true,
            audio: false,
            image_projection_backend:
                crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionBackend::None,
            audio_projection_backend:
                crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionBackend::None,
            max_images_per_prompt: 0,
            max_soft_tokens_per_image: 0,
            supports_mixed_embedding_prefill: false,
        }
    }

    #[test]
    fn bias_view_resolves_for_flagged_record() {
        let mut desc = make_descriptor();
        desc.projection_bias_segment_index = 4;
        let bindings = SealedMultimodalBindings::from_verified_parts(
            &make_segments_with_biases(512), // == scale segment length
            5,
            desc,
            summary(),
            caps(),
            ProjectionPrecision::Nf4Tile640,
            &[flagged_record(ProjectionRole::ImageProjection, 64)],
        )
        .expect("flagged record with sealed bias segment must bind");
        assert!(bindings.bias_segment.is_some(), "bias segment resolved");
        let b = bindings.image_projection().expect("projection binding");
        assert!(b.biases.is_some(), "record-level bias binding present");
        // The parallel-layout contract: bias view geometry IS the scale geometry.
        assert_eq!(b.bias_view_geometry(), Some((64, 32)));
    }

    #[test]
    fn v1_record_never_takes_the_bias_path() {
        // Bias segment sealed, but the record's flags == 0 (every v1 packer
        // wrote 0) — the record-level gate, not the descriptor index, is the
        // load-bearing guard.
        let mut desc = make_descriptor();
        desc.projection_bias_segment_index = 4;
        let bindings = SealedMultimodalBindings::from_verified_parts(
            &make_segments_with_biases(512),
            5,
            desc,
            summary(),
            caps(),
            ProjectionPrecision::Nf4Tile640,
            &[record(ProjectionRole::ImageProjection)],
        )
        .expect("v1 record must load unchanged");
        let b = bindings.image_projection().expect("projection binding");
        assert!(b.biases.is_none(), "flags==0 must never bind biases");
        assert_eq!(b.bias_view_geometry(), None);
    }

    #[test]
    fn v1_descriptor_zero_index_is_unreachable_behind_the_flag() {
        // v1 packers wrote 0 into the (then-reserved) index slot. 0 aliases a
        // real segment slot, which is exactly why the record flag gates the
        // path: with flags == 0 the aliased index must never be consulted.
        let desc = make_descriptor(); // bias index left at default 0
        let bindings = SealedMultimodalBindings::from_verified_parts(
            &make_segments(), // slot 0 is the WEIGHTS segment (the alias hazard)
            4,
            desc,
            summary(),
            caps(),
            ProjectionPrecision::Nf4Tile640,
            &[record(ProjectionRole::ImageProjection)],
        )
        .expect("v1 artifact must load");
        // resolve_optional_segment finds slot 0 has the WRONG kind for biases,
        // so the segment resolves to None — and the unflagged record never
        // asks. Bit-identical v1 behavior.
        assert!(bindings.bias_segment.is_none());
        assert!(bindings.image_projection().unwrap().biases.is_none());
    }

    #[test]
    fn flagged_record_without_bias_segment_is_rejected() {
        let desc = make_descriptor(); // no bias index (u16 default 0 → wrong kind → None)
        let err = SealedMultimodalBindings::from_verified_parts(
            &make_segments(),
            4,
            desc,
            summary(),
            caps(),
            ProjectionPrecision::Nf4Tile640,
            &[flagged_record(ProjectionRole::ImageProjection, 0)],
        )
        .expect_err("declared residency without a sealed segment must fail loudly");
        assert!(err.contains("FLAG_HAS_BIAS"), "actionable error: {err}");
    }

    #[test]
    fn bias_view_out_of_bounds_is_rejected() {
        let mut desc = make_descriptor();
        desc.projection_bias_segment_index = 4;
        let err = SealedMultimodalBindings::from_verified_parts(
            &make_segments_with_biases(64), // record view [64, 96) exceeds 64
            5,
            desc,
            summary(),
            caps(),
            ProjectionPrecision::Nf4Tile640,
            &[flagged_record(ProjectionRole::ImageProjection, 64)],
        )
        .expect_err("parallel view must fit the bias segment");
        assert!(err.contains("parallel-layout"), "actionable error: {err}");
    }

    #[test]
    fn descriptor_layout_is_stride_stable() {
        // The bias index reuses the former image_reserved slot: same offset,
        // same width — the descriptor's size and every later field offset are
        // unchanged, so v1 readers parse new artifacts and vice versa.
        assert_eq!(std::mem::size_of::<MultimodalInputDescriptorV1>(), 176);
        let d = MultimodalInputDescriptorV1::default();
        let base = &d as *const _ as usize;
        let off = (&d.projection_bias_segment_index as *const _ as usize) - base;
        assert_eq!(
            off, 30,
            "bias index must occupy the old image_reserved slot"
        );
        // Record stride unchanged too (the loader walks with size_of stride).
        assert_eq!(std::mem::size_of::<ProjectionTensorRecord>(), 80);
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
                    crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionBackend::Metal,
                audio_projection_backend:
                    crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionBackend::Metal,
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
                    crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionBackend::Metal,
                audio_projection_backend:
                    crate::ecs::compute_image::legacy_compute_image_runtime::multimodal::ProjectionBackend::Metal,
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
            magic: crate::ecs::compute_image::legacy_compute_image_compile::execution_graph::EXECUTION_GRAPH_MAGIC,
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
