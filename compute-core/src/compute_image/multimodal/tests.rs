use super::*;

fn make_valid_descriptor() -> MultimodalInputDescriptorV1 {
    let mut desc = MultimodalInputDescriptorV1::default();
    desc.magic = MULTIMODAL_DESCRIPTOR_MAGIC;
    desc.version = 1;
    desc.modality_mask = 2;
    desc.decoder_hidden_size = 3584;
    desc.vocabulary_size = 262144;
    desc.image_patch_size = 14;
    desc.image_pooling_kernel = 2;
    desc.image_channels = 3;
    desc.image_min_soft_tokens = 64;
    desc.image_default_soft_tokens = 280;
    desc.image_max_soft_tokens = 1024;
    desc.image_position_table_height = 64;
    desc.image_position_table_width = 64;
    desc.image_position_embedding_width = 3584;
    desc.projection_weight_segment_index = 0;
    desc.projection_scale_segment_index = 1;
    desc.auxiliary_weight_segment_index = 2;
    desc.position_embedding_segment_index = 3;
    desc.processor_contract_digest = [0xAB; 32];
    desc.tensor_layout_digest = [0xCD; 32];
    desc
}

#[test]
fn multimodal_descriptor_roundtrip() {
    let desc = make_valid_descriptor();
    assert_eq!(desc.magic, MULTIMODAL_DESCRIPTOR_MAGIC);
    assert_eq!(desc.version, 1);
    assert_eq!(desc.validate(), Ok(()));
}

#[test]
fn multimodal_descriptor_rejects_bad_magic() {
    let mut desc = make_valid_descriptor();
    desc.magic = *b"BADMAGIC";
    assert!(desc.validate().is_err());
}

#[test]
fn multimodal_descriptor_rejects_bad_version() {
    let mut desc = make_valid_descriptor();
    desc.version = 99;
    assert!(desc.validate().is_err());
}

#[test]
fn multimodal_descriptor_default_is_invalid() {
    let desc = MultimodalInputDescriptorV1::default();
    assert!(desc.validate().is_err());
}

#[test]
fn text_only_cimage_has_no_multimodal_capabilities() {
    let caps = MultimodalCapabilities::default();
    assert!(caps.text);
    assert!(!caps.image);
    assert!(!caps.audio);
    assert_eq!(caps.image_projection_backend, ProjectionBackend::None);
    assert_eq!(caps.max_soft_tokens_per_image, 0);
}

#[test]
fn image_prompt_requires_multimodal_descriptor() {
    let caps = MultimodalCapabilities::default();
    assert!(!caps.image);
    assert!(!caps.supports_mixed_embedding_prefill);
    let desc = make_valid_descriptor();
    assert_eq!(desc.modality_mask & 2, 2);
}

#[test]
fn projection_role_values_are_distinct() {
    use std::collections::HashSet;
    let roles = [
        ProjectionRole::ImagePatchEmbedding,
        ProjectionRole::ImageProjection,
        ProjectionRole::ImagePositionEmbedding,
        ProjectionRole::ImagePooling,
        ProjectionRole::AudioFrameEmbedding,
        ProjectionRole::AudioProjection,
        ProjectionRole::AudioPositionEmbedding,
    ];
    let mut seen = HashSet::new();
    for role in &roles {
        assert!(seen.insert(*role as u16), "duplicate role value: {:?}", role);
    }
}

#[test]
fn projection_tensor_record_default_is_zeroed() {
    let rec = ProjectionTensorRecord::default();
    assert_eq!(rec.logical_name_hash, 0);
    assert_eq!(rec.role, 0);
    assert_eq!(rec.weight_offset, 0);
    assert_eq!(rec.input_width, 0);
    assert_eq!(rec.output_width, 0);
}

#[test]
fn processor_contract_default_is_zeroed() {
    let contract = ImageProcessorContractV1::default();
    assert_eq!(contract.resize_policy, 0);
    assert_eq!(contract.patch_size, 0);
    assert_eq!(contract.default_soft_tokens, 0);
}

#[test]
fn input_modality_mask_bits_are_powers_of_two() {
    assert_eq!(InputModality::Text.as_mask_bit(), 1);
    assert_eq!(InputModality::Image.as_mask_bit(), 2);
    assert_eq!(InputModality::Audio.as_mask_bit(), 4);
}
