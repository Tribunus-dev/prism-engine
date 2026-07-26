use super::gemma4_unified::*;

#[test]
fn gemma4_12b_schema_matches_known_constants() {
    let schema = Gemma4UnifiedSchema::gemma4_12b_unified();
    assert_eq!(schema.hidden_size, GEMMA4_12B_UNIFIED_HIDDEN_SIZE);
    assert_eq!(schema.num_layers, GEMMA4_12B_UNIFIED_NUM_LAYERS);
    assert_eq!(
        schema.num_attention_heads,
        GEMMA4_12B_UNIFIED_NUM_ATTENTION_HEADS
    );
    assert_eq!(schema.num_key_value_heads, GEMMA4_12B_UNIFIED_NUM_KV_HEADS);
    assert_eq!(schema.vocabulary_size, GEMMA4_12B_UNIFIED_VOCABULARY_SIZE);
    assert!(schema.supports_text);
    assert!(schema.supports_image);
    assert!(schema.supports_audio);
    assert_eq!(GEMMA4_12B_UNIFIED_INTERMEDIATE_SIZE, 15360);
    assert_eq!(GEMMA4_12B_UNIFIED_HEAD_DIM, 256);
}

#[test]
fn gemma4_12b_schema_validates_architecture() {
    let schema = Gemma4UnifiedSchema::gemma4_12b_unified();
    assert!(schema.validate_architecture().is_ok());
}

#[test]
fn gemma4_schema_rejects_legacy_vision_tower() {
    let schema = Gemma4UnifiedSchema::gemma4_12b_unified();
    let names_with_vit = vec!["model.vision_tower.encoder.layers.0.weight".to_string()];
    assert!(schema.reject_legacy_vision_tower(&names_with_vit).is_err());

    let names_clean = vec!["model.layers.0.self_attn.q_proj.weight".to_string()];
    assert!(schema.reject_legacy_vision_tower(&names_clean).is_ok());
}

#[test]
fn gemma4_schema_rejects_siglip_tensors() {
    let schema = Gemma4UnifiedSchema::gemma4_12b_unified();
    let names = vec!["siglip.vision_model.encoder.layers.0.self_attn.q_proj.weight".to_string()];
    assert!(schema.reject_legacy_vision_tower(&names).is_err());
}

#[test]
fn classify_tensor_name_decoder_weights() {
    assert_eq!(
        classify_tensor_name("model.layers.0.self_attn.q_proj.weight"),
        TensorClassification::DecoderRequired
    );
    assert_eq!(
        classify_tensor_name("model.layers.0.mlp.gate_proj.weight"),
        TensorClassification::DecoderRequired
    );
}

#[test]
fn classify_tensor_name_unknown_below_threshold() {
    // A tensor name that doesn't match any known pattern
    assert_eq!(
        classify_tensor_name("model.some_random_tensor.weight"),
        TensorClassification::Unknown
    );
}

#[test]
fn classify_tensor_name_multimodal() {
    assert_eq!(
        classify_tensor_name("model.vision_embedder.patch_dense.weight"),
        TensorClassification::MultimodalImageRequired
    );
    assert_eq!(
        classify_tensor_name("model.embed_vision.embedding_projection.weight"),
        TensorClassification::MultimodalImageRequired
    );
}

#[test]
fn classify_tensor_name_vision_embedder() {
    assert_eq!(
        classify_tensor_name("model.vision_embedder.patch_dense.weight"),
        TensorClassification::MultimodalImageRequired
    );
    assert_eq!(
        classify_tensor_name("model.embed_vision.embedding_projection.weight"),
        TensorClassification::MultimodalImageRequired
    );
    assert_eq!(
        classify_tensor_name("model.vision_embedder.pos_embedding"),
        TensorClassification::MultimodalImageRequired
    );
}

#[test]
fn classify_tensor_name_audio_embedder() {
    assert_eq!(
        classify_tensor_name("model.embed_audio.embedding_projection.weight"),
        TensorClassification::MultimodalAudioRequired
    );
}

#[test]
fn classify_tensor_name_mtp() {
    assert_eq!(
        classify_tensor_name("model.mtp_projection.weight"),
        TensorClassification::MtpRequired
    );
    assert_eq!(
        classify_tensor_name("model.mtp_norm.weight"),
        TensorClassification::MtpRequired
    );
}

#[test]
fn classify_tensor_name_lm_head() {
    assert_eq!(
        classify_tensor_name("model.language_model.lm_head_projection.weight"),
        TensorClassification::LmHeadRequired
    );
}
