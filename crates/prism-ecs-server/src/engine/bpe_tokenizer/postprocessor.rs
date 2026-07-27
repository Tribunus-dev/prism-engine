//! Post-processing — insert special tokens (BOS/EOS/CLS/SEP) into the encoding.
//!
//! This module owns the canonical authority for post-processor construction
//! from `tokenizer.json` and the application of post-processing to a
//! per-word model-tokenized encoding. It does not own pre-tokenization,
//! model tokenization, decoding, or pipeline orchestration.
//!
//! Post-processing produces a (ids, type_ids, special_tokens_mask) triple.
//! The pipeline orchestrator in `loader.rs` merges that triple into an
//! `Encoding` with attention_mask and word_ids derived from the existing
//! per-word encoding.

use serde_json::Value;

/// Result of post-processing: (ids, type_ids, special_tokens_mask).
/// `None` is returned when the post-processor decides the input is already
/// complete (e.g., the special tokens are already present and the template
/// is a no-op).
pub(crate) type PostProcessResult = Option<(Vec<u32>, Vec<u32>, Vec<u32>)>;

#[derive(Debug, Clone)]
pub(crate) enum PostProcessorKind {
    /// No post-processing — just return the encoding as-is
    None,
    /// Template-based post-processor
    Template {
        single_template: Option<Template>,
        pair_template: Option<Template>,
    },
    /// Simple BOS/EOS appending (for older tokenizers without explicit template)
    BosEos { bos_id: u32, eos_id: u32 },
}

#[derive(Debug, Clone)]
pub(crate) struct Template {
    pub(crate) pieces: Vec<TemplatePiece>,
}

#[derive(Debug, Clone)]
pub(crate) enum TemplatePiece {
    /// A special token (BOS, EOS, SEP, CLS, etc.)
    Special { id: u32, type_id: u32 },
    /// The input sequence placeholder
    Sequence { index: usize, type_id: u32 },
}

impl PostProcessorKind {
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
        let ptype = v.get("type").and_then(Value::as_str).unwrap_or("");

        match ptype {
            "TemplateProcessing" => {
                let single_val = v
                    .get("single")
                    .ok_or_else(|| "TemplateProcessing missing 'single'".to_string())?;
                let pair_val = v.get("pair");

                let single = Template::from_json(single_val)?;
                let pair = match pair_val {
                    Some(pv) => Some(Template::from_json(pv)?),
                    None => None,
                };

                Ok(PostProcessorKind::Template {
                    single_template: Some(single),
                    pair_template: pair,
                })
            }
            "ByteLevel" => {
                // ByteLevel post-processor trims whitespace offsets — no-op for IDs
                Ok(PostProcessorKind::None)
            }
            "RobertaProcessing" | "BertProcessing" => {
                let sep = v
                    .get("sep")
                    .and_then(|s| s.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(2);
                let cls = v
                    .get("cls")
                    .and_then(|s| s.as_array())
                    .and_then(|arr| arr.get(0))
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(0);
                // Build a template: [CLS] $A [SEP] for single
                let single = Template {
                    pieces: vec![
                        TemplatePiece::Special {
                            id: cls,
                            type_id: 0,
                        },
                        TemplatePiece::Sequence {
                            index: 0,
                            type_id: 0,
                        },
                        TemplatePiece::Special {
                            id: sep,
                            type_id: 0,
                        },
                    ],
                };
                Ok(PostProcessorKind::Template {
                    single_template: Some(single),
                    pair_template: None,
                })
            }
            _ => Ok(PostProcessorKind::None),
        }
    }

    /// Apply the post-processor to a per-word model-tokenized encoding.
    /// `bos_token_id` / `eos_token_id` are passed in from the top-level
    /// `Tokenizer` so the `None` variant can fall back to BOS/EOS insertion
    /// when they are defined.
    pub(crate) fn post_process(
        &self,
        ids: &[u32],
        _encoding: &crate::engine::bpe_tokenizer::encoding::Encoding,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> PostProcessResult {
        match self {
            PostProcessorKind::None => {
                // If bos/eos are defined but there's no explicit template,
                // prepend/append them
                let mut new_ids = Vec::new();
                let mut type_ids = Vec::new();
                let mut special_mask = Vec::new();

                if let Some(bos) = bos_token_id {
                    if ids.is_empty() || ids[0] != bos {
                        new_ids.push(bos);
                        type_ids.push(0);
                        special_mask.push(1);
                    }
                }
                for &id in ids {
                    new_ids.push(id);
                    type_ids.push(0);
                    special_mask.push(0);
                }
                if let Some(eos) = eos_token_id {
                    if ids.last() != Some(&eos) {
                        new_ids.push(eos);
                        type_ids.push(0);
                        special_mask.push(1);
                    }
                }

                if new_ids.len() > ids.len() {
                    Some((new_ids, type_ids, special_mask))
                } else {
                    None
                }
            }
            PostProcessorKind::Template {
                single_template,
                pair_template: _,
            } => {
                if let Some(template) = single_template {
                    let mut new_ids = Vec::new();
                    let mut type_ids = Vec::new();
                    let mut special_mask = Vec::new();

                    for piece in &template.pieces {
                        match piece {
                            TemplatePiece::Special { id, type_id } => {
                                new_ids.push(*id);
                                type_ids.push(*type_id);
                                special_mask.push(1);
                            }
                            TemplatePiece::Sequence { index: _, type_id } => {
                                for &id in ids {
                                    new_ids.push(id);
                                    type_ids.push(*type_id);
                                    special_mask.push(0);
                                }
                            }
                        }
                    }

                    Some((new_ids, type_ids, special_mask))
                } else {
                    None
                }
            }
            PostProcessorKind::BosEos { bos_id, eos_id } => {
                let mut new_ids = vec![*bos_id];
                let mut type_ids = vec![0];
                let mut special_mask = vec![1];
                for &id in ids {
                    new_ids.push(id);
                    type_ids.push(0);
                    special_mask.push(0);
                }
                new_ids.push(*eos_id);
                type_ids.push(0);
                special_mask.push(1);
                Some((new_ids, type_ids, special_mask))
            }
        }
    }
}

impl Template {
    pub(crate) fn from_json(v: &Value) -> Result<Self, String> {
        let pieces = v
            .get("pieces")
            .or_else(|| v.get("sequence")) // some formats use "sequence" key
            .and_then(Value::as_array)
            .ok_or_else(|| "Template missing 'pieces'".to_string())?;

        let mut template_pieces = Vec::new();
        for piece_val in pieces {
            let ptype = piece_val.get("type").and_then(Value::as_str).unwrap_or("");

            match ptype {
                "SpecialToken" | "special_token" | "Special" => {
                    let id = piece_val
                        .get("id")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    let type_id = piece_val
                        .get("type_id")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    template_pieces.push(TemplatePiece::Special { id, type_id });
                }
                "Sequence" | "sequence" => {
                    let index = piece_val
                        .get("index")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize)
                        .unwrap_or(0);
                    let type_id = piece_val
                        .get("type_id")
                        .and_then(Value::as_u64)
                        .map(|v| v as u32)
                        .unwrap_or(0);
                    template_pieces.push(TemplatePiece::Sequence { index, type_id });
                }
                _ => {}
            }
        }

        Ok(Template {
            pieces: template_pieces,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::bpe_tokenizer::encoding::Encoding;

    fn empty_encoding() -> Encoding {
        Encoding::empty()
    }

    #[test]
    fn none_post_processor_inserts_bos_and_eos() {
        let pp = PostProcessorKind::None;
        let result = pp.post_process(&[10, 20, 30], &empty_encoding(), Some(1), Some(2));
        let (ids, type_ids, special) = result.expect("should produce a result");
        assert_eq!(ids, vec![1, 10, 20, 30, 2]);
        assert_eq!(type_ids, vec![0, 0, 0, 0, 0]);
        assert_eq!(special, vec![1, 0, 0, 0, 1]);
    }

    #[test]
    fn none_post_processor_no_bos_or_eos_returns_none() {
        let pp = PostProcessorKind::None;
        let result = pp.post_process(&[10, 20, 30], &empty_encoding(), None, None);
        assert!(result.is_none());
    }

    #[test]
    fn none_post_processor_does_not_double_insert_bos() {
        let pp = PostProcessorKind::None;
        let result = pp.post_process(&[1, 20, 30], &empty_encoding(), Some(1), Some(2));
        let (ids, _, _) = result.expect("should produce a result");
        // BOS already at index 0, do not prepend again
        assert_eq!(ids.first(), Some(&1));
    }

    #[test]
    fn none_post_processor_does_not_double_insert_eos() {
        let pp = PostProcessorKind::None;
        let result = pp.post_process(&[20, 30, 2], &empty_encoding(), Some(1), Some(2));
        let (ids, _, _) = result.expect("should produce a result");
        // EOS already at end, do not append again
        assert_eq!(ids.last(), Some(&2));
    }

    #[test]
    fn template_post_processor_renders_cls_sep() {
        let v = serde_json::json!({
            "type": "TemplateProcessing",
            "single": {
                "pieces": [
                    {"type": "SpecialToken", "id": 0, "type_id": 0},
                    {"type": "Sequence", "id": 0, "type_id": 0},
                    {"type": "SpecialToken", "id": 2, "type_id": 0}
                ]
            }
        });
        let pp = PostProcessorKind::from_json(&v).unwrap();
        let result = pp.post_process(&[10, 20], &empty_encoding(), None, None);
        let (ids, _, special) = result.expect("should produce a result");
        assert_eq!(ids, vec![0, 10, 20, 2]);
        assert_eq!(special, vec![1, 0, 0, 1]);
    }

    #[test]
    fn roberta_processing_constructs_cls_seq_sep_template() {
        let v = serde_json::json!({
            "type": "RobertaProcessing",
            "sep": [2, 2],
            "cls": [0, 0]
        });
        let pp = PostProcessorKind::from_json(&v).unwrap();
        let result = pp.post_process(&[10, 20], &empty_encoding(), None, None);
        let (ids, _, _) = result.expect("should produce a result");
        assert_eq!(ids, vec![0, 10, 20, 2]);
    }

    #[test]
    fn bert_processing_constructs_cls_seq_sep_template() {
        let v = serde_json::json!({
            "type": "BertProcessing",
            "sep": [102, 1],
            "cls": [101, 1]
        });
        let pp = PostProcessorKind::from_json(&v).unwrap();
        let result = pp.post_process(&[7592], &empty_encoding(), None, None);
        let (ids, _, _) = result.expect("should produce a result");
        assert_eq!(ids, vec![101, 7592, 102]);
    }

    #[test]
    fn byte_level_post_processor_is_noop_for_ids() {
        let v = serde_json::json!({"type": "ByteLevel"});
        let pp = PostProcessorKind::from_json(&v).unwrap();
        assert!(matches!(pp, PostProcessorKind::None));
    }

    #[test]
    fn unknown_post_processor_type_falls_back_to_none() {
        let v = serde_json::json!({"type": "FancyPostProcessor"});
        let pp = PostProcessorKind::from_json(&v).unwrap();
        assert!(matches!(pp, PostProcessorKind::None));
    }

    #[test]
    fn bos_eos_post_processor_always_wraps() {
        let pp = PostProcessorKind::BosEos { bos_id: 99, eos_id: 100 };
        let result = pp.post_process(&[10, 20], &empty_encoding(), None, None);
        let (ids, _, special) = result.expect("should produce a result");
        assert_eq!(ids, vec![99, 10, 20, 100]);
        assert_eq!(special, vec![1, 0, 0, 1]);
    }
}
