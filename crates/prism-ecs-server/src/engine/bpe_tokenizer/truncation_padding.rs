//! Truncation and padding — bound the encoding length and pad to a target.
//!
//! This module owns the canonical authority for the configuration shapes
//! (strategy, params) and the application logic for truncating an encoding
//! to a maximum length (with overflow windows) and padding it to a target
//! length. It does not own pre-tokenization, model tokenization, or
//! pipeline orchestration.

use crate::engine::bpe_tokenizer::encoding::Encoding;

/// Truncation strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationStrategy {
    /// Truncate only the first sequence (prompt A).
    OnlyFirst,
    /// Truncate only the second sequence (prompt B).
    OnlySecond,
    /// Truncate from the longer sequence first.
    LongestFirst,
}

/// Truncation parameters.
#[derive(Debug, Clone)]
pub struct TruncationParams {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub stride: usize,
}

/// Padding parameters.
#[derive(Debug, Clone)]
pub struct PaddingParams {
    pub pad_token_id: u32,
    pub pad_token: String,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_left: bool,
}

/// Apply the truncation strategy to the encoding in place. Returns the
/// overflow encodings (typically empty unless a stride was specified).
pub(crate) fn apply_truncation(
    encoding: &mut Encoding,
    params: &TruncationParams,
) -> Vec<Encoding> {
    if encoding.ids.len() <= params.max_length {
        return Vec::new();
    }

    let max = params.max_length;
    let stride = params.stride.min(max);
    let mut overflows: Vec<Encoding> = Vec::new();

    match params.strategy {
        TruncationStrategy::OnlyFirst | TruncationStrategy::LongestFirst => {
            // Truncate from the end
            if stride > 0 {
                let overflow_start = max.saturating_sub(stride);
                let overflow = Encoding {
                    ids: encoding.ids[overflow_start..].to_vec(),
                    attention_mask: vec![1; encoding.ids.len() - overflow_start],
                    type_ids: encoding.type_ids[overflow_start..].to_vec(),
                    word_ids: encoding.word_ids[overflow_start..].to_vec(),
                    special_tokens_mask: encoding.special_tokens_mask[overflow_start..]
                        .to_vec(),
                    overflowing: Vec::new(),
                };
                overflows.push(overflow);
            }
            encoding.ids.truncate(max);
            encoding.attention_mask.truncate(max);
            encoding.type_ids.truncate(max);
            encoding.word_ids.truncate(max);
            encoding.special_tokens_mask.truncate(max);
        }
        TruncationStrategy::OnlySecond => {
            // No second sequence to truncate for simple encode
            // In paired mode this would truncate the second sequence
            encoding.ids.truncate(max);
            encoding.attention_mask.truncate(max);
            encoding.type_ids.truncate(max);
            encoding.word_ids.truncate(max);
            encoding.special_tokens_mask.truncate(max);
        }
    }

    overflows
}

/// Apply padding to the encoding in place. No-op if `pad_to_multiple_of`
/// is `None` and the encoding is already at a stable length.
pub(crate) fn apply_padding(encoding: &mut Encoding, params: &PaddingParams) {
    let pad_id = params.pad_token_id;
    let current_len = encoding.ids.len();
    let target_len = match params.pad_to_multiple_of {
        Some(m) => {
            let rem = current_len % m;
            if rem == 0 {
                current_len
            } else {
                current_len + (m - rem)
            }
        }
        // No padding without pad_to_multiple_of or explicit length
        None => current_len,
    };

    if target_len <= current_len {
        return;
    }

    let pad_count = target_len - current_len;

    if params.pad_left {
        let mut new_ids = vec![pad_id; pad_count];
        let mut new_mask = vec![0u32; pad_count];
        let mut new_type = vec![0u32; pad_count];
        let mut new_word = vec![None; pad_count];
        let mut new_special = vec![1u32; pad_count];

        new_ids.extend_from_slice(&encoding.ids);
        new_mask.extend_from_slice(&encoding.attention_mask);
        new_type.extend_from_slice(&encoding.type_ids);
        new_word.extend_from_slice(&encoding.word_ids);
        new_special.extend_from_slice(&encoding.special_tokens_mask);

        encoding.ids = new_ids;
        encoding.attention_mask = new_mask;
        encoding.type_ids = new_type;
        encoding.word_ids = new_word;
        encoding.special_tokens_mask = new_special;
    } else {
        encoding
            .ids
            .extend(std::iter::repeat(pad_id).take(pad_count));
        encoding
            .attention_mask
            .extend(std::iter::repeat(0).take(pad_count));
        encoding
            .type_ids
            .extend(std::iter::repeat(0).take(pad_count));
        encoding
            .word_ids
            .extend(std::iter::repeat(None).take(pad_count));
        encoding
            .special_tokens_mask
            .extend(std::iter::repeat(1).take(pad_count));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoding(ids: Vec<u32>) -> Encoding {
        let mut e = Encoding::empty();
        for id in ids {
            e.push(id, Some(0), false, 0);
        }
        e
    }

    #[test]
    fn truncation_noop_when_within_max_length() {
        let mut e = make_encoding(vec![10, 20, 30]);
        let overflow = apply_truncation(
            &mut e,
            &TruncationParams {
                max_length: 5,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            },
        );
        assert!(overflow.is_empty());
        assert_eq!(e.ids, vec![10, 20, 30]);
    }

    #[test]
    fn truncation_truncates_to_max_length() {
        let mut e = make_encoding(vec![10, 20, 30, 40, 50]);
        let overflow = apply_truncation(
            &mut e,
            &TruncationParams {
                max_length: 2,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
            },
        );
        assert_eq!(e.ids, vec![10, 20]);
        // stride=0 → no overflow window
        assert!(overflow.is_empty());
    }

    #[test]
    fn truncation_with_stride_emits_overflow_window() {
        let mut e = make_encoding(vec![10, 20, 30, 40, 50]);
        let overflow = apply_truncation(
            &mut e,
            &TruncationParams {
                max_length: 3,
                strategy: TruncationStrategy::LongestFirst,
                stride: 1,
            },
        );
        assert_eq!(e.ids, vec![10, 20, 30]);
        // overflow window covers the stride (last `stride` tokens)
        assert_eq!(overflow.len(), 1);
        assert_eq!(overflow[0].ids, vec![30, 40, 50]);
    }

    #[test]
    fn truncation_only_first_behaves_like_longest_first() {
        let mut e = make_encoding(vec![10, 20, 30, 40, 50]);
        let overflow = apply_truncation(
            &mut e,
            &TruncationParams {
                max_length: 2,
                strategy: TruncationStrategy::OnlyFirst,
                stride: 0,
            },
        );
        assert_eq!(e.ids, vec![10, 20]);
        assert!(overflow.is_empty());
    }

    #[test]
    fn truncation_only_second_truncates_to_max_length() {
        let mut e = make_encoding(vec![10, 20, 30, 40, 50]);
        let overflow = apply_truncation(
            &mut e,
            &TruncationParams {
                max_length: 2,
                strategy: TruncationStrategy::OnlySecond,
                stride: 0,
            },
        );
        assert_eq!(e.ids, vec![10, 20]);
        assert!(overflow.is_empty());
    }

    #[test]
    fn padding_noop_when_already_at_target_multiple() {
        let mut e = make_encoding(vec![10, 20, 30, 40]); // 4 = multiple of 4
        apply_padding(
            &mut e,
            &PaddingParams {
                pad_token_id: 0,
                pad_token: "<pad>".to_string(),
                pad_to_multiple_of: Some(4),
                pad_left: false,
            },
        );
        assert_eq!(e.ids, vec![10, 20, 30, 40]);
    }

    #[test]
    fn padding_right_pads_to_multiple() {
        let mut e = make_encoding(vec![10, 20, 30]); // 3 → 8
        apply_padding(
            &mut e,
            &PaddingParams {
                pad_token_id: 0,
                pad_token: "<pad>".to_string(),
                pad_to_multiple_of: Some(8),
                pad_left: false,
            },
        );
        assert_eq!(e.ids, vec![10, 20, 30, 0, 0, 0, 0, 0]);
        assert_eq!(e.attention_mask, vec![1, 1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(e.special_tokens_mask, vec![0, 0, 0, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn padding_left_pads_to_multiple() {
        let mut e = make_encoding(vec![10, 20, 30]); // 3 → 8
        apply_padding(
            &mut e,
            &PaddingParams {
                pad_token_id: 0,
                pad_token: "<pad>".to_string(),
                pad_to_multiple_of: Some(8),
                pad_left: true,
            },
        );
        assert_eq!(e.ids[..5], vec![0, 0, 0, 0, 0]);
        assert_eq!(e.ids[5..], vec![10, 20, 30]);
        assert_eq!(e.attention_mask, vec![0, 0, 0, 0, 0, 1, 1, 1]);
    }

    #[test]
    fn padding_with_no_target_multiple_is_noop() {
        let mut e = make_encoding(vec![10, 20, 30]);
        apply_padding(
            &mut e,
            &PaddingParams {
                pad_token_id: 0,
                pad_token: "<pad>".to_string(),
                pad_to_multiple_of: None,
                pad_left: false,
            },
        );
        // No target → no padding
        assert_eq!(e.ids, vec![10, 20, 30]);
    }
}
