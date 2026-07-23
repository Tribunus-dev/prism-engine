//! ECS-native inference work metadata and admission calculations.
//!
//! The scheduler keeps this metadata on the canonical work entity.  It is
//! deliberately small and provider-neutral: the backend receives a serialized
//! work slice, while lifecycle ownership remains with [`KernelHandle`].

use prism_ecs_core::Component;
use serde::{Deserialize, Serialize};

/// The two token-generation phases understood by the ECS scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InferencePhase {
    Prefill,
    Decode,
}

/// Explicit token/KV work description attached to an inference work entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceWorkMetadata {
    pub phase: InferencePhase,
    pub prompt_tokens: u32,
    pub prefilled_tokens: u32,
    pub generated_tokens: u32,
    pub max_new_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub kv_epoch: u64,
    pub kv_tokens: u32,
    pub kv_capacity_tokens: u32,
    pub deadline_ms: u64,
    pub priority: u32,
}

impl Component for InferenceWorkMetadata {}

impl Default for InferenceWorkMetadata {
    fn default() -> Self {
        Self {
            phase: InferencePhase::Prefill,
            prompt_tokens: 0,
            prefilled_tokens: 0,
            generated_tokens: 0,
            max_new_tokens: 1,
            prefill_chunk_tokens: 1,
            kv_epoch: 0,
            kv_tokens: 0,
            kv_capacity_tokens: u32::MAX,
            deadline_ms: 0,
            priority: 0,
        }
    }
}

impl InferenceWorkMetadata {
    /// Parse the optional inference section of a work resource claim.
    ///
    /// An empty or invalid claim uses safe defaults so legacy work creation
    /// remains compatible. Explicit zero values are normalized to a usable
    /// one-token budget rather than creating permanently un-runnable work.
    pub fn from_resource_claim(claim: &str) -> Self {
        #[derive(Debug, Deserialize, Default)]
        struct Claim {
            #[serde(default)]
            prompt_tokens: u32,
            #[serde(default)]
            max_new_tokens: u32,
            #[serde(default)]
            prefill_chunk_tokens: u32,
            #[serde(default)]
            kv_epoch: u64,
            #[serde(default)]
            kv_tokens: u32,
            #[serde(default)]
            kv_capacity_tokens: u32,
            #[serde(default)]
            deadline_ms: u64,
            #[serde(default)]
            priority: u32,
        }

        let claim = serde_json::from_str::<Claim>(claim).unwrap_or_default();
        Self {
            prompt_tokens: claim.prompt_tokens,
            max_new_tokens: claim.max_new_tokens.max(1),
            prefill_chunk_tokens: claim.prefill_chunk_tokens.max(1),
            kv_epoch: claim.kv_epoch,
            kv_tokens: claim.kv_tokens,
            kv_capacity_tokens: if claim.kv_capacity_tokens == 0 {
                u32::MAX
            } else {
                claim.kv_capacity_tokens
            },
            deadline_ms: claim.deadline_ms,
            priority: claim.priority,
            ..Self::default()
        }
    }

    /// Number of tokens reserved by admission for this request.
    pub fn reserved_tokens(&self) -> u32 {
        self.kv_tokens
            .saturating_add(self.prompt_tokens)
            .saturating_add(self.max_new_tokens)
    }

    /// Number of KV tokens required if the request runs to its configured end.
    pub fn required_kv_tokens(&self) -> u32 {
        self.reserved_tokens()
    }

    /// Returns the next bounded prefill interval as `[start, end)`.
    pub fn next_prefill_chunk(&self, scheduler_limit: u32) -> (u32, u32) {
        let start = self.prefilled_tokens.min(self.prompt_tokens);
        let limit = self.prefill_chunk_tokens.min(scheduler_limit.max(1));
        let end = start.saturating_add(limit).min(self.prompt_tokens);
        (start, end)
    }

    pub fn deadline_expired(&self, now_ms: u64) -> bool {
        self.deadline_ms != 0 && now_ms >= self.deadline_ms
    }

    pub fn next_after_prefill(&self, scheduler_limit: u32) -> Self {
        let (_, end) = self.next_prefill_chunk(scheduler_limit);
        let mut next = *self;
        next.prefilled_tokens = end;
        next.kv_tokens = next
            .kv_tokens
            .saturating_add(end.saturating_sub(self.prefilled_tokens.min(self.prompt_tokens)));
        if next.prefilled_tokens >= next.prompt_tokens {
            next.phase = InferencePhase::Decode;
        }
        next
    }

    pub fn next_after_decode(&self) -> Self {
        let mut next = *self;
        next.generated_tokens = next.generated_tokens.saturating_add(1);
        next.kv_tokens = next.kv_tokens.saturating_add(1);
        next.phase = InferencePhase::Decode;
        next
    }

    pub fn is_complete_after_decode(&self) -> bool {
        self.generated_tokens.saturating_add(1) >= self.max_new_tokens
    }
}

/// Bounded policy for token/KV-aware admission and work slicing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InferenceAdmissionPolicy {
    pub max_inflight_tokens: u32,
    pub max_kv_tokens: u32,
    pub prefill_chunk_tokens: u32,
    pub max_prefill_tokens_per_tick: u32,
}

impl Default for InferenceAdmissionPolicy {
    fn default() -> Self {
        Self {
            max_inflight_tokens: 32 * 1024,
            max_kv_tokens: 32 * 1024,
            prefill_chunk_tokens: 256,
            max_prefill_tokens_per_tick: 1024,
        }
    }
}

impl InferenceAdmissionPolicy {
    pub fn validate(self) -> bool {
        self.max_inflight_tokens > 0
            && self.max_kv_tokens > 0
            && self.prefill_chunk_tokens > 0
            && self.max_prefill_tokens_per_tick > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_explicit_kv_epoch_and_normalizes_chunk_budget() {
        let metadata = InferenceWorkMetadata::from_resource_claim(
            r#"{"prompt_tokens":9,"max_new_tokens":3,"prefill_chunk_tokens":0,"kv_epoch":7,"kv_tokens":2,"kv_capacity_tokens":32}"#,
        );
        assert_eq!(metadata.kv_epoch, 7);
        assert_eq!(metadata.reserved_tokens(), 14);
        assert_eq!(metadata.prefill_chunk_tokens, 1);
        assert_eq!(metadata.next_prefill_chunk(4), (0, 1));
    }

    #[test]
    fn advances_prefill_in_bounded_chunks_then_decode_tokens() {
        let metadata = InferenceWorkMetadata {
            prompt_tokens: 5,
            prefill_chunk_tokens: 2,
            max_new_tokens: 2,
            kv_epoch: 11,
            ..InferenceWorkMetadata::default()
        };
        let prefilled = metadata.next_after_prefill(2);
        assert_eq!(prefilled.prefilled_tokens, 2);
        assert_eq!(prefilled.kv_tokens, 2);
        assert_eq!(prefilled.phase, InferencePhase::Prefill);
        let decoded = InferenceWorkMetadata {
            phase: InferencePhase::Decode,
            prefilled_tokens: 5,
            kv_tokens: 5,
            ..prefilled
        }
        .next_after_decode();
        assert_eq!(decoded.generated_tokens, 1);
        assert_eq!(decoded.kv_tokens, 6);
    }
}
