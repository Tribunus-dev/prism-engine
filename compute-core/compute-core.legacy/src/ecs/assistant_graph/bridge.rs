use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub enum BridgeValueType {
    UserTextInput,
    UserImageInput,
    TokenSequence,
    PerceptionFacts,
    RetrievalCandidates,
    DraftReasoningTrace,
    AssistantResponseState,
    SpeechPlan,
    SemanticSpeechTokens,
    AudioFrames,
    KvCacheView,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BridgeDecl {
    pub bridge_id: String,
    pub value_type: BridgeValueType,
    pub source_region: String,
    pub target_region: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantResponseState {
    pub text: String,
    pub language: String,
    pub turn_intent: TurnIntent,
    pub speaking_style: SpeakingStyle,
    pub emotional_register: EmotionalRegister,
    pub pacing: PacingPlan,
    pub emphasis_spans: Vec<EmphasisSpan>,
    pub pronunciation_hints: Vec<PronunciationHint>,
    pub confidence: f32,
    pub committed_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnIntent {
    Respond,
    Explain,
    Clarify,
    ExecuteTool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpeakingStyle {
    Neutral,
    Empathetic,
    Concise,
    Detailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmotionalRegister {
    Neutral,
    Warm,
    Urgent,
    Reflective,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PacingPlan {
    pub words_per_minute: f32,
    pub pause_after_sentence_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmphasisSpan {
    pub start: u32,
    pub length: u32,
    pub strength: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PronunciationHint {
    pub text: String,
    pub ipa: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechPlan {
    pub utterances: Vec<SpeechUtterance>,
    pub language: String,
    pub voice_id: Option<String>,
    pub pace: f32,
    pub emphasis_spans: Vec<EmphasisSpan>,
    pub pronunciation_hints: Vec<PronunciationHint>,
    pub source_response_epoch: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeechUtterance {
    pub text: String,
    pub duration_ms: u32,
}
