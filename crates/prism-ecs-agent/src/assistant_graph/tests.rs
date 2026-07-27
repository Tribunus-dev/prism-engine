//! Integration tests for the assistant graph structural validator and the
//! manifest / authority / bridge / route-graph / state-schema data types.
//!
//! Every `Gate N:` test exercises a single validator gate against a
//! fixture that has been perturbed in exactly one way. The serde tests
//! confirm that every variant of every enum survives a JSON roundtrip
//! and that the manifest itself can be re-parsed without loss.

use super::authority::{
    AuthorityRule, AuthorityRuleKind, RegionAuthorityPolicy,
};
use super::bridge::{
    BridgeDecl, BridgeValueType, EmotionalRegister, EmphasisSpan, PacingPlan, PronunciationHint,
    SpeakingStyle, SpeechPlan, SpeechUtterance, TurnIntent,
};
use super::graph::{AssistantRouteGraph, RouteEdge, RouteKind};
use super::manifest::{
    AssistantContract, AssistantGraphManifest, AssistantRegionDecl, AssistantRegionKind,
    RegionOutputAuthority,
};
use super::receipts::{
    AssistantGraphValidationReceipt, AssistantGraphValidationStatus,
};
use super::state::{
    RegionStateAccess, SharedStateSchema, StatePersistence, StateStoreDecl, StateStoreKind,
};
use super::validate::AssistantGraphValidator;
use super::AssistantResponseState;

// ---------------------------------------------------------------------------
// Helper — builds a minimal valid manifest
// ---------------------------------------------------------------------------

fn valid_manifest() -> AssistantGraphManifest {
    AssistantGraphManifest {
        graph_id: "test-graph-1".to_string(),
        schema_version: 1,
        assistant_contract: AssistantContract {
            contract_id: "contract-1".to_string(),
            max_active_regions: 5,
            requires_bridge_types: true,
            requires_authority: true,
        },
        regions: vec![
            AssistantRegionDecl {
                region_id: "reasoning-1".to_string(),
                partition_id: None,
                region_kind: AssistantRegionKind::ReasoningDecode,
                input_types: vec![],
                output_types: vec![
                    BridgeValueType::DraftReasoningTrace,
                    BridgeValueType::TokenSequence,
                ],
                state_access: RegionStateAccess {
                    read_stores: vec!["kv-store".to_string()],
                    write_stores: vec!["kv-store".to_string()],
                    requires_epoch_check: true,
                },
                authority: vec![
                    RegionOutputAuthority::DraftText,
                    RegionOutputAuthority::KvCacheUpdate,
                ],
            },
            AssistantRegionDecl {
                region_id: "reasoning-2".to_string(),
                partition_id: None,
                region_kind: AssistantRegionKind::ReasoningDecode,
                input_types: vec![BridgeValueType::DraftReasoningTrace],
                output_types: vec![BridgeValueType::AssistantResponseState],
                state_access: RegionStateAccess {
                    read_stores: vec!["kv-store".to_string()],
                    write_stores: vec!["draft-store".to_string()],
                    requires_epoch_check: true,
                },
                authority: vec![RegionOutputAuthority::DraftText],
            },
            AssistantRegionDecl {
                region_id: "finalizer-1".to_string(),
                partition_id: None,
                region_kind: AssistantRegionKind::ReasoningDecode,
                input_types: vec![BridgeValueType::AssistantResponseState],
                output_types: vec![BridgeValueType::AssistantResponseState],
                state_access: RegionStateAccess {
                    read_stores: vec!["draft-store".to_string()],
                    write_stores: vec!["response-store".to_string()],
                    requires_epoch_check: true,
                },
                authority: vec![RegionOutputAuthority::CommittedAssistantResponse],
            },
            AssistantRegionDecl {
                region_id: "tts-1".to_string(),
                partition_id: None,
                region_kind: AssistantRegionKind::SpeechSynthesis,
                input_types: vec![
                    BridgeValueType::AssistantResponseState,
                    BridgeValueType::SpeechPlan,
                ],
                output_types: vec![BridgeValueType::SpeechPlan],
                state_access: RegionStateAccess {
                    read_stores: vec!["response-store".to_string()],
                    write_stores: vec!["audio-cache".to_string()],
                    requires_epoch_check: false,
                },
                authority: vec![
                    RegionOutputAuthority::SpeechPlan,
                    RegionOutputAuthority::AudioFrames,
                ],
            },
        ],
        bridges: vec![
            BridgeDecl {
                bridge_id: "bridge-2".to_string(),
                value_type: BridgeValueType::DraftReasoningTrace,
                source_region: "reasoning-1".to_string(),
                target_region: "reasoning-2".to_string(),
            },
            BridgeDecl {
                bridge_id: "bridge-3a".to_string(),
                value_type: BridgeValueType::AssistantResponseState,
                source_region: "finalizer-1".to_string(),
                target_region: "tts-1".to_string(),
            },
            BridgeDecl {
                bridge_id: "bridge-3b".to_string(),
                value_type: BridgeValueType::AssistantResponseState,
                source_region: "reasoning-2".to_string(),
                target_region: "finalizer-1".to_string(),
            },
            BridgeDecl {
                bridge_id: "bridge-4".to_string(),
                value_type: BridgeValueType::SpeechPlan,
                source_region: "tts-1".to_string(),
                target_region: "tts-1".to_string(),
            },
            BridgeDecl {
                bridge_id: "bridge-5".to_string(),
                value_type: BridgeValueType::TokenSequence,
                source_region: "reasoning-1".to_string(),
                target_region: "reasoning-1".to_string(),
            },
        ],
        shared_state_schema: SharedStateSchema {
            stores: vec![
                StateStoreDecl {
                    store_id: "kv-store".to_string(),
                    store_kind: StateStoreKind::TextKvCache,
                    owner_region: "reasoning-1".to_string(),
                    dtype: "f32".to_string(),
                    max_bytes: 1_000_000,
                    persistence: StatePersistence::Session,
                },
                StateStoreDecl {
                    store_id: "response-store".to_string(),
                    store_kind: StateStoreKind::AssistantResponseState,
                    owner_region: "finalizer-1".to_string(),
                    dtype: "f32".to_string(),
                    max_bytes: 100_000,
                    persistence: StatePersistence::EpochOnly,
                },
                StateStoreDecl {
                    store_id: "draft-store".to_string(),
                    store_kind: StateStoreKind::RetrievalWorkingSet,
                    owner_region: "reasoning-2".to_string(),
                    dtype: "f32".to_string(),
                    max_bytes: 200_000,
                    persistence: StatePersistence::EpochOnly,
                },
                StateStoreDecl {
                    store_id: "audio-cache".to_string(),
                    store_kind: StateStoreKind::SpeechAcousticCache,
                    owner_region: "tts-1".to_string(),
                    dtype: "f32".to_string(),
                    max_bytes: 500_000,
                    persistence: StatePersistence::EpochOnly,
                },
            ],
        },
        route_graph: AssistantRouteGraph {
            edges: vec![
                RouteEdge {
                    from_region: "reasoning-1".to_string(),
                    to_region: "reasoning-2".to_string(),
                    allowed_types: vec![BridgeValueType::DraftReasoningTrace],
                },
                RouteEdge {
                    from_region: "finalizer-1".to_string(),
                    to_region: "tts-1".to_string(),
                    allowed_types: vec![
                        BridgeValueType::AssistantResponseState,
                        BridgeValueType::SpeechPlan,
                    ],
                },
                RouteEdge {
                    from_region: "reasoning-2".to_string(),
                    to_region: "finalizer-1".to_string(),
                    allowed_types: vec![BridgeValueType::AssistantResponseState],
                },
            ],
            route_kind: RouteKind::Sequential,
        },
        authority_policy: RegionAuthorityPolicy {
            policy_id: "policy-1".to_string(),
            rules: vec![AuthorityRule {
                rule_kind: AuthorityRuleKind::OnlyOneRegionMayEmit(
                    RegionOutputAuthority::CommittedAssistantResponse,
                ),
                reject_message: "Only one region can emit committed response".to_string(),
            }],
        },
    }
}

// ---------------------------------------------------------------------------
// Serde roundtrip tests
// ---------------------------------------------------------------------------

#[test]
fn test_serde_assistant_graph_manifest() {
    let m = valid_manifest();
    let json = serde_json::to_string(&m).unwrap();
    let back: AssistantGraphManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(m.graph_id, back.graph_id);
    assert_eq!(m.schema_version, back.schema_version);
    assert_eq!(m.regions.len(), back.regions.len());
    assert_eq!(m.bridges.len(), back.bridges.len());
}

#[test]
fn test_serde_assistant_contract() {
    let c = AssistantContract {
        contract_id: "c1".to_string(),
        max_active_regions: 5,
        requires_bridge_types: false,
        requires_authority: true,
    };
    let json = serde_json::to_string(&c).unwrap();
    let back: AssistantContract = serde_json::from_str(&json).unwrap();
    assert_eq!(c.contract_id, back.contract_id);
    assert_eq!(c.max_active_regions, back.max_active_regions);
}

#[test]
fn test_serde_assistant_region_decl() {
    let r = AssistantRegionDecl {
        region_id: "r1".to_string(),
        partition_id: Some("p1".to_string()),
        region_kind: AssistantRegionKind::VisionPerception,
        input_types: vec![BridgeValueType::UserImageInput],
        output_types: vec![BridgeValueType::PerceptionFacts],
        state_access: RegionStateAccess {
            read_stores: vec![],
            write_stores: vec![],
            requires_epoch_check: false,
        },
        authority: vec![RegionOutputAuthority::PerceptionFacts],
    };
    let json = serde_json::to_string(&r).unwrap();
    let back: AssistantRegionDecl = serde_json::from_str(&json).unwrap();
    assert_eq!(r.region_id, back.region_id);
    assert_eq!(back.region_kind, AssistantRegionKind::VisionPerception);
}

#[test]
fn test_serde_region_kind_all_variants() {
    use AssistantRegionKind::*;
    let variants = vec![
        ReasoningDecode,
        VisionPerception,
        EmbeddingRetrieval,
        SpeechSynthesis,
        ToolDecision,
        DecoderLayerProof,
        SyntheticStub,
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: AssistantRegionKind = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_region_output_authority_all_variants() {
    use RegionOutputAuthority::*;
    let variants = vec![
        PerceptionFacts,
        EmbeddingVectors,
        RetrievalCandidates,
        DraftText,
        ToolCallProposal,
        ToolCallDecision,
        CommittedAssistantResponse,
        SpeechPlan,
        AudioFrames,
        KvCacheUpdate,
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: RegionOutputAuthority = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_bridge_value_type_all_variants() {
    use BridgeValueType::*;
    let variants = vec![
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
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: BridgeValueType = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_assistant_route_graph() {
    let g = AssistantRouteGraph {
        edges: vec![RouteEdge {
            from_region: "a".to_string(),
            to_region: "b".to_string(),
            allowed_types: vec![BridgeValueType::TokenSequence],
        }],
        route_kind: RouteKind::Parallel,
    };
    let json = serde_json::to_string(&g).unwrap();
    let back: AssistantRouteGraph = serde_json::from_str(&json).unwrap();
    assert_eq!(g.route_kind, back.route_kind);
    assert_eq!(g.edges.len(), back.edges.len());
}

#[test]
fn test_serde_bridge_decl() {
    let b = BridgeDecl {
        bridge_id: "b1".to_string(),
        value_type: BridgeValueType::PerceptionFacts,
        source_region: "vision".to_string(),
        target_region: "reasoning".to_string(),
    };
    let json = serde_json::to_string(&b).unwrap();
    let back: BridgeDecl = serde_json::from_str(&json).unwrap();
    assert_eq!(b.bridge_id, back.bridge_id);
    assert_eq!(b.value_type, back.value_type);
}

#[test]
fn test_serde_assistant_response_state() {
    let s = AssistantResponseState {
        text: "Hello".to_string(),
        language: "en".to_string(),
        turn_intent: TurnIntent::Respond,
        speaking_style: SpeakingStyle::Neutral,
        emotional_register: EmotionalRegister::Warm,
        pacing: PacingPlan {
            words_per_minute: 150.0,
            pause_after_sentence_ms: 200,
        },
        emphasis_spans: vec![EmphasisSpan {
            start: 0,
            length: 5,
            strength: 0.8,
        }],
        pronunciation_hints: vec![PronunciationHint {
            text: "hello".to_string(),
            ipa: "həˈloʊ".to_string(),
        }],
        confidence: 0.95,
        committed_epoch: 42,
    };
    let json = serde_json::to_string(&s).unwrap();
    let back: AssistantResponseState = serde_json::from_str(&json).unwrap();
    assert_eq!(s.text, back.text);
    assert_eq!(s.confidence, back.confidence);
    assert_eq!(s.committed_epoch, back.committed_epoch);
}

#[test]
fn test_serde_speech_plan() {
    let sp = SpeechPlan {
        utterances: vec![SpeechUtterance {
            text: "Hello world".to_string(),
            duration_ms: 500,
        }],
        language: "en".to_string(),
        voice_id: Some("v1".to_string()),
        pace: 1.0,
        emphasis_spans: vec![],
        pronunciation_hints: vec![],
        source_response_epoch: 42,
    };
    let json = serde_json::to_string(&sp).unwrap();
    let back: SpeechPlan = serde_json::from_str(&json).unwrap();
    assert_eq!(sp.utterances.len(), back.utterances.len());
    assert_eq!(sp.language, back.language);
    assert_eq!(sp.source_response_epoch, back.source_response_epoch);
}

#[test]
fn test_serde_region_state_access() {
    let a = RegionStateAccess {
        read_stores: vec!["store-a".to_string()],
        write_stores: vec!["store-b".to_string()],
        requires_epoch_check: true,
    };
    let json = serde_json::to_string(&a).unwrap();
    let back: RegionStateAccess = serde_json::from_str(&json).unwrap();
    assert_eq!(a.read_stores, back.read_stores);
    assert_eq!(a.requires_epoch_check, back.requires_epoch_check);
}

#[test]
fn test_serde_shared_state_schema() {
    let s = SharedStateSchema {
        stores: vec![StateStoreDecl {
            store_id: "s1".to_string(),
            store_kind: StateStoreKind::AssistantResponseState,
            owner_region: "r1".to_string(),
            dtype: "f32".to_string(),
            max_bytes: 1024,
            persistence: StatePersistence::Session,
        }],
    };
    let json = serde_json::to_string(&s).unwrap();
    let back: SharedStateSchema = serde_json::from_str(&json).unwrap();
    assert_eq!(s.stores.len(), back.stores.len());
    assert_eq!(s.stores[0].store_id, back.stores[0].store_id);
}

#[test]
fn test_serde_state_store_kind_all_variants() {
    use StateStoreKind::*;
    let variants = vec![
        TextKvCache,
        VisionEmbeddingCache,
        SpeechSemanticCache,
        SpeechAcousticCache,
        RetrievalWorkingSet,
        AssistantResponseState,
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: StateStoreKind = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_authority_policy() {
    let p = RegionAuthorityPolicy {
        policy_id: "p1".to_string(),
        rules: vec![AuthorityRule {
            rule_kind: AuthorityRuleKind::OnlyOneRegionMayEmit(
                RegionOutputAuthority::CommittedAssistantResponse,
            ),
            reject_message: "only one".to_string(),
        }],
    };
    let json = serde_json::to_string(&p).unwrap();
    let back: RegionAuthorityPolicy = serde_json::from_str(&json).unwrap();
    assert_eq!(p.policy_id, back.policy_id);
    assert_eq!(p.rules.len(), back.rules.len());
}

#[test]
fn test_serde_authority_rule_kind_all_variants() {
    use AuthorityRuleKind::*;
    let variants = vec![
        OnlyOneRegionMayEmit(RegionOutputAuthority::PerceptionFacts),
        RegionMayNotMutate {
            region: "r1".to_string(),
            store_kind: "AssistantResponseState".to_string(),
        },
        TtsMustNotConsumeDraftText,
        RouteRequiresAuthority {
            source: RegionOutputAuthority::DraftText,
            target: RegionOutputAuthority::CommittedAssistantResponse,
        },
    ];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: AuthorityRuleKind = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_validation_receipt() {
    let r = AssistantGraphValidationReceipt {
        graph_id: "g1".to_string(),
        contract_valid: true,
        region_count: 3,
        bridge_count: 2,
        route_edges: 1,
        errors: vec![],
        warnings: vec!["warning-1".to_string()],
        validation_status: AssistantGraphValidationStatus::ValidWithWarnings,
    };
    let json = serde_json::to_string(&r).unwrap();
    let back: AssistantGraphValidationReceipt = serde_json::from_str(&json).unwrap();
    assert_eq!(r.graph_id, back.graph_id);
    assert_eq!(r.errors.len(), back.errors.len());
    assert_eq!(r.warnings.len(), back.warnings.len());
    assert_eq!(
        back.validation_status,
        AssistantGraphValidationStatus::ValidWithWarnings
    );
}

#[test]
fn test_serde_validation_status_all_variants() {
    use AssistantGraphValidationStatus::*;
    let variants = vec![Valid, ValidWithWarnings, Invalid];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: AssistantGraphValidationStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_turn_intent_all_variants() {
    use TurnIntent::*;
    let variants = vec![Respond, Explain, Clarify, ExecuteTool];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: TurnIntent = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_speaking_style_all_variants() {
    use SpeakingStyle::*;
    let variants = vec![Neutral, Empathetic, Concise, Detailed];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: SpeakingStyle = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_emotional_register_all_variants() {
    use EmotionalRegister::*;
    let variants = vec![Neutral, Warm, Urgent, Reflective];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: EmotionalRegister = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_state_persistence_all_variants() {
    use StatePersistence::*;
    let variants = vec![EpochOnly, Session, Permanent];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: StatePersistence = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

#[test]
fn test_serde_route_kind_all_variants() {
    use RouteKind::*;
    let variants = vec![Sequential, Parallel, Conditional];
    for v in &variants {
        let json = serde_json::to_string(v).unwrap();
        let back: RouteKind = serde_json::from_str(&json).unwrap();
        assert_eq!(*v, back);
    }
}

// ---------------------------------------------------------------------------
// Validator gate tests
// ---------------------------------------------------------------------------

#[test]
fn test_validator_valid_manifest() {
    let m = valid_manifest();
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Valid,
        "Expected Valid, got {:?} with errors: {:?}",
        receipt.validation_status,
        receipt.errors
    );
    assert!(receipt.errors.is_empty());
    assert!(receipt.contract_valid);
    assert_eq!(receipt.region_count, 4);
    assert_eq!(receipt.bridge_count, 5);
    assert_eq!(receipt.route_edges, 3);
}

#[test]
fn test_gate1_two_regions_emit_committed_response() {
    let mut m = valid_manifest();
    // Add a second region with CommittedAssistantResponse (same partition — none)
    m.regions.push(AssistantRegionDecl {
        region_id: "redundant-responder".to_string(),
        partition_id: None,
        region_kind: AssistantRegionKind::ToolDecision,
        input_types: vec![],
        output_types: vec![],
        state_access: RegionStateAccess {
            read_stores: vec![],
            write_stores: vec![],
            requires_epoch_check: false,
        },
        authority: vec![RegionOutputAuthority::CommittedAssistantResponse],
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate1 = receipt.errors.iter().any(|e| e.starts_with("Gate 1:"));
    assert!(
        has_gate1,
        "Expected Gate 1 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate2_tts_mutates_semantic_response() {
    let mut m = valid_manifest();
    // Point tts-1 to write to the response-store (which is AssistantResponseState kind)
    if let Some(tts) = m.regions.iter_mut().find(|r| r.region_id == "tts-1") {
        tts.state_access
            .write_stores
            .push("response-store".to_string());
    }
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate2 = receipt.errors.iter().any(|e| e.starts_with("Gate 2:"));
    assert!(
        has_gate2,
        "Expected Gate 2 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate3_vision_emits_committed_text() {
    let mut m = valid_manifest();
    m.regions.push(AssistantRegionDecl {
        region_id: "vision-1".to_string(),
        partition_id: None,
        region_kind: AssistantRegionKind::VisionPerception,
        input_types: vec![],
        output_types: vec![],
        state_access: RegionStateAccess {
            read_stores: vec![],
            write_stores: vec![],
            requires_epoch_check: false,
        },
        authority: vec![RegionOutputAuthority::CommittedAssistantResponse],
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate3 = receipt.errors.iter().any(|e| e.starts_with("Gate 3:"));
    assert!(
        has_gate3,
        "Expected Gate 3 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate4_embedding_writes_identity() {
    let mut m = valid_manifest();
    m.regions.push(AssistantRegionDecl {
        region_id: "embed-1".to_string(),
        partition_id: None,
        region_kind: AssistantRegionKind::EmbeddingRetrieval,
        input_types: vec![],
        output_types: vec![],
        state_access: RegionStateAccess {
            read_stores: vec![],
            write_stores: vec!["response-store".to_string()],
            requires_epoch_check: false,
        },
        authority: vec![RegionOutputAuthority::EmbeddingVectors],
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate4 = receipt.errors.iter().any(|e| e.starts_with("Gate 4:"));
    assert!(
        has_gate4,
        "Expected Gate 4 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate5_conflicting_writes() {
    let mut m = valid_manifest();
    // Make both reasoning-1 and reasoning-2 write to "kv-store"
    if let Some(r2) = m.regions.iter_mut().find(|r| r.region_id == "reasoning-2") {
        r2.state_access.write_stores.push("kv-store".to_string());
    }
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate5 = receipt.errors.iter().any(|e| e.starts_with("Gate 5:"));
    assert!(
        has_gate5,
        "Expected Gate 5 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate6_route_draft_text_mismatch() {
    let mut m = valid_manifest();
    // Add a route edge that allows DraftReasoningTrace to a region
    // that has CommittedAssistantResponse authority
    m.route_graph.edges.push(RouteEdge {
        from_region: "reasoning-1".to_string(),
        to_region: "finalizer-1".to_string(),
        allowed_types: vec![BridgeValueType::DraftReasoningTrace],
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate6 = receipt.errors.iter().any(|e| e.starts_with("Gate 6:"));
    assert!(
        has_gate6,
        "Expected Gate 6 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate7a_exceeds_max_active_regions() {
    let mut m = valid_manifest();
    m.assistant_contract.max_active_regions = 1; // only 1 allowed, we have 3
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate7a = receipt.errors.iter().any(|e| e.starts_with("Gate 7a:"));
    assert!(
        has_gate7a,
        "Expected Gate 7a error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate7b_requires_bridge_types_empty() {
    let mut m = valid_manifest();
    m.assistant_contract.requires_bridge_types = true;
    m.bridges.clear();
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate7b = receipt.errors.iter().any(|e| e.starts_with("Gate 7b:"));
    assert!(
        has_gate7b,
        "Expected Gate 7b error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate7c_requires_authority_missing() {
    let mut m = valid_manifest();
    m.assistant_contract.requires_authority = true;
    // Clear authority on the first region
    if let Some(r) = m.regions.first_mut() {
        r.authority.clear();
    }
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate7c = receipt.errors.iter().any(|e| e.starts_with("Gate 7c:"));
    assert!(
        has_gate7c,
        "Expected Gate 7c error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate8_bridge_unknown_region() {
    let mut m = valid_manifest();
    m.bridges.push(BridgeDecl {
        bridge_id: "bad-bridge".to_string(),
        value_type: BridgeValueType::UserTextInput,
        source_region: "ghost-region".to_string(),
        target_region: "reasoning-1".to_string(),
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate8 = receipt.errors.iter().any(|e| e.starts_with("Gate 8:"));
    assert!(
        has_gate8,
        "Expected Gate 8 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate9_region_type_unresolved() {
    let mut m = valid_manifest();
    // Add a region with an input type not backed by any bridge
    m.regions.push(AssistantRegionDecl {
        region_id: "orphan-region".to_string(),
        partition_id: Some("p2".to_string()),
        region_kind: AssistantRegionKind::DecoderLayerProof,
        input_types: vec![BridgeValueType::KvCacheView],
        output_types: vec![],
        state_access: RegionStateAccess {
            read_stores: vec![],
            write_stores: vec![],
            requires_epoch_check: false,
        },
        authority: vec![RegionOutputAuthority::KvCacheUpdate],
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate9 = receipt.errors.iter().any(|e| e.starts_with("Gate 9:"));
    assert!(
        has_gate9,
        "Expected Gate 9 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate10_duplicate_bridge_ids() {
    let mut m = valid_manifest();
    m.bridges.push(BridgeDecl {
        bridge_id: "bridge-2".to_string(),
        value_type: BridgeValueType::TokenSequence,
        source_region: "reasoning-1".to_string(),
        target_region: "finalizer-1".to_string(),
    });
    let receipt = AssistantGraphValidator::validate(&m);
    assert_eq!(
        receipt.validation_status,
        AssistantGraphValidationStatus::Invalid
    );
    let has_gate10 = receipt.errors.iter().any(|e| e.starts_with("Gate 10:"));
    assert!(
        has_gate10,
        "Expected Gate 10 error, got: {:?}",
        receipt.errors
    );
}

#[test]
fn test_gate_partitioned_committed_response_is_ok() {
    // Two regions in different partitions both emitting CommittedAssistantResponse should be OK
    let mut m = valid_manifest();
    m.regions.push(AssistantRegionDecl {
        region_id: "other-responder".to_string(),
        partition_id: Some("partition-2".to_string()),
        region_kind: AssistantRegionKind::ToolDecision,
        input_types: vec![],
        output_types: vec![],
        state_access: RegionStateAccess {
            read_stores: vec![],
            write_stores: vec![],
            requires_epoch_check: false,
        },
        authority: vec![RegionOutputAuthority::CommittedAssistantResponse],
    });
    let receipt = AssistantGraphValidator::validate(&m);
    // The original reasoning-2 also has CommittedAssistantResponse in partition=None,
    // but this new one is in a different partition — should be OK
    let gate1_errors: Vec<&String> = receipt
        .errors
        .iter()
        .filter(|e| e.starts_with("Gate 1:"))
        .collect();
    assert!(
        gate1_errors.is_empty(),
        "Expected no Gate 1 errors for different partitions, got: {:?}",
        gate1_errors
    );
}
