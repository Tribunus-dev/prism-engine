//! Semantic admission for agent/tool actions.
//!
//! Models remain probabilistic planners. This module provides the typed,
//! deterministic boundary that validates proposed actions before handlers are
//! allowed to produce side effects.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActionContract {
    pub tool: String,
    pub arguments: serde_json::Value,
    pub idempotency_key: Option<String>,
    pub expected_state: Option<String>,
    pub side_effects: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologyEntity {
    pub id: String,
    pub kind: String,
    pub properties: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OntologyRelation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticViolation {
    pub code: String,
    pub message: String,
    pub path: Option<String>,
    pub retryable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValidationReceipt {
    pub action_id: String,
    pub tool: String,
    pub accepted: bool,
    pub phase: String,
    pub violations: Vec<SemanticViolation>,
    pub state_before: Option<String>,
    pub state_after: Option<String>,
    pub side_effects_permitted: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StagedActionPhase {
    Planned,
    Validated,
    Committed,
    Aborted,
}

/// Explicit two-phase wrapper for side-effecting actions. Handlers can build
/// one during planning and only commit it after semantic validation succeeds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StagedAction {
    pub contract: ActionContract,
    pub receipt: ValidationReceipt,
    pub phase: StagedActionPhase,
}

impl StagedAction {
    pub fn plan(contract: ActionContract, receipt: ValidationReceipt) -> Self {
        let phase = if receipt.accepted {
            StagedActionPhase::Validated
        } else {
            StagedActionPhase::Aborted
        };
        Self {
            contract,
            receipt,
            phase,
        }
    }
    pub fn commit(&mut self) -> Result<&ValidationReceipt, SemanticViolation> {
        if self.phase != StagedActionPhase::Validated || !self.receipt.side_effects_permitted {
            return Err(SemanticViolation {
                code: "SIDE_EFFECT_NOT_PERMITTED".into(),
                message: "action must pass semantic admission before commit".into(),
                path: None,
                retryable: false,
            });
        }
        self.phase = StagedActionPhase::Committed;
        Ok(&self.receipt)
    }
    pub fn abort(&mut self) {
        self.phase = StagedActionPhase::Aborted;
    }
}

#[derive(Debug, Clone, Default)]
pub struct Ontology {
    entities: HashMap<String, OntologyEntity>,
    relations: Vec<OntologyRelation>,
    transitions: HashMap<(String, String), HashSet<String>>,
}

impl Ontology {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_entity(&mut self, entity: OntologyEntity) {
        self.entities.insert(entity.id.clone(), entity);
    }
    pub fn add_relation(&mut self, relation: OntologyRelation) {
        self.relations.push(relation);
    }

    pub fn allow_transition(&mut self, entity_kind: &str, from: &str, to: &str) {
        self.transitions
            .entry((entity_kind.to_string(), from.to_string()))
            .or_default()
            .insert(to.to_string());
    }

    pub fn validate(&self, action: &ActionContract) -> ValidationReceipt {
        let mut violations = Vec::new();
        if action.tool.trim().is_empty() {
            violations.push(SemanticViolation {
                code: "EMPTY_TOOL".into(),
                message: "tool name is required".into(),
                path: Some("tool".into()),
                retryable: false,
            });
        }
        if !action.arguments.is_object() {
            violations.push(SemanticViolation {
                code: "ARGUMENTS_NOT_OBJECT".into(),
                message: "tool arguments must be a JSON object".into(),
                path: Some("arguments".into()),
                retryable: true,
            });
        }
        if let Some(expected) = &action.expected_state {
            let entity_kind = action
                .arguments
                .get("entity_kind")
                .and_then(|v| v.as_str())
                .unwrap_or("tool");
            let next = action.arguments.get("next_state").and_then(|v| v.as_str());
            if let Some(next) = next {
                let allowed = self
                    .transitions
                    .get(&(entity_kind.to_string(), expected.clone()))
                    .map(|states| states.contains(next))
                    .unwrap_or(false);
                if !allowed {
                    violations.push(SemanticViolation {
                        code: "INVALID_STATE_TRANSITION".into(),
                        message: format!(
                            "{entity_kind} cannot transition from {expected} to {next}"
                        ),
                        path: Some("next_state".into()),
                        retryable: false,
                    });
                }
            }
        }
        let accepted = violations.is_empty();
        ValidationReceipt {
            action_id: uuid::Uuid::new_v4().to_string(),
            tool: action.tool.clone(),
            accepted,
            phase: "validated".into(),
            violations,
            state_before: action.expected_state.clone(),
            state_after: action
                .arguments
                .get("next_state")
                .and_then(|v| v.as_str())
                .map(str::to_owned),
            side_effects_permitted: accepted && action.side_effects,
        }
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
    pub fn relation_count(&self) -> usize {
        self.relations.len()
    }
}

/// Shared admission service used by the MCP scheduler and agent handlers.
#[derive(Clone, Default)]
pub struct SemanticAdmission {
    ontology: Arc<RwLock<Ontology>>,
}

/// Minimal deterministic JSON-schema gate for tool boundaries. It covers the
/// contract features Prism tools rely on today without making handlers depend
/// on a particular schema-validation crate.
pub fn validate_typed_arguments(
    schema: &serde_json::Value,
    args: &serde_json::Value,
) -> Vec<SemanticViolation> {
    let mut violations = Vec::new();
    let Some(object) = args.as_object() else {
        return vec![SemanticViolation {
            code: "ARGUMENTS_NOT_OBJECT".into(),
            message: "arguments must be an object".into(),
            path: None,
            retryable: true,
        }];
    };
    if let Some(required) = schema.get("required").and_then(|v| v.as_array()) {
        for name in required.iter().filter_map(|v| v.as_str()) {
            if !object.contains_key(name) {
                violations.push(SemanticViolation {
                    code: "MISSING_REQUIRED_ARGUMENT".into(),
                    message: format!("required argument '{name}' is missing"),
                    path: Some(name.into()),
                    retryable: true,
                });
            }
        }
    }
    if let Some(properties) = schema.get("properties").and_then(|v| v.as_object()) {
        for (name, definition) in properties {
            let Some(value) = object.get(name) else {
                continue;
            };
            let expected = definition.get("type").and_then(|v| v.as_str());
            let valid = match expected {
                Some("string") => value.is_string(),
                Some("integer") => value.as_i64().is_some() || value.as_u64().is_some(),
                Some("number") => value.is_number(),
                Some("boolean") => value.is_boolean(),
                Some("array") => value.is_array(),
                Some("object") => value.is_object(),
                _ => true,
            };
            if !valid {
                violations.push(SemanticViolation {
                    code: "ARGUMENT_TYPE_MISMATCH".into(),
                    message: format!("argument '{name}' does not match declared type {expected:?}"),
                    path: Some(name.clone()),
                    retryable: true,
                });
            }
        }
    }
    violations
}

impl SemanticAdmission {
    pub fn new() -> Self {
        let admission = Self::default();
        admission.with_ontology(|ontology| {
            ontology.allow_transition("job", "queued", "running");
            ontology.allow_transition("job", "running", "completed");
            ontology.allow_transition("job", "running", "failed");
            ontology.allow_transition("job", "running", "cancelled");
            ontology.allow_transition("model", "unloaded", "loaded");
            ontology.allow_transition("model", "loaded", "unloaded");
            ontology.allow_transition("inference", "created", "running");
            ontology.allow_transition("inference", "running", "completed");
            ontology.allow_transition("inference", "running", "cancelled");
            ontology.allow_transition("inference", "running", "failed");
        });
        admission
    }
    pub fn validate(&self, action: &ActionContract) -> ValidationReceipt {
        self.ontology
            .read()
            .map(|ontology| ontology.validate(action))
            .unwrap_or_else(|_| ValidationReceipt {
                action_id: uuid::Uuid::new_v4().to_string(),
                tool: action.tool.clone(),
                accepted: false,
                phase: "validator_unavailable".into(),
                violations: vec![SemanticViolation {
                    code: "ONTOLOGY_LOCKED".into(),
                    message: "semantic validator unavailable".into(),
                    path: None,
                    retryable: true,
                }],
                state_before: action.expected_state.clone(),
                state_after: None,
                side_effects_permitted: false,
            })
    }
    pub fn with_ontology<R>(&self, f: impl FnOnce(&mut Ontology) -> R) -> R {
        f(&mut self
            .ontology
            .write()
            .expect("semantic ontology lock poisoned"))
    }
}

#[derive(Debug, Clone)]
pub struct LoopBudget {
    pub max_steps: u32,
    pub max_tokens: u64,
    pub deadline: Instant,
}

impl LoopBudget {
    pub fn new(max_steps: u32, max_tokens: u64, timeout: Duration) -> Self {
        Self {
            max_steps,
            max_tokens,
            deadline: Instant::now() + timeout,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoopDecision {
    Continue,
    Escalate(String),
    Stop(String),
}

#[derive(Debug)]
pub struct LoopGuard {
    budget: LoopBudget,
    steps: u32,
    tokens: u64,
    states: HashSet<String>,
}

impl LoopGuard {
    pub fn new(budget: LoopBudget) -> Self {
        Self {
            budget,
            steps: 0,
            tokens: 0,
            states: HashSet::new(),
        }
    }
    pub fn observe(&mut self, state: &str, tokens: u64) -> LoopDecision {
        self.steps += 1;
        self.tokens = self.tokens.saturating_add(tokens);
        if !self.states.insert(state.to_string()) {
            return LoopDecision::Escalate("repeated agent state detected".into());
        }
        if self.steps > self.budget.max_steps {
            return LoopDecision::Stop("agent step budget exhausted".into());
        }
        if self.tokens > self.budget.max_tokens {
            return LoopDecision::Stop("agent token budget exhausted".into());
        }
        if Instant::now() >= self.budget.deadline {
            return LoopDecision::Stop("agent deadline exceeded".into());
        }
        LoopDecision::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_transition_is_rejected_before_side_effects() {
        let mut ontology = Ontology::new();
        ontology.allow_transition("job", "queued", "running");
        let action = ActionContract {
            tool: "run_job".into(),
            arguments: serde_json::json!({"entity_kind":"job","next_state":"completed"}),
            idempotency_key: None,
            expected_state: Some("queued".into()),
            side_effects: true,
        };
        let receipt = ontology.validate(&action);
        assert!(!receipt.accepted);
        assert!(!receipt.side_effects_permitted);
        assert_eq!(receipt.violations[0].code, "INVALID_STATE_TRANSITION");
    }

    #[test]
    fn loop_guard_detects_repetition_and_budget_exhaustion() {
        let mut guard = LoopGuard::new(LoopBudget::new(3, 100, Duration::from_secs(10)));
        assert_eq!(guard.observe("a", 1), LoopDecision::Continue);
        assert!(matches!(guard.observe("a", 1), LoopDecision::Escalate(_)));
    }

    #[test]
    fn staged_action_cannot_commit_after_rejection() {
        let contract = ActionContract {
            tool: "write".into(),
            arguments: serde_json::json!({}),
            idempotency_key: None,
            expected_state: None,
            side_effects: true,
        };
        let receipt = ValidationReceipt {
            action_id: "a".into(),
            tool: "write".into(),
            accepted: false,
            phase: "validated".into(),
            violations: vec![],
            state_before: None,
            state_after: None,
            side_effects_permitted: false,
        };
        let mut staged = StagedAction::plan(contract, receipt);
        assert!(staged.commit().is_err());
        assert_eq!(staged.phase, StagedActionPhase::Aborted);
    }

    #[test]
    fn typed_arguments_enforce_required_fields_and_types() {
        let schema = serde_json::json!({"type":"object","required":["count"],"properties":{"count":{"type":"integer"}}});
        assert_eq!(
            validate_typed_arguments(&schema, &serde_json::json!({"count":"one"}))[0].code,
            "ARGUMENT_TYPE_MISMATCH"
        );
        assert_eq!(
            validate_typed_arguments(&schema, &serde_json::json!({}))[0].code,
            "MISSING_REQUIRED_ARGUMENT"
        );
    }
}
