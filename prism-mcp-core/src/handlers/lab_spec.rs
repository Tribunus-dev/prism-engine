//! Experiment spec types: the definition of an experiment's structure, steps, and gates.

use serde::{Deserialize, Serialize};

/// The overall state of an experiment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum ExperimentState {
    /// Experiment has been defined but not started.
    #[default]
    Pending,
    /// Experiment is actively executing steps.
    Running,
    /// Experiment completed successfully (all steps passed all gates).
    Completed,
    /// Experiment was cancelled by the user.
    Cancelled,
    /// Experiment failed: a step or gate rejected the result.
    Failed,
}

/// A gate condition that must be satisfied for a step to be considered passing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateCondition {
    /// A named metric must exceed this threshold.
    MetricAbove { metric: String, threshold: f64 },
    /// A named metric must be below this threshold.
    MetricBelow { metric: String, threshold: f64 },
    /// A custom predicate evaluated by the daemon tool registry.
    Custom {
        tool_name: String,
        args: serde_json::Value,
    },
}

/// The execution state of a single experiment step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum StepState {
    /// Step hasn't run yet.
    #[default]
    Pending,
    /// Step is currently executing.
    Running,
    /// Step completed; the attached gates determined the outcome.
    Passed,
    /// Step completed but one or more gates rejected it.
    Failed,
    /// Step was skipped (e.g. a prior gate failed or experiment was cancelled).
    Skipped,
}

/// A single step in an experiment DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentStep {
    /// Unique name within the experiment (e.g. "calibrate", "validate", "benchmark").
    pub name: String,
    /// The tool registered in the daemon that implements this step.
    pub tool_name: String,
    /// Arguments forwarded to the tool on execution.
    pub args: serde_json::Value,
    /// Names of steps that must complete before this one runs.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Gate conditions checked after the step's tool returns.
    #[serde(default)]
    pub gates: Vec<GateCondition>,
    /// Current execution state.
    #[serde(default)]
    pub state: StepState,
    /// Optional human-readable result summary populated on completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_summary: Option<String>,
}

/// The full definition of an experiment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpec {
    /// Human-readable name.
    pub name: String,
    /// Free-form description.
    #[serde(default)]
    pub description: String,
    /// Ordered (by dependency resolution) list of steps.
    pub steps: Vec<ExperimentStep>,
    /// Global experiment state.
    #[serde(default)]
    pub state: ExperimentState,
    /// Serialized result payload from a completed/promoted experiment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Tags for filtering and search.
    #[serde(default)]
    pub tags: Vec<String>,
}

impl ExperimentSpec {
    /// Returns `true` if every step has reached a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.steps.iter().all(|s| {
            matches!(
                s.state,
                StepState::Passed | StepState::Failed | StepState::Skipped
            )
        })
    }

    /// Returns the names of steps whose dependencies are all satisfied
    /// and are still `Pending` (ready to execute).
    pub fn ready_steps(&self) -> Vec<&str> {
        let terminal: Vec<&str> = self
            .steps
            .iter()
            .filter(|s| {
                matches!(
                    s.state,
                    StepState::Passed | StepState::Skipped | StepState::Failed
                )
            })
            .map(|s| s.name.as_str())
            .collect();

        self.steps
            .iter()
            .filter(|s| s.state == StepState::Pending)
            .filter(|s| {
                s.depends_on
                    .iter()
                    .all(|dep| terminal.contains(&dep.as_str()))
            })
            .map(|s| s.name.as_str())
            .collect()
    }
    /// Returns the names of steps that should be auto-skipped because a
    /// dependency failed (making continued execution pointless).
    pub fn steps_to_skip(&self) -> Vec<&str> {
        let failed_deps: Vec<&str> = self
            .steps
            .iter()
            .filter(|s| s.state == StepState::Failed)
            .map(|s| s.name.as_str())
            .collect();

        self.steps
            .iter()
            .filter(|s| {
                s.state == StepState::Pending
                    && s.depends_on
                        .iter()
                        .any(|dep| failed_deps.contains(&dep.as_str()))
            })
            .map(|s| s.name.as_str())
            .collect()
    }
}

/// Payload used when promoting an experiment result to production.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionPayload {
    pub experiment_id: String,
    pub result: serde_json::Value,
    pub promoted_by: String,
    pub promoted_at: String,
}
