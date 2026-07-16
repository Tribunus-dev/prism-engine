//! prism_mcp_lab handlers — experiment lifecycle tools.

use crate::handlers::lab_spec::{ExperimentSpec, ExperimentState, ExperimentStep, GateCondition, StepState};
use anyhow::{Context, Result};
use crate::{
    DaemonState, ExperimentId, McpHandler, RequestContext, ToolRequest, ToolResult,
};
use serde_json::Value;

fn get_str<'a>(args: &'a Value, field: &str) -> Result<&'a str> {
    args.get(field)
        .and_then(|v| v.as_str())
        .with_context(|| format!("missing '{}'", field))
}

fn get_opt_str<'a>(args: &'a Value, field: &str) -> Option<&'a str> {
    args.get(field).and_then(|v| v.as_str())
}

fn text_result(v: Value) -> ToolResult {
    ToolResult::Text(v.to_string())
}

pub const MIGRATION_SQL: &str = "
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
    state TEXT NOT NULL DEFAULT 'pending', spec_json TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]', result_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
CREATE TABLE IF NOT EXISTS experiment_steps (
    id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    name TEXT NOT NULL, tool_name TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
    depends_on_json TEXT NOT NULL DEFAULT '[]', gates_json TEXT NOT NULL DEFAULT '[]',
    args_json TEXT NOT NULL DEFAULT '{}', result_summary TEXT, sort_order INTEGER NOT NULL DEFAULT 0
);
";

fn load_experiment(state: &DaemonState, id: &str) -> Result<(ExperimentSpec, String)> {
    let document = state
        .experiment_store
        .get_experiment(id)?
        .ok_or_else(|| anyhow::anyhow!("experiment not found: {id}"))?;
    Ok((
        serde_json::from_value(
            document
                .get("spec")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("experiment document missing spec"))?,
        )?,
        document
            .get("state")
            .and_then(Value::as_str)
            .unwrap_or("pending")
            .to_string(),
    ))
}

fn save_experiment_raw(state: &DaemonState, id: &str, spec_json: &str, es: &str) -> Result<()> {
    let mut document = state
        .experiment_store
        .get_experiment(id)?
        .ok_or_else(|| anyhow::anyhow!("experiment not found: {id}"))?;
    document["spec"] = serde_json::from_str(spec_json)?;
    document["state"] = Value::String(es.to_string());
    state.experiment_store.put_experiment(id, &document)?;
    Ok(())
}

fn update_exp_state(state: &DaemonState, id: &str, es: &str) -> Result<()> {
    let mut document = state
        .experiment_store
        .get_experiment(id)?
        .ok_or_else(|| anyhow::anyhow!("experiment not found: {id}"))?;
    document["state"] = Value::String(es.to_string());
    state.experiment_store.put_experiment(id, &document)?;
    Ok(())
}

fn update_step_state(
    state: &DaemonState,
    eid: &str,
    sn: &str,
    st: &str,
    smr: Option<&str>,
) -> Result<()> {
    let mut document = state
        .experiment_store
        .get_experiment(eid)?
        .ok_or_else(|| anyhow::anyhow!("experiment not found: {eid}"))?;
    if let Some(steps) = document.get_mut("steps").and_then(Value::as_array_mut) {
        if let Some(step) = steps
            .iter_mut()
            .find(|step| step.get("name").and_then(Value::as_str) == Some(sn))
        {
            step["state"] = Value::String(st.to_string());
            step["result_summary"] = smr.map(Value::from).unwrap_or(Value::Null);
        }
    }
    state.experiment_store.put_experiment(eid, &document)?;
    Ok(())
}

fn eval_gates(gates: &[GateCondition], result: &Value) -> bool {
    gates.iter().all(|g| match g {
        GateCondition::MetricAbove { metric, threshold } => result
            .get(metric)
            .and_then(|v| v.as_f64())
            .map(|v| v > *threshold)
            .unwrap_or(false),
        GateCondition::MetricBelow { metric, threshold } => result
            .get(metric)
            .and_then(|v| v.as_f64())
            .map(|v| v < *threshold)
            .unwrap_or(false),
        GateCondition::Custom { .. } => true,
    })
}

fn parse_gates(val: &Value) -> Result<Vec<GateCondition>> {
    val.as_array()
        .context("gates not array")?
        .iter()
        .map(|g| {
            if let Some(a) = g.get("metric_above") {
                Ok(GateCondition::MetricAbove {
                    metric: a
                        .get("metric")
                        .and_then(|v| v.as_str())
                        .context("no metric")?
                        .to_string(),
                    threshold: a
                        .get("threshold")
                        .and_then(|v| v.as_f64())
                        .context("no threshold")?,
                })
            } else if let Some(b) = g.get("metric_below") {
                Ok(GateCondition::MetricBelow {
                    metric: b
                        .get("metric")
                        .and_then(|v| v.as_str())
                        .context("no metric")?
                        .to_string(),
                    threshold: b
                        .get("threshold")
                        .and_then(|v| v.as_f64())
                        .context("no threshold")?,
                })
            } else if let Some(c) = g.get("custom") {
                Ok(GateCondition::Custom {
                    tool_name: c
                        .get("tool_name")
                        .and_then(|v| v.as_str())
                        .context("no tool_name")?
                        .to_string(),
                    args: c.get("args").cloned().unwrap_or_default(),
                })
            } else {
                anyhow::bail!("bad gate: need metric_above, metric_below, or custom")
            }
        })
        .collect()
}

// 1: create_experiment
pub struct CreateExperiment;
impl McpHandler for CreateExperiment {
    fn name(&self) -> &'static str {
        "create_experiment"
    }
    fn description(&self) -> &'static str {
        "Define an experiment with steps and gates"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{
        "name":{"type":"string"},"description":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},
        "steps":{"type":"array","items":{"type":"object","properties":{
            "name":{"type":"string"},"tool_name":{"type":"string"},
            "args":{"type":"object"},"depends_on":{"type":"array","items":{"type":"string"}},
            "gates":{"type":"array","items":{"type":"object"}}
        },"required":["name","tool_name"]}}
    },"required":["name","steps"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let name = get_str(req.args, "name")?;
        let desc = req
            .args
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let tags: Vec<String> = req
            .args
            .get("tags")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let raw = req
            .args
            .get("steps")
            .and_then(|v| v.as_array())
            .context("missing steps")?
            .clone();
        let steps: Vec<ExperimentStep> = raw
            .into_iter()
            .enumerate()
            .map(|(i, s)| {
                let sn = s
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&format!("s{}", i))
                    .to_string();
                let tn = s
                    .get("tool_name")
                    .and_then(|v| v.as_str())
                    .context(format!("step{} no tool", i))?
                    .to_string();
                Ok(ExperimentStep {
                    name: sn,
                    tool_name: tn,
                    args: s.get("args").cloned().unwrap_or_default(),
                    depends_on: s
                        .get("depends_on")
                        .and_then(|v| serde_json::from_value(v.clone()).ok())
                        .unwrap_or_default(),
                    gates: s
                        .get("gates")
                        .map(|g| parse_gates(g))
                        .transpose()?
                        .unwrap_or_default(),
                    state: StepState::Pending,
                    result_summary: None,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut seen = std::collections::HashSet::new();
        for s in &steps {
            if !seen.insert(&s.name) {
                anyhow::bail!("dup step: {}", s.name);
            }
        }
        let spec = ExperimentSpec {
            name: name.to_string(),
            description: desc.to_string(),
            steps,
            state: ExperimentState::Pending,
            result: None,
            tags,
        };
        let id = ExperimentId::new();
        let sc = spec.steps.len();
        let steps = spec.steps.iter().map(|step| serde_json::json!({"name":step.name,"tool_name":step.tool_name,"state":"pending","depends_on":step.depends_on,"gates":step.gates,"args":step.args,"result_summary":null})).collect::<Vec<_>>();
        state.experiment_store.put_experiment(
            &id.to_string(),
            &serde_json::json!({"spec":spec,"state":"pending","steps":steps}),
        )?;
        Ok(text_result(
            serde_json::json!({"experiment_id":id.to_string(),"name":spec.name,"step_count":sc}),
        ))
    }
}

// 2: run_experiment
pub struct RunExperiment;
impl McpHandler for RunExperiment {
    fn name(&self) -> &'static str {
        "run_experiment"
    }
    fn description(&self) -> &'static str {
        "Execute experiment DAG via daemon tools"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{
        "experiment_id":{"type":"string"},"max_steps":{"type":"integer"}
    },"required":["experiment_id"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let id = get_str(req.args, "experiment_id")?;
        let max: usize = req
            .args
            .get("max_steps")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(usize::MAX);
        let (mut spec, dbst) = load_experiment(state, id)?;
        if dbst != "pending" && dbst != "running" {
            anyhow::bail!("state is '{}'", dbst);
        }
        if dbst == "pending" {
            update_exp_state(state, id, "running")?;
            spec.state = ExperimentState::Running;
        }
        let ready: Vec<String> = spec
            .ready_steps()
            .into_iter()
            .map(|s| s.to_string())
            .take(max)
            .collect();
        if ready.is_empty() {
            if spec.is_terminal() {
                let ap = spec.steps.iter().all(|s| s.state == StepState::Passed);
                let fs = if ap { "completed" } else { "failed" };
                spec.state = if ap {
                    ExperimentState::Completed
                } else {
                    ExperimentState::Failed
                };
                save_experiment_raw(state, id, &serde_json::to_string(&spec)?, fs)?;
                return Ok(text_result(
                    serde_json::json!({"experiment_id":id,"dispatched":[],"state":fs}),
                ));
            }
            return Ok(text_result(
                serde_json::json!({"experiment_id":id,"dispatched":[],"state":"running"}),
            ));
        }
        let mut dispatched = Vec::new();
        let mut results = Vec::new();
        for sn in &ready {
            let step = spec.steps.iter_mut().find(|s| s.name == *sn).unwrap();
            step.state = StepState::Running;
            update_step_state(state, id, sn, "running", None)?;
            let outcome = match state.tools.get(step.tool_name.as_str()) {
                Some(h) => h.call(ToolRequest { args: &step.args }, _ctx, state),
                None => Err(anyhow::anyhow!("tool '{}' not found", step.tool_name)),
            };
            let (sst, summary) = match outcome {
                Ok(tr) => {
                    let txt = match &tr {
                        ToolResult::Text(t) => t.clone(),
                    };
                    let rv: Value =
                        serde_json::from_str(&txt).unwrap_or(Value::String(txt.clone()));
                    (
                        if eval_gates(&step.gates, &rv) {
                            StepState::Passed
                        } else {
                            StepState::Failed
                        },
                        Some(txt),
                    )
                }
                Err(e) => (StepState::Failed, Some(format!("Error: {}", e))),
            };
            step.state = sst.clone();
            step.result_summary = summary.clone();
            let st = match sst {
                StepState::Passed => "passed",
                StepState::Failed => "failed",
                _ => "unknown",
            };
            update_step_state(state, id, sn, st, summary.as_deref())?;
            dispatched.push(serde_json::json!({"step":sn,"state":st,"tool":step.tool_name}));
            results.push(sst);
        }
        let es = if spec.is_terminal() {
            if results.iter().all(|r| *r == StepState::Passed) {
                "completed"
            } else {
                "failed"
            }
        } else {
            "running"
        };
        save_experiment_raw(state, id, &serde_json::to_string(&spec)?, es)?;
        Ok(text_result(
            serde_json::json!({"experiment_id":id,"state":es,"dispatched":dispatched}),
        ))
    }
}

// 3: get_experiment
pub struct GetExperiment;
impl McpHandler for GetExperiment {
    fn name(&self) -> &'static str {
        "get_experiment"
    }
    fn description(&self) -> &'static str {
        "Query experiment state by ID"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{"experiment_id":{"type":"string"}},"required":["experiment_id"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let id = get_str(req.args, "experiment_id")?;
        let (spec, db_state) = load_experiment(state, id)?;
        let document = state
            .experiment_store
            .get_experiment(id)?
            .ok_or_else(|| anyhow::anyhow!("experiment not found: {id}"))?;
        let steps: Vec<Value> = document
            .get("steps")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        Ok(text_result(
            serde_json::json!({"experiment_id":id,"name":spec.name,"description":spec.description,"state":db_state,"result":spec.result,"steps":steps}),
        ))
    }
}

// 4: list_experiments
pub struct ListExperiments;
impl McpHandler for ListExperiments {
    fn name(&self) -> &'static str {
        "list_experiments"
    }
    fn description(&self) -> &'static str {
        "List experiments with optional state/tag filters"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{"state":{"type":"string"},"tag":{"type":"string"},"limit":{"type":"integer"},"offset":{"type":"integer"}}})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let sf = get_opt_str(req.args, "state");
        let tf = get_opt_str(req.args, "tag");
        let limit: i64 = req.args.get("limit").and_then(|v| v.as_u64()).unwrap_or(50) as i64;
        let off: i64 = req.args.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as i64;
        let mut rows: Vec<Value> = state
            .experiment_store
            .list_experiments()?
            .into_iter()
            .filter_map(|(id, document)| {
                let state = document
                    .get("state")
                    .and_then(Value::as_str)
                    .unwrap_or("pending");
                let spec = document.get("spec")?;
                if sf.is_some_and(|filter| filter != state) {
                    return None;
                }
                if tf.is_some_and(|filter| {
                    !spec
                        .get("tags")
                        .map_or(false, |tags| tags.to_string().contains(filter))
                }) {
                    return None;
                }
                Some(serde_json::json!({"experiment_id":id,"name":spec.get("name"),"state":state}))
            })
            .collect();
        let start = (off as usize).min(rows.len());
        rows = rows.into_iter().skip(start).take(limit as usize).collect();
        Ok(text_result(
            serde_json::json!({"experiments":rows,"count":rows.len()}),
        ))
    }
}

// 5: cancel_experiment
pub struct CancelExperiment;
impl McpHandler for CancelExperiment {
    fn name(&self) -> &'static str {
        "cancel_experiment"
    }
    fn description(&self) -> &'static str {
        "Cancel a running experiment"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{"experiment_id":{"type":"string"}},"required":["experiment_id"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let id = get_str(req.args, "experiment_id")?;
        let (mut spec, dbst) = load_experiment(state, id)?;
        if dbst != "running" && dbst != "pending" {
            anyhow::bail!("state is '{}'", dbst);
        }
        for s in &mut spec.steps {
            if s.state == StepState::Pending || s.state == StepState::Running {
                s.state = StepState::Skipped;
                update_step_state(state, id, &s.name, "skipped", None)?;
            }
        }
        spec.state = ExperimentState::Cancelled;
        save_experiment_raw(state, id, &serde_json::to_string(&spec)?, "cancelled")?;
        Ok(text_result(
            serde_json::json!({"experiment_id":id,"state":"cancelled"}),
        ))
    }
}

// 6: compare_experiments
pub struct CompareExperiments;
impl McpHandler for CompareExperiments {
    fn name(&self) -> &'static str {
        "compare_experiments"
    }
    fn description(&self) -> &'static str {
        "Compare two experiment results"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{"experiment_id_a":{"type":"string"},"experiment_id_b":{"type":"string"}},"required":["experiment_id_a","experiment_id_b"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let ia = get_str(req.args, "experiment_id_a")?;
        let ib = get_str(req.args, "experiment_id_b")?;
        let (sa, _) = load_experiment(state, ia)?;
        let (sb, _) = load_experiment(state, ib)?;
        let mut ma: std::collections::HashMap<&str, &ExperimentStep> =
            std::collections::HashMap::new();
        let mut mb = std::collections::HashMap::new();
        for s in &sa.steps {
            ma.insert(s.name.as_str(), s);
        }
        for s in &sb.steps {
            mb.insert(s.name.as_str(), s);
        }
        let keys: std::collections::BTreeSet<&str> = ma.keys().chain(mb.keys()).copied().collect();
        let comp: Vec<Value> = keys.iter().map(|k| serde_json::json!({
            "step":k,"state_a":ma.get(k).map(|s|format!("{:?}",s.state)),"state_b":mb.get(k).map(|s|format!("{:?}",s.state)),
        })).collect();
        Ok(text_result(
            serde_json::json!({"id_a":ia,"id_b":ib,"comparison":comp}),
        ))
    }
}

// 7: promote_experiment_result
pub struct PromoteExperimentResult;
impl McpHandler for PromoteExperimentResult {
    fn name(&self) -> &'static str {
        "promote_experiment_result"
    }
    fn description(&self) -> &'static str {
        "Promote experiment result to production"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{"experiment_id":{"type":"string"},"promoted_by":{"type":"string"}},"required":["experiment_id","promoted_by"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let id = get_str(req.args, "experiment_id")?;
        let pb = get_str(req.args, "promoted_by")?;
        let (spec, dbst) = load_experiment(state, id)?;
        if dbst != "completed" {
            anyhow::bail!("only completed can be promoted");
        }
        let rv = serde_json::json!({"experiment_name":spec.name,"promoted_by":pb,"promoted_at":chrono::Utc::now().to_rfc3339()});
        let mut document = state
            .experiment_store
            .get_experiment(id)?
            .ok_or_else(|| anyhow::anyhow!("experiment not found: {id}"))?;
        document["result"] = rv.clone();
        state.experiment_store.put_experiment(id, &document)?;
        Ok(text_result(
            serde_json::json!({"experiment_id":id,"promoted_by":pb,"result":rv}),
        ))
    }
}

// 8: resume_experiment
pub struct ResumeExperiment;
impl McpHandler for ResumeExperiment {
    fn name(&self) -> &'static str {
        "resume_experiment"
    }
    fn description(&self) -> &'static str {
        "Resume from last completed step"
    }
    fn input_schema(&self) -> Value {
        serde_json::json!({"type":"object","properties":{"experiment_id":{"type":"string"}},"required":["experiment_id"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        state: &DaemonState,
    ) -> Result<ToolResult> {
        let id = get_str(req.args, "experiment_id")?;
        let (mut spec, dbst) = load_experiment(state, id)?;
        if dbst != "failed" && dbst != "cancelled" && dbst != "completed" {
            anyhow::bail!("can't resume from '{}'", dbst);
        }
        let mut rc = 0u32;
        for s in &mut spec.steps {
            if s.state == StepState::Failed || s.state == StepState::Running {
                s.state = StepState::Pending;
                s.result_summary = None;
                update_step_state(state, id, &s.name, "pending", None)?;
                rc += 1;
            }
        }
        spec.state = ExperimentState::Running;
        save_experiment_raw(state, id, &serde_json::to_string(&spec)?, "running")?;
        Ok(text_result(
            serde_json::json!({"experiment_id":id,"state":"running","reset_count":rc}),
        ))
    }
}
