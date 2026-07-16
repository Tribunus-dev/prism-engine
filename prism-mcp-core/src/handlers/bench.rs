use crate::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use std::time::Instant;
use uuid::Uuid;
/// SQL migration for benchmark baselines table.
pub const MIGRATION: &str = "CREATE TABLE IF NOT EXISTS benchmark_baselines (baseline_name TEXT PRIMARY KEY, created_at TEXT NOT NULL, baseline_report_id TEXT NOT NULL);";

fn init(s: &DaemonState) -> anyhow::Result<()> {
    let _ = s;
    Ok(())
}
fn a<'a>(v: &'a Value, k: &str) -> anyhow::Result<&'a str> {
    v.get(k)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("{k} required"))
}
fn out(v: Value) -> anyhow::Result<ToolResult> {
    Ok(ToolResult::text(serde_json::to_string_pretty(&v)?))
}
pub struct CreateBenchmarkPlanHandler;
impl McpHandler for CreateBenchmarkPlanHandler {
    fn name(&self) -> &'static str {
        "create_benchmark_plan"
    }
    fn description(&self) -> &'static str {
        "Persist a benchmark plan."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"name":{"type":"string"},"command":{"type":"array"},"samples":{"type":"integer","default":1}},"required":["name"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let name = a(r.args, "name")?;
        let id = Uuid::new_v4().to_string();
        s.benchmark_store.put_plan(&id, name, r.args)?;
        out(json!({"plan_id":id,"name":name,"status":"created"}))
    }
}
pub struct RunBenchmarkHandler;
impl McpHandler for RunBenchmarkHandler {
    fn name(&self) -> &'static str {
        "run_benchmark"
    }
    fn description(&self) -> &'static str {
        "Execute the command in a persisted benchmark plan and record timing."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"plan_id":{"type":"string"}},"required":["plan_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let id = a(r.args, "plan_id")?.to_owned();
        let owner = uuid::Uuid::new_v4().to_string();
        let lease_key = format!("benchmark:{id}");
        if !s.resource_leases.acquire(&lease_key, &owner, 900)? {
            anyhow::bail!("benchmark plan is already running: {id}");
        }
        let result = (|| -> anyhow::Result<ToolResult> {
            let v = s
                .benchmark_store
                .get_plan(&id)?
                .ok_or_else(|| anyhow::anyhow!("benchmark plan not found: {id}"))?;
            let cmd = v
                .get("command")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).collect::<Vec<_>>());
            let start = Instant::now();
            let (code, output) = if let Some(parts) = cmd {
                if parts.is_empty() {
                    anyhow::bail!("command cannot be empty")
                }
                let o = std::process::Command::new(parts[0])
                    .args(&parts[1..])
                    .output()?;
                (
                    o.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&o.stdout).to_string()
                        + &String::from_utf8_lossy(&o.stderr),
                )
            } else {
                (0, "no command supplied; plan validation only".into())
            };
            let ms = start.elapsed().as_secs_f64() * 1000.;
            let rid = Uuid::new_v4().to_string();
            s.projection_store
                .record_benchmark(&rid, &id, ms, code, &output)?;
            out(
                json!({"report_id":rid,"plan_id":id,"elapsed_ms":ms,"exit_code":code,"output":output}),
            )
        })();
        let release_result = s.resource_leases.release(&lease_key, &owner);
        release_result?;
        result
    }
}
pub struct CompareBenchmarksHandler;
impl McpHandler for CompareBenchmarksHandler {
    fn name(&self) -> &'static str {
        "compare_benchmarks"
    }
    fn description(&self) -> &'static str {
        "Compare recorded benchmark reports."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"baseline_report_id":{"type":"string"},"candidate_report_id":{"type":"string"}},"required":["baseline_report_id","candidate_report_id"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let a_id = a(r.args, "baseline_report_id")?.to_owned();
        let b_id = a(r.args, "candidate_report_id")?.to_owned();
        let x = s
            .benchmark_store
            .get_report(&a_id)?
            .ok_or_else(|| anyhow::anyhow!("baseline report not found"))?;
        let y = s
            .benchmark_store
            .get_report(&b_id)?
            .ok_or_else(|| anyhow::anyhow!("candidate report not found"))?;
        out(
            json!({"baseline":[a_id,x.0,x.1],"candidate":[b_id,y.0,y.1],"delta_ms":y.0-x.0,"improved":y.0<x.0,"both_succeeded":x.1==0&&y.1==0}),
        )
    }
}
pub struct DetectPerformanceRegressionHandler;
impl McpHandler for DetectPerformanceRegressionHandler {
    fn name(&self) -> &'static str {
        "detect_performance_regression"
    }
    fn description(&self) -> &'static str {
        "Compare a report with a named baseline using a 5 percent threshold."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"report_id":{"type":"string"},"baseline_name":{"type":"string"}},"required":["report_id","baseline_name"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let rid = a(r.args, "report_id")?.to_owned();
        let name = a(r.args, "baseline_name")?.to_owned();
        let bid = s
            .benchmark_store
            .get_baseline(&name)?
            .ok_or_else(|| anyhow::anyhow!("baseline not found: {name}"))?;
        let base = s
            .benchmark_store
            .get_report(&bid)?
            .ok_or_else(|| anyhow::anyhow!("baseline report not found"))?
            .0;
        let cur = s
            .benchmark_store
            .get_report(&rid)?
            .ok_or_else(|| anyhow::anyhow!("report not found"))?
            .0;
        out(
            json!({"report_id":rid,"baseline":name,"baseline_ms":base,"report_ms":cur,"regression":cur>base*1.05,"delta_percent":(cur-base)/base*100.}),
        )
    }
}
pub struct PromoteBaselineHandler;
impl McpHandler for PromoteBaselineHandler {
    fn name(&self) -> &'static str {
        "promote_baseline"
    }
    fn description(&self) -> &'static str {
        "Persist a benchmark report as a named baseline."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"report_id":{"type":"string"},"baseline_name":{"type":"string"}},"required":["report_id","baseline_name"]})
    }
    fn call(
        &self,
        r: ToolRequest<'_>,
        _: &RequestContext,
        s: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        init(s)?;
        let rid = a(r.args, "report_id")?;
        let name = a(r.args, "baseline_name")?;
        if s.benchmark_store.get_report(rid)?.is_none() {
            anyhow::bail!("report not found: {rid}")
        }
        s.benchmark_store.put_baseline(name, rid)?;
        out(json!({"baseline_name":name,"report_id":rid,"status":"promoted"}))
    }
}
