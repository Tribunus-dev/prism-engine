use prism_mcp_core::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use std::time::Instant;
use uuid::Uuid;
fn init(s: &DaemonState) -> anyhow::Result<()> {
    s.db.with_writer(|c|{c.execute_batch("CREATE TABLE IF NOT EXISTS prism_bench_plans(id TEXT PRIMARY KEY,name TEXT NOT NULL,spec TEXT NOT NULL); CREATE TABLE IF NOT EXISTS prism_bench_reports(id TEXT PRIMARY KEY,plan_id TEXT NOT NULL,elapsed_ms REAL NOT NULL,exit_code INTEGER NOT NULL,output TEXT NOT NULL); CREATE TABLE IF NOT EXISTS prism_bench_baselines(name TEXT PRIMARY KEY,report_id TEXT NOT NULL)")?;Ok(())})
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
        s.db.with_writer(|c| {
            c.execute(
                "INSERT INTO prism_bench_plans VALUES(?1,?2,?3)",
                [&id, name, &serde_json::to_string(r.args)?],
            )?;
            Ok(())
        })?;
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
        let spec = s.db.with_reader(|c| {
            Ok(c.query_row(
                "SELECT spec FROM prism_bench_plans WHERE id=?1",
                [&id],
                |row| row.get::<_, String>(0),
            )?)
        })?;
        let v: Value = serde_json::from_str(&spec)?;
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
        s.db.with_writer(|c| {
            c.execute(
                "INSERT INTO prism_bench_reports VALUES(?1,?2,?3,?4,?5)",
                rusqlite::params![rid, id, ms, code, output],
            )?;
            Ok(())
        })?;
        out(json!({"report_id":rid,"plan_id":id,"elapsed_ms":ms,"exit_code":code,"output":output}))
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
        let rows = s.db.with_reader(|c| {
            let mut q = c.prepare(
                "SELECT id,elapsed_ms,exit_code FROM prism_bench_reports WHERE id IN(?1,?2)",
            )?;
            let mut x = Vec::new();
            for z in q.query_map([&a_id, &b_id], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, f64>(1)?,
                    r.get::<_, i64>(2)?,
                ))
            })? {
                x.push(z?)
            }
            Ok(x)
        })?;
        if rows.len() != 2 {
            anyhow::bail!("both reports must exist")
        }
        let x = rows.iter().find(|r| r.0 == a_id).unwrap();
        let y = rows.iter().find(|r| r.0 == b_id).unwrap();
        out(
            json!({"baseline":x,"candidate":y,"delta_ms":y.1-x.1,"improved":y.1<x.1,"both_succeeded":x.2==0&&y.2==0}),
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
        let bid = s.db.with_reader(|c| {
            Ok(c.query_row(
                "SELECT report_id FROM prism_bench_baselines WHERE name=?1",
                [&name],
                |row| row.get::<_, String>(0),
            )?)
        })?;
        let vals = s.db.with_reader(|c| {
            let mut q =
                c.prepare("SELECT id,elapsed_ms FROM prism_bench_reports WHERE id IN(?1,?2)")?;
            let mut x = Vec::new();
            for z in q.query_map([&bid, &rid], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?))
            })? {
                x.push(z?)
            }
            Ok(x)
        })?;
        if vals.len() != 2 {
            anyhow::bail!("baseline or report not found")
        }
        let base = vals.iter().find(|x| x.0 == bid).unwrap().1;
        let cur = vals.iter().find(|x| x.0 == rid).unwrap().1;
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
        let exists = s.db.with_reader(|c| {
            Ok(c.query_row(
                "SELECT COUNT(*) FROM prism_bench_reports WHERE id=?1",
                [rid],
                |row| row.get::<_, i64>(0),
            )?)
        })?;
        if exists == 0 {
            anyhow::bail!("report not found: {rid}")
        }
        s.db.with_writer(|c|{c.execute("INSERT INTO prism_bench_baselines VALUES(?1,?2) ON CONFLICT(name) DO UPDATE SET report_id=excluded.report_id",[name,rid])?;Ok(())})?;
        out(json!({"baseline_name":name,"report_id":rid,"status":"promoted"}))
    }
}
