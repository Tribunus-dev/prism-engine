use crate::{DaemonState, McpHandler, RequestContext, ToolRequest, ToolResult};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

fn required_path<'a>(args: &'a Value, key: &str) -> anyhow::Result<&'a Path> {
    let raw = args.get(key).and_then(Value::as_str).unwrap_or("");
    if raw.is_empty() {
        anyhow::bail!("{key} required");
    }
    Ok(Path::new(raw))
}

fn manifest(path: &Path) -> anyhow::Result<Value> {
    let bytes = fs::read(path)?;
    if bytes.starts_with(b"{") {
        Ok(serde_json::from_slice(&bytes)?)
    } else {
        anyhow::bail!(
            "{} is not a JSON model manifest; tensor metadata cannot be inferred",
            path.display()
        )
    }
}

fn tensors(doc: &Value) -> Vec<(String, Value)> {
    match doc.get("tensors") {
        Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| {
                v.get("name")
                    .and_then(Value::as_str)
                    .map(|n| (n.to_owned(), v.clone()))
            })
            .collect(),
        _ => Vec::new(),
    }
}

pub struct InspectModelHandler;
impl McpHandler for InspectModelHandler {
    fn name(&self) -> &'static str {
        "inspect_model"
    }
    fn description(&self) -> &'static str {
        "Inspect a model file: read header, detect format."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"source":{"type":"string"}},"required":["source"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = required_path(req.args, "source")?;
        let meta = fs::metadata(path)?;
        let format = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown");
        let mut result =
            json!({"source":path,"size_bytes":meta.len(),"format":format,"readable":true});
        if format == "json" {
            result["tensor_count"] = json!(tensors(&manifest(path)?).len());
        }
        Ok(ToolResult::text(serde_json::to_string_pretty(&result)?))
    }
}

pub struct ListModelTensorsHandler;
impl McpHandler for ListModelTensorsHandler {
    fn name(&self) -> &'static str {
        "list_model_tensors"
    }
    fn description(&self) -> &'static str {
        "List tensor metadata from a JSON model manifest."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"model_id":{"type":"string"},"class_filter":{"type":"string"},"limit":{"type":"integer","default":50}},"required":["model_id"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = required_path(req.args, "model_id")?;
        let filter = req.args.get("class_filter").and_then(Value::as_str);
        let limit = req.args.get("limit").and_then(Value::as_u64).unwrap_or(50) as usize;
        let items: Vec<Value> = tensors(&manifest(path)?)
            .into_iter()
            .filter(|(_, v)| {
                filter
                    .map(|f| v.get("class").and_then(Value::as_str) == Some(f))
                    .unwrap_or(true)
            })
            .take(limit)
            .map(|(name, value)| json!({"name":name,"metadata":value}))
            .collect();
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"model":path,"count":items.len(),"tensors":items}),
        )?))
    }
}

pub struct GetModelTensorHandler;
impl McpHandler for GetModelTensorHandler {
    fn name(&self) -> &'static str {
        "get_model_tensor"
    }
    fn description(&self) -> &'static str {
        "Get tensor metadata by tensor ID from a JSON model manifest."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"tensor_id":{"type":"string"},"model_manifest":{"type":"string"}},"required":["tensor_id","model_manifest"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let id = req
            .args
            .get("tensor_id")
            .and_then(Value::as_str)
            .unwrap_or("");
        let path = required_path(req.args, "model_manifest")?;
        let tensor = tensors(&manifest(path)?)
            .into_iter()
            .find(|(name, _)| name == id)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {id}"))?;
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"name":tensor.0,"metadata":tensor.1}),
        )?))
    }
}

pub struct ClassifyModelTensorsHandler;
impl McpHandler for ClassifyModelTensorsHandler {
    fn name(&self) -> &'static str {
        "classify_model_tensors"
    }
    fn description(&self) -> &'static str {
        "Classify tensors by manifest metadata and name patterns."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"model_manifest":{"type":"string"}},"required":["model_manifest"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = required_path(req.args, "model_manifest")?;
        let classified: Vec<Value> = tensors(&manifest(path)?)
            .into_iter()
            .map(|(name, value)| {
                let class = value
                    .get("class")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| {
                        if name.contains("embed") {
                            "embedding".into()
                        } else if name.contains("norm") {
                            "normalization".into()
                        } else if name.contains("lm_head") {
                            "output".into()
                        } else {
                            "unknown".into()
                        }
                    });
                json!({"name":name,"class":class,"metadata":value})
            })
            .collect();
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"model":path,"tensors":classified}),
        )?))
    }
}

pub struct CompareModelsHandler;
impl McpHandler for CompareModelsHandler {
    fn name(&self) -> &'static str {
        "compare_models"
    }
    fn description(&self) -> &'static str {
        "Compare two JSON model manifests by tensor names and metadata."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"manifest_a":{"type":"string"},"manifest_b":{"type":"string"}},"required":["manifest_a","manifest_b"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let a = required_path(req.args, "manifest_a")?;
        let b = required_path(req.args, "manifest_b")?;
        let ta = tensors(&manifest(a)?);
        let tb = tensors(&manifest(b)?);
        let names_a: std::collections::BTreeSet<_> = ta.iter().map(|(n, _)| n).collect();
        let names_b: std::collections::BTreeSet<_> = tb.iter().map(|(n, _)| n).collect();
        let only_a: Vec<_> = names_a.difference(&names_b).map(|s| (*s).clone()).collect();
        let only_b: Vec<_> = names_b.difference(&names_a).map(|s| (*s).clone()).collect();
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"manifest_a":a,"manifest_b":b,"tensor_count_a":ta.len(),"tensor_count_b":tb.len(),"only_a":only_a,"only_b":only_b,"same_tensor_names":only_a.is_empty() && only_b.is_empty()}),
        )?))
    }
}

pub struct EstimateModelMemoryHandler;
impl McpHandler for EstimateModelMemoryHandler {
    fn name(&self) -> &'static str {
        "estimate_model_memory"
    }
    fn description(&self) -> &'static str {
        "Estimate model memory from tensor shapes and element sizes."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"model_manifest":{"type":"string"}},"required":["model_manifest"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = required_path(req.args, "model_manifest")?;
        let mut total = 0u64;
        let mut unknown = Vec::new();
        for (name, value) in tensors(&manifest(path)?) {
            let shape: Option<Vec<u64>> = value
                .get("shape")
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_u64).collect());
            let bytes = value.get("bytes").and_then(Value::as_u64).or_else(|| {
                shape.as_ref().map(|s| {
                    s.iter().product::<u64>()
                        * value
                            .get("element_size")
                            .and_then(Value::as_u64)
                            .unwrap_or(2)
                })
            });
            if let Some(b) = bytes {
                total += b;
            } else {
                unknown.push(name);
            }
        }
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"model":path,"estimated_bytes":total,"estimated_mib":total as f64 / 1_048_576.0,"unknown_tensors":unknown}),
        )?))
    }
}

pub struct ValidateModelAssetsHandler;
impl McpHandler for ValidateModelAssetsHandler {
    fn name(&self) -> &'static str {
        "validate_model_assets"
    }
    fn description(&self) -> &'static str {
        "Validate model manifest and referenced asset files."
    }
    fn input_schema(&self) -> Value {
        json!({"type":"object","properties":{"model_manifest":{"type":"string"}},"required":["model_manifest"]})
    }
    fn call(
        &self,
        req: ToolRequest<'_>,
        _ctx: &RequestContext,
        _state: &DaemonState,
    ) -> anyhow::Result<ToolResult> {
        let path = required_path(req.args, "model_manifest")?;
        let doc = manifest(path)?;
        let base = path.parent().unwrap_or(Path::new("."));
        let mut missing = Vec::new();
        if let Some(assets) = doc.get("assets").and_then(Value::as_array) {
            for asset in assets.iter().filter_map(Value::as_str) {
                if !base.join(asset).is_file() {
                    missing.push(asset);
                }
            }
        }
        let valid = missing.is_empty();
        Ok(ToolResult::text(serde_json::to_string_pretty(
            &json!({"manifest":path,"valid":valid,"missing_assets":missing,"tensor_count":tensors(&doc).len()}),
        )?))
    }
}
