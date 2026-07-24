use prism_ecs_compile::SemanticRegionSpec;
use prism_ecs_ir::semantic_region::{
    RegionRepresentationAssignment, SemanticRegionId, SemanticRegionPlan,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
struct DemoReceipt {
    schema: String,
    commit_sha: String,
    model_dir: String,
    tensor: String,
    tensor_shape: Vec<u64>,
    partition_digest: String,
    plan_digest: String,
    region_count: usize,
    assignments: Vec<RegionRepresentationAssignment>,
    claim_classes: BTreeMap<String, String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let model_dir = required_value(&args, "--model-dir")?;
    let tensor = required_value(&args, "--tensor")?;
    let spec_path = required_value(&args, "--spec")?;
    let json_out = required_value(&args, "--json-out")?;
    let assignments = repeated_values(&args, "--assign");

    let model_dir = PathBuf::from(model_dir);
    let mapped_shape = mapped_tensor_shape(&model_dir, &tensor)?;
    let spec = SemanticRegionSpec::load(&spec_path)?;
    let (partition, discovery) = spec.into_partition(&tensor, &mapped_shape, &spec_path)?;

    let mut by_role = BTreeMap::new();
    for region in &partition.regions {
        by_role.insert(role_name(&region.role), region.id.clone());
    }

    let mut plan_assignments = Vec::new();
    for value in assignments {
        let (role, representation) = value
            .split_once('=')
            .ok_or_else(|| format!("invalid --assign value: {value}"))?;
        let region = by_role
            .get(role)
            .ok_or_else(|| format!("assignment role not found in partition: {role}"))?;
        plan_assignments.push(RegionRepresentationAssignment {
            region: SemanticRegionId(region.0.clone()),
            representation: representation.to_string(),
            codec: None,
            preferred_lane: None,
            residency: None,
            assignment_evidence: vec!["explicit-demo-assignment".into()],
        });
    }

    let plan = SemanticRegionPlan {
        partition,
        assignments: plan_assignments,
        compile_verified: true,
        plan_digest: String::new(),
    }
    .seal()?;

    let receipt = DemoReceipt {
        schema: "prism.semantic-region.plan-receipt.v1".into(),
        commit_sha: option_env!("GIT_COMMIT_SHA").unwrap_or("unknown").into(),
        model_dir: model_dir.display().to_string(),
        tensor: tensor.clone(),
        tensor_shape: mapped_shape.clone(),
        partition_digest: discovery.partition_digest.clone(),
        plan_digest: plan.plan_digest.clone(),
        region_count: plan.partition.regions.len(),
        assignments: plan.assignments.clone(),
        claim_classes: BTreeMap::from([
            (
                "tensor_source".into(),
                "repository-backed mapped checkpoint".into(),
            ),
            ("boundaries".into(), "explicit architecture contract".into()),
            ("legality".into(), "compile-verified".into()),
            ("numerical_quality".into(), "unproven".into()),
            ("execution_performance".into(), "unmeasured".into()),
        ]),
    };

    fs::write(&json_out, serde_json::to_vec_pretty(&receipt)?)?;

    println!("Tensor: {tensor}");
    println!("Shape: {mapped_shape:?}");
    println!(
        "Semantic partition: {} regions",
        plan.partition.regions.len()
    );
    println!("Coverage: {}", discovery.coverage);
    println!("Overlap: {}", discovery.overlap);
    println!("Plan:");
    for assignment in &plan.assignments {
        let region = plan
            .partition
            .regions
            .iter()
            .find(|region| region.id == assignment.region)
            .expect("verified assignment region");
        println!(
            "  {:<18} {:?} -> {}",
            role_name(&region.role),
            region.selector,
            assignment.representation
        );
    }
    println!("Plan digest: {}", plan.plan_digest);
    println!("Evidence:");
    println!("  tensor source: repository-backed mapped checkpoint");
    println!("  boundaries: explicit architecture contract");
    println!("  legality: compile-verified");
    println!("  numerical quality: unproven");
    println!("  execution performance: unmeasured");
    println!("Receipt: {json_out}");
    Ok(())
}

fn required_value(args: &[String], flag: &str) -> Result<String, Box<dyn std::error::Error>> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .ok_or_else(|| format!("missing required argument {flag}").into())
}

fn repeated_values(args: &[String], flag: &str) -> Vec<String> {
    args.windows(2)
        .filter(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
        .collect()
}

fn mapped_tensor_shape(
    model_dir: &Path,
    tensor: &str,
) -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_slice(&fs::read(&index_path)?)?;
    let shard = index
        .get("weight_map")
        .and_then(|map| map.get(tensor))
        .and_then(|value| value.as_str())
        .ok_or_else(|| format!("tensor {tensor} not present in {}", index_path.display()))?;
    let shard_path = model_dir.join(shard);
    let bytes = fs::read(&shard_path)?;
    let safetensors = safetensors::SafeTensors::deserialize(&bytes)?;
    let view = safetensors.tensor(tensor)?;
    Ok(view.shape().iter().map(|&dim| dim as u64).collect())
}

fn role_name(role: &prism_ecs_ir::semantic_region::RegionRole) -> String {
    use prism_ecs_ir::semantic_region::RegionRole;
    match role {
        RegionRole::QueryProjection => "query_projection".into(),
        RegionRole::KeyProjection => "key_projection".into(),
        RegionRole::ValueProjection => "value_projection".into(),
        RegionRole::GateProjection => "gate_projection".into(),
        RegionRole::UpProjection => "up_projection".into(),
        RegionRole::DownProjection => "down_projection".into(),
        RegionRole::Router => "router".into(),
        RegionRole::SharedExpert => "shared_expert".into(),
        RegionRole::Generic { label } => format!("generic:{label}"),
        other => format!("{other:?}").to_lowercase(),
    }
}
