//! Persistent, bounded reuse index for promoted representation profiles.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RepresentationProfile {
    pub format: String,
    pub role: String,
    pub shape: Vec<usize>,
    pub reusable_tensors: Vec<String>,
    pub source_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct TensorFamilySignature {
    pub role: String,
    pub rank: usize,
    pub normalized_shape: Vec<usize>,
    pub component: String,
    pub dtype: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFamilyPolicy {
    pub signature: TensorFamilySignature,
    pub champion: String,
    pub members: Vec<String>,
    pub verification_members: Vec<String>,
    pub outliers: Vec<String>,
    pub format: String,
    #[serde(default)]
    pub outlier_formats: BTreeMap<String, String>,
}

/// Cluster catalog entries into reusable structural families. The first
/// member is the champion; verification members are deliberately sampled from
/// the rest so reuse never becomes an unconditional assignment.
pub fn cluster_tensor_families(
    tensors: &prism_ecs_source::TensorCatalog,
) -> BTreeMap<TensorFamilySignature, TensorFamilyPolicy> {
    let mut grouped: BTreeMap<TensorFamilySignature, Vec<String>> = BTreeMap::new();
    for tensor in &tensors.tensors {
        grouped
            .entry(tensor_family_signature(tensor))
            .or_default()
            .push(tensor.name.clone());
    }
    grouped
        .into_iter()
        .map(|(signature, mut members)| {
            members.sort();
            let champion = members.first().cloned().unwrap_or_default();
            let verification_members = members
                .iter()
                .skip(1)
                .step_by(8)
                .take(32)
                .cloned()
                .collect();
            (
                signature.clone(),
                TensorFamilyPolicy {
                    signature,
                    champion,
                    members,
                    verification_members,
                    outliers: Vec::new(),
                    format: "unassigned".into(),
                    outlier_formats: BTreeMap::new(),
                },
            )
        })
        .collect()
}

/// Apply a validated family format while retaining explicit per-tensor
/// assignments. Outliers are excluded and must receive their own evaluation.
pub fn apply_family_formats(
    plan: &mut prism_ecs_ir::evolution::compile_plan::FormatPlan,
    tensors: &prism_ecs_source::TensorCatalog,
    policies: &BTreeMap<TensorFamilySignature, TensorFamilyPolicy>,
) {
    for (signature, policy) in policies {
        if policy.format == "unassigned" {
            continue;
        }
        for name in &policy.members {
            if policy.outliers.iter().any(|outlier| outlier == name) {
                if let Some(format) = policy
                    .outlier_formats
                    .get(name)
                    .and_then(|format| parse_tensor_format(format))
                {
                    plan.per_tensor.insert(name.clone(), format);
                } else {
                    plan.per_tensor.remove(name);
                }
                continue;
            }
            if tensors
                .get(name)
                .is_some_and(|tensor| tensor_family_signature(tensor) == *signature)
            {
                if let Some(format) = parse_tensor_format(&policy.format) {
                    plan.per_tensor.insert(name.clone(), format);
                }
            }
        }
    }
}

/// Classify a verified canary against its family champion. A failed gate is
/// persisted as an outlier so later compilations cannot recycle the champion
/// representation into that tensor without re-evaluation.
pub fn record_outlier(
    policy: &mut TensorFamilyPolicy,
    tensor: impl Into<String>,
    divergence: f64,
    max_divergence: f64,
) {
    let tensor = tensor.into();
    if divergence.is_finite()
        && divergence > max_divergence
        && !policy.outliers.iter().any(|name| name == &tensor)
    {
        policy.outliers.push(tensor);
    }
}

fn parse_tensor_format(
    value: &str,
) -> Option<prism_ecs_ir::evolution::mutation_table::TensorFormat> {
    use prism_ecs_ir::evolution::mutation_table::TensorFormat;
    Some(match value {
        "Fp16" => TensorFormat::Fp16,
        "Bf16" => TensorFormat::Bf16,
        "Int8" => TensorFormat::Int8,
        "Int4" => TensorFormat::Int4,
        "Nf4" => TensorFormat::Nf4,
        "Nf8" => TensorFormat::Nf8,
        "Ternary158" => TensorFormat::Ternary158,
        "Binary1" => TensorFormat::Binary1,
        _ => return None,
    })
}

pub fn tensor_family_signature(
    tensor: &prism_ecs_source::TensorDescriptor,
) -> TensorFamilySignature {
    let lower = tensor.name.to_ascii_lowercase();
    let component = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
        "router",
        "norm",
        "embed",
        "lm_head",
    ]
    .iter()
    .find(|part| lower.contains(**part))
    .copied()
    .unwrap_or("other")
    .to_string();
    TensorFamilySignature {
        role: prism_ecs_ir::evolution::compile_plan::classify_tensor(&tensor.name),
        rank: tensor.shape.len(),
        normalized_shape: tensor
            .shape
            .iter()
            .map(|dim| dim.next_power_of_two())
            .collect(),
        component,
        dtype: tensor.dtype.clone(),
    }
}

pub fn record_promoted_profiles(
    model_dir: &Path,
    plan: &prism_ecs_ir::evolution::compile_plan::FormatPlan,
    tensors: &prism_ecs_source::TensorCatalog,
) -> Result<(), String> {
    let path = model_dir.join(".prism-representation-cache.json");
    let mut profiles: BTreeMap<String, RepresentationProfile> = fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default();
    for (name, format) in &plan.per_tensor {
        let Some(tensor) = tensors.get(name) else {
            continue;
        };
        let signature = tensor_family_signature(tensor);
        let key = format!("family:{:?}", signature);
        let profile = profiles
            .entry(key)
            .or_insert_with(|| RepresentationProfile {
                format: format!("{format:?}"),
                role: prism_ecs_ir::evolution::compile_plan::classify_tensor(name),
                shape: tensor.shape.clone(),
                reusable_tensors: Vec::new(),
                source_count: 0,
            });
        profile.source_count += 1;
        if profile.reusable_tensors.len() < 4096
            && !profile.reusable_tensors.iter().any(|n| n == name)
        {
            profile.reusable_tensors.push(name.clone());
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&profiles).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

pub fn persist_family_policies(
    model_dir: &Path,
    plan: &prism_ecs_ir::evolution::compile_plan::FormatPlan,
    tensors: &prism_ecs_source::TensorCatalog,
) -> Result<(), String> {
    let mut policies = cluster_tensor_families(tensors);
    for policy in policies.values_mut() {
        if let Some(format) = policy
            .members
            .iter()
            .find_map(|name| plan.per_tensor.get(name))
        {
            policy.format = format!("{format:?}");
        }
    }
    persist_family_policy_map(model_dir, &policies)
}

pub fn persist_family_policy_map(
    model_dir: &Path,
    policies: &BTreeMap<TensorFamilySignature, TensorFamilyPolicy>,
) -> Result<(), String> {
    let path = model_dir.join(".prism-tensor-families.json");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(
        path,
        serde_json::to_vec_pretty(&policies).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}
