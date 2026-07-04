//! Level 2 Core ML bundle compiler for distillation teacher regions.
//!
//! Distillation uses a stateless dense teacher projection as the Core ML
//! reference path. The generated bundles mirror the deterministic synthetic
//! teacher weights used by Level 1 so the Level 2 bridge can load a real
//! `.mlmodelc` instead of falling back immediately.

use crate::coreai_pipeline;
use crate::mil_builder::MilBuilder;
use crate::mlpackage::ModelMeta;
use coreml_proto::proto::mil_spec;
use std::fs;
use std::path::{Path, PathBuf};

pub const TEACHER_INPUT_NAME: &str = "hidden_states";
pub const TEACHER_OUTPUT_NAME: &str = "matmul_1";

pub fn teacher_region_digest(microbatch: usize) -> String {
    format!("teacher-region-{microbatch:04x}")
}

pub fn required_teacher_region_digests(total_microbatches: usize) -> Vec<String> {
    (1..=total_microbatches)
        .map(teacher_region_digest)
        .collect()
}

fn teacher_weight_values(hidden_dim: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; hidden_dim * hidden_dim];
    for out_idx in 0..hidden_dim {
        for in_idx in 0..hidden_dim {
            let teacher_linear_idx = out_idx * hidden_dim + in_idx;
            let value = ((teacher_linear_idx as f64).sin() * 0.01) as f32;
            values[in_idx * hidden_dim + out_idx] = value;
        }
    }
    values
}

fn build_teacher_program(hidden_dim: usize) -> Result<mil_spec::Program, String> {
    MilBuilder::new("main")
        .input(
            TEACHER_INPUT_NAME,
            mil_spec::DataType::Float32,
            &[1, hidden_dim as i64],
        )
        .const_f32(
            "teacher_weight",
            &teacher_weight_values(hidden_dim),
            &[hidden_dim as i64, hidden_dim as i64],
        )
        .matmul(TEACHER_INPUT_NAME, "teacher_weight_0")
        .output(TEACHER_OUTPUT_NAME)
        .build()
        .map_err(|e| format!("build teacher MIL program: {e}"))
}

fn teacher_meta(hidden_dim: usize) -> ModelMeta {
    ModelMeta {
        model_name: "distill-teacher-region".into(),
        function_name: "main".into(),
        short_description: "Distill Level 2 teacher projection".into(),
        version: "1.0".into(),
        author: "Tribunus Compute".into(),
        output_name: TEACHER_OUTPUT_NAME.into(),
        inputs: vec![(TEACHER_INPUT_NAME.into(), vec![1, hidden_dim as i64])],
        outputs: vec![(TEACHER_OUTPUT_NAME.into(), vec![1, hidden_dim as i64])],
    }
}

fn compiled_model_ready(path: &Path) -> bool {
    path.is_dir() && path.join("metadata.json").exists() && path.join("model.mil").exists()
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), String> {
    if dst.exists() {
        fs::remove_dir_all(dst).map_err(|e| format!("remove {}: {e}", dst.display()))?;
    }
    fs::create_dir_all(dst).map_err(|e| format!("mkdir {}: {e}", dst.display()))?;

    for entry in fs::read_dir(src).map_err(|e| format!("read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if from.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to)
                .map_err(|e| format!("copy {} -> {}: {e}", from.display(), to.display()))?;
        }
    }

    Ok(())
}

fn compile_template_bundle(hidden_dim: usize) -> Result<tempfile::TempDir, String> {
    let compile_dir = tempfile::tempdir().map_err(|e| format!("tempdir: {e}"))?;
    let receipt = coreai_pipeline::build_and_compile(
        build_teacher_program(hidden_dim)?,
        &teacher_meta(hidden_dim),
        compile_dir.path(),
        "teacher-region-template",
        "cpuAndNeuralEngine",
    )?;

    let compiled_path = Path::new(&receipt.compiled_modelc_path);
    if !compiled_model_ready(compiled_path) {
        return Err(format!(
            "compiled template bundle is incomplete at {}",
            compiled_path.display()
        ));
    }

    Ok(compile_dir)
}

pub fn ensure_teacher_bundles(
    output_dir: &Path,
    hidden_dim: usize,
    total_microbatches: usize,
) -> Result<Vec<PathBuf>, String> {
    fs::create_dir_all(output_dir)
        .map_err(|e| format!("create model output dir {}: {e}", output_dir.display()))?;

    let digests = required_teacher_region_digests(total_microbatches);
    let mut targets = Vec::with_capacity(digests.len());
    let mut missing = Vec::new();

    for digest in &digests {
        let target = output_dir.join(format!("{digest}.mlmodelc"));
        if !compiled_model_ready(&target) {
            missing.push(target.clone());
        }
        targets.push(target);
    }

    if missing.is_empty() {
        return Ok(targets);
    }

    let compile_dir = compile_template_bundle(hidden_dim)?;
    let template = Path::new(compile_dir.path()).join("teacher-region-template.modelc");
    let compiled_template = find_compiled_bundle(&template)?;

    for missing_target in missing {
        copy_dir_all(&compiled_template, &missing_target)?;
    }

    Ok(targets)
}

fn find_compiled_bundle(root: &Path) -> Result<PathBuf, String> {
    if compiled_model_ready(root) {
        return Ok(root.to_path_buf());
    }

    for entry in fs::read_dir(root).map_err(|e| format!("read {}: {e}", root.display()))? {
        let entry = entry.map_err(|e| format!("dir entry: {e}"))?;
        let path = entry.path();
        if path.is_dir() {
            if let Ok(found) = find_compiled_bundle(&path) {
                return Ok(found);
            }
        }
    }

    Err(format!(
        "no compiled .mlmodelc bundle found in {}",
        root.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teacher_region_digest_sequence_matches_scheduler() {
        assert_eq!(
            required_teacher_region_digests(4),
            vec![
                "teacher-region-0001".to_string(),
                "teacher-region-0002".to_string(),
                "teacher-region-0003".to_string(),
                "teacher-region-0004".to_string(),
            ]
        );
    }

    #[test]
    fn teacher_weights_are_transposed_for_matmul() {
        let weights = teacher_weight_values(3);
        assert_eq!(weights.len(), 9);
        let expected = ((0f64).sin() * 0.01) as f32;
        assert_eq!(weights[0], expected);
        let teacher_row_major_idx = 1 * 3 + 2;
        let expected_transposed = ((teacher_row_major_idx as f64).sin() * 0.01) as f32;
        assert_eq!(weights[2 * 3 + 1], expected_transposed);
    }
}
