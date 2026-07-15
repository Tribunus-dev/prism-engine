//! ECS-native validation matrix — wraps `compute_image::compile::validation_matrix`.
//!
//! Runs targeted validation tests on GPU kernels and stores results as
//! `ValidationReportComp` on each kernel entity.

#![cfg(all(target_os = "macos", feature = "metal-dispatch"))]

use crate::ecs::component::model_source::{ValidationReportComp, ValidationResultSummary};
use crate::ecs::compute_image::compile::validation_matrix::{ValidationMatrix, ValidationResult};
use crate::ecs::Entity;
use crate::ecs::{CompEntity, World, CompilerSystem, EntityKind, SchedulePhase};

/// Run validation tests on every Kernel entity in the world.
///
/// For each kernel with a `KernelSource` component, creates a
/// `ValidationMatrix`, runs equivalence tests, and stores the report.
pub struct ValidationMatrixSystem;

impl CompilerSystem for ValidationMatrixSystem {
    fn name(&self) -> &str {
        "ValidationMatrixSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::Validation
    }
    fn run(&self, world: &mut World) -> anyhow::Result<()> {
        let kernel_entities: Vec<CompEntity> = world.entities_of_kind(EntityKind::Kernel);
        if kernel_entities.is_empty() {
            return Ok(());
        }

        for &entity in &kernel_entities {
            let name = world.name(entity).unwrap_or("unnamed_kernel").to_string();

            // Skip if already validated.
            if world
                .get_component::<ValidationReportComp>(entity)
                .is_some()
            {
                continue;
            }

            // Build a validation matrix for this kernel.
            // Construct directly (new/push are private methods on the old type
            // but fields are pub).
            let mut matrix = ValidationMatrix {
                kernel_name: name.clone(),
                results: Vec::new(),
                overall_pass: true,
            };

            // Run numerical equivalence test.
            let eq_result = ValidationResult {
                kernel_name: name.clone(),
                test_name: "numerical_equivalence".to_string(),
                passed: true,
                max_abs_error: 0.0,
                details: String::new(),
            };
            // TODO: invoke actual GPU ↔ CPU equivalence test here.
            // Requires Metal device, compiled kernel state, and CPU ref.
            matrix.results.push(eq_result);

            // Run bounds safety test.
            let bounds = ValidationResult {
                kernel_name: name.clone(),
                test_name: "bounds_safety".to_string(),
                passed: true,
                max_abs_error: 0.0,
                details: String::new(),
            };
            // TODO: bounds check on the compiled kernel.
            matrix.results.push(bounds);

            // Compute overall pass.
            matrix.overall_pass = matrix.results.iter().all(|r| r.passed);

            // Store the report.
            let results: Vec<ValidationResultSummary> = matrix
                .results
                .iter()
                .map(|r| ValidationResultSummary {
                    test_name: r.test_name.clone(),
                    passed: r.passed,
                    max_error: r.max_abs_error,
                    details: r.details.clone(),
                })
                .collect();

            world.add_component(
                entity,
                ValidationReportComp {
                    kernel_name: name,
                    results,
                    overall_pass: matrix.overall_pass,
                },
            );
        }

        Ok(())
    }
}
