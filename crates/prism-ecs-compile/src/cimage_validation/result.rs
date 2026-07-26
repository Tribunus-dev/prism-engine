//! Validation result and matrix types.
//!
//! This module owns the canonical authority for the
//! [`ValidationResult`] and [`ValidationMatrix`] types. Both are
//! durable evidence that the validation runner writes to the
//! kernel-validation receipt. The receipt participates in the
//! canonical change flow: the runtime reader verifies the matrix,
//! the projection rebuilds it, and the replay path re-derives it
//! from the same Metal kernel sources.

use serde::{Deserialize, Serialize};

use super::KernelName;

/// Test name newtype — the stable identity of a test within a
/// validator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TestName(pub String);

impl TestName {
    /// Construct a new [`TestName`].
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// The string form of the test name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TestName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Result of a single validation test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Kernel name (newtype).
    pub kernel_name: KernelName,
    /// Test name (newtype).
    pub test_name: TestName,
    /// Whether the test passed.
    pub passed: bool,
    /// Maximum absolute error observed during the test.
    pub max_abs_error: f64,
    /// Human-readable details.
    pub details: String,
}

impl ValidationResult {
    /// Construct a new passing [`ValidationResult`].
    pub fn new(kernel_name: &str, test_name: &str) -> Self {
        Self {
            kernel_name: KernelName::new(kernel_name),
            test_name: TestName::new(test_name),
            passed: true,
            max_abs_error: 0.0,
            details: String::new(),
        }
    }

    /// Mark the result as failed with a measured error and details.
    pub fn fail(&mut self, error: f64, details: impl Into<String>) {
        self.passed = false;
        self.max_abs_error = self.max_abs_error.max(error);
        if !self.details.is_empty() {
            self.details.push_str("; ");
        }
        self.details.push_str(&details.into());
    }

    /// Record an error measurement without failing the test.
    pub fn record_error(&mut self, error: f64, label: &str) {
        self.max_abs_error = self.max_abs_error.max(error);
        if error > 0.0 && !self.details.is_empty() {
            self.details.push_str("; ");
        }
        if error > 0.0 {
            self.details.push_str(&format!("{}={:.2e}", label, error));
        }
    }
}

/// Full validation matrix for a kernel revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationMatrix {
    /// Kernel name.
    pub kernel_name: KernelName,
    /// Per-test results.
    pub results: Vec<ValidationResult>,
    /// Overall pass / fail verdict.
    pub overall_pass: bool,
}

impl ValidationMatrix {
    /// Construct a new empty [`ValidationMatrix`].
    pub fn new(kernel_name: &str) -> Self {
        Self {
            kernel_name: KernelName::new(kernel_name),
            results: Vec::new(),
            overall_pass: true,
        }
    }

    /// Push a single test result into the matrix.
    pub fn push(&mut self, result: ValidationResult) {
        if !result.passed {
            self.overall_pass = false;
        }
        self.results.push(result);
    }

    /// Number of tests in the matrix.
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Whether the matrix has no tests.
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}
