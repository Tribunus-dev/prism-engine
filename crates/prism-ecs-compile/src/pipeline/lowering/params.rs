//! `pipeline::lowering::params` — Core ML lowering parameter types.
//!
//! This file owns the canonical authority for the Core ML lowering
//! surface: opcodes, scheduled ops, precision and shape policies,
//! targets, parameter schemas, and structured diagnostics. The
//! `coreml_proto`-specific [`MilValueRef`] and [`TensorMeta`] live in
//! the engine; the constitutional surface provides the typed contract.

use prism_ecs_backend::routing::{OperationId, TensorId};

// ── Opcode ─────────────────────────────────────────────────────────────────

/// Fieldless opcode for the Core ML operation registry.
/// No data-bearing attributes — those belong on the `ScheduledOp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    /// Constant value (weights, biases).
    Constant,
    /// Identity copy.
    Identity,
    /// Element-wise add.
    Add,
    /// Element-wise multiply.
    Multiply,
    /// Matrix multiply.
    Matmul,
    /// Reshape to a new shape.
    Reshape,
    /// Permute axes.
    Transpose,
    /// Softmax.
    Softmax,
    /// SiLU activation.
    Silu,
}

impl Opcode {
    /// Human-readable name for diagnostics.
    #[allow(non_camel_case_types)]
    pub fn name(&self) -> &'static str {
        match self {
            Opcode::Constant => "constant",
            Opcode::Identity => "identity",
            Opcode::Add => "add",
            Opcode::Multiply => "multiply",
            Opcode::Matmul => "matmul",
            Opcode::Reshape => "reshape",
            Opcode::Transpose => "transpose",
            Opcode::Softmax => "softmax",
            Opcode::Silu => "silu",
        }
    }
}

// ── ScheduledOp ────────────────────────────────────────────────────────────

/// A concrete operation in a scheduled region, with Core ML-specific
/// attributes attached.
#[derive(Debug, Clone)]
pub struct ScheduledOp {
    /// Stable operation identifier.
    pub op_id: OperationId,
    /// Core ML opcode.
    pub opcode: Opcode,
    /// Input tensor IDs.
    pub inputs: Vec<TensorId>,
    /// Output tensor IDs.
    pub outputs: Vec<TensorId>,
    /// Op-specific attributes.
    pub attrs: OpAttrs,
}

/// Per-op attributes for the 9-op envelope.
#[derive(Debug, Clone)]
pub enum OpAttrs {
    /// Constant tensor with row-major F32 data.
    Constant {
        /// Row-major F32 data.
        data: Vec<f32>,
        /// Shape of the constant tensor.
        shape: Vec<u32>,
    },
    /// Identity (no attributes).
    Identity,
    /// Add (no attributes).
    Add,
    /// Multiply (no attributes).
    Multiply,
    /// Matmul.
    Matmul {
        /// Transpose left operand.
        transpose_x: bool,
        /// Transpose right operand.
        transpose_y: bool,
    },
    /// Reshape.
    Reshape {
        /// Target shape; one dimension may be -1 (inferred).
        target_shape: Vec<i64>,
    },
    /// Transpose.
    Transpose {
        /// Full permutation (e.g. `[1, 0]` for 2D transpose).
        permutation: Vec<u32>,
    },
    /// Softmax.
    Softmax {
        /// Normalized axis.
        axis: i64,
    },
    /// SiLU (no attributes).
    Silu,
}

// ── Precision policy ──────────────────────────────────────────────────────

/// Three distinct precision concerns in one policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionPolicy {
    /// Compute F32, weights F32, interface F32.
    F32,
    /// Fp16 (refused in the legacy gate; accepted in production).
    Fp16,
}

impl PrecisionPolicy {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            PrecisionPolicy::F32 => "fp32",
            PrecisionPolicy::Fp16 => "fp16",
        }
    }

    /// Returns Ok if this precision is supported by the legacy gate.
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            PrecisionPolicy::F32 => Ok(()),
            PrecisionPolicy::Fp16 => Err("FP16 not supported in this gate"),
        }
    }
}

// ── Shape policy ──────────────────────────────────────────────────────────

/// How a tensor shape is constrained.
#[derive(Debug, Clone)]
pub enum ShapePolicy {
    /// Fixed shape; the only policy accepted in the legacy gate.
    Fixed(Vec<u32>),
    /// Bounded shape with default/min/max.
    Bounded {
        /// Default shape.
        default: Vec<u32>,
        /// Minimum shape.
        min: Vec<u32>,
        /// Maximum shape.
        max: Vec<u32>,
    },
    /// Enumerated set of alternative shapes.
    Enumerated {
        /// Default shape.
        default: Vec<u32>,
        /// Alternative shapes.
        alternatives: Vec<Vec<u32>>,
    },
    /// Symbolic shape with named dimensions.
    Symbolic {
        /// Named symbolic dimensions.
        named_dims: Vec<NamedDim>,
    },
}

impl ShapePolicy {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ShapePolicy::Fixed(_) => "fixed",
            ShapePolicy::Bounded { .. } => "bounded",
            ShapePolicy::Enumerated { .. } => "enumerated",
            ShapePolicy::Symbolic { .. } => "symbolic",
        }
    }

    /// Returns Ok only for `Fixed`. Others return a structured refusal.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            ShapePolicy::Fixed(_) => Ok(()),
            ShapePolicy::Bounded { .. } => Err("bounded shapes not supported in this gate".into()),
            ShapePolicy::Enumerated { .. } => {
                Err("enumerated shapes not supported in this gate".into())
            }
            ShapePolicy::Symbolic { .. } => {
                Err("symbolic shapes not supported in this gate".into())
            }
        }
    }
}

/// A named symbolic dimension (for future use).
#[derive(Debug, Clone)]
pub struct NamedDim {
    /// Dimension name.
    pub name: String,
    /// Dimension size.
    pub size: u32,
}

// ── Storage encoding ──────────────────────────────────────────────────────

/// How weight data is stored in a constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageEncoding {
    /// F32 little-endian.
    F32LittleEndian,
    /// Fp16 little-endian.
    Fp16LittleEndian,
    /// Unsigned 8-bit.
    U8,
    /// Signed 32-bit.
    I32,
}

impl StorageEncoding {
    /// Short name.
    pub fn name(&self) -> &'static str {
        match self {
            StorageEncoding::F32LittleEndian => "fp32le",
            StorageEncoding::Fp16LittleEndian => "fp16le",
            StorageEncoding::U8 => "u8",
            StorageEncoding::I32 => "i32",
        }
    }
}

// ── Target model ──────────────────────────────────────────────────────────

/// Validated target profile — indivisible compatibility row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreAiTarget {
    /// macOS 13 / iOS 16 / Core ML 6.
    MacOS13,
    /// macOS 14 / iOS 17 / Core ML 7.
    MacOS14,
    /// macOS 15 / iOS 18 / Core ML 8.
    MacOS15,
}

impl CoreAiTarget {
    /// Default gate target.
    pub fn default_gate_target() -> Self {
        CoreAiTarget::MacOS13
    }

    /// MIL spec version.
    pub fn spec_version(&self) -> u32 {
        match self {
            CoreAiTarget::MacOS13 => 7,
            CoreAiTarget::MacOS14 => 8,
            CoreAiTarget::MacOS15 => 9,
        }
    }

    /// Deployment-target string.
    pub fn deployment_target(&self) -> &'static str {
        match self {
            CoreAiTarget::MacOS13 => "macOS13",
            CoreAiTarget::MacOS14 => "macOS14",
            CoreAiTarget::MacOS15 => "macOS15",
        }
    }

    /// Opset identifier.
    pub fn opset_identifier(&self) -> &'static str {
        match self {
            CoreAiTarget::MacOS13 => "CoreML6",
            CoreAiTarget::MacOS14 => "CoreML7",
            CoreAiTarget::MacOS15 => "CoreML8",
        }
    }
}

// ── OpParamSchema ─────────────────────────────────────────────────────────

/// Typed parameter schema mapping scheduled attributes to MIL input
/// bindings. The `coreml_proto::mil_spec::Value` reference is
/// preserved as a string for the constitutional surface; the engine's
/// full schema is hardware-gated.
#[derive(Debug, Clone)]
pub struct OpParamSchema {
    /// Constant-value inputs emitted alongside tensor inputs.
    pub constant_inputs: Vec<(String, String)>,
    /// Tensor inputs resolved from value bindings.
    pub tensor_inputs: Vec<(String, TensorId)>,
}

// ── LoweringDiagnostic ─────────────────────────────────────────────────────

/// Structured diagnostic from the lowering pass.
#[derive(Debug, Clone)]
pub enum LoweringDiagnostic {
    /// An operation is not supported by this lowering target.
    UnsupportedOp {
        /// Operation identifier.
        op_id: OperationId,
        /// Operation opcode.
        opcode: Opcode,
        /// Reason for non-support.
        reason: String,
        /// Optional remediation suggestion.
        suggestion: Option<String>,
    },
    /// A shape policy is not supported by this target.
    ShapePolicyUnsupported {
        /// Operation identifier.
        op_id: OperationId,
        /// Policy name.
        policy: String,
    },
    /// An op's tensor shape does not match the expected shape.
    ShapeMismatch {
        /// Operation identifier.
        op_id: OperationId,
        /// Tensor identifier.
        tensor: TensorId,
        /// Expected shape.
        expected: String,
        /// Found shape.
        found: String,
    },
    /// A requested precision is not supported.
    PrecisionUnsupported {
        /// Operation identifier.
        op_id: OperationId,
        /// Requested precision name.
        requested: String,
        /// Supported precision names.
        supported: Vec<String>,
    },
    /// A hard constraint was violated.
    ConstraintViolation {
        /// Operation identifier.
        op_id: OperationId,
        /// Constraint name.
        constraint: String,
        /// Detail message.
        detail: String,
    },
    /// A non-fatal warning.
    Warning {
        /// Operation identifier.
        op_id: OperationId,
        /// Warning message.
        message: String,
    },
}

impl LoweringDiagnostic {
    /// Whether this diagnostic is fatal (not a warning).
    pub fn is_fatal(&self) -> bool {
        !matches!(self, LoweringDiagnostic::Warning { .. })
    }

    /// Human-readable message.
    pub fn message(&self) -> String {
        match self {
            LoweringDiagnostic::UnsupportedOp {
                op_id,
                opcode,
                reason,
                ..
            } => {
                format!("op {op_id:?} ({}): {reason}", opcode.name())
            }
            LoweringDiagnostic::ShapePolicyUnsupported { op_id, policy } => {
                format!("op {op_id:?}: shape policy '{policy}' not supported")
            }
            LoweringDiagnostic::ShapeMismatch {
                op_id,
                tensor,
                expected,
                found,
            } => {
                format!(
                    "op {op_id:?}: tensor {tensor:?} expected {expected}, found {found}"
                )
            }
            LoweringDiagnostic::PrecisionUnsupported {
                op_id,
                requested,
                supported,
            } => {
                format!(
                    "op {op_id:?}: precision '{requested}' not supported (supported: {supported:?})"
                )
            }
            LoweringDiagnostic::ConstraintViolation {
                op_id,
                constraint,
                detail,
            } => {
                format!("op {op_id:?}: constraint '{constraint}' violated: {detail}")
            }
            LoweringDiagnostic::Warning { op_id, message } => {
                format!("op {op_id:?}: {message}")
            }
        }
    }
}

// ── CoreAiLoweringError ─────────────────────────────────────────────────────

/// Structured error from the Core ML lowering pass.
#[derive(Debug, Clone)]
pub struct CoreAiLoweringError {
    /// Region identity this error belongs to.
    pub region_identity: String,
    /// Fatal diagnostics.
    pub fatal: Vec<LoweringDiagnostic>,
    /// Non-fatal warnings.
    pub warnings: Vec<LoweringDiagnostic>,
    /// Optional source location.
    pub source: Option<String>,
}

impl CoreAiLoweringError {
    /// Create a new error for the given region.
    pub fn new(region_identity: &str) -> Self {
        Self {
            region_identity: region_identity.to_string(),
            fatal: Vec::new(),
            warnings: Vec::new(),
            source: None,
        }
    }

    /// Append a fatal diagnostic.
    pub fn with_fatal(mut self, d: LoweringDiagnostic) -> Self {
        self.fatal.push(d);
        self
    }

    /// Append a warning.
    pub fn with_warning(mut self, d: LoweringDiagnostic) -> Self {
        self.warnings.push(d);
        self
    }

    /// Set the source location.
    pub fn with_source(mut self, s: String) -> Self {
        self.source = Some(s);
        self
    }
}

impl std::fmt::Display for CoreAiLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CoreAiLoweringError [{}]", self.region_identity)?;
        for d in &self.fatal {
            write!(f, "\n  fatal: {}", d.message())?;
        }
        for d in &self.warnings {
            write!(f, "\n  warning: {}", d.message())?;
        }
        if let Some(ref s) = self.source {
            write!(f, "\n  source: {s}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opcode_names_are_stable() {
        assert_eq!(Opcode::Matmul.name(), "matmul");
        assert_eq!(Opcode::Softmax.name(), "softmax");
    }

    #[test]
    fn precision_policy_validates_f32() {
        assert!(PrecisionPolicy::F32.validate().is_ok());
        assert!(PrecisionPolicy::Fp16.validate().is_err());
    }

    #[test]
    fn shape_policy_validates_only_fixed() {
        assert!(ShapePolicy::Fixed(vec![1, 4]).validate().is_ok());
        assert!(ShapePolicy::Bounded {
            default: vec![1, 4],
            min: vec![1, 1],
            max: vec![1, 8],
        }
        .validate()
        .is_err());
    }

    #[test]
    fn coreai_target_spec_versions() {
        assert_eq!(CoreAiTarget::MacOS13.spec_version(), 7);
        assert_eq!(CoreAiTarget::MacOS14.spec_version(), 8);
        assert_eq!(CoreAiTarget::MacOS15.spec_version(), 9);
    }

    #[test]
    fn lowering_diagnostic_is_fatal() {
        let warn = LoweringDiagnostic::Warning {
            op_id: OperationId(1),
            message: "hi".into(),
        };
        assert!(!warn.is_fatal());

        let fatal = LoweringDiagnostic::ShapePolicyUnsupported {
            op_id: OperationId(1),
            policy: "bounded".into(),
        };
        assert!(fatal.is_fatal());
    }
}
