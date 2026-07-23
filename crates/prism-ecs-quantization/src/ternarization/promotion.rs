use prism_ecs_ir::evolution::TernaryAdmissionLimits;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendPass {
    pub attempted: bool,
    pub passed: bool,
}

impl BackendPass {
    pub fn unavailable() -> Self {
        Self {
            attempted: false,
            passed: false,
        }
    }

    pub fn passed() -> Self {
        Self {
            attempted: true,
            passed: true,
        }
    }
}

/// Evidence required to admit a native ternary CImage.
///
/// Behavioral measurements are optional at the representation boundary so
/// older receipts can still be decoded, but admission requires every value
/// to be present, finite, and within the reference gates. `None` therefore
/// represents missing provenance rather than zero error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NativeTernaryPromotionEvidence {
    pub cpu_canary: BackendPass,
    pub accelerate_reconstruction: BackendPass,
    pub metal_packed: BackendPass,
    pub ane_static: BackendPass,
    pub cimage_replay: BackendPass,
    pub behavioral_reference: BackendPass,
    #[serde(default)]
    pub activation_error: Option<f64>,
    #[serde(default)]
    pub logit_divergence: Option<f64>,
    #[serde(default)]
    pub task_loss: Option<f64>,
    #[serde(default)]
    pub router_agreement: Option<f64>,
    #[serde(default)]
    pub router_margin_error: Option<f64>,
    #[serde(default)]
    pub logit_cross_entropy: Option<f64>,
    #[serde(default)]
    pub generation_loss: Option<f64>,
    #[serde(default)]
    pub expert_balance_error: Option<f64>,
    pub ane_selected: bool,
    pub packed_abi_digest: String,
    pub reference_digest: String,
}

impl NativeTernaryPromotionEvidence {
    pub fn eligible(&self) -> bool {
        self.reject_reason().is_none()
    }

    pub fn reject_reason(&self) -> Option<String> {
        self.reject_reason_with_limits(&TernaryAdmissionLimits::default())
    }

    pub fn reject_reason_with_limits(&self, limits: &TernaryAdmissionLimits) -> Option<String> {
        if !self.cpu_canary.attempted || !self.cpu_canary.passed {
            Some("CPU canary failed or was not measured".into())
        } else if !self.accelerate_reconstruction.attempted
            || !self.accelerate_reconstruction.passed
        {
            Some("Accelerate reconstruction failed or was not measured".into())
        } else if !self.metal_packed.attempted || !self.metal_packed.passed {
            Some("Metal packed validation failed or was not measured".into())
        } else if self.ane_selected && (!self.ane_static.attempted || !self.ane_static.passed) {
            Some("ANE static validation failed or was not measured".into())
        } else if !self.cimage_replay.attempted || !self.cimage_replay.passed {
            Some("CImage replay failed or was not measured".into())
        } else {
            self.behavioral_reject_reason(limits)
        }
    }

    /// Validate only the reference-behavior portion of the receipt. This is
    /// used before replay is attached so a failed threshold can be reported
    /// without converting it into a generic replay failure.
    pub fn behavioral_reject_reason(&self, limits: &TernaryAdmissionLimits) -> Option<String> {
        if !self.behavioral_reference.attempted {
            return Some("behavioral reference failed or was not measured".into());
        }

        let numeric_reason = upper_gate(
            "activation_error",
            self.activation_error,
            limits.max_activation_error,
        )
        .or_else(|| {
            upper_gate(
                "logit_divergence",
                self.logit_divergence,
                limits.max_logit_divergence,
            )
        })
        .or_else(|| upper_gate("task_loss", self.task_loss, limits.max_task_loss))
        .or_else(|| {
            lower_gate(
                "router_agreement",
                self.router_agreement,
                limits.min_router_agreement,
            )
        })
        .or_else(|| {
            upper_gate(
                "router_margin_error",
                self.router_margin_error,
                limits.max_router_margin_error,
            )
        })
        .or_else(|| {
            upper_gate(
                "logit_cross_entropy",
                self.logit_cross_entropy,
                limits.max_logit_cross_entropy,
            )
        })
        .or_else(|| {
            upper_gate(
                "generation_loss",
                self.generation_loss,
                limits.max_generation_loss,
            )
        })
        .or_else(|| {
            upper_gate(
                "expert_balance_error",
                self.expert_balance_error,
                limits.max_expert_balance_error,
            )
        });
        numeric_reason.or_else(|| {
            (!self.behavioral_reference.passed)
                .then(|| "behavioral reference failed its required gates".into())
        })
    }
}

fn upper_gate(name: &str, value: Option<f64>, limit: f64) -> Option<String> {
    match value {
        Some(value) if value.is_finite() && value <= limit => None,
        Some(value) if !value.is_finite() => {
            Some(format!("behavioral {name} is not finite: {value:?}"))
        }
        Some(value) => Some(format!("behavioral {name} {value} exceeds maximum {limit}")),
        None => Some(format!("behavioral {name} measurement is missing")),
    }
}

fn lower_gate(name: &str, value: Option<f64>, limit: f64) -> Option<String> {
    match value {
        Some(value) if value.is_finite() && value >= limit => None,
        Some(value) if !value.is_finite() => {
            Some(format!("behavioral {name} is not finite: {value:?}"))
        }
        Some(value) => Some(format!(
            "behavioral {name} {value} is below minimum {limit}"
        )),
        None => Some(format!("behavioral {name} measurement is missing")),
    }
}
