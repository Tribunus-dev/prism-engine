/// Holdout gate — evaluates engram quality on held-out data.
/// Plan Section 6 table: "Rollout: Held-out behavioral quality does not
/// regress beyond policy."
#[derive(Debug, Clone)]
pub struct HoldoutGate {
    pub max_quality_regression: f64,
    pub max_interference: f64,
}

impl HoldoutGate {
    pub fn new(max_regression: f64, max_interference: f64) -> Self {
        Self {
            max_quality_regression: max_regression,
            max_interference: max_interference,
        }
    }

    pub fn evaluate_quality(
        &self,
        predicted: &[f32],
        target: &[f32],
        baseline_loss: f64,
    ) -> Result<(), String> {
        let mut sq_error = 0.0f64;
        for (p, t) in predicted.iter().zip(target.iter()) {
            let err = (*p - *t) as f64;
            sq_error += err * err;
        }
        let holdout_loss = sq_error / predicted.len() as f64;
        let regression = holdout_loss - baseline_loss;
        if regression > self.max_quality_regression {
            return Err(format!(
                "holdout regression {} > {}",
                regression, self.max_quality_regression
            ));
        }
        Ok(())
    }

    /// Interference gate — preserves unrelated behavior.
    /// Plan: "Interference: Preserve unrelated behavior."
    pub fn evaluate_interference(
        &self,
        unaffected_before: &[f32],
        unaffected_after: &[f32],
    ) -> Result<(), String> {
        let mut sq_error = 0.0f64;
        for (b, a) in unaffected_before.iter().zip(unaffected_after.iter()) {
            let err = (*b - *a) as f64;
            sq_error += err * err;
        }
        let interference = (sq_error / unaffected_before.len() as f64).sqrt();
        if interference > self.max_interference {
            return Err(format!(
                "interference {} > {}",
                interference, self.max_interference
            ));
        }
        Ok(())
    }
}
