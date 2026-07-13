//! Engram application — CPU and Metal runtime for applying trained engrams.
//!
//! An engram's payload is applied to activations at a specific region of the
//! execution graph according to the `EngramApplication` mode (additive,
//! multiplicative, low-rank projection, latent prefix, or adapter activation).

use crate::ecs::training_target::spec::EngramApplication;

/// Apply an engram payload to activations on the CPU.
///
/// The `payload` byte slice is interpreted according to the `application` mode:
/// - `AdditiveResidual` — payload is `f32` residuals added element-wise.
/// - `MultiplicativeModulation` — payload is `f32` scales multiplied
///   element-wise.
/// - `LowRankProjection` — placeholder for LoRA-style A/B matrix application.
/// - `LatentPrefix` — placeholder.
/// - `AdapterActivation` — placeholder.
pub fn apply_cpu(
    application: &EngramApplication,
    activations: &mut [f32],
    payload: &[u8],
) -> Result<(), String> {
    if payload.len() % std::mem::size_of::<f32>() != 0 {
        return Err("engram payload is not f32 aligned".into());
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunk is four bytes")))
        .collect();
    match application {
        EngramApplication::AdditiveResidual => {
            if values.len() != activations.len() {
                return Err(format!(
                    "additive engram width {} does not match activations {}",
                    values.len(),
                    activations.len()
                ));
            }
            for (a, r) in activations.iter_mut().zip(&values) {
                *a += r;
            }
            Ok(())
        }
        EngramApplication::MultiplicativeModulation => {
            if values.len() != activations.len() {
                return Err(format!(
                    "multiplicative engram width {} does not match activations {}",
                    values.len(),
                    activations.len()
                ));
            }
            for (a, s) in activations.iter_mut().zip(&values) {
                *a *= s;
            }
            Ok(())
        }
        EngramApplication::LowRankProjection => {
            // Simple low-rank adaptation (LoRA-style)
            // payload = [A_matrix_bytes, B_matrix_bytes]
            // TODO: actual matrix-multiply when the decomposition format is settled.
            Ok(())
        }
        EngramApplication::LatentPrefix => {
            // TODO: prepend or splice latent prefix tokens.
            Ok(())
        }
        EngramApplication::AdapterActivation => {
            // TODO: run a small adapter MLP.
            Ok(())
        }
    }
}

/// Apply an engram payload to activations on Metal GPU.
///
/// Currently returns an error — the Metal engram kernel has not been compiled
/// yet. Use `apply_cpu` for functional application.
#[cfg(feature = "metal-dispatch")]
pub fn apply_metal(
    device: &metal::Device,
    application: &EngramApplication,
    activations: &metal::Buffer,
    _payload: &[u8],
) -> Result<(), String> {
    let _ = device;
    let _ = application;
    let _ = activations;
    Err("Metal engram application not implemented — use CPU path".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn additive_application_requires_matching_width() {
        let mut activations = vec![1.0, 2.0];
        apply_cpu(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &payload(&[0.5, -1.0]),
        )
        .unwrap();
        assert_eq!(activations, vec![1.5, 1.0]);
        assert!(apply_cpu(
            &EngramApplication::AdditiveResidual,
            &mut activations,
            &payload(&[1.0]),
        )
        .is_err());
    }
}

/// Non-Metal stub when `metal-dispatch` is disabled.
#[cfg(not(feature = "metal-dispatch"))]
pub fn apply_metal(
    _device: &std::ffi::c_void,
    _application: &EngramApplication,
    _activations: &std::ffi::c_void,
    _payload: &[u8],
) -> Result<(), String> {
    Err("Metal engram application not available — use CPU path".into())
}
