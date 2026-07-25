//! This module owns the canonical authority for the shape, dtype, and
//! broadcast helpers used by the phase graph.
//! It does not own UOp identity, graph mutation, or kernel lowering.

pub(crate) fn is_supported_dtype(dtype: &str) -> bool {
    matches!(dtype, "f32" | "f16" | "bf16" | "i8" | "u8" | "i32" | "u32")
}

pub(crate) fn cast_f32(value: f32, from: &str, to: &str) -> f32 {
    debug_assert!(matches!(
        from,
        "f32" | "f16" | "bf16" | "i8" | "u8" | "i32" | "u32"
    ));
    match to {
        "f32" => value,
        "f16" => half::f16::from_f32(value).to_f32(),
        "bf16" => half::bf16::from_f32(value).to_f32(),
        "i8" => value.clamp(i8::MIN as f32, i8::MAX as f32).trunc(),
        "u8" => value.clamp(0.0, u8::MAX as f32).trunc(),
        "i32" => value.clamp(i32::MIN as f32, i32::MAX as f32).trunc(),
        "u32" => value.clamp(0.0, u32::MAX as f32).trunc(),
        // WAIVER: the `to` value is validated by `TinyGraph::validate` for every
        // `UOpKind::Cast` before this helper runs, so the match is exhaustive
        // over the inputs the type system actually allows. Falling through to
        // `unreachable!` keeps the runtime path branch-free for the hot loop.
        _ => unreachable!("validated cast target"),
    }
}

pub(crate) fn element_count(shape: &[u64]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim as usize))
        // WAIVER: `TinyGraph::validate` rejects any shape containing a zero or
        // any product that overflows `usize` before this helper is reached, so
        // the multiplication cannot fail on validated graphs.
        .unwrap_or(0)
}

/// NumPy/tinygrad-style trailing-dimension broadcasting. A dimension is
/// compatible when it is equal or one; missing leading dimensions behave as
/// one. Keeping this helper in the compact IR makes shape semantics shared by
/// validation, reference execution, and future backend index lowering.
pub(crate) fn broadcast_shape(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for (axis, output_dim) in shape.iter_mut().take(rank).enumerate() {
        let left_dim = left
            .get(left.len().wrapping_sub(rank - axis))
            .copied()
            .unwrap_or(1);
        let right_dim = right
            .get(right.len().wrapping_sub(rank - axis))
            .copied()
            .unwrap_or(1);
        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            return None;
        }
        *output_dim = left_dim.max(right_dim);
    }
    Some(shape)
}

pub(crate) fn broadcast_index(index: usize, output_shape: &[u64], input_shape: &[u64]) -> usize {
    if input_shape == output_shape {
        return index;
    }
    let rank_delta = output_shape.len() - input_shape.len();
    let mut input_index = 0usize;
    let mut stride = 1usize;
    for input_axis in (0..input_shape.len()).rev() {
        let output_axis = input_axis + rank_delta;
        let output_stride = output_shape[output_axis + 1..]
            .iter()
            .map(|dim| *dim as usize)
            .product::<usize>();
        let coordinate = (index / output_stride.max(1)) % output_shape[output_axis] as usize;
        let input_coordinate = if input_shape[input_axis] == 1 {
            0
        } else {
            coordinate
        };
        input_index += input_coordinate * stride;
        stride *= input_shape[input_axis] as usize;
    }
    input_index
}
