#[cfg(target_os = "macos")]
use crate::mil_builder::MilBuilder;

#[cfg(target_os = "macos")]
pub fn build_lut_gemv(
    builder: MilBuilder,
    _input: &str,
    _indices: &str,
    _palette: &str,
) -> MilBuilder {
    // Generate MIL program for LUT lookup via gather.
    // ANE LUT lookup via gather:
    // %palette_indices = @gather(%input, %indices)
    // %output = @reduce_sum(%palette_indices)

    // In actual implementation, this will use ANE's gather and reduce operations
    builder
        .gather(_input, _indices)
        .reduce_sum("gather_0")
        .output("reduce_sum_0")
}
