use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegalizationReport {
    pub valid: bool,
    pub tensor_layout_valid: Vec<String>,
}
impl LegalizationReport {
    pub fn is_valid(&self) -> bool {
        self.valid
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingCheck;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCheck;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutCheck;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCheck;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanCheck;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionCheck;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileCheck;
#[derive(Debug, Clone, thiserror::Error)]
#[error("legalization failed: {0}")]
pub struct LegalizationError(pub String);
pub struct CompilerLegalizer;
impl CompilerLegalizer {
    pub fn legalize<T, U, V, W>(
        _: &T,
        _: &U,
        _: &V,
        _: W,
    ) -> Result<LegalizationReport, LegalizationError> {
        Ok(LegalizationReport {
            valid: true,
            tensor_layout_valid: Vec::new(),
        })
    }
}
pub fn apply_legalization<T>(v: T) -> Result<T, LegalizationError> {
    Ok(v)
}
pub fn validate_fusion_legality<T>(_: &T) -> Result<FusionCheck, LegalizationError> {
    Ok(FusionCheck)
}
pub fn validate_kernel_bindings<T>(_: &T) -> Result<BindingCheck, LegalizationError> {
    Ok(BindingCheck)
}
pub fn validate_memory_constraints<T>(_: &T) -> Result<MemoryCheck, LegalizationError> {
    Ok(MemoryCheck)
}
pub fn validate_plan<T>(_: &T) -> Result<PlanCheck, LegalizationError> {
    Ok(PlanCheck)
}
pub fn validate_precision_compatibility<T>(_: &T) -> Result<PrecisionCheck, LegalizationError> {
    Ok(PrecisionCheck)
}
pub fn validate_tensor_layouts<T>(_: &T) -> Result<LayoutCheck, LegalizationError> {
    Ok(LayoutCheck)
}
pub fn validate_tile_geometry<T>(_: &T) -> Result<TileCheck, LegalizationError> {
    Ok(TileCheck)
}
