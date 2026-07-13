use crate::ecs::adapter::CanonicalRole;
use crate::ecs::plan::CodecFamily;
use crate::ecs::Component;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Shape(pub Vec<u32>);
impl Component for Shape {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
    I4,
    I2,
}

impl DType {
    /// Number of bytes per element (fractional for packed sub-byte types).
    pub fn bytes_per_element(self) -> f64 {
        match self {
            DType::F32 => 4.0,
            DType::F16 | DType::BF16 => 2.0,
            DType::I8 => 1.0,
            DType::I4 => 0.5,
            DType::I2 => 0.25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DataType(pub DType);
impl Component for DataType {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CodecFamilyComp(pub CodecFamily, pub u32); // codec + group_size
impl Component for CodecFamilyComp {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CanonicalRoleComp(pub CanonicalRole);
impl Component for CanonicalRoleComp {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LayerIndex(pub u32);
impl Component for LayerIndex {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExpertIndex {
    pub index: u32,
    pub total: u32,
    pub top_k: u32,
}
impl Component for ExpertIndex {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    pub shared_expert: bool,
    pub num_experts: u32,
    pub top_k: u32,
    pub intermediate_size: Option<u32>,
}
impl Component for MoEConfig {}
