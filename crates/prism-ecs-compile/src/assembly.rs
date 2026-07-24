use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyRequest;
pub struct AssemblyModelSource;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyReceipt;
pub fn assemble(_r: AssemblyRequest) -> Result<AssemblyReceipt, String> {
    Ok(AssemblyReceipt)
}
