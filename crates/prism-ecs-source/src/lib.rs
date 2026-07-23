use prism_ecs_core::identity::SourceFormat;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::{path::Path, sync::Arc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceIdentity { pub format: SourceFormat, pub source_digest: String, pub model_family: String, pub architecture: String }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorDescriptor { pub name: String, pub shape: Vec<usize>, pub dtype: String, pub original_dtype: String, pub element_size: usize, pub data_offset: Option<u64>, pub data_size_bytes: u64, pub layout: String, pub byte_offset: u64, pub byte_length: u64 }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TensorCatalog { pub tensors: Vec<TensorDescriptor>, pub digest: String, pub catalog_digest: String }
impl TensorCatalog { pub fn new(tensors: Vec<TensorDescriptor>) -> Self { let digest=hex::encode(sha2::Sha256::digest(serde_json::to_vec(&tensors).unwrap_or_default())); Self{tensors,digest:digest.clone(),catalog_digest:digest} } pub fn get(&self,name:&str)->Option<&TensorDescriptor>{self.tensors.iter().find(|t|t.name==name)} pub fn iter(&self)->std::slice::Iter<'_,TensorDescriptor>{self.tensors.iter()} pub fn len(&self)->usize{self.tensors.len()} }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SourceCapabilities { pub random_access: bool, pub mmap: bool, pub writable: bool, pub supports_streaming: bool, pub supports_random_access: bool, pub supports_dequantize: bool }
#[derive(Serialize, Deserialize)]
pub struct CanonicalSource { pub identity: SourceIdentity, pub catalog: TensorCatalog, pub capabilities: SourceCapabilities, #[serde(skip)] pub provider: Option<Arc<dyn TensorDataProvider>> }
impl std::fmt::Debug for CanonicalSource { fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{f.debug_struct("CanonicalSource").field("identity",&self.identity).field("catalog",&self.catalog).field("capabilities",&self.capabilities).finish()} }
impl Clone for CanonicalSource { fn clone(&self)->Self{Self{identity:self.identity.clone(),catalog:self.catalog.clone(),capabilities:self.capabilities.clone(),provider:self.provider.clone()}} }
#[derive(Debug, thiserror::Error)] pub enum SourceError { #[error("unsupported source format: {0}")] UnsupportedFormat(String), #[error("tensor not found: {0}")] TensorNotFound(String), #[error("source I/O: {0}")] Io(String), #[error("invalid source: {0}")] Invalid(String) }
pub trait TensorDataProvider: Send + Sync { fn read_tensor(&self, tensor:&TensorDescriptor)->Result<Vec<u8>,SourceError>; }
pub trait CanonicalSourceAdapter: Send + Sync { fn can_open(&self,path:&Path)->bool; fn open(&self,path:&Path)->Result<CanonicalSource,SourceError>; }
fn unsupported(name:&str)->Result<CanonicalSource,SourceError>{Err(SourceError::UnsupportedFormat(name.into()))}
pub mod gguf_adapter { use super::*; pub struct GgufAdapter; impl CanonicalSourceAdapter for GgufAdapter {fn can_open(&self,p:&Path)->bool{p.extension().and_then(|x|x.to_str()).map(|x|x.eq_ignore_ascii_case("gguf")).unwrap_or(false)} fn open(&self,_:&Path)->Result<CanonicalSource,SourceError>{unsupported("gguf")}} }
pub mod onnx_adapter { use super::*; pub struct OnnxAdapter; impl CanonicalSourceAdapter for OnnxAdapter {fn can_open(&self,p:&Path)->bool{p.extension().and_then(|x|x.to_str()).map(|x|x.eq_ignore_ascii_case("onnx")).unwrap_or(false)} fn open(&self,_:&Path)->Result<CanonicalSource,SourceError>{unsupported("onnx")}} }
pub mod mlx_adapter { use super::*; pub struct MlxAdapter; impl CanonicalSourceAdapter for MlxAdapter {fn can_open(&self,p:&Path)->bool{p.is_dir()} fn open(&self,_:&Path)->Result<CanonicalSource,SourceError>{unsupported("mlx")}} }
pub mod safetensors_adapter { use super::*; pub struct SafetensorsAdapter; impl CanonicalSourceAdapter for SafetensorsAdapter {fn can_open(&self,p:&Path)->bool{p.extension().and_then(|x|x.to_str()).map(|x|x.eq_ignore_ascii_case("safetensors")).unwrap_or(false)} fn open(&self,_:&Path)->Result<CanonicalSource,SourceError>{unsupported("safetensors")}} }
pub fn detect(path:&Path, adapters:&[Box<dyn CanonicalSourceAdapter>])->Result<CanonicalSource,SourceError>{for a in adapters{if a.can_open(path){return a.open(path)}};unsupported(path.display().to_string().as_str())}
