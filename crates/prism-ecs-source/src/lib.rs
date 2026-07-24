use prism_ecs_core::identity::SourceFormat;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceIdentity {
    pub format: SourceFormat,
    pub source_digest: String,
    pub model_family: String,
    pub architecture: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorDescriptor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub original_dtype: String,
    pub element_size: usize,
    pub data_offset: Option<u64>,
    pub data_size_bytes: u64,
    pub layout: String,
    pub byte_offset: u64,
    pub byte_length: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct TensorCatalog {
    pub tensors: Vec<TensorDescriptor>,
    pub digest: String,
    pub catalog_digest: String,
}
impl TensorCatalog {
    pub fn new(tensors: Vec<TensorDescriptor>) -> Self {
        let digest = hex::encode(sha2::Sha256::digest(
            serde_json::to_vec(&tensors).unwrap_or_default(),
        ));
        Self {
            tensors,
            digest: digest.clone(),
            catalog_digest: digest,
        }
    }
    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|t| t.name == name)
    }
    pub fn iter(&self) -> std::slice::Iter<'_, TensorDescriptor> {
        self.tensors.iter()
    }
    pub fn len(&self) -> usize {
        self.tensors.len()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SourceCapabilities {
    pub random_access: bool,
    pub mmap: bool,
    pub writable: bool,
    pub supports_streaming: bool,
    pub supports_random_access: bool,
    pub supports_dequantize: bool,
}
#[derive(Serialize, Deserialize)]
pub struct CanonicalSource {
    pub identity: SourceIdentity,
    pub catalog: TensorCatalog,
    pub capabilities: SourceCapabilities,
    #[serde(skip)]
    pub provider: Option<Arc<dyn TensorDataProvider>>,
}
impl std::fmt::Debug for CanonicalSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalSource")
            .field("identity", &self.identity)
            .field("catalog", &self.catalog)
            .field("capabilities", &self.capabilities)
            .finish()
    }
}
impl Clone for CanonicalSource {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity.clone(),
            catalog: self.catalog.clone(),
            capabilities: self.capabilities.clone(),
            provider: self.provider.clone(),
        }
    }
}
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("unsupported source format: {0}")]
    UnsupportedFormat(String),
    #[error("tensor not found: {0}")]
    TensorNotFound(String),
    #[error("source I/O: {0}")]
    Io(String),
    #[error("invalid source: {0}")]
    Invalid(String),
}
pub trait TensorDataProvider: Send + Sync {
    fn read_tensor(&self, tensor: &TensorDescriptor) -> Result<Vec<u8>, SourceError>;
}
pub trait CanonicalSourceAdapter: Send + Sync {
    fn can_open(&self, path: &Path) -> bool;
    fn open(&self, path: &Path) -> Result<CanonicalSource, SourceError>;
}
fn unsupported(name: &str) -> Result<CanonicalSource, SourceError> {
    Err(SourceError::UnsupportedFormat(name.into()))
}
pub mod gguf_adapter {
    use super::*;
    pub struct GgufAdapter;
    impl CanonicalSourceAdapter for GgufAdapter {
        fn can_open(&self, p: &Path) -> bool {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        }
        fn open(&self, _: &Path) -> Result<CanonicalSource, SourceError> {
            unsupported("gguf")
        }
    }
}
pub mod onnx_adapter {
    use super::*;
    pub struct OnnxAdapter;
    impl CanonicalSourceAdapter for OnnxAdapter {
        fn can_open(&self, p: &Path) -> bool {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("onnx"))
                .unwrap_or(false)
        }
        fn open(&self, _: &Path) -> Result<CanonicalSource, SourceError> {
            unsupported("onnx")
        }
    }
}
fn dtype_size(dtype: &str) -> usize {
    match dtype {
        "F64" | "I64" | "U64" => 8,
        "F32" | "I32" | "U32" => 4,
        "F16" | "BF16" | "I16" | "U16" => 2,
        "BOOL" | "U8" | "I8" => 1,
        _ => 1,
    }
}

#[derive(Clone)]
struct SafeTensorEntry {
    path: PathBuf,
    offset: u64,
    length: u64,
}
struct SafeTensorDataProvider {
    entries: HashMap<String, SafeTensorEntry>,
}
impl TensorDataProvider for SafeTensorDataProvider {
    fn read_tensor(&self, tensor: &TensorDescriptor) -> Result<Vec<u8>, SourceError> {
        let entry = self
            .entries
            .get(&tensor.name)
            .ok_or_else(|| SourceError::TensorNotFound(tensor.name.clone()))?;
        let mut file = File::open(&entry.path).map_err(|e| SourceError::Io(e.to_string()))?;
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| SourceError::Io(e.to_string()))?;
        let mut data = vec![0u8; entry.length as usize];
        file.read_exact(&mut data)
            .map_err(|e| SourceError::Io(e.to_string()))?;
        Ok(data)
    }
}

fn open_safetensors_dir(path: &Path) -> Result<CanonicalSource, SourceError> {
    let mut shards = std::fs::read_dir(path)
        .map_err(|e| SourceError::Io(e.to_string()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect::<Vec<_>>();
    shards.sort();
    if shards.is_empty() {
        return Err(SourceError::UnsupportedFormat(format!(
            "no safetensors shards in {}",
            path.display()
        )));
    }
    let mut tensors = Vec::new();
    let mut entries = HashMap::new();
    let mut digest = sha2::Sha256::new();
    for shard in shards {
        let mut file = File::open(&shard).map_err(|e| SourceError::Io(e.to_string()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)
            .map_err(|e| SourceError::Invalid(e.to_string()))?;
        let header_len = u64::from_le_bytes(len) as usize;
        let mut bytes = vec![0u8; header_len];
        file.read_exact(&mut bytes)
            .map_err(|e| SourceError::Invalid(e.to_string()))?;
        digest.update(&bytes);
        let header: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| SourceError::Invalid(e.to_string()))?;
        let object = header
            .as_object()
            .ok_or_else(|| SourceError::Invalid("safetensors header is not an object".into()))?;
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let shape = value
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| SourceError::Invalid(format!("missing shape for {name}")))?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect::<Vec<_>>();
            let dtype = value
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("U8")
                .to_string();
            let offsets = value
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| SourceError::Invalid(format!("missing data offsets for {name}")))?;
            let start = offsets
                .first()
                .and_then(|v| v.as_u64())
                .ok_or_else(|| SourceError::Invalid(format!("invalid start for {name}")))?;
            let end = offsets
                .get(1)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| SourceError::Invalid(format!("invalid end for {name}")))?;
            let descriptor = TensorDescriptor {
                name: name.clone(),
                shape,
                dtype: dtype.clone(),
                original_dtype: dtype.clone(),
                element_size: dtype_size(&dtype),
                data_offset: Some(8 + header_len as u64 + start),
                data_size_bytes: end - start,
                layout: "row_major".into(),
                byte_offset: 8 + header_len as u64 + start,
                byte_length: end - start,
            };
            entries.insert(
                name.clone(),
                SafeTensorEntry {
                    path: shard.clone(),
                    offset: descriptor.byte_offset,
                    length: descriptor.byte_length,
                },
            );
            tensors.push(descriptor);
        }
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let catalog = TensorCatalog::new(tensors);
    let source_digest = hex::encode(digest.finalize());
    Ok(CanonicalSource {
        identity: SourceIdentity {
            format: SourceFormat::SafeTensors,
            source_digest,
            model_family: "unknown".into(),
            architecture: "safetensors".into(),
        },
        catalog,
        capabilities: SourceCapabilities {
            random_access: true,
            mmap: false,
            writable: false,
            supports_streaming: true,
            supports_random_access: true,
            supports_dequantize: false,
        },
        provider: Some(Arc::new(SafeTensorDataProvider { entries })),
    })
}

pub mod mlx_adapter {
    use super::*;
    pub struct MlxAdapter;
    impl CanonicalSourceAdapter for MlxAdapter {
        fn can_open(&self, p: &Path) -> bool {
            p.is_dir()
                && p.read_dir()
                    .ok()
                    .map(|e| {
                        e.flatten()
                            .any(|x| x.path().extension().is_some_and(|v| v == "safetensors"))
                    })
                    .unwrap_or(false)
        }
        fn open(&self, p: &Path) -> Result<CanonicalSource, SourceError> {
            open_safetensors_dir(p)
        }
    }
}
pub mod safetensors_adapter {
    use super::*;
    pub struct SafetensorsAdapter;
    impl CanonicalSourceAdapter for SafetensorsAdapter {
        fn can_open(&self, p: &Path) -> bool {
            p.extension()
                .and_then(|x| x.to_str())
                .map(|x| x.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false)
        }
        fn open(&self, p: &Path) -> Result<CanonicalSource, SourceError> {
            if p.is_dir() {
                open_safetensors_dir(p)
            } else {
                unsupported("single safetensors file")
            }
        }
    }
}
pub fn detect(
    path: &Path,
    adapters: &[Box<dyn CanonicalSourceAdapter>],
) -> Result<CanonicalSource, SourceError> {
    for a in adapters {
        if a.can_open(path) {
            return a.open(path);
        }
    }
    unsupported(path.display().to_string().as_str())
}
