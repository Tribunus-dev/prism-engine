//! A small, deterministic tensor container used at Prism's ECS boundaries,
//! plus the canonical backend-neutral evaluation surface for codec-correct
//! candidates.
//!
//! The container deliberately stores metadata separately from bytes.  It is
//! suitable for model sidecars and CImage payloads, but does not pretend to be
//! a model format: safetensors/GGUF ingestion remains responsible for reading
//! those formats and can pass their validated tensor bytes here.
//!
//! The [`evaluator`] submodule owns the canonical authority for the
//! heterogeneous evaluation surface: backend-neutral contract, codec-correct
//! fixtures, immutable evidence, admission decisions, and the system that
//! coordinates evaluation lanes. See [`evaluator::HeterogeneousEvaluatorSystem`]
//! for the entry point.
//!
//! The [`lut`] submodule owns the canonical authority for the lookup-table
//! (LUT) codec: model graph descriptors, palettized matrix format, FP16 math
//! kernels, and INT8 KV-cache quantization helpers. See [`lut::graph`] for
//! the entry point.

pub mod evaluator;
pub mod lut;

use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, fs, path::Path};
use thiserror::Error;

const MAGIC: &[u8; 8] = b"PRMTCOD\0";
const VERSION: u16 = 1;
const HEADER_BYTES: usize = 8 + 2 + 4 + 8 + 32;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("invalid tensor metadata: {0}")]
    InvalidMetadata(String),
    #[error("invalid codec envelope: {0}")]
    InvalidEnvelope(String),
    #[error("metadata encoding failed: {0}")]
    Metadata(#[from] serde_json::Error),
    #[error("I/O failed: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl DType {
    pub const fn byte_width(self) -> usize {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl TensorMetadata {
    pub fn element_count(&self) -> Result<usize, CodecError> {
        self.shape
            .iter()
            .try_fold(1usize, |n, &d| {
                usize::try_from(d).ok().and_then(|d| n.checked_mul(d))
            })
            .ok_or_else(|| CodecError::InvalidMetadata("shape overflows host usize".into()))
    }
    pub fn byte_len(&self) -> Result<usize, CodecError> {
        self.element_count()?
            .checked_mul(self.dtype.byte_width())
            .ok_or_else(|| CodecError::InvalidMetadata("byte length overflows host usize".into()))
    }
    fn validate(&self) -> Result<(), CodecError> {
        if self.name.is_empty() {
            return Err(CodecError::InvalidMetadata("tensor name is empty".into()));
        }
        if self.name.len() > 4096 {
            return Err(CodecError::InvalidMetadata(
                "tensor name is too long".into(),
            ));
        }
        self.byte_len().map(|_| ())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub metadata: TensorMetadata,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(metadata: TensorMetadata, data: Vec<u8>) -> Result<Self, CodecError> {
        metadata.validate()?;
        let expected = metadata.byte_len()?;
        if data.len() != expected {
            return Err(CodecError::InvalidMetadata(format!(
                "{} declares {expected} bytes, got {}",
                metadata.name,
                data.len()
            )));
        }
        Ok(Self { metadata, data })
    }
    pub fn from_bytes(
        name: impl Into<String>,
        dtype: DType,
        shape: Vec<u64>,
        data: Vec<u8>,
    ) -> Result<Self, CodecError> {
        Self::new(
            TensorMetadata {
                name: name.into(),
                dtype,
                shape,
            },
            data,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorCodec;

impl TensorCodec {
    pub fn encode(tensor: &Tensor) -> Result<Vec<u8>, CodecError> {
        tensor.metadata.validate()?;
        if tensor.data.len() != tensor.metadata.byte_len()? {
            return Err(CodecError::InvalidMetadata(
                "data length does not match metadata".into(),
            ));
        }
        let meta = serde_json::to_vec(&tensor.metadata)?;
        let meta_len = u32::try_from(meta.len())
            .map_err(|_| CodecError::InvalidMetadata("metadata is too large".into()))?;
        let data_len = u64::try_from(tensor.data.len())
            .map_err(|_| CodecError::InvalidMetadata("tensor is too large".into()))?;
        let digest = blake3::hash(&tensor.data);
        let mut out = Vec::with_capacity(HEADER_BYTES + meta.len() + tensor.data.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&meta_len.to_le_bytes());
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend_from_slice(digest.as_bytes());
        out.extend_from_slice(&meta);
        out.extend_from_slice(&tensor.data);
        Ok(out)
    }
    pub fn decode(bytes: &[u8]) -> Result<Tensor, CodecError> {
        if bytes.len() < HEADER_BYTES {
            return Err(CodecError::InvalidEnvelope("truncated header".into()));
        }
        if &bytes[..8] != MAGIC {
            return Err(CodecError::InvalidEnvelope("bad magic".into()));
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        if version != VERSION {
            return Err(CodecError::InvalidEnvelope(format!(
                "unsupported version {version}"
            )));
        }
        let ml = u32::from_le_bytes(bytes[10..14].try_into().unwrap()) as usize;
        let dl = u64::from_le_bytes(bytes[14..22].try_into().unwrap()) as usize;
        let end_meta = HEADER_BYTES
            .checked_add(ml)
            .ok_or_else(|| CodecError::InvalidEnvelope("metadata length overflow".into()))?;
        let end = end_meta
            .checked_add(dl)
            .ok_or_else(|| CodecError::InvalidEnvelope("data length overflow".into()))?;
        if end != bytes.len() {
            return Err(CodecError::InvalidEnvelope(
                "length does not match envelope".into(),
            ));
        }
        let metadata: TensorMetadata = serde_json::from_slice(&bytes[HEADER_BYTES..end_meta])?;
        let data = &bytes[end_meta..];
        if blake3::hash(data).as_bytes() != &bytes[22..54] {
            return Err(CodecError::InvalidEnvelope(
                "payload checksum mismatch".into(),
            ));
        }
        Tensor::new(metadata, data.to_vec())
    }
    pub fn write(path: impl AsRef<Path>, tensor: &Tensor) -> Result<(), CodecError> {
        fs::write(path, Self::encode(tensor)?).map_err(Into::into)
    }
    pub fn read(path: impl AsRef<Path>) -> Result<Tensor, CodecError> {
        Self::decode(&fs::read(path)?)
    }
}

pub fn encode(tensor: &Tensor) -> Result<Vec<u8>, CodecError> {
    TensorCodec::encode(tensor)
}
pub fn decode(bytes: &[u8]) -> Result<Tensor, CodecError> {
    TensorCodec::decode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn round_trip_preserves_metadata_and_bytes() {
        let t = Tensor::from_bytes(
            "w",
            DType::F32,
            vec![2, 2],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        )
        .unwrap();
        assert_eq!(
            TensorCodec::decode(&TensorCodec::encode(&t).unwrap()).unwrap(),
            t
        );
    }
    #[test]
    fn rejects_wrong_size() {
        assert!(Tensor::from_bytes("x", DType::I16, vec![2], vec![0]).is_err());
    }
    #[test]
    fn rejects_corruption() {
        let t = Tensor::from_bytes("x", DType::U8, vec![2], vec![1, 2]).unwrap();
        let mut b = encode(&t).unwrap();
        *b.last_mut().unwrap() ^= 1;
        assert!(decode(&b).is_err());
    }
    #[test]
    fn file_round_trip() {
        let p = tempfile::NamedTempFile::new().unwrap();
        let t = Tensor::from_bytes("x", DType::U8, vec![3], vec![1, 2, 3]).unwrap();
        TensorCodec::write(p.path(), &t).unwrap();
        assert_eq!(TensorCodec::read(p.path()).unwrap(), t);
    }
}
