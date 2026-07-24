//! Pure Rust safetensors format parser and serializer.
//!
//! The safetensors format is:
//!
//!   [8 bytes: header_len (u64 LE)]
//!   [header_len bytes: JSON, padded to 8-byte alignment]
//!   [tensor data, at offsets specified in JSON]
//!
//! The JSON header may contain a `"__metadata__"` key (optional, JSON object)
//! and top-level tensor entries. Each tensor entry has:
//!
//! ```json
//! {
//!   "dtype": "F32",
//!   "shape": [768, 768],
//!   "data_offsets": [0, 2304]
//! }
//! ```
//!
//! Offsets in `data_offsets` are relative to the start of the data section
//! (the byte immediately after the header). The data section starts at an
//! 8-byte aligned offset from the beginning of the file.

use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// ── Dtype ────────────────────────────────────────────────────────────────

/// Element types supported by the safetensors format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    F64,
}

impl Dtype {
    /// Parse a dtype string from the safetensors JSON header.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "F32" => Ok(Dtype::F32),
            "F16" => Ok(Dtype::F16),
            "BF16" => Ok(Dtype::BF16),
            "I64" => Ok(Dtype::I64),
            "I32" => Ok(Dtype::I32),
            "I16" => Ok(Dtype::I16),
            "I8" => Ok(Dtype::I8),
            "U64" => Ok(Dtype::U64),
            "U32" => Ok(Dtype::U32),
            "U16" => Ok(Dtype::U16),
            "U8" => Ok(Dtype::U8),
            "F64" => Ok(Dtype::F64),
            other => Err(format!("unknown safetensors dtype: {other}")),
        }
    }

    /// Return the canonical dtype string used in safetensors JSON headers.
    pub fn as_str(&self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U64 => "U64",
            Dtype::U32 => "U32",
            Dtype::U16 => "U16",
            Dtype::U8 => "U8",
            Dtype::F64 => "F64",
        }
    }

    /// Return the number of bytes per element for this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 | Dtype::U16 => 2,
            Dtype::I64 | Dtype::U64 | Dtype::F64 => 8,
            Dtype::I8 | Dtype::U8 => 1,
        }
    }
}

// ── TensorView ──────────────────────────────────────────────────────────

/// A view into a single tensor's data, borrowed from the original buffer.
#[derive(Debug)]
pub struct TensorView<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Create a new tensor view from individual parts.
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Self {
        TensorView { dtype, shape, data }
    }

    pub fn dtype(&self) -> &Dtype {
        &self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Number of elements in this tensor (product of shape dimensions).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total byte size of the tensor data.
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.dtype.element_size()
    }
}

// ── TensorMetaInfo / TensorMetadata ─────────────────────────────────────

/// Metadata for a single tensor, parsed from the header without borrowing data.
#[derive(Debug, Clone)]
pub struct TensorMetaInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte offsets relative to the start of the data section.
    pub data_offsets: (usize, usize),
}

/// Collected tensor metadata from the header only.
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub tensors: HashMap<String, TensorMetaInfo>,
}

impl TensorMetadata {
    /// Iterate over all (name, info) pairs.
    pub fn tensors(&self) -> impl Iterator<Item = (&str, &TensorMetaInfo)> {
        self.tensors.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of tensors in this metadata.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true if there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

// ── SafeTensors ─────────────────────────────────────────────────────────

/// A parsed safetensors file with borrowed tensor views into the original buffer.
#[derive(Debug)]
pub struct SafeTensors<'a> {
    tensors: HashMap<String, TensorView<'a>>,
    metadata: HashMap<String, String>,
}

/// A named tensor with owned data for serialization.
pub struct NamedTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl<'a> SafeTensors<'a> {
    // ── Deserialization ─────────────────────────────────────────

    /// Parse the full safetensors buffer, borrowing tensor data from it.
    pub fn deserialize(data: &'a [u8]) -> Result<Self, String> {
        let data_start = Self::read_header_len(data)?;
        let (metadata_map, tensor_meta) = Self::parse_json_header(&data[8..data_start])?;

        let mut tensors: HashMap<String, TensorView<'a>> = HashMap::new();
        for (name, info) in &tensor_meta.tensors {
            let start = data_start + info.data_offsets.0;
            let end = data_start + info.data_offsets.1;
            if end > data.len() {
                return Err(format!(
                    "tensor '{name}' data_offsets [{}, {}] out of bounds (data len {})",
                    info.data_offsets.0,
                    info.data_offsets.1,
                    data.len() - data_start
                ));
            }
            tensors.insert(
                name.clone(),
                TensorView {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    data: &data[start..end],
                },
            );
        }

        Ok(SafeTensors {
            tensors,
            metadata: metadata_map,
        })
    }

    /// Read only the header portion (metadata + tensor metadata) without
    /// borrowing tensor data. Returns `(metadata, TensorMetadata)`.
    pub fn read_metadata(
        data: &'a [u8],
    ) -> Result<(HashMap<String, String>, TensorMetadata), String> {
        let data_start = Self::read_header_len(data)?;
        Self::parse_json_header(&data[8..data_start])
    }

    /// Read the 8-byte header length field, skip any padding bytes between
    /// the header and data section, and return the byte offset where the data
    /// section starts.
    ///
    /// The safetensors spec requires the data section to be 8-byte aligned
    /// from the start of the file, so we skip any alignment padding after
    /// the JSON header before declaring the data section.
    fn read_header_len(data: &[u8]) -> Result<usize, String> {
        if data.len() < 8 {
            return Err(format!(
                "file too small: {} bytes, need at least 8 for header length field",
                data.len()
            ));
        }
        let header_len_bytes: [u8; 8] =
            data[..8].try_into().map_err(|_| "bad header length read")?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // The header occupies bytes 8..(8 + header_len). The data section
        // should start at the next 8-byte aligned boundary.
        let header_end = 8 + header_len;
        if header_end > data.len() {
            return Err(format!(
                "header claims {} bytes but file is only {} bytes",
                header_len,
                data.len() - 8
            ));
        }

        // Align data_start to 8 bytes.
        let data_start = (header_end + 7) & !7;
        if data_start > data.len() {
            return Err(format!(
                "aligned header end at {} exceeds file length of {}",
                data_start,
                data.len()
            ));
        }

        Ok(data_start)
    }

    /// Parse the JSON header bytes (without the 8-byte length prefix) into
    /// metadata map and tensor metadata.
    fn parse_json_header(
        header_bytes: &[u8],
    ) -> Result<(HashMap<String, String>, TensorMetadata), String> {
        let end = header_bytes
            .iter()
            .rposition(|byte| !matches!(byte, 0 | b' ' | b'\n' | b'\r' | b'\t'))
            .map(|i| i + 1)
            .unwrap_or(0);
        let header_bytes = &header_bytes[..end];
        let header: Value = serde_json::from_slice(header_bytes)
            .map_err(|e| format!("invalid JSON header: {e}"))?;

        let obj = header.as_object().ok_or_else(|| {
            format!(
                "expected JSON object in safetensors header, got {}",
                json_type_name(&header)
            )
        })?;

        // Extract metadata (optional). The safetensors format uses both
        // `__metadata__` (preferred) and `metadata` (legacy) keys.
        let metadata_map = Self::extract_metadata_obj(obj);

        // Every key that is not a reserved name is a tensor name.
        let mut tensor_map: HashMap<String, TensorMetaInfo> = HashMap::new();
        for (key, value) in obj {
            if is_reserved_header_key(key) {
                continue;
            }
            let entry = value.as_object().ok_or_else(|| {
                format!(
                    "tensor '{key}': expected object, got {}",
                    json_type_name(value)
                )
            })?;

            let dtype_str = entry
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("tensor '{key}': missing or invalid 'dtype'"))?;
            let dtype = Dtype::from_str(dtype_str)?;

            let shape: Vec<usize> = entry
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("tensor '{key}': missing or invalid 'shape'"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .ok_or_else(|| format!("tensor '{key}': non-integer in shape"))
                        .map(|u| u as usize)
                })
                .collect::<Result<Vec<_>, String>>()?;

            let offsets = entry
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("tensor '{key}': missing or invalid 'data_offsets'"))?;
            if offsets.len() != 2 {
                return Err(format!(
                    "tensor '{key}': data_offsets must have exactly 2 elements, got {}",
                    offsets.len()
                ));
            }
            let offset_start = offsets[0]
                .as_u64()
                .ok_or_else(|| format!("tensor '{key}': data_offsets[0] is not a u64"))?
                as usize;
            let offset_end = offsets[1]
                .as_u64()
                .ok_or_else(|| format!("tensor '{key}': data_offsets[1] is not a u64"))?
                as usize;

            if offset_end < offset_start {
                return Err(format!(
                    "tensor '{key}': data_offsets end < start ({} < {})",
                    offset_end, offset_start
                ));
            }

            // Validate element count matches data size.
            let expected_bytes: usize = shape.iter().product::<usize>() * dtype.element_size();
            let actual_bytes = offset_end - offset_start;
            if actual_bytes != expected_bytes {
                return Err(format!(
                    "tensor '{key}': shape {:?} with dtype {:?} expects {expected_bytes} bytes, \
                     but data_offsets span {actual_bytes} bytes",
                    shape, dtype
                ));
            }

            tensor_map.insert(
                key.clone(),
                TensorMetaInfo {
                    dtype,
                    shape,
                    data_offsets: (offset_start, offset_end),
                },
            );
        }

        Ok((
            metadata_map,
            TensorMetadata {
                tensors: tensor_map,
            },
        ))
    }

    /// Extract the metadata object from the header JSON, supporting both
    /// `__metadata__` (preferred, per spec) and `metadata` (legacy) keys.
    fn extract_metadata_obj(obj: &serde_json::Map<String, Value>) -> HashMap<String, String> {
        // Prefer `__metadata__`, fall back to `metadata`.
        let meta_val = obj.get("__metadata__").or_else(|| obj.get("metadata"));

        match meta_val {
            Some(Value::Object(m)) => {
                let mut map = HashMap::new();
                for (k, v) in m {
                    let val_str = match v {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        Value::Bool(b) => b.to_string(),
                        other => format!("{other}"),
                    };
                    map.insert(k.clone(), val_str);
                }
                map
            }
            _ => HashMap::new(),
        }
    }

    // ── Serialization ───────────────────────────────────────────

    /// Serialize a collection of named tensors + optional metadata into the
    /// safetensors binary format.
    ///
    /// `tensors` is a slice of `(name, dtype, shape, data)` tuples. The header
    /// is padded to 8-byte alignment as required by the spec.
    pub fn serialize(
        tensors: &[(&str, Dtype, &[usize], &[u8])],
        metadata: &HashMap<String, String>,
    ) -> Result<Vec<u8>, String> {
        Self::serialize_impl(tensors, metadata)
    }

    /// Serialize and write to a file path.
    pub fn serialize_to_file(
        tensors: &[(&str, Dtype, &[usize], &[u8])],
        metadata: &HashMap<String, String>,
        path: impl AsRef<Path>,
    ) -> Result<(), String> {
        let buf = Self::serialize_impl(tensors, metadata)?;
        fs::write(path.as_ref(), &buf).map_err(|e| format!("write safetensors: {e}"))
    }

    /// Internal serialization logic.
    fn serialize_impl(
        tensors: &[(&str, Dtype, &[usize], &[u8])],
        metadata: &HashMap<String, String>,
    ) -> Result<Vec<u8>, String> {
        // Validate each tensor's data size matches its shape + dtype.
        for (name, dtype, shape, data) in tensors {
            let expected: usize = shape.iter().product::<usize>() * dtype.element_size();
            if data.len() != expected {
                return Err(format!(
                    "tensor '{name}': shape {:?} with dtype {:?} expects {expected} bytes, \
                     got {}",
                    shape,
                    dtype,
                    data.len()
                ));
            }
        }

        // Build JSON header.
        let mut json_map = serde_json::Map::new();

        // Include metadata as `__metadata__` key.
        if !metadata.is_empty() {
            let mut meta_obj = serde_json::Map::new();
            for (k, v) in metadata {
                meta_obj.insert(k.clone(), Value::String(v.clone()));
            }
            json_map.insert("__metadata__".to_string(), Value::Object(meta_obj));
        }

        // Build tensor entries and data section simultaneously.
        let mut data_section = Vec::new();
        for (name, dtype, shape, data) in tensors {
            let start = data_section.len();
            data_section.extend_from_slice(data);
            let end = data_section.len();

            let shape_vec: Vec<Value> = shape.iter().map(|&d| Value::Number(d.into())).collect();
            let entry = serde_json::json!({
                "dtype": dtype.as_str(),
                "shape": shape_vec,
                "data_offsets": [start, end],
            });
            json_map.insert(name.to_string(), entry);
        }

        let header_json = Value::Object(json_map);
        let header_bytes =
            serde_json::to_vec(&header_json).map_err(|e| format!("serialize header JSON: {e}"))?;

        // Pad header to 8-byte alignment: 8 (length field) + header_bytes
        // must be a multiple of 8 so that the data section starts 8-byte aligned.
        let raw_len = 8 + header_bytes.len();
        let padded_len = (raw_len + 7) & !7;

        let mut buf = Vec::with_capacity(padded_len + data_section.len());

        // Write header length (the length of the JSON portion only, per spec).
        let header_len = header_bytes.len() as u64;
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        // Pad with zero bytes (or spaces; zeros are conventional).
        buf.resize(padded_len, 0u8);
        buf.extend_from_slice(&data_section);

        Ok(buf)
    }

    // ── Accessors ───────────────────────────────────────────────

    /// Get a tensor view by name.
    pub fn tensor(&self, name: &str) -> Result<&TensorView<'a>, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' not found"))
    }

    /// Return all tensor names.
    pub fn names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Iterate over all (name, view) pairs.
    pub fn tensors(&self) -> impl Iterator<Item = (&String, &TensorView<'a>)> {
        self.tensors.iter()
    }

    /// Return the number of tensors.
    pub fn num_tensors(&self) -> usize {
        self.tensors.len()
    }

    /// Return the metadata map parsed from the header.
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Return a human-readable type name for a JSON value.
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Keys in the JSON header that are reserved (not tensor names).
fn is_reserved_header_key(key: &str) -> bool {
    matches!(key, "__metadata__" | "metadata")
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn write_safetensors(tensors: &[(&str, Dtype, &[u32], &[u8])]) -> Vec<u8> {
        let mut json_map = serde_json::Map::new();
        let mut data_section = Vec::new();

        for (name, dtype, shape, tensor_data) in tensors {
            let start = data_section.len();
            data_section.extend_from_slice(tensor_data);
            let end = data_section.len();

            let entry = serde_json::json!({
                "dtype": dtype.as_str(),
                "shape": shape,
                "data_offsets": [start, end],
            });
            json_map.insert(name.to_string(), entry);
        }

        let header = serde_json::Value::Object(json_map);
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;

        // Pad to 8-byte alignment.
        let raw_len = 8 + header_bytes.len();
        let padded_len = (raw_len + 7) & !7;

        let mut buf = Vec::with_capacity(padded_len + data_section.len());
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.resize(padded_len, 0u8);
        buf.extend_from_slice(&data_section);
        buf
    }

    // ── Deserialize tests ───────────────────────────────────────

    #[test]
    fn test_deserialize_single_tensor() {
        let data = vec![0u8; 16]; // 4 f32 values × 4 bytes
        let shape: &[u32] = &[4];
        let buf = write_safetensors(&[("weights", Dtype::F32, shape, &data)]);

        let st = SafeTensors::deserialize(&buf).unwrap();
        assert_eq!(st.num_tensors(), 1);

        let view = st.tensor("weights").unwrap();
        assert_eq!(*view.dtype(), Dtype::F32);
        assert_eq!(view.shape(), &[4usize]);
        assert_eq!(view.data(), &data);
    }

    #[test]
    fn test_deserialize_multiple_tensors() {
        let w1 = vec![1u8, 2, 3, 4];
        let w2 = vec![5u8, 6];
        let buf = write_safetensors(&[("a", Dtype::F32, &[1], &w1), ("b", Dtype::BF16, &[1], &w2)]);

        let st = SafeTensors::deserialize(&buf).unwrap();
        assert_eq!(st.num_tensors(), 2);

        let names: Vec<&String> = st.names();
        assert!(names.contains(&&"a".to_string()));
        assert!(names.contains(&&"b".to_string()));

        let a = st.tensor("a").unwrap();
        assert_eq!(*a.dtype(), Dtype::F32);
        assert_eq!(a.data(), &w1);

        let b = st.tensor("b").unwrap();
        assert_eq!(*b.dtype(), Dtype::BF16);
        assert_eq!(b.data(), &w2);
    }

    #[test]
    fn test_tensors_iterator() {
        let w1 = vec![1u8, 2, 3, 4];
        let w2 = vec![5u8, 6, 7, 8];
        let buf = write_safetensors(&[("x", Dtype::I32, &[1], &w1), ("y", Dtype::F32, &[1], &w2)]);

        let st = SafeTensors::deserialize(&buf).unwrap();
        let mut count = 0;
        for (name, view) in st.tensors() {
            assert!(name == "x" || name == "y");
            assert!(view.dtype() == &Dtype::I32 || view.dtype() == &Dtype::F32);
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn test_read_metadata() {
        let data = vec![1u8, 2, 3, 4];
        let buf = write_safetensors(&[("t", Dtype::I16, &[2], &data)]);

        let (meta, tensor_meta) = SafeTensors::read_metadata(&buf).unwrap();
        assert!(meta.is_empty());
        assert_eq!(tensor_meta.len(), 1);

        let (name, info) = tensor_meta.tensors().next().unwrap();
        assert_eq!(name, "t");
        assert_eq!(info.dtype, Dtype::I16);
        assert_eq!(info.shape, vec![2usize]);
        assert_eq!(info.data_offsets, (0, 4));
    }

    #[test]
    fn test_too_small_file() {
        let err = SafeTensors::deserialize(&[0u8; 3]).unwrap_err();
        assert!(err.contains("too small"), "got: {err}");
    }

    #[test]
    fn test_tensor_not_found() {
        let buf = write_safetensors(&[("a", Dtype::U8, &[1], &[42])]);
        let st = SafeTensors::deserialize(&buf).unwrap();
        let err = st.tensor("nonexistent").unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn test_invalid_json_header() {
        let mut buf = vec![5u8, 0, 0, 0, 0, 0, 0, 0];
        buf.extend_from_slice(b"{{{{{");
        // Pad to align
        buf.resize((buf.len() + 7) & !7, 0);
        let err = SafeTensors::deserialize(&buf).unwrap_err();
        assert!(err.contains("invalid JSON"), "got: {err}");
    }

    #[test]
    fn test_unsupported_dtype() {
        let header = serde_json::json!({
            "t": {"dtype": "F128", "shape": [1], "data_offsets": [0, 8]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;
        let raw_len = 8 + header_bytes.len();
        let padded_len = (raw_len + 7) & !7;

        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.resize(padded_len, 0u8);
        buf.extend_from_slice(&[0u8; 8]);

        let err = SafeTensors::deserialize(&buf).unwrap_err();
        assert!(err.contains("unknown"), "got: {err}");
    }

    #[test]
    fn test_dtype_element_size() {
        assert_eq!(Dtype::F32.element_size(), 4);
        assert_eq!(Dtype::F64.element_size(), 8);
        assert_eq!(Dtype::BF16.element_size(), 2);
        assert_eq!(Dtype::F16.element_size(), 2);
        assert_eq!(Dtype::I64.element_size(), 8);
        assert_eq!(Dtype::U64.element_size(), 8);
        assert_eq!(Dtype::I32.element_size(), 4);
        assert_eq!(Dtype::I16.element_size(), 2);
        assert_eq!(Dtype::I8.element_size(), 1);
        assert_eq!(Dtype::U8.element_size(), 1);
    }

    #[test]
    fn test_dtype_roundtrip_str() {
        let dtypes = [
            Dtype::F32,
            Dtype::F16,
            Dtype::BF16,
            Dtype::I64,
            Dtype::I32,
            Dtype::I16,
            Dtype::I8,
            Dtype::U64,
            Dtype::U32,
            Dtype::U16,
            Dtype::U8,
            Dtype::F64,
        ];
        for dt in &dtypes {
            assert_eq!(Dtype::from_str(dt.as_str()).unwrap(), *dt);
        }
    }

    // ── Metadata key tests ──────────────────────────────────────

    #[test]
    fn test_metadata_with_metadata_key() {
        let header = serde_json::json!({
            "metadata": {"foo": "bar", "baz": "qux"},
            "t": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;
        let raw_len = 8 + header_bytes.len();
        let padded_len = (raw_len + 7) & !7;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.resize(padded_len, 0u8);
        buf.push(42);

        let (meta, _tm) = SafeTensors::read_metadata(&buf).unwrap();
        assert_eq!(meta.get("foo").unwrap(), "bar");
        assert_eq!(meta.get("baz").unwrap(), "qux");
    }

    #[test]
    fn test_metadata_with_dundermetadata_key() {
        let header = serde_json::json!({
            "__metadata__": {"version": "1.0", "format": "pt"},
            "t": {"dtype": "I32", "shape": [2], "data_offsets": [0, 8]}
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;
        let raw_len = 8 + header_bytes.len();
        let padded_len = (raw_len + 7) & !7;
        let mut buf = Vec::new();
        buf.extend_from_slice(&header_len.to_le_bytes());
        buf.extend_from_slice(&header_bytes);
        buf.resize(padded_len, 0u8);
        buf.extend_from_slice(&[1u8, 0, 0, 0, 2u8, 0, 0, 0]);

        let (meta, _tm) = SafeTensors::read_metadata(&buf).unwrap();
        assert_eq!(meta.get("version").unwrap(), "1.0");
        assert_eq!(meta.get("format").unwrap(), "pt");
    }

    // ── Serialize tests ─────────────────────────────────────────

    #[test]
    fn test_serialize_roundtrip() {
        let data_a = vec![1u8, 2, 3, 4];
        let data_b = vec![5u8, 6, 7, 8, 9, 10, 11, 12];
        let tensors: &[(&str, Dtype, &[usize], &[u8])] = &[
            ("a", Dtype::F32, &[1], &data_a),
            ("b", Dtype::F64, &[1], &data_b),
        ];
        let meta = HashMap::new();

        let buf = SafeTensors::serialize(tensors, &meta).unwrap();

        // Parse it back and verify.
        let st = SafeTensors::deserialize(&buf).unwrap();
        assert_eq!(st.num_tensors(), 2);

        let va = st.tensor("a").unwrap();
        assert_eq!(*va.dtype(), Dtype::F32);
        assert_eq!(va.data(), &data_a);

        let vb = st.tensor("b").unwrap();
        assert_eq!(*vb.dtype(), Dtype::F64);
        assert_eq!(vb.data(), &data_b);
    }

    #[test]
    fn test_serialize_with_metadata() {
        let data = vec![0u8; 8];
        let tensors: &[(&str, Dtype, &[usize], &[u8])] = &[("w", Dtype::I64, &[1], &data)];
        let mut meta = HashMap::new();
        meta.insert("foo".to_string(), "bar".to_string());

        let buf = SafeTensors::serialize(tensors, &meta).unwrap();

        // Parse back and check metadata.
        let (parsed_meta, tm) = SafeTensors::read_metadata(&buf).unwrap();
        assert_eq!(parsed_meta.get("foo").unwrap(), "bar");
        assert_eq!(tm.len(), 1);
    }

    #[test]
    fn test_serialize_8byte_alignment() {
        let data = vec![0u8; 4];
        let tensors: &[(&str, Dtype, &[usize], &[u8])] = &[("x", Dtype::F32, &[1], &data)];

        let buf = SafeTensors::serialize(tensors, &HashMap::new()).unwrap();

        // Data section must start at an 8-byte aligned offset.
        let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
        let header_end = 8 + header_len;
        let data_start = (header_end + 7) & !7;
        assert_eq!(data_start, (header_end + 7) & !7);
        assert_eq!(data_start % 8, 0, "data_start {} not 8-aligned", data_start);
        assert_eq!(buf[data_start..], data);

        // The file should roundtrip correctly.
        let st = SafeTensors::deserialize(&buf).unwrap();
        assert_eq!(st.num_tensors(), 1);
        assert_eq!(st.tensor("x").unwrap().data(), &data);
    }

    #[test]
    fn test_serialize_data_size_mismatch() {
        let data = vec![0u8; 4]; // 4 bytes, but F64 expects 8 per element
        let tensors: &[(&str, Dtype, &[usize], &[u8])] = &[("bad", Dtype::F64, &[1], &data)];

        let err = SafeTensors::serialize(tensors, &HashMap::new()).unwrap_err();
        assert!(err.contains("expects 8 bytes"), "got: {err}");
    }

    #[test]
    fn test_serialize_multiple_tensors() {
        let data = vec![0u8; 4];
        let tensors: &[(&str, Dtype, &[usize], &[u8])] = &[
            ("w1", Dtype::F32, &[1], &data),
            ("w2", Dtype::I32, &[1], &data),
        ];

        let buf = SafeTensors::serialize(tensors, &HashMap::new()).unwrap();
        let st = SafeTensors::deserialize(&buf).unwrap();
        assert_eq!(st.num_tensors(), 2);
        assert!(st.tensor("w1").is_ok());
        assert!(st.tensor("w2").is_ok());
    }

    // ── TensorView::new test ────────────────────────────────────

    #[test]
    fn test_tensor_view_new() {
        let data = vec![1u8, 2, 3, 4];
        let tv = TensorView::new(Dtype::U32, vec![1], &data);
        assert_eq!(*tv.dtype(), Dtype::U32);
        assert_eq!(tv.shape(), &[1usize]);
        assert_eq!(tv.data(), &[1u8, 2, 3, 4]);
        assert_eq!(tv.num_elements(), 1);
        assert_eq!(tv.byte_size(), 4);
    }

    // ── Edge case tests ─────────────────────────────────────────

    #[test]
    fn test_empty_file() {
        let err = SafeTensors::deserialize(&[]).unwrap_err();
        assert!(err.contains("too small"), "got: {err}");
    }

    #[test]
    fn test_zero_length_header() {
        let mut buf = vec![0u8; 8]; // header_len = 0, header starts and ends at offset 8
                                    // Pad to 8 bytes (8 is already 8-aligned)
                                    // data section: a single byte
        buf.push(42u8);
        let err = SafeTensors::deserialize(&buf).unwrap_err();
        assert!(err.contains("invalid JSON"), "got: {err}");
    }
}
