//! ONNX model adapter — minimal protobuf parser + TensorProvider for `.onnx` files.
//!
//! Parses ONNX protobuf binary files using a hand-written streaming decoder
//! (no protobuf code-gen dependencies). Extracts tensor names, shapes, and
//! raw weight data from the GraphProto initializers, model inputs, and outputs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use prism_ecs_core::identity::{TensorInfo, TensorProvider, TensorReader};

// ── Public types ─────────────────────────────────────────────────────────

/// Metadata describing an ONNX model's graph structure.
#[derive(Debug, Clone)]
pub struct OnnxModelDescriptor {
    /// Names of graph input tensors (from ValueInfoProto).
    pub graph_inputs: Vec<String>,
    /// Names of graph output tensors (from ValueInfoProto).
    pub graph_outputs: Vec<String>,
    /// Initializer weights keyed by tensor name.
    pub initializers: HashMap<String, OnnxTensorInfo>,
    /// Graph nodes (operators).
    pub ops: Vec<OnnxOperator>,
    /// Inferred hidden dimension size.
    pub hidden_size: usize,
    /// Inferred number of transformer layers.
    pub num_layers: usize,
    /// Inferred number of attention heads.
    pub num_heads: usize,
    /// Inferred vocabulary size (from embedding table).
    pub vocab_size: usize,
}

/// Shape and type info for a single ONNX tensor.
#[derive(Debug, Clone)]
pub struct OnnxTensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub data_type: OnnxDataType,
}

/// ONNX tensor data types (the subset we care about).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxDataType {
    Float,
    Float16,
    Int8,
    Int32,
    Int64,
    Undefined,
}

/// Lightweight representation of graph operators.
#[derive(Debug, Clone)]
pub enum OnnxOperator {
    MatMul {
        input: String,
        weight: String,
        output: String,
    },
    Add {
        inputs: Vec<String>,
        output: String,
    },
    Reshape {
        input: String,
        shape: String,
        output: String,
    },
    /// Catch-all for unmodeled ops.
    Other {
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    },
}

// ── OnnxModelProvider ────────────────────────────────────────────────────

/// Provides tensor access from a single `.onnx` file.
///
/// Parses the ONNX protobuf at construction time, extracts all initializer
/// tensor data, and serves them through [`TensorProvider`].
pub struct OnnxModelProvider {
    #[allow(dead_code)]
    file_path: PathBuf,
    /// Full protobuf file contents (loaded once).
    file_data: Vec<u8>,
    /// Indexed tensor info.
    tensors: Vec<OnnxTensorInfo>,
    /// Tensor name → index in `tensors`.
    name_index: HashMap<String, usize>,
    /// Byte offsets into `file_data` for each initializer's raw tensor data.
    /// Maps tensor name → (offset, length) within `file_data`.
    raw_offsets: HashMap<String, (usize, usize)>,
    /// Parsed graph metadata.
    metadata: OnnxModelDescriptor,
}

impl OnnxModelProvider {
    /// Parse an ONNX file and index all initializer tensors.
    pub fn new(path: &Path) -> Result<Self, String> {
        let file_data =
            std::fs::read(path).map_err(|e| format!("read ONNX file {}: {e}", path.display()))?;

        let mut parser = ProtoParser::new(&file_data);
        let model = parser.parse_model_proto()?;

        let graph_inputs = model.graph_input_names;
        let graph_outputs = model.graph_output_names;
        let ops = model.operators;
        let tensors = model.initializers;
        let initializers: HashMap<String, OnnxTensorInfo> = tensors
            .iter()
            .map(|t| (t.name.clone(), t.clone()))
            .collect();

        // Build name index and raw-data offset map.
        let mut name_index: HashMap<String, usize> = HashMap::new();
        let mut raw_offsets: HashMap<String, (usize, usize)> = HashMap::new();
        for (i, t) in tensors.iter().enumerate() {
            name_index.insert(t.name.clone(), i);
        }
        for offset_entry in &model.raw_data_offsets {
            raw_offsets.insert(offset_entry.0.clone(), (offset_entry.1, offset_entry.2));
        }

        // Infer model architecture metadata from tensor shapes.
        let (hidden_size, num_layers, num_heads, vocab_size) =
            infer_model_dims(&tensors, &graph_inputs);

        let metadata = OnnxModelDescriptor {
            graph_inputs,
            graph_outputs,
            initializers,
            ops,
            hidden_size,
            num_layers,
            num_heads,
            vocab_size,
        };

        Ok(Self {
            file_path: path.to_path_buf(),
            file_data,
            tensors,
            name_index,
            raw_offsets,
            metadata,
        })
    }

    /// Access the parsed model metadata.
    pub fn metadata(&self) -> &OnnxModelDescriptor {
        &self.metadata
    }

    /// Read the raw bytes of an initializer tensor by name.
    fn read_tensor_raw(&self, name: &str) -> Result<Vec<u8>, String> {
        let idx = self
            .name_index
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' not found in ONNX file"))?;
        let info = &self.tensors[*idx];

        // First try raw_data (most common for modern ONNX exports).
        if let Some(&(offset, len)) = self.raw_offsets.get(name) {
            if offset + len <= self.file_data.len() {
                return Ok(self.file_data[offset..offset + len].to_vec());
            }
        }

        // Fall back: reconstruct from typed repeated fields.
        // (float_data field 5, int32_data field 6, int64_data field 9)
        // Re-parse the TensorProto for typed data.
        self.extract_typed_tensor_data(name, info)
    }

    /// Re-extract typed data (float_data / int32_data / int64_data) by
    /// re-parsing the TensorProto section. This is the fallback when raw_data
    /// is absent (older ONNX files).
    fn extract_typed_tensor_data(
        &self,
        name: &str,
        info: &OnnxTensorInfo,
    ) -> Result<Vec<u8>, String> {
        // We need to find the tensor proto section again. Since we already parsed
        // it once, we can search by re-scanning for the field-7 (name) match.
        // But a simpler approach: re-read the file and scan GraphProto initializers.
        let mut parser = ProtoParser::new(&self.file_data);
        let field_entries = parser.collect_initializer_sections();
        for (offset, len) in &field_entries {
            if *offset + *len > self.file_data.len() {
                continue;
            }
            let chunk = &self.file_data[*offset..*offset + *len];
            // Check if this TensorProto has our name at field 7.
            if let Some(n) = extract_string_field(chunk, 7) {
                if n == name {
                    // Found it — extract typed data.
                    return extract_typed_tensor_bytes(chunk, info);
                }
            }
        }
        Err(format!(
            "cannot locate tensor '{}' raw or typed data in ONNX file",
            name
        ))
    }
}

impl TensorProvider for OnnxModelProvider {
    fn list_tensors(&self) -> Result<Vec<TensorInfo>, String> {
        let infos: Vec<TensorInfo> = self
            .tensors
            .iter()
            .map(|t| TensorInfo {
                name: t.name.clone(),
                shape: t.dimensions.clone(),
                dtype: format!("{:?}", t.data_type).to_lowercase(),
                size_bytes: t.dimensions.iter().product::<usize>() as u64
                    * match t.data_type {
                        OnnxDataType::Float => 4,
                        OnnxDataType::Float16 => 2,
                        OnnxDataType::Int8 => 1,
                        OnnxDataType::Int32 => 4,
                        OnnxDataType::Int64 => 8,
                        OnnxDataType::Undefined => 4,
                    },
            })
            .collect();
        Ok(infos)
    }

    fn open_tensor(&self, name: &str) -> Result<Box<dyn TensorReader>, String> {
        let raw = self.read_tensor_raw(name)?;
        let idx = self
            .name_index
            .get(name)
            .ok_or_else(|| format!("tensor '{name}' not found"))?;
        let shape = self.tensors[*idx].dimensions.clone();
        Ok(Box::new(OnnxTensorReader {
            data: raw,
            shape,
            pos: 0,
        }))
    }
}

// ── TensorReader ─────────────────────────────────────────────────────────

/// Reader that yields bytes from an ONNX initializer tensor.
struct OnnxTensorReader {
    data: Vec<u8>,
    shape: Vec<usize>,
    pos: usize,
}

impl TensorReader for OnnxTensorReader {
    fn read_chunk(&mut self, buffer: &mut [u8]) -> Result<usize, String> {
        let remaining = self.data.len() - self.pos;
        if remaining == 0 {
            return Ok(0);
        }
        let to_read = buffer.len().min(remaining);
        buffer[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
        self.pos += to_read;
        Ok(to_read)
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn size_bytes(&self) -> u64 {
        self.data.len() as u64
    }
}

// ── Minimal protobuf binary decoder ──────────────────────────────────────

/// Wire types for protobuf field encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireType {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    StartGroup = 3,
    EndGroup = 4,
    Fixed32 = 5,
}

impl WireType {
    fn from_u8(v: u8) -> Option<WireType> {
        match v {
            0 => Some(WireType::Varint),
            1 => Some(WireType::Fixed64),
            2 => Some(WireType::LengthDelimited),
            3 => Some(WireType::StartGroup),
            4 => Some(WireType::EndGroup),
            5 => Some(WireType::Fixed32),
            _ => None,
        }
    }
}

/// Internal parsed representation of an ONNX model.
struct ParsedModel {
    graph_input_names: Vec<String>,
    graph_output_names: Vec<String>,
    initializers: Vec<OnnxTensorInfo>,
    operators: Vec<OnnxOperator>,
    /// (tensor_name, offset, length) for each initializer's raw_data blob.
    raw_data_offsets: Vec<(String, usize, usize)>,
}

/// Streaming read cursor over a byte slice.
struct ProtoParser<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtoParser<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Read a varint-encoded integer.
    fn read_varint(&mut self) -> Result<u64, String> {
        let mut value: u64 = 0;
        let mut shift: u32 = 0;
        loop {
            if self.pos >= self.data.len() {
                return Err("unexpected EOF in varint".to_string());
            }
            let byte = self.data[self.pos];
            self.pos += 1;
            value |= ((byte & 0x7F) as u64) << shift;
            shift += 7;
            if shift > 63 {
                return Err("varint too long".to_string());
            }
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
    }

    /// Read a fixed 32-bit little-endian value.
    fn read_fixed32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() {
            return Err("unexpected EOF in fixed32".to_string());
        }
        let bytes: [u8; 4] = self.data[self.pos..self.pos + 4].try_into().unwrap();
        self.pos += 4;
        Ok(u32::from_le_bytes(bytes))
    }

    /// Read a fixed 64-bit little-endian value.
    fn read_fixed64(&mut self) -> Result<u64, String> {
        if self.pos + 8 > self.data.len() {
            return Err("unexpected EOF in fixed64".to_string());
        }
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        Ok(u64::from_le_bytes(bytes))
    }

    /// Read a length-delimited field: varint length then bytes.
    fn read_length_delimited(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() {
            return Err(format!(
                "unexpected EOF in length-delimited field: need {} bytes at pos {} of {}",
                len,
                self.pos,
                self.data.len()
            ));
        }
        let slice = &self.data[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    /// Read the next field tag: `(field_number, wire_type)`.
    /// Returns `None` when we reach the end of the message.
    fn read_tag(&mut self) -> Result<Option<(u64, WireType)>, String> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let varint = self.read_varint()?;
        let field_number = varint >> 3;
        let wire_type_val = (varint & 0x07) as u8;
        let wire_type = WireType::from_u8(wire_type_val)
            .ok_or_else(|| format!("unknown wire type {wire_type_val}"))?;
        Ok(Some((field_number, wire_type)))
    }

    /// Skip a field's value (advance past its bytes).
    fn skip_field(&mut self, wire_type: WireType) -> Result<(), String> {
        match wire_type {
            WireType::Varint => {
                self.read_varint()?;
            }
            WireType::Fixed64 => {
                self.read_fixed64()?;
            }
            WireType::LengthDelimited => {
                self.read_length_delimited()?;
            }
            WireType::StartGroup => {
                // Skip everything until matching EndGroup.
                let mut depth = 1u32;
                while depth > 0 {
                    if self.pos >= self.data.len() {
                        return Err("unexpected EOF skipping group".to_string());
                    }
                    let tag = self.read_varint()?;
                    let wt = WireType::from_u8((tag & 0x07) as u8)
                        .ok_or_else(|| format!("unknown wire type in group: {}", tag & 0x07))?;
                    match wt {
                        WireType::StartGroup => depth += 1,
                        WireType::EndGroup => depth -= 1,
                        _ => self.skip_field(wt)?,
                    }
                }
            }
            WireType::EndGroup => {
                // Should not encounter EndGroup as a top-level skip target.
                return Err("unexpected EndGroup".to_string());
            }
            WireType::Fixed32 => {
                self.read_fixed32()?;
            }
        }
        Ok(())
    }

    // ── High-level parsing ─────────────────────────────────────────

    /// Parse the top-level ModelProto message.
    fn parse_model_proto(&mut self) -> Result<ParsedModel, String> {
        let mut graph_input_names: Vec<String> = Vec::new();
        let mut graph_output_names: Vec<String> = Vec::new();
        let mut initializers: Vec<OnnxTensorInfo> = Vec::new();
        let mut operators: Vec<OnnxOperator> = Vec::new();
        let mut raw_data_offsets: Vec<(String, usize, usize)> = Vec::new();

        loop {
            let tag = self.read_tag()?;
            let (field_number, wire_type) = match tag {
                Some(t) => t,
                None => break,
            };

            match field_number {
                // ModelProto fields (we only care about graph = field 7):
                // 1: int64 ir_version
                // 2: repeated OperatorSetIdProto opset_import
                // 4: string producer_name
                // 5: string producer_version
                // 6: string domain
                // 7: GraphProto graph
                7 if wire_type == WireType::LengthDelimited => {
                    let graph_bytes = self.read_length_delimited()?;
                    let mut gp = ProtoParser::new(graph_bytes);
                    let (ins, outs, inits, ops, raw_offsets) = gp.parse_graph_proto()?;
                    graph_input_names = ins;
                    graph_output_names = outs;
                    initializers = inits;
                    operators = ops;
                    raw_data_offsets = raw_offsets;
                }
                _ => {
                    self.skip_field(wire_type)?;
                }
            }
        }

        Ok(ParsedModel {
            graph_input_names,
            graph_output_names,
            initializers,
            operators,
            raw_data_offsets,
        })
    }

    /// Parse a GraphProto message.
    fn parse_graph_proto(
        &mut self,
    ) -> Result<
        (
            Vec<String>,
            Vec<String>,
            Vec<OnnxTensorInfo>,
            Vec<OnnxOperator>,
            Vec<(String, usize, usize)>,
        ),
        String,
    > {
        let mut input_names: Vec<String> = Vec::new();
        let mut output_names: Vec<String> = Vec::new();
        let mut initializers: Vec<OnnxTensorInfo> = Vec::new();
        let mut operators: Vec<OnnxOperator> = Vec::new();
        let mut raw_data_offsets: Vec<(String, usize, usize)> = Vec::new();

        loop {
            let tag = self.read_tag()?;
            let (field_number, wire_type) = match tag {
                Some(t) => t,
                None => break,
            };

            match field_number {
                // GraphProto fields:
                // 1: string name (skip)
                // 2: repeated NodeProto node
                // 5: repeated TensorProto initializer
                // 11: repeated ValueInfoProto input
                // 12: repeated ValueInfoProto output
                11 if wire_type == WireType::LengthDelimited => {
                    let vi_bytes = self.read_length_delimited()?;
                    let name = extract_value_info_name(vi_bytes);
                    if let Some(n) = name {
                        input_names.push(n);
                    }
                }
                12 if wire_type == WireType::LengthDelimited => {
                    let vi_bytes = self.read_length_delimited()?;
                    let name = extract_value_info_name(vi_bytes);
                    if let Some(n) = name {
                        output_names.push(n);
                    }
                }
                5 if wire_type == WireType::LengthDelimited => {
                    let tensor_bytes = self.read_length_delimited()?;
                    let _save_offset = self.pos; // offset is in the outer buffer's frame
                    let outer_offset = self.data.as_ptr() as usize;
                    let tensor_start = tensor_bytes.as_ptr() as usize;
                    let offset_in_file = tensor_start.wrapping_sub(outer_offset);
                    let (info, raw_offset, raw_len) =
                        parse_tensor_proto(tensor_bytes, offset_in_file)?;
                    initializers.push(info);
                    if let (Some(ro), Some(rl)) = (raw_offset, raw_len) {
                        raw_data_offsets.push((initializers.last().unwrap().name.clone(), ro, rl));
                    }
                }
                2 if wire_type == WireType::LengthDelimited => {
                    let node_bytes = self.read_length_delimited()?;
                    let op = parse_node_proto(node_bytes);
                    operators.push(op);
                }
                _ => {
                    self.skip_field(wire_type)?;
                }
            }
        }

        Ok((
            input_names,
            output_names,
            initializers,
            operators,
            raw_data_offsets,
        ))
    }

    /// Scan the file for all TensorProto sections (for re-parse fallback).
    /// Returns `(offset_into_file, length)` for each TensorProto body.
    fn collect_initializer_sections(&mut self) -> Vec<(usize, usize)> {
        let mut sections = Vec::new();
        self.pos = 0;

        // Navigate: ModelProto -> field 7 (GraphProto) -> field 5 (TensorProto)
        loop {
            let tag = match self.read_tag() {
                Ok(Some(t)) => t,
                _ => break,
            };
            let (field_number, wire_type) = tag;

            if field_number == 7 && wire_type == WireType::LengthDelimited {
                let graph_bytes = match self.read_length_delimited() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut gp = ProtoParser::new(graph_bytes);
                loop {
                    let gtag = match gp.read_tag() {
                        Ok(Some(t)) => t,
                        _ => break,
                    };
                    let (gf, gw) = gtag;
                    if gf == 5 && gw == WireType::LengthDelimited {
                        let tensor_bytes = match gp.read_length_delimited() {
                            Ok(b) => b,
                            Err(_) => break,
                        };
                        let start = tensor_bytes.as_ptr() as usize;
                        let base = self.data.as_ptr() as usize;
                        let offset = start.wrapping_sub(base);
                        sections.push((offset, tensor_bytes.len()));
                    } else {
                        let _ = gp.skip_field(gw);
                    }
                }
                break;
            } else {
                let _ = self.skip_field(wire_type);
            }
        }

        sections
    }
}

// ── ValueInfoProto parser ────────────────────────────────────────────────

/// Extract the `name` field from a ValueInfoProto message.
fn extract_value_info_name(data: &[u8]) -> Option<String> {
    let mut p = ProtoParser::new(data);
    loop {
        match p.read_tag() {
            Ok(Some((field_number, wire_type))) => {
                if field_number == 1 && wire_type == WireType::LengthDelimited {
                    match p.read_length_delimited() {
                        Ok(s) => return Some(String::from_utf8_lossy(s).to_string()),
                        Err(_) => return None,
                    }
                } else {
                    let _ = p.skip_field(wire_type);
                }
            }
            _ => return None,
        }
    }
}

// ── TensorProto parser ───────────────────────────────────────────────────

/// Parse a TensorProto message.
///
/// Returns `(tensor_info, raw_data_offset_in_file, raw_data_length)`.
/// `raw_data_offset` is relative to the start of the file (computed from
/// `outer_file_offset` + relative offset within this TensorProto).
fn parse_tensor_proto(
    data: &[u8],
    offset_in_file: usize,
) -> Result<(OnnxTensorInfo, Option<usize>, Option<usize>), String> {
    let mut dims: Vec<i64> = Vec::new();
    let mut data_type: i32 = 0;
    let mut name: Option<String> = None;
    let mut raw_data_offset: Option<usize> = None;
    let mut raw_data_len: Option<usize> = None;
    let mut float_data: Vec<f32> = Vec::new();
    let mut int32_data: Vec<i32> = Vec::new();
    let mut int64_data: Vec<i64> = Vec::new();

    let mut p = ProtoParser::new(data);
    loop {
        let tag = match p.read_tag() {
            Ok(Some(t)) => t,
            Ok(None) => break,
            Err(_) => break,
        };
        let (field_number, wire_type) = tag;

        match field_number {
            // field 1: repeated int64 dims (packed varint)
            1 if wire_type == WireType::LengthDelimited => {
                // Packed repeated int64.
                let bytes = match p.read_length_delimited() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut vp = ProtoParser::new(bytes);
                loop {
                    match vp.read_varint() {
                        Ok(v) => dims.push(v as i64),
                        Err(_) => break,
                    }
                }
            }
            1 if wire_type == WireType::Varint => {
                // Non-packed repeated int64 (unusual but valid).
                match p.read_varint() {
                    Ok(v) => dims.push(v as i64),
                    Err(_) => break,
                }
            }
            // field 2: int32 data_type
            2 if wire_type == WireType::Varint => match p.read_varint() {
                Ok(v) => data_type = v as i32,
                Err(_) => break,
            },
            // field 5: repeated float float_data
            5 if wire_type == WireType::LengthDelimited => {
                // Packed float data.
                let bytes = match p.read_length_delimited() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut fp = ProtoParser::new(bytes);
                while fp.remaining() >= 4 {
                    match fp.read_fixed32() {
                        Ok(v) => float_data.push(f32::from_le_bytes(v.to_le_bytes())),
                        Err(_) => break,
                    }
                }
            }
            5 if wire_type == WireType::Fixed32 => match p.read_fixed32() {
                Ok(v) => float_data.push(f32::from_le_bytes(v.to_le_bytes())),
                Err(_) => break,
            },
            // field 6: repeated int32 int32_data
            6 if wire_type == WireType::LengthDelimited => {
                let bytes = match p.read_length_delimited() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut fp = ProtoParser::new(bytes);
                loop {
                    match fp.read_varint() {
                        Ok(v) => int32_data.push(v as i32),
                        Err(_) => break,
                    }
                }
            }
            6 if wire_type == WireType::Varint => match p.read_varint() {
                Ok(v) => int32_data.push(v as i32),
                Err(_) => break,
            },
            // field 7: string name
            7 if wire_type == WireType::LengthDelimited => match p.read_length_delimited() {
                Ok(s) => name = Some(String::from_utf8_lossy(s).to_string()),
                Err(_) => break,
            },
            // field 9: repeated int64 int64_data
            9 if wire_type == WireType::LengthDelimited => {
                let bytes = match p.read_length_delimited() {
                    Ok(b) => b,
                    Err(_) => break,
                };
                let mut fp = ProtoParser::new(bytes);
                loop {
                    match fp.read_varint() {
                        Ok(v) => int64_data.push(v as i64),
                        Err(_) => break,
                    }
                }
            }
            9 if wire_type == WireType::Varint => match p.read_varint() {
                Ok(v) => int64_data.push(v as i64),
                Err(_) => break,
            },
            // field 12: bytes raw_data (the raw tensor content)
            12 if wire_type == WireType::LengthDelimited => {
                let raw_start = offset_in_file + p.pos;
                match p.read_length_delimited() {
                    Ok(slice) => {
                        raw_data_offset = Some(raw_start);
                        raw_data_len = Some(slice.len());
                    }
                    Err(_) => break,
                }
            }
            _ => {
                let _ = p.skip_field(wire_type);
            }
        }
    }

    let name = name.unwrap_or_else(|| "<unnamed>".to_string());
    let dimensions: Vec<usize> = dims.into_iter().map(|d| d as usize).collect();
    let dt = onnx_data_type_to_enum(data_type);

    // If raw_data wasn't present, compute size from typed data fields.
    if raw_data_offset.is_none() {
        if !float_data.is_empty() {
            let _bytes: Vec<u8> = float_data.iter().flat_map(|f| f.to_le_bytes()).collect();
            // Store synthetic raw_data by writing through — we can't store
            // a synthetic offset in the file. Instead we'll handle this in
            // read_tensor_raw by re-parsing.
        }
    }

    Ok((
        OnnxTensorInfo {
            name,
            dimensions,
            data_type: dt,
        },
        raw_data_offset,
        raw_data_len,
    ))
}

/// Extract typed tensor bytes from a TensorProto section (fallback when
/// raw_data field 12 is absent).
fn extract_typed_tensor_bytes(data: &[u8], info: &OnnxTensorInfo) -> Result<Vec<u8>, String> {
    let mut result = Vec::new();
    let mut p = ProtoParser::new(data);

    loop {
        match p.read_tag() {
            Ok(Some((field_number, wire_type))) => match field_number {
                // field 5: float_data
                5 if wire_type == WireType::LengthDelimited => {
                    let bytes = p.read_length_delimited()?;
                    let mut fp = ProtoParser::new(bytes);
                    while fp.remaining() >= 4 {
                        let v = fp.read_fixed32()?;
                        result.extend_from_slice(&v.to_le_bytes());
                    }
                }
                5 if wire_type == WireType::Fixed32 => {
                    let v = p.read_fixed32()?;
                    result.extend_from_slice(&v.to_le_bytes());
                }
                // field 6: int32_data
                6 if wire_type == WireType::LengthDelimited => {
                    let bytes = p.read_length_delimited()?;
                    let mut fp = ProtoParser::new(bytes);
                    loop {
                        match fp.read_varint() {
                            Ok(v) => result.extend_from_slice(&(v as i32).to_le_bytes()),
                            Err(_) => break,
                        }
                    }
                }
                6 if wire_type == WireType::Varint => {
                    let v = p.read_varint()?;
                    result.extend_from_slice(&(v as i32).to_le_bytes());
                }
                // field 9: int64_data
                9 if wire_type == WireType::LengthDelimited => {
                    let bytes = p.read_length_delimited()?;
                    let mut fp = ProtoParser::new(bytes);
                    loop {
                        match fp.read_varint() {
                            Ok(v) => result.extend_from_slice(&v.to_le_bytes()),
                            Err(_) => break,
                        }
                    }
                }
                9 if wire_type == WireType::Varint => {
                    let v = p.read_varint()?;
                    result.extend_from_slice(&v.to_le_bytes());
                }
                _ => {
                    p.skip_field(wire_type)?;
                }
            },
            Ok(None) => break,
            Err(_) => break,
        }
    }

    if result.is_empty() {
        return Err(format!("tensor '{}' has no typed data fields", info.name));
    }
    Ok(result)
}

// ── NodeProto parser ─────────────────────────────────────────────────────

/// Parse a NodeProto message into an `OnnxOperator`.
fn parse_node_proto(data: &[u8]) -> OnnxOperator {
    let mut inputs: Vec<String> = Vec::new();
    let mut outputs: Vec<String> = Vec::new();
    let mut op_type: Option<String> = None;

    let mut p = ProtoParser::new(data);
    loop {
        match p.read_tag() {
            Ok(Some((field_number, wire_type))) => match field_number {
                // field 1: repeated string input
                1 if wire_type == WireType::LengthDelimited => {
                    if let Ok(s) = p.read_length_delimited() {
                        inputs.push(String::from_utf8_lossy(s).to_string());
                    }
                }
                // field 2: repeated string output
                2 if wire_type == WireType::LengthDelimited => {
                    if let Ok(s) = p.read_length_delimited() {
                        outputs.push(String::from_utf8_lossy(s).to_string());
                    }
                }
                // field 4: string op_type
                4 if wire_type == WireType::LengthDelimited => {
                    if let Ok(s) = p.read_length_delimited() {
                        op_type = Some(String::from_utf8_lossy(s).to_string());
                    }
                }
                _ => {
                    let _ = p.skip_field(wire_type);
                }
            },
            _ => break,
        }
    }

    let op_type = op_type.unwrap_or_else(|| "Unknown".to_string());

    match op_type.as_str() {
        "MatMul" | "MatMulInteger" => OnnxOperator::MatMul {
            input: inputs.first().cloned().unwrap_or_default(),
            weight: inputs.get(1).cloned().unwrap_or_default(),
            output: outputs.first().cloned().unwrap_or_default(),
        },
        "Add" => OnnxOperator::Add {
            inputs,
            output: outputs.first().cloned().unwrap_or_default(),
        },
        "Reshape" => OnnxOperator::Reshape {
            input: inputs.first().cloned().unwrap_or_default(),
            shape: inputs.get(1).cloned().unwrap_or_default(),
            output: outputs.first().cloned().unwrap_or_default(),
        },
        _ => OnnxOperator::Other {
            op_type,
            inputs,
            outputs,
        },
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Map ONNX TensorProto.DataType enum value to our internal type.
fn onnx_data_type_to_enum(dt: i32) -> OnnxDataType {
    match dt {
        1 => OnnxDataType::Float,
        10 => OnnxDataType::Float16,
        3 => OnnxDataType::Int8,
        6 => OnnxDataType::Int32,
        7 => OnnxDataType::Int64,
        _ => OnnxDataType::Undefined,
    }
}

/// Extract a string field with the given field number from a protobuf message.
fn extract_string_field(data: &[u8], field_num: u64) -> Option<String> {
    let mut p = ProtoParser::new(data);
    loop {
        match p.read_tag() {
            Ok(Some((fn_, wt))) => {
                if fn_ == field_num && wt == WireType::LengthDelimited {
                    return match p.read_length_delimited() {
                        Ok(s) => Some(String::from_utf8_lossy(s).to_string()),
                        Err(_) => None,
                    };
                } else {
                    let _ = p.skip_field(wt);
                }
            }
            _ => return None,
        }
    }
}

/// Infer transformer architecture dimensions from the tensor collection.
fn infer_model_dims(
    tensors: &[OnnxTensorInfo],
    _inputs: &[String],
) -> (usize, usize, usize, usize) {
    let mut hidden_size: usize = 0;
    let mut num_layers: usize = 0;
    let mut num_heads: usize = 0;
    let mut vocab_size: usize = 0;

    // Detect hidden_size from weight matrix shapes (e.g. attention output).
    for t in tensors {
        let name_lower = t.name.to_lowercase();
        let dims = &t.dimensions;

        // Vocabulary size: look for embedding tables or lm_head weights.
        if dims.len() == 2 {
            if name_lower.contains("embed") || name_lower.contains("tok_embeddings") {
                vocab_size = dims[0].max(dims[1]);
                hidden_size = hidden_size.max(dims[0].min(dims[1]));
            }
            if name_lower.contains("lm_head") || name_lower.contains("embed_out") {
                vocab_size = dims[0].max(dims[1]);
            }
            // Detect hidden_size from attention output weight (out_proj / o_proj).
            if (name_lower.contains("out_proj")
                || name_lower.contains("o_proj")
                || name_lower.contains("attention.output"))
                && !name_lower.contains("weight_quant")
            {
                hidden_size = hidden_size.max(dims[1]);
            }
            // Detect from query/key/value projection weights.
            if (name_lower.contains("q_proj")
                || name_lower.contains("k_proj")
                || name_lower.contains("v_proj")
                || name_lower.contains("query")
                || name_lower.contains("key")
                || name_lower.contains("value"))
                && !name_lower.contains("weight_quant")
            {
                // q_proj has shape (hidden_size, hidden_size) or (num_heads * head_dim, hidden_size)
                hidden_size = hidden_size.max(dims[1]);
            }
        }

        // Count layers by counting self-attention or MLP layers.
        if name_lower.contains(".attention.")
            || name_lower.contains("self_attn.")
            || name_lower.contains("layers.")
        {
            // Try to extract layer number.
            if let Some(layer_str) = name_lower.split('.').find(|s| s.parse::<usize>().is_ok()) {
                if let Ok(n) = layer_str.parse::<usize>() {
                    num_layers = num_layers.max(n + 1);
                }
            }
        }
        // Also detect NumBlocks.* (Microsoft's format).
        if name_lower.contains("numblocks") || name_lower.contains("num_layers") {
            if dims.len() == 1 && dims[0] > 0 {
                num_layers = dims[0];
            }
        }

        // Detect num_heads from weight shapes.
        if name_lower.contains("num_heads") || name_lower.contains("n_head") {
            if !dims.is_empty() && dims[0] > 0 {
                num_heads = dims[0];
            }
        }
        if name_lower.contains("q_proj") || name_lower.contains("query.weight") {
            if dims.len() == 2 && hidden_size > 0 && dims[0] > 0 {
                // q_proj shape = (hidden_size, hidden_size) or (num_heads*head_dim, hidden_size)
                // Try to infer num_heads from the ratio.
                if dims[0] >= hidden_size && dims[0] % hidden_size == 0 {
                    // Multi-query: dims[0] / hidden_size = num_heads for GQA
                } else if hidden_size > 0 && dims[0] % 64 == 0 {
                    let inferred = dims[0] / 64;
                    num_heads = num_heads.max(inferred);
                }
            }
        }
    }

    // Fallback: if we found hidden_size, make sure num_heads is plausible.
    if hidden_size > 0 && num_heads == 0 {
        // Common head counts: 8, 12, 16, 32
        for h in &[32usize, 16, 12, 8] {
            if hidden_size % h == 0 {
                num_heads = *h;
                break;
            }
        }
    }
    if num_heads == 0 {
        num_heads = 32; // conservative default
    }

    // Fallback for layers.
    if num_layers == 0 {
        num_layers = 32; // conservative default for medium-sized models
    }

    (hidden_size, num_layers, num_heads, vocab_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_decoding() {
        let data = [0x96, 0x01]; // 150 in varint
        let mut p = ProtoParser::new(&data);
        assert_eq!(p.read_varint().unwrap(), 150);
    }

    #[test]
    fn test_small_varint() {
        let data = [42];
        let mut p = ProtoParser::new(&data);
        assert_eq!(p.read_varint().unwrap(), 42);
    }

    #[test]
    fn test_tag_decoding() {
        // Field 1, wire type 2 (length-delimited): tag = (1 << 3) | 2 = 10
        let data = [0x0A, 0x03, 0x41, 0x42, 0x43];
        let mut p = ProtoParser::new(&data);
        let tag = p.read_tag().unwrap().unwrap();
        assert_eq!(tag.0, 1); // field 1
        assert_eq!(tag.1, WireType::LengthDelimited); // wire type 2
        let val = p.read_length_delimited().unwrap();
        assert_eq!(val, b"ABC");
    }

    #[test]
    fn test_length_delimited() {
        let data = [3, 0x41, 0x42, 0x43];
        let mut p = ProtoParser::new(&data);
        let val = p.read_length_delimited().unwrap();
        assert_eq!(val, b"ABC");
    }

    #[test]
    fn test_onnx_data_type_enum() {
        assert_eq!(onnx_data_type_to_enum(1), OnnxDataType::Float);
        assert_eq!(onnx_data_type_to_enum(10), OnnxDataType::Float16);
        assert_eq!(onnx_data_type_to_enum(3), OnnxDataType::Int8);
        assert_eq!(onnx_data_type_to_enum(6), OnnxDataType::Int32);
        assert_eq!(onnx_data_type_to_enum(7), OnnxDataType::Int64);
        assert_eq!(onnx_data_type_to_enum(0), OnnxDataType::Undefined);
    }

    #[test]
    fn test_extract_string_field() {
        // A minimal proto with field 7 (string) = "test_tensor"
        // Tag for field 7, LEN: (7 << 3) | 2 = 58 = 0x3A
        // Length: 11
        // Data: "test_tensor"
        let data = [
            0x3A, 0x0B, b't', b'e', b's', b't', b'_', b't', b'e', b'n', b's', b'o', b'r',
        ];
        let result = extract_string_field(&data, 7);
        assert_eq!(result, Some("test_tensor".to_string()));
    }

    #[test]
    fn test_infer_model_dims() {
        let tensors = vec![
            OnnxTensorInfo {
                name: "model.layers.0.attention.q_proj.weight".to_string(),
                dimensions: vec![4096, 4096],
                data_type: OnnxDataType::Float,
            },
            OnnxTensorInfo {
                name: "model.layers.0.attention.o_proj.weight".to_string(),
                dimensions: vec![4096, 4096],
                data_type: OnnxDataType::Float,
            },
            OnnxTensorInfo {
                name: "model.embed_tokens.weight".to_string(),
                dimensions: vec![32000, 4096],
                data_type: OnnxDataType::Float,
            },
        ];
        let (hidden, layers, heads, vocab) = infer_model_dims(&tensors, &[]);
        assert_eq!(hidden, 4096);
        assert_eq!(vocab, 32000);
        assert!(layers >= 1);
        assert!(heads > 0);
    }
}

// ── Public API re-exports ────────────────────────────────────────────
