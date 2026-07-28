//! `MatrixWeightBindingV1` — the per-tensor format contract between the
//! compiler's packing pass and every runtime dispatch path.
//!
//! Each weight matrix gets a binding after the admission pipeline selects a
//! representation. The binding stores segment offsets, dimensions, kernel
//! ABI digest, and the metadata/sidecar/residual descriptors. The wire
//! format is canonical little-endian; `MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH`
//! is the on-disk size.
//!
//! Authority: per-matrix packing contract. Pure data + canonical
//! (de)serialisation. No engine-coupled dependencies.

/// V1 MatrixWeightBinding wire size in bytes (canonical little-endian).
pub const MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH: usize = 2
    + 4
    + 16
    + 1
    + 2
    + 32
    + 4
    + 4
    + 2
    + 4
    + 2
    + 1
    + 1
    + 1
    + 8
    + 8
    + 4
    + 1
    + 8
    + 8
    + 2
    + 1
    + 8
    + 8
    + 1
    + 1
    + 4
    + 1
    + 8
    + 8
    + 4;

/// V1 MatrixWeightBinding with canonical LE serialization.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MatrixWeightBindingV1 {
    /// Wire version (must be 1).
    pub binding_wire_version: u16,
    /// Index into the bindings array.
    pub matrix_id: u32,
    /// Stable tensor identifier (16 bytes).
    pub tensor_id: [u8; 16],
    /// Runtime representation class discriminant (0..=3).
    pub representation: u8,
    /// Representation-specific version (e.g. 640 for tile640).
    pub representation_version: u16,
    /// SHA-256 digest of the kernel ABI this binding targets.
    pub kernel_abi_digest: [u8; 32],
    /// Input feature count.
    pub in_features: u32,
    /// Output feature count.
    pub out_features: u32,
    /// Reduction tile size (640 for tile640, 0 for RawF32).
    pub reduction_tile_size: u16,
    /// Number of tiles per output channel.
    pub tiles_per_output_channel: u32,
    /// Number of tail reduction elements (non-tile-aligned remainder).
    pub tail_reduction_count: u16,
    /// Macro layout discriminant.
    pub macro_layout: u8,
    /// Tail-handling policy discriminant.
    pub tail_handling: u8,
    /// Segment index for the codes payload.
    pub code_segment: u8,
    /// Byte offset into the codes segment.
    pub code_offset: u64,
    /// Byte length of the codes payload.
    pub code_length: u64,
    /// Stride between consecutive tiles in the codes segment.
    pub code_tile_stride_bytes: u32,
    /// Segment index for the tile metadata payload.
    pub metadata_segment: u8,
    /// Byte offset into the metadata segment.
    pub metadata_offset: u64,
    /// Byte length of the metadata payload.
    pub metadata_length: u64,
    /// Stride between consecutive tiles in the metadata segment.
    pub metadata_tile_stride_bytes: u16,
    /// Segment index for the reduction-axis sidecar (0xFF = none).
    pub sidecar_segment: u8,
    /// Byte offset into the sidecar segment.
    pub sidecar_offset: u64,
    /// Byte length of the sidecar payload.
    pub sidecar_length: u64,
    /// Sidecar kind discriminant.
    pub sidecar_kind: u8,
    /// Per-element encoding format of the sidecar.
    pub sidecar_element_format: u8,
    /// Number of sidecar values.
    pub sidecar_count: u32,
    /// Segment index for the residual payload (0xFF = none).
    pub residual_segment: u8,
    /// Byte offset into the residual segment.
    pub residual_offset: u64,
    /// Byte length of the residual payload.
    pub residual_length: u64,
    /// Required alignment in bytes (for GPU / DMA).
    pub required_alignment_bytes: u32,
}

impl MatrixWeightBindingV1 {
    /// Construct a new `MatrixWeightBindingV1` and validate it.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        binding_wire_version: u16,
        matrix_id: u32,
        tensor_id: [u8; 16],
        representation: u8,
        representation_version: u16,
        kernel_abi_digest: [u8; 32],
        in_features: u32,
        out_features: u32,
        reduction_tile_size: u16,
        tiles_per_output_channel: u32,
        tail_reduction_count: u16,
        macro_layout: u8,
        tail_handling: u8,
        code_segment: u8,
        code_offset: u64,
        code_length: u64,
        code_tile_stride_bytes: u32,
        metadata_segment: u8,
        metadata_offset: u64,
        metadata_length: u64,
        metadata_tile_stride_bytes: u16,
        sidecar_segment: u8,
        sidecar_offset: u64,
        sidecar_length: u64,
        sidecar_kind: u8,
        sidecar_element_format: u8,
        sidecar_count: u32,
        residual_segment: u8,
        residual_offset: u64,
        residual_length: u64,
        required_alignment_bytes: u32,
    ) -> Result<Self, String> {
        let bind = Self {
            binding_wire_version,
            matrix_id,
            tensor_id,
            representation,
            representation_version,
            kernel_abi_digest,
            in_features,
            out_features,
            reduction_tile_size,
            tiles_per_output_channel,
            tail_reduction_count,
            macro_layout,
            tail_handling,
            code_segment,
            code_offset,
            code_length,
            code_tile_stride_bytes,
            metadata_segment,
            metadata_offset,
            metadata_length,
            metadata_tile_stride_bytes,
            sidecar_segment,
            sidecar_offset,
            sidecar_length,
            sidecar_kind,
            sidecar_element_format,
            sidecar_count,
            residual_segment,
            residual_offset,
            residual_length,
            required_alignment_bytes,
        };
        bind.validate()?;
        Ok(bind)
    }

    /// Validate structural invariants for this binding.
    pub fn validate(&self) -> Result<(), String> {
        if self.binding_wire_version != 1 {
            return Err(format!(
                "binding_wire_version must be 1, got {}",
                self.binding_wire_version
            ));
        }
        if self.representation > 3 {
            return Err(format!(
                "representation must be 0..=3 (valid RuntimeRepresentationClass), got {}",
                self.representation
            ));
        }
        if self.representation == 3 {
            // RawF32 special rules
            if self.reduction_tile_size != 0 {
                return Err(format!(
                    "RawF32 (representation==3) requires reduction_tile_size == 0, got {}",
                    self.reduction_tile_size
                ));
            }
            if self.tiles_per_output_channel != 0 {
                return Err(format!(
                    "RawF32 (representation==3) requires tiles_per_output_channel == 0, got {}",
                    self.tiles_per_output_channel
                ));
            }
            let expected_code_len = (self.in_features as u64)
                .checked_mul(self.out_features as u64)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| "overflow computing expected code_length for RawF32".to_string())?;
            if self.code_length != expected_code_len {
                return Err(format!(
                    "RawF32 (representation==3) requires code_length == in_features * out_features * 4, got {}, expected {}",
                    self.code_length, expected_code_len
                ));
            }
            if self.metadata_length != 0 {
                return Err(format!(
                    "RawF32 (representation==3) requires metadata_length == 0, got {}",
                    self.metadata_length
                ));
            }
            if self.sidecar_length != 0 {
                return Err(format!(
                    "RawF32 (representation==3) requires sidecar_length == 0, got {}",
                    self.sidecar_length
                ));
            }
        }
        Ok(())
    }
}

/// Serialize a `MatrixWeightBindingV1` in canonical little-endian format.
pub fn write_matrix_weight_binding_v1_le<W: std::io::Write>(
    w: &mut W,
    b: &MatrixWeightBindingV1,
) -> std::io::Result<()> {
    w.write_all(&b.binding_wire_version.to_le_bytes())?;
    w.write_all(&b.matrix_id.to_le_bytes())?;
    w.write_all(&b.tensor_id)?;
    w.write_all(&[b.representation])?;
    w.write_all(&b.representation_version.to_le_bytes())?;
    w.write_all(&b.kernel_abi_digest)?;
    w.write_all(&b.in_features.to_le_bytes())?;
    w.write_all(&b.out_features.to_le_bytes())?;
    w.write_all(&b.reduction_tile_size.to_le_bytes())?;
    w.write_all(&b.tiles_per_output_channel.to_le_bytes())?;
    w.write_all(&b.tail_reduction_count.to_le_bytes())?;
    w.write_all(&[b.macro_layout])?;
    w.write_all(&[b.tail_handling])?;
    w.write_all(&[b.code_segment])?;
    w.write_all(&b.code_offset.to_le_bytes())?;
    w.write_all(&b.code_length.to_le_bytes())?;
    w.write_all(&b.code_tile_stride_bytes.to_le_bytes())?;
    w.write_all(&[b.metadata_segment])?;
    w.write_all(&b.metadata_offset.to_le_bytes())?;
    w.write_all(&b.metadata_length.to_le_bytes())?;
    w.write_all(&b.metadata_tile_stride_bytes.to_le_bytes())?;
    w.write_all(&[b.sidecar_segment])?;
    w.write_all(&b.sidecar_offset.to_le_bytes())?;
    w.write_all(&b.sidecar_length.to_le_bytes())?;
    w.write_all(&[b.sidecar_kind])?;
    w.write_all(&[b.sidecar_element_format])?;
    w.write_all(&b.sidecar_count.to_le_bytes())?;
    w.write_all(&[b.residual_segment])?;
    w.write_all(&b.residual_offset.to_le_bytes())?;
    w.write_all(&b.residual_length.to_le_bytes())?;
    w.write_all(&b.required_alignment_bytes.to_le_bytes())?;
    Ok(())
}

/// Parse a `MatrixWeightBindingV1` from a byte slice (canonical LE).
pub fn read_matrix_weight_binding_v1_le(data: &[u8]) -> Result<MatrixWeightBindingV1, String> {
    if data.len() < MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH {
        return Err(format!(
            "MatrixWeightBindingV1 too small: {} < {}",
            data.len(),
            MATRIX_WEIGHT_BINDING_V1_BYTE_LENGTH
        ));
    }
    let mut off = 0usize;
    let mut read = |n: usize| -> Result<&[u8], String> {
        if off + n > data.len() {
            return Err(format!("binding short read at offset {off}, need {n} bytes"));
        }
        let s = &data[off..off + n];
        off += n;
        Ok(s)
    };
    let bv_bytes: [u8; 2] = read(2)?
        .try_into()
        .map_err(|_| "binding_wire_version conversion".to_string())?;
    let bv = u16::from_le_bytes(bv_bytes);
    if bv != 1 {
        return Err(format!(
            "unknown MatrixWeightBindingV1 wire version: {}",
            bv
        ));
    }
    let rep = read(1)?[0];
    if rep > 3 {
        return Err(format!("unknown representation discriminant: {}", rep));
    }
    let rt_bytes: [u8; 2] = read(2)?
        .try_into()
        .map_err(|_| "representation_version conversion".to_string())?;
    let rt = u16::from_le_bytes(rt_bytes);
    if rep <= 2 && rt != 640 {
        return Err(format!(
            "quantized format requires reduction_tile_size=640, got {}",
            rt
        ));
    }
    let ifeat_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "in_features conversion".to_string())?;
    let ofeat_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "out_features conversion".to_string())?;
    let ifeat = u32::from_le_bytes(ifeat_bytes);
    let ofeat = u32::from_le_bytes(ofeat_bytes);
    let _ = read(4)?; // tiles_per_output_channel (derivable)
    let trc_bytes: [u8; 2] = read(2)?
        .try_into()
        .map_err(|_| "tail_reduction_count conversion".to_string())?;
    let trc = u16::from_le_bytes(trc_bytes);
    if rep <= 2 && trc != (ifeat % 640) as u16 {
        return Err("tail_reduction_count mismatch".into());
    }
    let matrix_id_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "matrix_id conversion".to_string())?;
    let tensor_id: [u8; 16] = read(16)?
        .try_into()
        .map_err(|_| "tensor_id conversion".to_string())?;
    let rv_bytes: [u8; 2] = read(2)?
        .try_into()
        .map_err(|_| "representation_version 2 conversion".to_string())?;
    let kernel_abi_digest: [u8; 32] = read(32)?
        .try_into()
        .map_err(|_| "kernel_abi_digest conversion".to_string())?;
    let tiles_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "tiles_per_output_channel conversion".to_string())?;
    let macro_layout = read(1)?[0];
    let tail_handling = read(1)?[0];
    let code_segment = read(1)?[0];
    let code_offset_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "code_offset conversion".to_string())?;
    let code_length_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "code_length conversion".to_string())?;
    let code_stride_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "code_tile_stride_bytes conversion".to_string())?;
    let metadata_segment = read(1)?[0];
    let metadata_offset_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "metadata_offset conversion".to_string())?;
    let metadata_length_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "metadata_length conversion".to_string())?;
    let metadata_stride_bytes: [u8; 2] = read(2)?
        .try_into()
        .map_err(|_| "metadata_tile_stride_bytes conversion".to_string())?;
    let sidecar_segment = read(1)?[0];
    let sidecar_offset_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "sidecar_offset conversion".to_string())?;
    let sidecar_length_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "sidecar_length conversion".to_string())?;
    let sidecar_kind = read(1)?[0];
    let sidecar_element_format = read(1)?[0];
    let sidecar_count_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "sidecar_count conversion".to_string())?;
    let residual_segment = read(1)?[0];
    let residual_offset_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "residual_offset conversion".to_string())?;
    let residual_length_bytes: [u8; 8] = read(8)?
        .try_into()
        .map_err(|_| "residual_length conversion".to_string())?;
    let required_alignment_bytes_bytes: [u8; 4] = read(4)?
        .try_into()
        .map_err(|_| "required_alignment_bytes conversion".to_string())?;
    Ok(MatrixWeightBindingV1 {
        binding_wire_version: bv,
        matrix_id: u32::from_le_bytes(matrix_id_bytes),
        tensor_id,
        representation: rep,
        representation_version: u16::from_le_bytes(rv_bytes),
        kernel_abi_digest,
        in_features: ifeat,
        out_features: ofeat,
        reduction_tile_size: rt,
        tiles_per_output_channel: u32::from_le_bytes(tiles_bytes),
        tail_reduction_count: trc,
        macro_layout,
        tail_handling,
        code_segment,
        code_offset: u64::from_le_bytes(code_offset_bytes),
        code_length: u64::from_le_bytes(code_length_bytes),
        code_tile_stride_bytes: u32::from_le_bytes(code_stride_bytes),
        metadata_segment,
        metadata_offset: u64::from_le_bytes(metadata_offset_bytes),
        metadata_length: u64::from_le_bytes(metadata_length_bytes),
        metadata_tile_stride_bytes: u16::from_le_bytes(metadata_stride_bytes),
        sidecar_segment,
        sidecar_offset: u64::from_le_bytes(sidecar_offset_bytes),
        sidecar_length: u64::from_le_bytes(sidecar_length_bytes),
        sidecar_kind,
        sidecar_element_format,
        sidecar_count: u32::from_le_bytes(sidecar_count_bytes),
        residual_segment,
        residual_offset: u64::from_le_bytes(residual_offset_bytes),
        residual_length: u64::from_le_bytes(residual_length_bytes),
        required_alignment_bytes: u32::from_le_bytes(required_alignment_bytes_bytes),
    })
}
