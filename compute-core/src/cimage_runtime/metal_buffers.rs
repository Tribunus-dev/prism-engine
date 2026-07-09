//! Metal buffer store for cimage runtime.
//!
//! Allocates MTLBuffers from resolved tensor payloads and scratch plans.
//! The buffer names match the conventions established by `lower_mlp.rs` so
//! that `MetalRegionEncoder` can bind them by name.

use std::collections::BTreeMap;

use bytemuck::{Pod, Zeroable};
use metal::MTLResourceOptions;

use crate::execution_plan::CodecFamily;

use super::error::{CImageRuntimeError, CImageRuntimeResult};
use super::lower_mlp::CImageMlpRegionPlan;
use super::resolver::ResolvedMlpShardRuntime;
use super::tensor_store::{RuntimeTensorPayload, RuntimeTensorStore};

// ── Shader-visible constants ──────────────────────────────────────────────

/// MlpKernelConstants mirroring the Metal shader's constant buffer layout.
///
/// The shader reads this as a 64-byte `constant struct` bound at a dedicated
/// slot. Fields mirror the CIMAGE_MLP_CONSTANTS buffer layout in the Metal
/// template.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MlpKernelConstants {
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub group_size: u32,
    pub codec_id: u32,
    pub epsilon: f32,
    pub _pad0: [u32; 11],
}

// Safety: repr(C) and all fields are plain-old-data.
unsafe impl Pod for MlpKernelConstants {}
unsafe impl Zeroable for MlpKernelConstants {}

impl MlpKernelConstants {
    /// Byte size of the constants struct as seen by the Metal shader.
    pub const BYTE_SIZE: u64 = 64;
}

// ── Buffer store ──────────────────────────────────────────────────────────

/// A store of allocated MTLBuffers for CImage region execution.
///
/// Each buffer is identified by a string name matching the binding plan
/// produced by `MlpShardRegionBuilder` / `lower_mlp.rs`. The store owns the
/// `Device` handle so it outlives any temporary device reference.
pub struct MetalCImageBufferStore {
    pub device: metal::Device,
    pub buffers: BTreeMap<String, metal::Buffer>,
    pub byte_lengths: BTreeMap<String, u64>,
}

impl MetalCImageBufferStore {
    /// Create a new empty buffer store backed by the given Metal device.
    pub fn new(device: &metal::Device) -> Self {
        Self {
            device: device.clone(),
            buffers: BTreeMap::new(),
            byte_lengths: BTreeMap::new(),
        }
    }

    /// Allocate all buffers needed to execute an MLP shard region.
    ///
    /// Creates the following buffer groups:
    ///
    /// 1. **Input / output** — `hidden_in` (filled with `input` f32 data) and
    ///    `hidden_out` (zeroed).
    /// 2. **Weight buffers** — one per tensor in the resolved store:
    ///    - `rmsnorm_weight` (raw f32, same as the tensor key).
    ///    - `{proj}_codes`, `{proj}_scales`, `{proj}_biases` for each
    ///      projection tensor (gate_proj, up_proj, down_proj).  RawF32
    ///      payloads write into the `_codes` buffer and zero-fill the
    ///      `_scales`/`_biases` companions.
    /// 3. **Constants** — `mlp_constants` filled from `MlpKernelConstants`.
    /// 4. **Scratch** — one buffer per `ScratchBufferInfo` entry in the
    ///    arena plan.
    pub fn allocate_from_resolved_shard(
        &mut self,
        shard: &ResolvedMlpShardRuntime,
        plan: &CImageMlpRegionPlan,
        input: &[f32],
    ) -> CImageRuntimeResult<()> {
        let hidden_dim = shard.hidden_dim;
        let _intermediate_dim = shard.intermediate_dim;

        // 1. Input/output buffers.
        let hidden_in_len = (hidden_dim * 4) as u64;
        let hidden_in_buf = self.alloc_buffer("hidden_in", hidden_in_len)?;
        // Write input f32 data.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr() as *const u8,
                hidden_in_buf.contents() as *mut u8,
                hidden_in_len as usize,
            );
        }
        hidden_in_buf.did_modify_range(metal::NSRange::new(0, hidden_in_len));

        let _hidden_out = self.alloc_buffer("hidden_out", (hidden_dim * 4) as u64)?;

        // 2. Weight buffers from resolved tensor payloads.
        //
        // Buffer naming follows the conventions in `define_persistent_buffers`:
        //
        //   Tensor key          → Buffer name(s)
        //   ───────────           ──────────────
        //   rmsnorm_weight      → rmsnorm_weight (single, always f32)
        //   gate_proj           → gate_proj_codes / gate_proj_scales / gate_proj_biases
        //   up_proj             → up_proj_codes   / up_proj_scales   / up_proj_biases
        //   down_proj           → down_proj_codes / down_proj_scales / down_proj_biases
        for (_id, tensor) in &shard.tensors.tensors {
            self.allocate_tensor_buffers(&tensor.tensor_key, &tensor.payload)?;
        }

        // 3. Constants buffer.
        let codec_id = Self::detect_shard_codec_id(&shard.tensors);
        let group_size = Self::detect_shard_group_size(&shard.tensors);

        let constants = MlpKernelConstants {
            hidden_dim: hidden_dim as u32,
            intermediate_dim: shard.intermediate_dim as u32,
            group_size,
            codec_id,
            epsilon: 1e-6,
            _pad0: [0u32; 11],
        };

        let constants_bytes: &[u8] = bytemuck::bytes_of(&constants);
        debug_assert_eq!(constants_bytes.len(), 64);
        let mlp_constants_buf = self.alloc_buffer("mlp_constants", 64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                constants_bytes.as_ptr(),
                mlp_constants_buf.contents() as *mut u8,
                64,
            );
        }
        mlp_constants_buf.did_modify_range(metal::NSRange::new(0, 64));

        // 4. Scratch buffers from the arena plan.
        for scratch in &plan.arena_plan.scratch_buffers {
            self.alloc_buffer(&scratch.buffer_id, scratch.byte_size)?;
        }

        Ok(())
    }

    /// Get a reference to a named buffer.
    ///
    /// Returns `None` if no buffer with that name has been allocated.
    pub fn get_buffer(&self, id: &str) -> Option<&metal::Buffer> {
        self.buffers.get(id)
    }

    /// Get the byte length of a named buffer, if it has been allocated.
    pub fn get_byte_length(&self, id: &str) -> Option<u64> {
        self.byte_lengths.get(id).copied()
    }

    /// Read back `count` f32 values from a named buffer.
    ///
    /// # Panics
    ///
    /// Panics if the buffer is not found.  Silently clamps `count` to the
    /// number of complete f32 values available in the buffer.
    pub fn readback_f32(&self, id: &str, count: usize) -> Vec<f32> {
        let buf = self.buffers.get(id).expect("buffer not found");
        let ptr = buf.contents() as *const f32;
        let max_count = (buf.length() / 4) as usize;
        let n = count.min(max_count);
        unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Allocate a named `StorageModeShared` buffer and register it.
    fn alloc_buffer(&mut self, name: &str, length: u64) -> CImageRuntimeResult<metal::Buffer> {
        let buf = self
            .device
            .new_buffer(length, MTLResourceOptions::StorageModeShared);
        self.buffers.insert(name.to_string(), buf.clone());
        self.byte_lengths.insert(name.to_string(), length);
        Ok(buf)
    }

    /// Allocate weight buffers for a single tensor based on its payload kind.
    ///
    /// For `rmsnorm_weight` the buffer name matches the tensor key directly.
    /// For projection tensors the buffer set is `{key}_codes` / `{key}_scales`
    /// / `{key}_biases` — raw f32 data goes into `_codes` while `_scales`
    /// and `_biases` are zeroed.
    fn allocate_tensor_buffers(
        &mut self,
        tensor_key: &str,
        payload: &RuntimeTensorPayload,
    ) -> CImageRuntimeResult<()> {
        match payload {
            // ── RawF32 ─────────────────────────────────────────────────────
            RuntimeTensorPayload::RawF32(data) => {
                let is_rmsnorm = tensor_key == "rmsnorm_weight";
                if is_rmsnorm {
                    // Single buffer named after the tensor key.
                    let byte_len = (data.len() * 4) as u64;
                    let buf = self.alloc_buffer(tensor_key, byte_len)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr() as *const u8,
                            buf.contents() as *mut u8,
                            byte_len as usize,
                        );
                    }
                    buf.did_modify_range(metal::NSRange::new(0, byte_len));
                } else {
                    // Projection: write f32 data into `_codes`, zero-init the
                    // companion `_scales` and `_biases` buffers.
                    let codes_name = format!("{tensor_key}_codes");
                    let scales_name = format!("{tensor_key}_scales");
                    let biases_name = format!("{tensor_key}_biases");

                    // The persistent buffer defs allocate scales/biases at
                    // `out_dim * 4` bytes each (one f32 per output neuron).
                    let raw_len = (data.len() * 4) as u64;

                    let codes_buf = self.alloc_buffer(&codes_name, raw_len)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr() as *const u8,
                            codes_buf.contents() as *mut u8,
                            raw_len as usize,
                        );
                    }
                    codes_buf.did_modify_range(metal::NSRange::new(0, raw_len));

                    // scales / biases — allocate as zeroed (unused for f32).
                    let _ = self.alloc_buffer(&scales_name, (data.len() as u64) * 4)?;
                    let _ = self.alloc_buffer(&biases_name, (data.len() as u64) * 4)?;
                }
            }

            // ── FP16 ───────────────────────────────────────────────────────
            RuntimeTensorPayload::Fp16(data) => {
                let is_rmsnorm = tensor_key == "rmsnorm_weight";
                if is_rmsnorm {
                    let byte_len = (data.len() * 2) as u64;
                    let buf = self.alloc_buffer(tensor_key, byte_len)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr() as *const u8,
                            buf.contents() as *mut u8,
                            byte_len as usize,
                        );
                    }
                    buf.did_modify_range(metal::NSRange::new(0, byte_len));
                } else {
                    // Projection: write f16 data into `_codes`, zero-init
                    // the companion `_scales` and `_biases` buffers.
                    let codes_name = format!("{tensor_key}_codes");
                    let scales_name = format!("{tensor_key}_scales");
                    let biases_name = format!("{tensor_key}_biases");
                    let byte_len = (data.len() * 2) as u64;
                    let buf = self.alloc_buffer(&codes_name, byte_len)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr() as *const u8,
                            buf.contents() as *mut u8,
                            byte_len as usize,
                        );
                    }
                    buf.did_modify_range(metal::NSRange::new(0, byte_len));
                    // scales / biases — allocate as zeroed.
                    let _ = self.alloc_buffer(&scales_name, (data.len() as u64) * 2)?;
                    let _ = self.alloc_buffer(&biases_name, (data.len() as u64) * 2)?;
                }
            }

            // ── INT8 tile640 ──────────────────────────────────────────────
            RuntimeTensorPayload::Int8Packed {
                codes,
                scales,
                biases,
            } => {
                let codes_name = format!("{tensor_key}_codes");
                let scales_name = format!("{tensor_key}_scales");
                let biases_name = format!("{tensor_key}_biases");

                let codes_buf = self.alloc_buffer(&codes_name, codes.len() as u64)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        codes.as_ptr(),
                        codes_buf.contents() as *mut u8,
                        codes.len(),
                    );
                }
                codes_buf.did_modify_range(metal::NSRange::new(0, codes.len() as u64));

                self.alloc_f32_buffer(&scales_name, scales)?;
                self.alloc_f32_buffer(&biases_name, biases)?;
            }

            // ── NF4 tile640 ────────────────────────────────────────────────
            RuntimeTensorPayload::Nf4Packed {
                codes,
                scales,
                biases,
                ..
            } => {
                let codes_name = format!("{tensor_key}_codes");
                let scales_name = format!("{tensor_key}_scales");
                let biases_name = format!("{tensor_key}_biases");

                let codes_buf = self.alloc_buffer(&codes_name, codes.len() as u64)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        codes.as_ptr(),
                        codes_buf.contents() as *mut u8,
                        codes.len(),
                    );
                }
                codes_buf.did_modify_range(metal::NSRange::new(0, codes.len() as u64));

                self.alloc_f32_buffer(&scales_name, scales)?;
                self.alloc_f32_buffer(&biases_name, biases)?;
            }

            // ── Mixed precision ────────────────────────────────────────────
            RuntimeTensorPayload::MixedPrecision { .. } => {
                return Err(CImageRuntimeError::UnsupportedCodec(CodecFamily::Mixed));
            }
        }

        Ok(())
    }

    /// Allocate a buffer and fill it with f32 data.
    fn alloc_f32_buffer(&mut self, name: &str, data: &[f32]) -> CImageRuntimeResult<metal::Buffer> {
        let byte_len = (data.len() * 4) as u64;
        let buf = self.alloc_buffer(name, byte_len)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                byte_len as usize,
            );
        }
        buf.did_modify_range(metal::NSRange::new(0, byte_len));
        Ok(buf)
    }

    /// Determine the shard-level codec identifier.
    ///
    /// Prefers the `gate_proj` tensor if present, otherwise scans all tensors
    /// and returns the first non-RawF32 codec (defaulting to RawF32 = 0).
    fn detect_shard_codec_id(tensors: &RuntimeTensorStore) -> u32 {
        for (_id, tensor) in &tensors.tensors {
            if tensor.tensor_key.contains("gate_proj") {
                return codec_family_to_id(tensor.codec);
            }
        }
        for (_id, tensor) in &tensors.tensors {
            let id = codec_family_to_id(tensor.codec);
            if id != 0 {
                return id;
            }
        }
        0 // RawF32
    }

    /// Determine the shard-level group size for codec-packed tensors.
    ///
    /// Returns the NF4 group_size from the first Nf4Packed payload found, or
    /// 640 for INT8, or 0 for no quantization.
    fn detect_shard_group_size(tensors: &RuntimeTensorStore) -> u32 {
        for (_id, tensor) in &tensors.tensors {
            match &tensor.payload {
                RuntimeTensorPayload::Nf4Packed { group_size, .. } => return *group_size as u32,
                RuntimeTensorPayload::Int8Packed { .. } => return 640,
                _ => {}
            }
        }
        0
    }
}

// ── Ternary GEMV constants ─────────────────────────────────────────

/// Mirror of the Metal shader's TernaryGemvConstants constant buffer.
///
/// The shader reads this as a fixed-size struct at buffer index 4.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TernaryGemvConstants {
    pub rows: u32,
    pub cols: u32,
    pub group_size: u32,
    pub groups_per_row: u32,
    pub bytes_per_group: u32,
    pub output_dtype: u32,
    pub padding: [u32; 3],
}

// Safety: repr(C) and all fields are plain-old-data.
unsafe impl Pod for TernaryGemvConstants {}
unsafe impl Zeroable for TernaryGemvConstants {}

impl TernaryGemvConstants {
    /// Byte size of the constants struct as seen by the Metal shader.
    pub const BYTE_SIZE: u64 = 36;
}

// ── Codec family → shader enum mapping ────────────────────────────────

fn codec_family_to_id(codec: CodecFamily) -> u32 {
    match codec {
        CodecFamily::RawF32 => 0,
        CodecFamily::Nf4 => 1,
        CodecFamily::Int8 => 2,
        CodecFamily::Fp16 => 3,
        CodecFamily::SymInt4 => 4,
        CodecFamily::Ternary => 5,
        CodecFamily::Ternary1_58 => 7,
        CodecFamily::Mixed => 6,
        CodecFamily::Q8_0 => 8,
        CodecFamily::Q4_K => 9,
        CodecFamily::Q2_K => 10,
        CodecFamily::IQ2_XXS => 11,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cimage::*;
    use crate::cimage_runtime::lower_mlp::MlpShardRegionBuilder;
    use crate::cimage_runtime::resolver::CImageRuntimeResolver;
    use crate::cimage_runtime::tensor_store::MlpRegionExecutionMode;

    /// Build a resolved shard + region plan for testing.
    ///
    /// Returns the tempdir (keeps the cimage file alive), the resolved shard,
    /// and the region plan.
    fn build_test_shard_and_plan(
        codec: CodecFamily,
    ) -> (
        tempfile::TempDir,
        ResolvedMlpShardRuntime,
        CImageMlpRegionPlan,
    ) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.cimage");

        let config = SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: codec,
                up_codec: codec,
                down_codec: codec,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: false,
            },
        };

        let pending = MlpShardBuilder::build_synthetic_mlp_shard(config).unwrap();
        CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts)
            .unwrap();
        let loaded = CImageLoader::load_v0(&path).unwrap();
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(&loaded).unwrap();
        let plan = MlpShardRegionBuilder::build_region(
            &resolved.tensors,
            resolved.hidden_dim,
            resolved.intermediate_dim,
            MlpRegionExecutionMode::StagedKernels,
        )
        .unwrap();

        (dir, resolved, plan)
    }

    // ── Tests requiring Metal hardware ─────────────────────────────────────

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_buffer_store_allocates_named_buffers() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::RawF32);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![0.5f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        // Check mandatory buffers exist.
        assert!(store.get_buffer("hidden_in").is_some(), "missing hidden_in");
        assert!(
            store.get_buffer("hidden_out").is_some(),
            "missing hidden_out"
        );
        assert!(
            store.get_buffer("rmsnorm_weight").is_some(),
            "missing rmsnorm_weight"
        );

        // Projection triples.
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            let codes = format!("{proj}_codes");
            let scales = format!("{proj}_scales");
            let biases = format!("{proj}_biases");
            assert!(store.get_buffer(&codes).is_some(), "missing {codes}");
            assert!(store.get_buffer(&scales).is_some(), "missing {scales}");
            assert!(store.get_buffer(&biases).is_some(), "missing {biases}");
        }

        // Constants.
        assert!(
            store.get_buffer("mlp_constants").is_some(),
            "missing mlp_constants"
        );

        // Scratch buffers.
        for scratch in &plan.arena_plan.scratch_buffers {
            assert!(
                store.get_buffer(&scratch.buffer_id).is_some(),
                "missing scratch {}",
                scratch.buffer_id
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_buffer_byte_lengths_match_allocation() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::RawF32);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![0.5f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        // hidden_in = hidden_dim * 4
        assert_eq!(
            store.get_byte_length("hidden_in"),
            Some((shard.hidden_dim * 4) as u64)
        );
        // hidden_out = hidden_dim * 4
        assert_eq!(
            store.get_byte_length("hidden_out"),
            Some((shard.hidden_dim * 4) as u64)
        );
        // rmsnorm_weight = hidden_dim * 4
        assert_eq!(
            store.get_byte_length("rmsnorm_weight"),
            Some((shard.hidden_dim * 4) as u64)
        );
        // mlp_constants = 64
        assert_eq!(store.get_byte_length("mlp_constants"), Some(64));
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_readback_f32_returns_input() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::RawF32);
        let device = metal::Device::system_default().expect("no Metal device");

        let input: Vec<f32> = (0..shard.hidden_dim).map(|i| i as f32 * 0.25).collect();
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        let readback = store.readback_f32("hidden_in", shard.hidden_dim);
        assert_eq!(readback.len(), shard.hidden_dim);
        for (a, b) in readback.iter().zip(input.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "readback mismatch: got {a}, expected {b}"
            );
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_readback_clamps_count_to_buffer_capacity() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::RawF32);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![1.0f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        // Request more than the buffer holds — should clamp.
        let readback = store.readback_f32("hidden_in", shard.hidden_dim * 10);
        assert_eq!(readback.len(), shard.hidden_dim);
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_constants_buffer_contents() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::RawF32);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![0.5f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        let buf = store.get_buffer("mlp_constants").unwrap();
        let constants: &MlpKernelConstants = bytemuck::from_bytes(
            unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, 64) }
                .try_into()
                .unwrap(),
        );

        assert_eq!(constants.hidden_dim, 64);
        assert_eq!(constants.intermediate_dim, 128);
        assert_eq!(constants.codec_id, 0); // RawF32
        assert!((constants.epsilon - 1e-6).abs() < f32::EPSILON);
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_buffer_store_codec_id_nf4() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::Nf4);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![0.5f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        let buf = store.get_buffer("mlp_constants").unwrap();
        let constants: &MlpKernelConstants = bytemuck::from_bytes(
            unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, 64) }
                .try_into()
                .unwrap(),
        );

        assert_eq!(constants.codec_id, 1, "expected NF4 codec id");
        assert_eq!(constants.group_size, 32, "expected NF4 group size");
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_buffer_store_codec_id_int8() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::Int8);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![0.5f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        let buf = store.get_buffer("mlp_constants").unwrap();
        let constants: &MlpKernelConstants = bytemuck::from_bytes(
            unsafe { std::slice::from_raw_parts(buf.contents() as *const u8, 64) }
                .try_into()
                .unwrap(),
        );

        assert_eq!(constants.codec_id, 2, "expected INT8 codec id");
        assert_eq!(constants.group_size, 640, "expected INT8 group size");
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    #[test]
    fn test_readback_of_weight_buffers() {
        let (_dir, shard, plan) = build_test_shard_and_plan(CodecFamily::RawF32);
        let device = metal::Device::system_default().expect("no Metal device");

        let input = vec![1.0f32; shard.hidden_dim];
        let mut store = MetalCImageBufferStore::new(&device);
        store
            .allocate_from_resolved_shard(&shard, &plan, &input)
            .expect("allocation failed");

        // rmsnorm_weight should be non-zero f32 data.
        let weight = store.readback_f32("rmsnorm_weight", shard.hidden_dim);
        assert!(!weight.is_empty());
        // The first few values should match the deterministic seed.
        assert!(
            weight.iter().any(|&v| v != 0.0),
            "weight should not be all zeros"
        );
    }

    #[cfg(feature = "metal-dispatch")]
    #[test]
    fn test_ternary_gemv_constants_byte_size() {
        use bytemuck::bytes_of;
        let c = TernaryGemvConstants {
            rows: 128,
            cols: 4096,
            group_size: 32,
            groups_per_row: 128,
            bytes_per_group: 16,
            output_dtype: 0,
            padding: [0; 3],
        };
        assert_eq!(
            bytes_of(&c).len(),
            TernaryGemvConstants::BYTE_SIZE as usize,
            "TernaryGemvConstants byte size mismatch"
        );
    }
}
