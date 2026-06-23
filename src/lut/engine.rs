//! Prism Engine — unified inference runtime for `.cimage` models.

use std::collections::HashMap;
use std::path::Path;
use crate::lut::compiler::CompiledTensor;
use crate::lut::graph::{ActivationFunction, ComputeNode, ModelGraph, TensorRole};

#[derive(Debug, Default)]
pub struct InferenceStats {
    pub prompt_tokens: usize,
    pub generated_tokens: Vec<u32>,
    pub total_time_ms: f64,
}

// ── Metal backend ───────────────────────────────────────────────────────

#[cfg(feature = "metal-dispatch")]
mod metal_backend {
    use metal::*;
    use std::collections::HashMap;
    include!(concat!(env!("OUT_DIR"), "/embedded_metallib.rs"));
    const MAX_SEQ: u64 = 4096;
    pub struct MetalBackend {
        pub device: Device,
        pub library: Library,
        pub command_queue: CommandQueue,
        pub pipeline: ComputePipelineState,
        pub attn_pipeline: ComputePipelineState,
        pub weight_bufs: HashMap<String, (Buffer, u32, u32)>,
        pub scratch: Buffer,
    pub kv_bufs: Vec<(Buffer, Buffer)>,
    pub kv_stride: u64,
    pub kv_offsets: Vec<u64>,
    }
    impl MetalBackend {
        pub fn new(tensors: &HashMap<String, crate::lut::compiler::CompiledTensor>,
            mh: u64, _mi: u64, num_layers: u32, kv_stride: u64) -> Result<Self, String> {
            let device = Device::system_default().ok_or("No Metal device")?;
            let library = device.new_library_with_data(KERNEL_BYTES)
                .map_err(|e| format!("embedded lib: {e:?}"))?;
            let function = library.get_function("palettized_gemv", None).map_err(|e| format!("fn: {e:?}"))?;
            let pipeline = device.new_compute_pipeline_state_with_function(
                &function)
                .map_err(|e| format!("ps: {e:?}"))?;
            let attn_fn = library.get_function("attention_decode", None).map_err(|e| format!("attn fn: {e:?}"))?;
            let attn_pipeline = device.new_compute_pipeline_state_with_function(
                &attn_fn)
                .map_err(|e| format!("attn ps: {e:?}"))?;
            let cq = device.new_command_queue();
            let scratch = device.new_buffer(mh.max(_mi) * 2, MTLResourceOptions::StorageModeShared);
            let layer_cap = MAX_SEQ * kv_stride;
            let mut kv_bufs = Vec::with_capacity(num_layers as usize);
            for _ in 0..num_layers {
                kv_bufs.push((
                    device.new_buffer(layer_cap, MTLResourceOptions::StorageModePrivate),
                    device.new_buffer(layer_cap, MTLResourceOptions::StorageModePrivate),
                ));
            }
            let kv_offsets = vec![0u64; num_layers as usize];
            let mut wb = HashMap::new();
            for (k, ct) in tensors.iter() {
                let b = unsafe { device.new_buffer_with_data(ct.payload.as_ptr() as *const std::ffi::c_void,
                    ct.payload.len() as u64, MTLResourceOptions::StorageModeShared) };
                wb.insert(k.clone(), (b, ct.dim_m, ct.dim_n));
            }

            Ok(MetalBackend { device, library, command_queue: cq, pipeline, attn_pipeline, weight_bufs: wb, scratch, kv_bufs, kv_stride, kv_offsets })
        }
        pub fn gemv(&self, key: &str, inp: &[u16], out: &mut [u16]) -> Result<(), String> {
            let (wb, dm, dn) = self.weight_bufs.get(key).ok_or_else(|| format!("miss {key}"))?;
            let il = inp.len() * 2;
            let ol = out.len() * 2;
            unsafe { std::ptr::copy_nonoverlapping(inp.as_ptr() as *const u8, self.scratch.contents() as *mut u8, il); }
            let cb = self.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipeline);
            enc.set_buffer(0, Some(&self.scratch), 0);
            enc.set_buffer(1, Some(wb), 0);
            enc.set_buffer(2, Some(wb), (*dm as u64) * 16 * 2);
            enc.set_buffer(3, Some(&self.scratch), il as u64);
            enc.set_bytes(4, 4, dn as *const u32 as *const _);
            enc.set_bytes(5, 4, dm as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(*dm as u64, 1, 1), MTLSize::new(64, 1, 1));
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
            unsafe { std::ptr::copy_nonoverlapping(
                (self.scratch.contents() as *const u8).add(il), out.as_mut_ptr() as *mut u8, ol); }
            Ok(())
        }

        /// GPU-accelerated GQA attention: Q@K^T→softmax→@V
        /// Each thread handles one head. Q, K, V are FP16 Vec<u16> slices.
        pub fn attention(
            &self,
            q: &[u16],
            k: &[u16],
            v: &[u16],
            seq_len: usize,
            num_heads: usize,
            kv_heads: usize,
            head_dim: usize,
        ) -> Vec<u16> {
            let kv_dim = kv_heads * head_dim;
            let out_len = num_heads * head_dim;
            if k.len() < seq_len * kv_dim || v.len() < seq_len * kv_dim {
                return vec![0u16; out_len];
            }
            let qb = q.len() * 2;
            let kb = k.len() * 2;
            let vb = v.len() * 2;
            let ob = out_len * 2;
            let needed = qb + kb + vb + ob;
            if self.scratch.length() < needed as u64 { return vec![0u16; out_len]; }
            unsafe {
                let ptr = self.scratch.contents() as *mut u8;
                std::ptr::copy_nonoverlapping(q.as_ptr() as *const u8, ptr, qb);
                std::ptr::copy_nonoverlapping(k.as_ptr() as *const u8, ptr.add(qb), kb);
                std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, ptr.add(qb + kb), vb);
            }
            let cb = self.command_queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.attn_pipeline);
            enc.set_buffer(0, Some(&self.scratch), 0);
            enc.set_buffer(1, Some(&self.scratch), qb as u64);
            enc.set_buffer(2, Some(&self.scratch), (qb + kb) as u64);
            enc.set_buffer(3, Some(&self.scratch), (qb + kb + vb) as u64);
            let sl = seq_len as u32; let nh = num_heads as u32; let nkv = kv_heads as u32; let hd = head_dim as u32;
            enc.set_bytes(4, 4, &sl as *const u32 as *const _);
            enc.set_bytes(5, 4, &nh as *const u32 as *const _);
            enc.set_bytes(6, 4, &nkv as *const u32 as *const _);
            enc.set_bytes(7, 4, &hd as *const u32 as *const _);
            let stride = (self.kv_stride / 2) as u32; // elements per token in buffer
            enc.set_bytes(8, 4, &stride as *const u32 as *const _);
            enc.dispatch_thread_groups(MTLSize::new(num_heads as u64, 1, 1), MTLSize::new(1, 1, 1));
            enc.end_encoding(); cb.commit(); cb.wait_until_completed();
            let mut out = vec![0u16; out_len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (self.scratch.contents() as *const u8).add(qb + kb + vb),
                    out.as_mut_ptr() as *mut u8, ob);
            }
            out
        }

        /// Copy FP16 K/V for one token to the GPU cache.
        pub fn append_kv(&self, layer: usize, k: &[u16], v: &[u16], token_idx: u64) {
            if layer >= self.kv_bufs.len() || self.kv_stride == 0 { return; }
            let off = token_idx * self.kv_stride;
            if off + self.kv_stride > MAX_SEQ * self.kv_stride { return; }
            let (ref k_buf, ref v_buf) = self.kv_bufs[layer];
            let kb = k.len() as u64 * 2;
            unsafe {
                std::ptr::copy_nonoverlapping(k.as_ptr() as *const u8,
                    self.scratch.contents() as *mut u8, kb as usize);
                std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8,
                    (self.scratch.contents() as *mut u8).add(kb as usize), kb as usize);
            }
            let cb = self.command_queue.new_command_buffer();
            let blit = cb.new_blit_command_encoder();
            blit.copy_from_buffer(&self.scratch, 0, k_buf, off, kb);
            blit.copy_from_buffer(&self.scratch, kb, v_buf, off, kb);
            blit.end_encoding();
            cb.commit(); cb.wait_until_completed();
        }
    }
}




#[cfg(all(target_os = "macos", feature = "ane"))]
struct AneBackend {
    model: crate::ane::coreml_bridge::CoreMlModel,
    ctx: crate::ane::coreml_state::StatefulPrefillContext,
    chunk_size: u32,
}

// ── Main engine ─────────────────────────────────────────────────────────

pub struct PrismEngine {
    graph: ModelGraph,
    tensors: HashMap<String, CompiledTensor>,
    /// Compile-time execution plan (loaded from .cimage header).
    pub plan: Option<crate::lut::graph::ExecutionPlan>,
    #[cfg(feature = "metal-dispatch")] metal: Option<metal_backend::MetalBackend>,
    #[cfg(all(target_os = "macos", feature = "ane"))] ane: Option<AneBackend>,
}

/// INT8 KV cache with per-token inline scale.
/// Each token block = [scale_f32: 4 bytes LE][kv_dim × i8 packed as u8].
struct KVCache { k: Vec<Vec<u8>>, v: Vec<Vec<u8>>, seq_lens: Vec<usize>, kv_dim: usize }

impl KVCache {
    fn new(nl: usize, kd: usize) -> Self {
        KVCache { k: vec![Vec::new(); nl], v: vec![Vec::new(); nl], seq_lens: vec![0; nl], kv_dim: kd }
    }
    fn append(&mut self, l: usize, kc: &[u16], vc: &[u16]) {
        fn q(data: &[u16]) -> Vec<u8> {
            let token_size = data.len() + 4;
            let mut out = Vec::with_capacity(token_size);
            if data.is_empty() { return out; }
            let max_abs = data.iter().fold(0.0f32, |a, &v| a.max(half::f16::from_bits(v).to_f32().abs()));
            let scale = if max_abs > 1e-10 { 127.0 / max_abs } else { 1.0 };
            out.extend_from_slice(&scale.to_le_bytes());
            for &v in data { let f = half::f16::from_bits(v).to_f32();
                out.push(((f * scale).round().clamp(-128.0, 127.0) as i8) as u8); }
            out
        }
        self.k[l].extend_from_slice(&q(kc));
        self.v[l].extend_from_slice(&q(vc));
        self.seq_lens[l] += 1;
    }
    fn get_k(&self, l: usize) -> Vec<u16> { dequant_inline(&self.k[l], self.kv_dim) }
    fn get_v(&self, l: usize) -> Vec<u16> { dequant_inline(&self.v[l], self.kv_dim) }
    fn seq_len(&self, l: usize) -> usize { self.seq_lens[l] }
}

fn dequant_inline(data: &[u8], kv_dim: usize) -> Vec<u16> {
    let ts = kv_dim + 4;
    if data.len() < ts { return Vec::new(); }
    let nt = data.len() / ts;
    let mut out = Vec::with_capacity(nt * kv_dim);
    for t in 0..nt {
        let o = t * ts;
        let s = f32::from_le_bytes([data[o], data[o+1], data[o+2], data[o+3]]);
        for j in 0..kv_dim {
            out.push(half::f16::from_f32(((data[o + 4 + j] as i8) as f32) * (1.0 / s)).to_bits());
        }
    }
    out
}

/// KVCache config from execution plan: Int8 (CPU attention) or Fp16 (Metal attention).
#[derive(Clone)]
enum KVCacheMode {
    /// INT8 per-token with inline scale (default for CPU attention).
    Int8 { k: Vec<Vec<u8>>, v: Vec<Vec<u8>> },
    /// FP16 buffer for Metal attention (K/V stored as u16 on CPU, uploaded to GPU).
    Fp16 { k: Vec<Vec<u16>>, v: Vec<Vec<u16>> },
}

impl PrismEngine {
    pub fn load(path: &Path, graph: ModelGraph) -> Result<Self, String> {
        let reader = crate::quantization::cimage::CImageReader::open(path)?;
        let pal = graph.palettized_tensors();
        let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
        let mut tensors = HashMap::new();
        for tb in &pal {
            if let Some(rec) = reader.tensor(&tb.key) {
                let p = data[rec.offset as usize..][..rec.size as usize].to_vec();
                tensors.insert(tb.key.clone(), CompiledTensor { key: tb.key.clone(),
                    dim_m: tb.dim_m, dim_n: tb.dim_n, payload: p, effective_bpp: 0.0 });
            }
        }
        for node in &graph.nodes {
            if let ComputeNode::TokenEmbedding { key, .. } = node {
                if !tensors.contains_key(key) {
                    if let Some(rec) = reader.tensor(key) {
                        let p = data[rec.offset as usize..][..rec.size as usize].to_vec();
                        tensors.insert(key.clone(), CompiledTensor { key: key.clone(),
                            dim_m: rec.dim_m, dim_n: rec.dim_n, payload: p, effective_bpp: 0.0 });
                    }
                }
                break;
            }
        }
        // Load execution plan from .cimage header.
        let plan = reader.header.execution_plan.as_ref().and_then(|json| {
            serde_json::from_str::<crate::lut::graph::ExecutionPlan>(json).ok()
        });

        #[cfg(all(target_os = "macos", feature = "ane"))]
        let mut ane: Option<AneBackend> = None;
        // Extract and load ANE prefill model from .cimage blob.
        #[cfg(all(target_os = "macos", feature = "ane"))]
        if let Ok(blob) = reader.read_blob("_ane_prefill") {
            let tmp = tempfile::tempdir().map_err(|e| format!("tmpdir: {e}"))?;
            let mlmodelc_dir = tmp.path().join("ane_prefill.mlmodelc");
            if crate::ane::unpack_mlmodelc(&blob, &mlmodelc_dir).is_ok() {
                let model_path = mlmodelc_dir.to_string_lossy().to_string();
                let model = crate::ane::coreml_bridge::CoreMlModel::load(&model_path);
                if let Ok(model) = model {
                    let ctx = crate::ane::coreml_state::StatefulPrefillContext::new(model.ptr);
                    if let Ok(ctx) = ctx {
                        ane = Some(AneBackend { model, ctx, chunk_size: 32 });
                        eprintln!("[prism] ANE prefill loaded");
                    }
                }
            }
        }

        Ok(PrismEngine { graph, tensors,
            plan,
            #[cfg(feature = "metal-dispatch")] metal: None,
            #[cfg(all(target_os = "macos", feature = "ane"))] ane,
        })
    }
    pub fn from_memory(graph: ModelGraph, tensors: HashMap<String, CompiledTensor>) -> Self {
        PrismEngine { graph, tensors,
            plan: None,
            #[cfg(feature = "metal-dispatch")] metal: None,
            #[cfg(all(target_os = "macos", feature = "ane"))] ane: None,
        }
    }
    #[cfg(feature = "metal-dispatch")]
    pub fn with_metal(&mut self) -> Result<(), String> {
        let mx = self.graph.nodes.iter().filter_map(|n| match n {
            ComputeNode::PalettizedMatmul { tensor, .. } => Some(tensor.dim_n.max(tensor.dim_m)),
            _ => None,
        }).max().unwrap_or(4096) as u64;
                let nl = self.graph.num_layers as u32;
        let max_kvd = self.graph.nodes.iter().filter_map(|n| match n {
            ComputeNode::ScaledDotProductAttention { num_kv_heads, head_dim, .. } =>
                Some((*num_kv_heads * *head_dim) as u64),
            ComputeNode::LinearAttention { num_heads, head_dim, .. } =>
                Some((*num_heads * *head_dim) as u64),
            _ => None,
        }).max().unwrap_or(2048);
        self.metal = Some(metal_backend::MetalBackend::new(&self.tensors, mx, mx, nl, max_kvd * 2)?);
        eprintln!("[prism] Metal enabled"); Ok(())
    }
    #[cfg(all(target_os = "macos", feature = "ane"))]
    pub fn with_ane(&mut self, mc_dir: &str, cs: u32) -> Result<(), String> {
        let mp = format!("{}/k_cache_{}.mlmodelc", mc_dir, cs);
        let model = crate::ane::coreml_bridge::CoreMlModel::load(&mp)?;
        let ctx = crate::ane::coreml_state::StatefulPrefillContext::new(model.ptr)?;
        self.ane = Some(AneBackend { model, ctx, chunk_size: cs });
        eprintln!("[prism] ANE enabled (chunk={cs})"); Ok(())
    }

    #[cfg(all(target_os = "macos", feature = "ane"))]
    /// Standalone embedding lookup for ANE prefill (no &self needed).
    fn ane_embed(
        token: u32,
        tensors: &HashMap<String, crate::lut::compiler::CompiledTensor>,
    ) -> Result<Vec<u16>, String> {
        // Find embedding tensor by matching known key patterns.
        let (key, ct) = tensors.iter().find(|(k, _)| {
            k.as_str().contains("embed") || k.as_str().contains("tok_embeddings") || k.as_str().contains("wte") || k.as_str() == "t"
        }).ok_or_else(|| "no embedding tensor in ANE path".to_string())?;
        let hd = ct.dim_n as usize;
        let t = token as usize;
        if t >= ct.dim_m as usize { return Ok(vec![0u16; hd]); }
        let cb = ct.dim_m as usize * 16 * 2;
        let mut c = [0u16; 16];
        for i in 0..16 { c[i] = u16::from_le_bytes([ct.payload[t*32+i*2], ct.payload[t*32+i*2+1]]); }
        let io = cb + t * (hd / 2);
        let mut v = Vec::with_capacity(hd);
        for wi in 0..hd / 8 {
            let o = io + wi * 4;
            let pw = u32::from_le_bytes([ct.payload[o], ct.payload[o+1], ct.payload[o+2], ct.payload[o+3]]);
            for j in 0..8 { v.push(c[((pw >> (j*4)) & 0x0F) as usize]); }
        }
        Ok(v)
    }

    #[cfg(all(target_os = "macos", feature = "ane"))]
    fn ane_prefill(
        prompt: &[u32],
        ane: &mut AneBackend,
        graph: &ModelGraph,
        tensors: &HashMap<String, crate::lut::compiler::CompiledTensor>,
    ) -> Result<Vec<u16>, String> {
        use crate::ane::arena::{Arena, Dtype};

        let cs = ane.chunk_size as usize;
        let ed = graph.nodes.iter().find_map(|n| match n {
            ComputeNode::TokenEmbedding { hidden_dim, .. } => Some(*hidden_dim as usize),
            _ => None,
        }).unwrap_or(896);
        let hd = graph.nodes.iter().find_map(|n| match n {
            ComputeNode::ScaledDotProductAttention { head_dim, .. } => Some(*head_dim as usize),
            _ => None,
        }).unwrap_or(64);
        let nh = graph.nodes.iter().find_map(|n| match n {
            ComputeNode::ScaledDotProductAttention { num_heads, .. } => Some(*num_heads as usize),
            _ => None,
        }).unwrap_or(32);
        let nkv = graph.nodes.iter().find_map(|n| match n {
            ComputeNode::ScaledDotProductAttention { num_kv_heads, .. } => Some(*num_kv_heads as usize),
            _ => None,
        }).unwrap_or(nh);
        let kd = nkv * hd;

        let t0 = std::time::Instant::now();
        eprintln!("[prism:ane] Prefill {} tokens (chunk={}, ed={}, kd={})",
            prompt.len(), cs, ed, kd);

        // Phase 1: Embed all prompt tokens on CPU for ANE input.
        let embeddings: Vec<Vec<u16>> = prompt.iter()
            .map(|&t| Self::ane_embed(t, tensors))
            .collect::<Result<Vec<_>, _>>()?;

        let mut last_hidden = vec![0u16; ed.max(1)];

        // Phase 2: Run ANE prefill in chunks.
        for ci in (0..prompt.len()).step_by(cs) {
            let ce = (ci + cs).min(prompt.len());
            let nt = ce - ci;

            // Build IOSurface input arena with FP16 embeddings.
            let in_arena = Arena::new(nt as u32, ed as u32, Dtype::Float16)?;
            in_arena.lock()?;
            unsafe {
                let dst = in_arena.info.base_address as *mut u16;
                for (i, emb) in embeddings[ci..ce].iter().enumerate() {
                    std::ptr::copy_nonoverlapping(emb.as_ptr(), dst.add(i * ed), ed);
                }
            }
            in_arena.unlock()?;

            // Output arena: last token's hidden state.
            let mut out_arena = Arena::new(1, ed as u32, Dtype::Float16)?;

            // Run ANE stateful prefill for this chunk.
            ane.ctx.prefill_chunk(
                ane.model.ptr,
                &in_arena.info,
                &mut out_arena.info,
                nt as u32,
                ed as u32,
                ed as u32,
            )?;

            // Extract last hidden state from IOSurface output arena.
            out_arena.lock()?;
            unsafe {
                let src = out_arena.info.base_address as *const u16;
                last_hidden = std::slice::from_raw_parts(src, ed).to_vec();
            }
            out_arena.unlock()?;

            eprintln!("  [prism:ane] chunk {}: {} tokens ({:.1}s)",
                ci / cs, nt, t0.elapsed().as_secs_f64());
        }

        eprintln!("[prism:ane] Done ({:.1}s)", t0.elapsed().as_secs_f64());
        Ok(last_hidden)
    }

    /// Apply LM head projection to get logits from hidden state.
    /// Tied embeddings: `hidden @ embed^T = logits` via GEMV.
    fn lm_head_projection(&self, h: &[u16]) -> Vec<u16> {
        for node in &self.graph.nodes {
            if let ComputeNode::LanguageModelHead { tensor } = node {
                if let Some(ct) = self.tensors.get(&tensor.key) {
                    if h.len() == tensor.dim_n as usize {
                        return self.gemv(h, tensor, &ct.payload);
                    }
                }
            }
        }
        // Tied embedding head: find embedding tensor and GEMV hidden @ embed^T.
        if let Some(k) = self.graph.nodes.iter().find_map(|n| match n {
            ComputeNode::TokenEmbedding { key, .. } => Some(key.clone()), _ => None
        }) {
            if let Some(ct) = self.tensors.get(&k) {
                let tb = crate::lut::graph::TensorBlueprint { key: k, dim_m: ct.dim_m, dim_n: ct.dim_n };
                return self.gemv(h, &tb, &ct.payload);
            }
        }
        vec![]
    }

    pub fn generate(&mut self, prompt: &[u32], mt: usize) -> Result<InferenceStats, String> {
        let t0 = std::time::Instant::now();
        let nl = self.graph.num_layers as usize;
        let kd = self.graph.nodes.iter().find_map(|n| match n {
            ComputeNode::ScaledDotProductAttention { num_kv_heads, head_dim, .. } =>
                Some((*num_kv_heads * *head_dim) as usize),
            ComputeNode::PalettizedMatmul { role: TensorRole::KProj, tensor } => Some(tensor.dim_m as usize),
            _ => None,
        }).unwrap_or(896);
        let mut kv = KVCache::new(nl, kd);
        let mut pos = 0i64;

        #[cfg(all(target_os = "macos", feature = "ane"))]
        let last_h: Option<Vec<u16>> = {
            // Isolate ANE borrow to this block — dropped before KV fill.
            let ane = self.ane.as_mut();
            if let Some(ane) = ane {
                let lh = Self::ane_prefill(prompt, ane, &self.graph, &self.tensors)?;
                pos = prompt.len() as i64;
                Some(lh)
            } else { None }
        };

        #[cfg(not(all(target_os = "macos", feature = "ane")))]
        self.prefill_cpu(prompt, &mut kv, &mut pos)?;

        #[cfg(all(target_os = "macos", feature = "ane"))]
        if last_h.is_some() {
            // KV cache fill: recompute K/V projections for all prompt tokens.
            for &token in prompt {
                let h = self.embed(token)?;
                let mut li = 0usize;
                for node in &self.graph.nodes {
                    if let ComputeNode::PalettizedMatmul { role, tensor } = node {
                        if let Some(ct) = self.tensors.get(&tensor.key) {
                            if h.len() != tensor.dim_n as usize { continue; }
                            let out = self.gemv(&h, tensor, &ct.payload);
                            match role {
                                TensorRole::KProj => kv.append(li, &out, &[]),
                                TensorRole::VProj => {
                                    let max_abs = out.iter().fold(0.0f32, |a, &v|
                                        a.max(half::f16::from_bits(v).to_f32().abs()));
                                    let scale = if max_abs > 1e-10 { 127.0 / max_abs } else { 1.0 };
                                    kv.v[li].extend_from_slice(&scale.to_le_bytes());
                                    for &v in &out {
                                        let f = half::f16::from_bits(v).to_f32();
                                        kv.v[li].push(((f * scale).round().clamp(-128.0, 127.0) as i8) as u8);
                                    }
                                    kv.seq_lens[li] = kv.k[li].len() / (kv.kv_dim + 4).max(1);
                                }
                                TensorRole::DownProj => li += 1,
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        let mut li = 0usize;
        let mut nt = *prompt.last().unwrap_or(&0);
        let mut gen = Vec::with_capacity(mt);
        // First token: use last hidden from ANE prefill directly (avoids redundant forward pass).
        #[cfg(all(target_os = "macos", feature = "ane"))]
        if let Some(ref lh) = last_h {
            // Apply LM head to ANE's last hidden state to get logits.
            let lm_head = self.lm_head_projection(lh);
            nt = self.argmax(&lm_head)?;
            gen.push(nt);
            pos += 1;
            if gen.len() >= mt { return Ok(InferenceStats { prompt_tokens: prompt.len(), generated_tokens: gen,
                total_time_ms: t0.elapsed().as_secs_f64() * 1000.0 }); }
        }
        for _ in 0..mt {
            let h = self.step(nt, pos, &mut li, &mut kv)?; pos += 1;
            nt = self.argmax(&h)?; gen.push(nt);
        }
        Ok(InferenceStats { prompt_tokens: prompt.len(), generated_tokens: gen,
            total_time_ms: t0.elapsed().as_secs_f64() * 1000.0 })
    }

    fn prefill_cpu(&self, p: &[u32], kv: &mut KVCache, pos: &mut i64) -> Result<(), String> {
        let mut li = 0usize;
        for &t in p { self.step(t, *pos, &mut li, kv)?; *pos += 1; }
        Ok(())
    }
    fn step(&self, token: u32, pos: i64, li: &mut usize, kv: &mut KVCache) -> Result<Vec<u16>, String> {
        let mut h = self.embed(token)?;
        let mut hr = h.clone();
        let mut q: Option<Vec<u16>> = None;
        let mut fused_qkv: Option<Vec<u16>> = None;
        let mut gate: Option<Vec<u16>> = None;
        let mut up: Option<Vec<u16>> = None;
        let mut last_k: Option<Vec<u16>> = None;
        *li = 0;
        for node in &self.graph.nodes {
            match node {
                ComputeNode::TokenEmbedding { .. } => {}
                ComputeNode::Norm { eps, .. } => rms_norm_inplace(&mut h, *eps),
                ComputeNode::PalettizedMatmul { role, tensor } => {
                    let Some(ct) = self.tensors.get(&tensor.key) else { continue; };
                    if h.len() != tensor.dim_n as usize { continue; }
                    let out = self.gemv(&h, tensor, &ct.payload);
                    match role {
                        TensorRole::QProj => q = Some(out),
                        TensorRole::FusedQkvProj => fused_qkv = Some(out),
                        TensorRole::KProj => { kv.append(*li, &out, &[]); last_k = Some(out); },
                        TensorRole::VProj => {
                            // INT8 KV: quantize V token via same q() used in append
                            let max_abs = out.iter().fold(0.0f32, |a, &v| a.max(half::f16::from_bits(v).to_f32().abs()));
                            let scale = if max_abs > 1e-10 { 127.0 / max_abs } else { 1.0 };
                            kv.v[*li].extend_from_slice(&scale.to_le_bytes());
                            for &v in &out {
                                let f = half::f16::from_bits(v).to_f32();
                                kv.v[*li].push(((f * scale).round().clamp(-128.0, 127.0) as i8) as u8);
                            }
                            kv.seq_lens[*li] = kv.k[*li].len() / (kv.kv_dim + 4).max(1);
                            #[cfg(feature = "metal-dispatch")]
                            if let (Some(ref k_gpu), Some(ref m)) = (last_k.take(), self.metal.as_ref()) {
                                m.append_kv(*li, k_gpu, &out, kv.seq_lens[*li] as u64);
                            }
                        }
                        TensorRole::OProj => { h = out; vec_add_inplace(&mut h, &hr); hr = h.clone(); }
                        TensorRole::GateProj => gate = Some(out),
                        TensorRole::UpProj => up = Some(out),
                        TensorRole::DownProj => {
                            if let Some(ref act) = gate {
                                if act.len() == tensor.dim_n as usize {
                                    h = self.gemv(act, tensor, &ct.payload);
                                    vec_add_inplace(&mut h, &hr); hr = h.clone();
                                }
                            }
                            gate = None; up = None; *li += 1;
                        }
                        _ => {}
                    }
                }
                ComputeNode::RotaryEmbedding { head_dim, rope_theta } => {
                    if let Some(ref mut qv) = q { rope_inplace(qv, pos, *head_dim as usize, *rope_theta); }
                }
                ComputeNode::ScaledDotProductAttention { num_heads, num_kv_heads, head_dim: hd } => {
                    if let Some(qv) = q.take() {
                        let sl = kv.seq_len(*li);
                        let kc = kv.get_k(*li); let vc = kv.get_v(*li);
                        if sl > 0 && kc.len() >= sl * kv.kv_dim {
                            h = self.attention(&qv, &kc, &vc, *num_heads as usize, *num_kv_heads as usize, *hd as usize, sl);
                        } else { h = qv; }
                    }
                }
                ComputeNode::MRoPE { head_dim, rope_theta, .. } => {
                    if let Some(ref mut qv) = q {
                        rope_inplace(qv, pos, *head_dim as usize, *rope_theta);
                    }
                    if let Some(ref mut fqkv) = fused_qkv {
                        let q_dim = *head_dim as usize;
                        // Apply RoPE to Q and K portions of fused QKV
                        if fqkv.len() >= q_dim * 3 {
                            rope_inplace(&mut fqkv[..q_dim], pos, q_dim, *rope_theta);
                            rope_inplace(&mut fqkv[q_dim..q_dim*2], pos, q_dim, *rope_theta);
                        }
                    }
                }
                ComputeNode::LinearAttention { num_heads, num_kv_heads, head_dim: hd } => {
                    if let Some(fqkv) = fused_qkv.take() {
                        let nh = *num_heads as usize;
                        let nkv = *num_kv_heads as usize;
                        let hdm = *hd as usize;
                        let q_dim = nh * hdm;
                        // Linear attention uses separate K/V head dims from config.
                        // For Qwen3.5: linear_num_kv_heads=16, linear_key_head_dim=128
                        // The fused QKV is Q + K + V where Q uses self-attention dims.
                        // Infer KV dim from remaining fused output.
                        let kv_dim = if fqkv.len() > q_dim { (fqkv.len() - q_dim) / 2 } else { q_dim };
                        if fqkv.len() < q_dim + 2 * kv_dim { continue; }
                        // Split fused QKV into Q, K, V slices (already RoPE'd by MRoPE handler)
                        let q = fqkv[..q_dim].to_vec();
                        let k = fqkv[q_dim..q_dim + kv_dim].to_vec();
                        let v = fqkv[q_dim + kv_dim..q_dim + 2*kv_dim].to_vec();
                        // Store K and V in INT8 cache
                        kv.append(*li, &k, &[]);
                        let max_abs = v.iter().fold(0.0f32, |a, &val|
                            a.max(half::f16::from_bits(val).to_f32().abs()));
                        let scale = if max_abs > 1e-10 { 127.0 / max_abs } else { 1.0 };
                        kv.v[*li].extend_from_slice(&scale.to_le_bytes());
                        for &val in &v {
                            let f = half::f16::from_bits(val).to_f32();
                            kv.v[*li].push(((f * scale).round().clamp(-128.0, 127.0) as i8) as u8);
                        }
                        kv.seq_lens[*li] = kv.k[*li].len() / (kv.kv_dim + 4).max(1);
                        // Softmax attention with cache
                        let sl = kv.seq_len(*li);
                        let kc = kv.get_k(*li);
                        let vc = kv.get_v(*li);
                        if sl > 0 && kc.len() >= sl * kv.kv_dim {
                            h = self.attention(&q, &kc, &vc, nh, nkv, hdm, sl);
                        } else { h = q; }
                    }
                }
                ComputeNode::AttentionOutputGate { key, dim } => {
                    if let Some(ct) = self.tensors.get(key) {
                        if h.len() == *dim as usize {
                            let out = self.gemv(&h, &crate::lut::graph::TensorBlueprint {
                                key: key.clone(), dim_m: *dim, dim_n: h.len() as u32,
                            }, &ct.payload);
                            // Same residual pattern as OProj
                            h = out;
                            vec_add_inplace(&mut h, &hr);
                            hr = h.clone();
                        }
                    }
                }
                ComputeNode::SharedKVProjection { tensor } => {
                    if let Some(ct) = self.tensors.get(&tensor.key) {
                        if h.len() == tensor.dim_n as usize {
                            let out = self.gemv(&h, tensor, &ct.payload);
                            // Shared K/V: store same values for both K and V
                            kv.append(*li, &out, &out);
                        }
                    }
                }
                ComputeNode::MultiTokenPredictionHead { tensor, depth: _ } => {
                    if let Some(ct) = self.tensors.get(&tensor.key) {
                        if h.len() == tensor.dim_n as usize {
                            // MTP head: separate LM head for MTP depth, produce logits
                            h = self.gemv(&h, tensor, &ct.payload);
                        }
                    }
                }
                ComputeNode::Activation { func } => match func {
                    ActivationFunction::Silu => {
                        if let Some(ref mut g) = gate {
                            silu_inplace(g);
                            if let Some(ref u) = up {
                                for i in 0..g.len().min(u.len()) {
                                    let a = half::f16::from_bits(g[i]).to_f32();
                                    let b = half::f16::from_bits(u[i]).to_f32();
                                    g[i] = half::f16::from_f32(a * b).to_bits();
                                }
                            }
                        }
                    }
                    ActivationFunction::Gelu => { if let Some(ref mut g) = gate { gelu_inplace(g); } }
                },
                ComputeNode::LanguageModelHead { tensor } => {
                    if let Some(ct) = self.tensors.get(&tensor.key) {
                        if h.len() == tensor.dim_n as usize { h = self.gemv(&h, tensor, &ct.payload); }
                    }
                }
                _ => {}
            }
        }
        let has_lm = self.graph.nodes.iter().any(|n| matches!(n, ComputeNode::LanguageModelHead { .. }));
        if !has_lm {
            if let Some(k) = self.graph.nodes.iter().find_map(|n| match n {
                ComputeNode::TokenEmbedding { key, .. } => Some(key.clone()), _ => None
            }) {
                if let Some(ct) = self.tensors.get(&k) {
                    let tb = crate::lut::graph::TensorBlueprint { key: k.clone(), dim_m: ct.dim_m, dim_n: ct.dim_n };
                    h = self.gemv(&h, &tb, &ct.payload);
                }
            }
        }
        Ok(h)
    }

    fn embed(&self, token: u32) -> Result<Vec<u16>, String> {
        let ek = self.graph.nodes.iter().find_map(|n| match n {
            ComputeNode::TokenEmbedding { key, .. } => Some(key.clone()), _ => None
        }).ok_or("no emb")?;
        let ct = self.tensors.get(&ek).ok_or_else(|| format!("emb miss: {ek}"))?;
        let hd = ct.dim_n as usize; let t = token as usize;
        if t >= ct.dim_m as usize { return Ok(vec![0u16; hd]); }
        let cb = ct.dim_m as usize * 16 * 2;
        let mut c = [0u16; 16];
        for i in 0..16 { c[i] = u16::from_le_bytes([ct.payload[t*32+i*2], ct.payload[t*32+i*2+1]]); }
        let io = cb + t * (hd / 2);
        let mut v = Vec::with_capacity(hd);
        for wi in 0..hd / 8 {
            let o = io + wi * 4;
            let pw = u32::from_le_bytes([ct.payload[o], ct.payload[o+1], ct.payload[o+2], ct.payload[o+3]]);
            for j in 0..8 { v.push(c[((pw >> (j*4)) & 0x0F) as usize]); }
        }
        Ok(v)
    }

    fn argmax(&self, logits: &[u16]) -> Result<u32, String> {
        let mut b = 0u32; let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            let f = half::f16::from_bits(v).to_f32();
            if f > bv { bv = f; b = i as u32; }
        }
        Ok(b)
    }

    pub fn graph(&self) -> &ModelGraph { &self.graph }
    pub fn tensor(&self, key: &str) -> Option<&CompiledTensor> { self.tensors.get(key) }
    pub fn embedding_dim(&self) -> u32 {
        self.graph.nodes.iter().find_map(|n| match n {
            ComputeNode::TokenEmbedding { hidden_dim, .. } => Some(*hidden_dim), _ => None
        }).unwrap_or(896)
    }
    pub fn head_dim(&self) -> u32 {
        self.graph.nodes.iter().find_map(|n| match n {
            ComputeNode::ScaledDotProductAttention { head_dim, .. } => Some(*head_dim),
            ComputeNode::RotaryEmbedding { head_dim, .. } => Some(*head_dim), _ => None
        }).unwrap_or(64)
    }

    fn gemv(&self, input: &[u16], tensor: &crate::lut::graph::TensorBlueprint, _payload: &[u8]) -> Vec<u16> {
        let mut out = vec![0u16; tensor.dim_m as usize];
        #[cfg(feature = "metal-dispatch")]
        if let Some(ref m) = self.metal { if m.gemv(&tensor.key, input, &mut out).is_ok() { return out; } }
        lut_gemv_cpu(input, _payload, tensor.dim_m, tensor.dim_n)
    }

    /// GPU-accelerated attention with CPU fallback.
    fn attention(&self, q: &[u16], k: &[u16], v: &[u16], nh: usize, nkv: usize, hd: usize, sl: usize) -> Vec<u16> {
        #[cfg(feature = "metal-dispatch")]
        if let Some(ref m) = self.metal {
            if sl <= 4096 {
                return m.attention(q, k, v, sl, nh, nkv, hd);
            }
        }
        attention_cpu(q, k, v, nh, nkv, hd, sl)
    }
}

// ── CPU operations ──────────────────────────────────────────────────────

fn lut_gemv_cpu(inp: &[u16], p: &[u8], dm: u32, dn: u32) -> Vec<u16> {
    let m = dm as usize; let n = dn as usize; let cbb = m * 16 * 2;
    let mut o = vec![0u16; m];
    for r in 0..m {
        let mut cb = [0u16; 16];
        for i in 0..16 { cb[i] = u16::from_le_bytes([p[r*32+i*2], p[r*32+i*2+1]]); }
        let io = cbb + r * (n / 2);
        let mut acc = 0.0f32;
        for wi in 0..n / 8 {
            let o2 = io + wi * 4;
            let pw = u32::from_le_bytes([p[o2], p[o2+1], p[o2+2], p[o2+3]]);
            for j in 0..8 {
                acc += half::f16::from_bits(inp[wi*8+j]).to_f32() * half::f16::from_bits(cb[((pw>>(j*4))&0x0F) as usize]).to_f32();
            }
        }
        o[r] = half::f16::from_f32(acc).to_bits();
    }
    o
}

fn rms_norm_inplace(x: &mut [u16], eps: f32) {
    let inv = 1.0 / (x.iter().map(|&v| { let f = half::f16::from_bits(v).to_f32(); f*f }).sum::<f32>()
        / x.len() as f32 + eps).sqrt();
    for v in x.iter_mut() { let f = half::f16::from_bits(*v).to_f32(); *v = half::f16::from_f32(f * inv).to_bits(); }
}

fn vec_add_inplace(a: &mut [u16], b: &[u16]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        let fa = half::f16::from_bits(*av).to_f32();
        let fb = half::f16::from_bits(bv).to_f32();
        *av = half::f16::from_f32(fa + fb).to_bits();
    }
}

fn silu_inplace(x: &mut [u16]) {
    for v in x.iter_mut() { let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(f / (1.0 + (-f).exp())).to_bits(); }
}

fn gelu_inplace(x: &mut [u16]) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() { let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(0.5 * f * (1.0 + (s * (f + 0.044715 * f * f * f)).tanh())).to_bits(); }
}

fn attention_cpu(q: &[u16], kc: &[u16], vc: &[u16], nh: usize, nkv: usize, hd: usize, sl: usize) -> Vec<u16> {
    let g = nh / nkv.max(1); let kvd = nkv * hd; let mut o = vec![0u16; nh * hd];
    for h in 0..nh {
        let kh = h / g; let qb = h * hd;
        let mut sc = vec![0.0f32; sl];
        for p in 0..sl {
            let kb = p * kvd + kh * hd;
            let mut s = 0.0f32;
            for d in 0..hd { s += half::f16::from_bits(q[qb+d]).to_f32() * half::f16::from_bits(kc[kb+d]).to_f32(); }
            sc[p] = s / (hd as f32).sqrt();
        }
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut ex = vec![0.0f32; sl]; let mut es = 0.0f32;
        for (i, &s) in sc.iter().enumerate() { let e = (s - mx).exp(); ex[i] = e; es += e; }
        let inv = 1.0 / (es + 1e-10);
        for d in 0..hd {
            let mut ac = 0.0f32;
            for p in 0..sl { ac += ex[p] * half::f16::from_bits(vc[p * kvd + kh * hd + d]).to_f32() * inv; }
            o[qb + d] = half::f16::from_f32(ac).to_bits();
        }
    }
    o
}

fn rope_inplace(x: &mut [u16], pos: i64, hd: usize, th: f32) {
    for i in (0..x.len()).step_by(hd) {
        let h = hd / 2;
        for j in 0..h {
            let a = half::f16::from_bits(x[i+j]).to_f32();
            let b = half::f16::from_bits(x[i+j+h]).to_f32();
            let ang = (pos as f32) * (10000.0f32).powf(-2.0 * j as f32 / hd as f32);
            let (sa, ca) = ang.sin_cos();
            x[i+j] = half::f16::from_f32(a*ca - b*sa).to_bits();
            x[i+j+h] = half::f16::from_f32(a*sa + b*ca).to_bits();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut::graph::UnifiedConfig;
    #[test]
    fn test_lut_gemv() {
        let mut p = Vec::new();
        for _ in 0..16 { p.extend_from_slice(&0x3c00u16.to_le_bytes()); }
        for _ in 0..16 { p.extend_from_slice(&0x4000u16.to_le_bytes()); }
        for _ in 0..2 { p.extend_from_slice(&[0x00; 4]); }
        let o = lut_gemv_cpu(&[0x3c00u16; 8], &p, 2, 8);
        let v = half::f16::from_bits(o[0]).to_f32();
        assert!((v - 8.0).abs() < 0.01);
    }
    #[test]
    fn test_rms() {
        let mut x = vec![0x3c00u16; 4]; rms_norm_inplace(&mut x, 1e-6);
        for &v in &x { assert!((half::f16::from_bits(v).to_f32() - 1.0).abs() < 1e-4); }
    }
    #[test]
    fn test_embed_oob() {
        let e = PrismEngine {
            graph: ModelGraph { nodes: vec![ComputeNode::TokenEmbedding { key: "t".into(), vocab_size: 10, hidden_dim: 8 }], num_layers: 0 },
            tensors: {
                let mut m = HashMap::new();
                let mut p = Vec::new();
                for _ in 0..2*16 { p.extend_from_slice(&0x3c00u16.to_le_bytes()); }
                for _ in 0..2*(8/2) { p.push(0); }
                m.insert("t".into(), CompiledTensor { key: "t".into(), dim_m: 2, dim_n: 8, payload: p, effective_bpp: 4.0 });
                m
            },
            plan: None,
            #[cfg(feature = "metal-dispatch")] metal: None,
            #[cfg(all(target_os = "macos", feature = "ane"))] ane: None,
        };
        assert_eq!(e.embed(0).unwrap().len(), 8);
        assert_eq!(e.embed(5).unwrap().len(), 8);
    }
}
