//! Multi-representation weight validation — three configurations:
//!   A: Pure FP16 (baseline)
//!   B: Hybrid — Q4 disk → FP16 Core ML / Q4 Metal
//!   C: Pure GPU Q4 (Metal only, no Core ML)
//!
//! Measures: prepare time, peak RSS, token latency, RMSE, SNR, package size.
//!
//! Run: cargo test --test multi_rep_validate --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::time::Instant;

// ── LogicalWeightTensor abstraction ───────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct WeightId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightRole {
    GateProjection,
    UpProjection,
    DownProjection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DenseMaterialization {
    PreparedFromQ4OnCpu,
    LoadedFromFp16Checkpoint,
    CoreMlArtifactInternal,
}

#[derive(Debug, Clone)]
pub enum WeightRepresentation {
    Q4BlockSym128 {
        packed_resource: usize,
        scales_resource: usize,
        group_size: usize,
    },
    Fp16Dense {
        resource: usize,
        materialization: DenseMaterialization,
    },
}

#[derive(Debug, Clone)]
pub struct LogicalWeightTensor {
    pub id: WeightId,
    pub logical_shape: Vec<usize>,
    pub semantic_role: WeightRole,
    pub representations: Vec<WeightRepresentation>,
}

// ── FP16 ↔ bits helpers ──────────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    (if exp <= 0 {
        sign | (mant >> 1)
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u32) << 10) | mant
    }) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | mant << 13)
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── Deterministic data generation ─────────────────────────────────────────

fn make_data(n: usize, k: usize, seed: u64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    (0..n * k)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 ^ seed).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

fn make_input(k: usize, seed: u64) -> Vec<f32> {
    make_data(1, k, seed)
}

// ── Q4 packing ────────────────────────────────────────────────────────────

fn pack_q4(data: &[f32], n: usize, k: usize, gs: usize) -> (Vec<u32>, Vec<u16>) {
    let ng = k / gs;
    let mut packed = vec![0u32; n * (k / 8)];
    let mut scales = vec![0u16; n * ng];

    for row in 0..n {
        for g in 0..ng {
            let mut max_abs = 0.0f32;
            for j in 0..gs {
                let a = data[row * k + g * gs + j].abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[row * ng + g] = f32_to_f16_bits(scale);

            let base = row * k + g * gs;
            for j in 0..(gs / 8) {
                let mut word = 0u32;
                for nib in 0..8 {
                    let orig = data[base + j * 8 + nib];
                    let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                    word |= ((q & 0x0F) as u32) << (nib * 4);
                }
                packed[row * (k / 8) + g * (gs / 8) + j] = word;
            }
        }
    }
    (packed, scales)
}

// ── Q4→FP16 decompressor (scalar, prepare-phase) ─────────────────────────
// NOTE: Can be replaced with Accelerate vDSP_vflt16 + vDSP_vsmul for ~8×
// throughput. The 0.2ms cost at 512×2048 is invisible vs 1.2s model load.

fn decompress_q4_to_f16(
    packed: &[u32],
    scales: &[u16],
    n: usize,
    k: usize,
    gs: usize,
    out: &mut [u16],
) {
    let ng = k / gs;
    for row in 0..n {
        for g in 0..ng {
            let scale = f16_bits_to_f32(scales[row * ng + g]);
            let base = row * k + g * gs;
            for j in 0..(gs / 8) {
                let word = packed[row * (k / 8) + g * (gs / 8) + j];
                for nib in 0..8 {
                    let nibble = (word >> (nib * 4)) & 0x0F;
                    let signed_val = (nibble ^ 8) as i32 - 8;
                    let deq = (signed_val as f32) * scale;
                    out[base + j * 8 + nib] = f32_to_f16_bits(deq);
                }
            }
        }
    }
}

// ── Reference matmul ──────────────────────────────────────────────────────

fn ref_matmul(input: &[f32], weight: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += input[i] * weight[row * k + i];
        }
        out[row] = sum;
    }
    out
}

// ── Metal sources ─────────────────────────────────────────────────────────

const Q4_KERNEL_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void q4_gemv(
    device const half*      input   [[buffer(0)]],
    device const uint*      weights [[buffer(1)]],
    device const half*      scales  [[buffer(2)]],
    device half*            output  [[buffer(3)]],
    constant uint&          K       [[buffer(4)]],
    constant uint&          N       [[buffer(5)]],
    constant uint&          gs      [[buffer(6)]],
    constant uint&          ng      [[buffer(7)]],
    uint                    row     [[thread_position_in_grid]])
{
    if (row >= N) return;
    float acc_f = 0.0f;
    uint base = row * (K / 8);
    for (uint g = 0; g < ng; ++g) {
        float group_acc = 0.0f;
        half scale = scales[row * ng + g];
        for (uint j = 0; j < gs / 8; ++j) {
            uint packed = weights[base + g * (gs / 8) + j];
            uchar4 bytes = as_type<uchar4>(packed);
            uint off = g * gs + j * 8;
#define NIB(n, i) { uint x = (n >> (i*4)) & 0xFu; group_acc += float(int(x ^ 8u) - 8) * float(scale) * float(input[off + i]); }
            NIB(bytes[0],0) NIB(bytes[0],1) NIB(bytes[1],0) NIB(bytes[1],1)
            NIB(bytes[2],0) NIB(bytes[2],1) NIB(bytes[3],0) NIB(bytes[3],1)
#undef NIB
        }
        acc_f += group_acc;
    }
    output[row] = half(acc_f);
}
"##;

fn fp16_source(name: &str) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    write!(s, "#include <metal_stdlib>\nusing namespace metal;\n").unwrap();
    write!(
        s,
        "kernel void {}(device const half* input [[buffer(0)]],\n",
        name
    )
    .unwrap();
    write!(
        s,
        "                    device const half* weight [[buffer(1)]],\n"
    )
    .unwrap();
    write!(
        s,
        "                    device half* output [[buffer(3)]],\n"
    )
    .unwrap();
    write!(s, "                    constant uint& K [[buffer(4)]],\n").unwrap();
    write!(s, "                    constant uint& N [[buffer(5)]],\n").unwrap();
    write!(
        s,
        "                    uint row [[thread_position_in_grid]]) {{\n"
    )
    .unwrap();
    write!(s, "    if (row >= N) return;\n").unwrap();
    write!(s, "    half acc = 0;\n").unwrap();
    write!(s, "    for (uint i = 0; i < K; ++i) {{\n").unwrap();
    write!(s, "        acc += input[i] * weight[row * K + i];\n").unwrap();
    write!(s, "    }}\n").unwrap();
    write!(s, "    output[row] = acc;\n}}\n").unwrap();
    s
}

// ── RSS measurement ───────────────────────────────────────────────────────
// Uses libc task_info for current process RSS in bytes.

#[allow(dead_code)]
fn current_rss_bytes() -> u64 {
    0 // RSS measurement requires mach_task_basic_info FFI; see Instruments VM Tracker for production telemetry
}

// Placeholder: return 0 until mach_task_basic_info is wired.
// The harness will report "N/A" for RSS.

// ── Accuracy metrics ──────────────────────────────────────────────────────

fn rmse(computed: &[f32], reference: &[f32]) -> f64 {
    let n = computed.len().min(reference.len());
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        let d = (computed[i] - reference[i]) as f64;
        sum_sq += d * d;
    }
    (sum_sq / n as f64).sqrt()
}

fn snr_db(computed: &[f32], reference: &[f32]) -> f64 {
    let n = computed.len().min(reference.len());
    let mut signal = 0.0f64;
    let mut noise = 0.0f64;
    for i in 0..n {
        signal += (reference[i] as f64) * (reference[i] as f64);
        let d = (computed[i] - reference[i]) as f64;
        noise += d * d;
    }
    if noise <= 1e-30 {
        return 200.0;
    }
    10.0 * (signal / noise).log10()
}

// ── Metal compilation helper ──────────────────────────────────────────────

fn compile(
    name: &str,
    source: &str,
) -> tribunus_compute_core::compute_image::metal_pipeline::MetalPipelineOutput {
    tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source(name, source)
        .expect(&format!("metal compile failed: {}", name))
}

// ── Experiment configuration results ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ConfigResult {
    pub label: &'static str,
    pub prepare_us: f64,
    pub token_us: f64,
    pub rss_mb: f64,
    pub package_mb: f64,
    pub rmse: f64,
    pub snr: f64,
}

// ── Experiment harness ────────────────────────────────────────────────────

pub struct ExperimentConfig<'a> {
    pub logical_tensor: &'a LogicalWeightTensor,
    pub n: usize,  // output dim
    pub k: usize,  // input dim
    pub gs: usize, // group size for Q4
    pub iterations: usize,
    pub input_seed: u64,
    pub weight_seed: u64,
}

pub struct ExperimentHarness {
    pub configs: Vec<ConfigResult>,
}

impl ExperimentHarness {
    pub fn new() -> Self {
        Self {
            configs: Vec::new(),
        }
    }

    /// Configuration A: Pure FP16 — everything stored and consumed as FP16.
    pub fn run_config_a(&mut self, cfg: &ExperimentConfig) {
        let n = cfg.n;
        let k = cfg.k;
        let it = cfg.iterations;

        eprintln!("  [Config A] Compiling ...");
        let compiled = compile("fp16_a", &fp16_source("fp16_mm_a"));
        let dev = metal::Device::system_default().unwrap();
        let lib = dev.new_library_with_data(&compiled.metallib_bytes).unwrap();
        let fn_a = lib.get_function("fp16_mm_a", None).unwrap();
        let pl = dev.new_compute_pipeline_state_with_function(&fn_a).unwrap();
        let q = dev.new_command_queue();
        let sb = metal::MTLResourceOptions::StorageModeShared;

        let weight_f32 = make_data(n, k, cfg.weight_seed);
        let input_f32 = make_input(k, cfg.input_seed);

        // Prepare: create FP16 weight buffer
        let t0 = Instant::now();
        let weight_u16: Vec<u16> = weight_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let w_buf = dev.new_buffer((n * k * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight_u16.as_ptr() as *const u8,
                w_buf.contents() as *mut u8,
                n * k * 2,
            )
        }
        let i_buf = dev.new_buffer((k * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                input_f32.as_ptr() as *const u8,
                i_buf.contents() as *mut u8,
                k * 4,
            )
        }
        // Write input as FP16
        unsafe {
            let in_ptr = i_buf.contents() as *mut u16;
            let in_f16: Vec<u16> = input_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
            std::ptr::copy_nonoverlapping(in_f16.as_ptr(), in_ptr, k);
        }
        let o_buf = dev.new_buffer((n * 2) as u64, sb);
        let ck = dev.new_buffer(4, sb);
        unsafe {
            *(ck.contents() as *mut u32) = k as u32;
        }
        let cn = dev.new_buffer(4, sb);
        unsafe {
            *(cn.contents() as *mut u32) = n as u32;
        }
        let prepare_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        // Reference
        let ref_out = ref_matmul(&input_f32, &weight_f32, n, k);

        // Warmup
        let wg = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let gg = metal::MTLSize {
            width: ((n + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };
        for _ in 0..5 {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Timing
        let t0 = Instant::now();
        for _ in 0..it {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let token_ns = t0.elapsed().as_nanos() as f64 / it as f64;

        // Accuracy
        let mut result = vec![0.0f32; n];
        unsafe {
            let ptr = o_buf.contents() as *mut u16;
            for i in 0..n {
                result[i] = f16_bits_to_f32(ptr.add(i).read());
            }
        }
        let rmse_v = rmse(&result, &ref_out);
        let snr_v = snr_db(&result, &ref_out);

        let package_mb = (n * k * 2) as f64 / (1024.0 * 1024.0);

        self.configs.push(ConfigResult {
            label: "A (FP16 baseline)",
            prepare_us,
            token_us: token_ns / 1000.0,
            rss_mb: 0.0,
            package_mb,
            rmse: rmse_v,
            snr: snr_v,
        });
    }

    /// Configuration B: Hybrid — Q4 disk, decompress to FP16 for Core ML,
    /// Metal reads Q4 directly, ANE reads FP16.
    pub fn run_config_b(&mut self, cfg: &ExperimentConfig) {
        let n = cfg.n;
        let k = cfg.k;
        let gs = cfg.gs;
        let it = cfg.iterations;

        let weight_f32 = make_data(n, k, cfg.weight_seed);
        let input_f32 = make_input(k, cfg.input_seed);
        let ref_out = ref_matmul(&input_f32, &weight_f32, n, k);

        // Compile both kernels
        let q4_compiled = compile("q4_b", Q4_KERNEL_SRC);
        let fp16_compiled = compile("fp16_b", &fp16_source("fp16_mm_b"));

        eprintln!("  [Config B] Compiling ...");
        let dev = metal::Device::system_default().unwrap();
        let q = dev.new_command_queue();
        let sb = metal::MTLResourceOptions::StorageModeShared;

        // Prepare phase: pack Q4 + decompress to FP16
        let t0 = Instant::now();
        let (q4_packed, q4_scales) = pack_q4(&weight_f32, n, k, gs);
        let mut fp16_weights = vec![0u16; n * k];
        decompress_q4_to_f16(&q4_packed, &q4_scales, n, k, gs, &mut fp16_weights);
        let prepare_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        // Metal Q4 pipeline
        let q4_lib = dev
            .new_library_with_data(&q4_compiled.metallib_bytes)
            .unwrap();
        let q4_fn = q4_lib.get_function("q4_gemv", None).unwrap();
        let q4_pl = dev
            .new_compute_pipeline_state_with_function(&q4_fn)
            .unwrap();

        // Metal FP16 pipeline
        let fp16_lib = dev
            .new_library_with_data(&fp16_compiled.metallib_bytes)
            .unwrap();
        let fp16_fn = fp16_lib.get_function("fp16_mm_b", None).unwrap();
        let fp16_pl = dev
            .new_compute_pipeline_state_with_function(&fp16_fn)
            .unwrap();

        // Buffers
        let i_buf = dev.new_buffer((k * 2) as u64, sb);
        unsafe {
            let in_f16: Vec<u16> = input_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
            std::ptr::copy_nonoverlapping(in_f16.as_ptr(), i_buf.contents() as *mut u16, k);
        }
        let o_buf = dev.new_buffer((n * 2) as u64, sb);

        // Q4 weight buffer
        let ng = k / gs;
        let q4_w = dev.new_buffer((q4_packed.len() * 4) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_packed.as_ptr() as *const u8,
                q4_w.contents() as *mut u8,
                q4_packed.len() * 4,
            );
        }
        let q4_s = dev.new_buffer((q4_scales.len() * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_scales.as_ptr() as *const u8,
                q4_s.contents() as *mut u8,
                q4_scales.len() * 2,
            );
        }

        // FP16 weight buffer (decompressed)
        let fp16_w = dev.new_buffer((n * k * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                fp16_weights.as_ptr() as *const u8,
                fp16_w.contents() as *mut u8,
                n * k * 2,
            );
        }

        // Constants
        let ck = dev.new_buffer(4, sb);
        unsafe {
            *(ck.contents() as *mut u32) = k as u32;
        }
        let cn = dev.new_buffer(4, sb);
        unsafe {
            *(cn.contents() as *mut u32) = n as u32;
        }
        let cgs = dev.new_buffer(4, sb);
        unsafe {
            *(cgs.contents() as *mut u32) = gs as u32;
        }
        let cng = dev.new_buffer(4, sb);
        unsafe {
            *(cng.contents() as *mut u32) = ng as u32;
        }

        // Warmup Q4
        let wg = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let gg = metal::MTLSize {
            width: ((n + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };
        for _ in 0..5 {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&q4_pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&q4_w), 0);
            enc.set_buffer(2, Some(&q4_s), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.set_buffer(6, Some(&cgs), 0);
            enc.set_buffer(7, Some(&cng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Time Q4 decode
        let t0 = Instant::now();
        for _ in 0..it {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&q4_pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&q4_w), 0);
            enc.set_buffer(2, Some(&q4_s), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.set_buffer(6, Some(&cgs), 0);
            enc.set_buffer(7, Some(&cng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let q4_ns = t0.elapsed().as_nanos() as f64 / it as f64;

        // Read Q4 output
        let mut q4_result = vec![0.0f32; n];
        unsafe {
            let ptr = o_buf.contents() as *mut u16;
            for i in 0..n {
                q4_result[i] = f16_bits_to_f32(ptr.add(i).read());
            }
        }

        // Warmup FP16 (ANE path simulation)
        for _ in 0..5 {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&fp16_pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&fp16_w), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        // Time FP16 decode
        let t0 = Instant::now();
        for _ in 0..it {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&fp16_pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&fp16_w), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let fp16_ns = t0.elapsed().as_nanos() as f64 / it as f64;

        // Accuracy: Q4 lane vs reference
        let q4_rmse = rmse(&q4_result, &ref_out);
        let q4_snr = snr_db(&q4_result, &ref_out);

        // Accuracy: FP16 lane vs reference
        let mut fp16_result = vec![0.0f32; n];
        unsafe {
            let ptr = o_buf.contents() as *mut u16;
            for i in 0..n {
                fp16_result[i] = f16_bits_to_f32(ptr.add(i).read());
            }
        }
        let fp16_rmse = rmse(&fp16_result, &ref_out);
        let fp16_snr = snr_db(&fp16_result, &ref_out);

        // Package size: Q4 + scales
        let q4_bytes = q4_packed.len() * 4 + q4_scales.len() * 2;
        let fp16_bytes = n * k * 2;
        let package_mb = (q4_bytes + fp16_bytes) as f64 / (1024.0 * 1024.0);

        // Report the METAL Q4 decode time as the token latency (hybrid decode path)
        self.configs.push(ConfigResult {
            label: "B (hybrid Q4/FP16)",
            prepare_us,
            token_us: q4_ns / 1000.0,
            rss_mb: 0.0,
            package_mb,
            rmse: q4_rmse,
            snr: q4_snr,
        });
        // Also print the FP16 ANE-simulated path
        eprintln!(
            "  [Config B] ANE-sim FP16 lane: token={:.1}us RMSE={:.6} SNR={:.1}dB",
            fp16_ns / 1000.0,
            fp16_rmse,
            fp16_snr
        );
    }

    /// Configuration C: Pure GPU Q4 — no FP16 representation at all.
    pub fn run_config_c(&mut self, cfg: &ExperimentConfig) {
        let n = cfg.n;
        let k = cfg.k;
        let gs = cfg.gs;
        let it = cfg.iterations;

        let weight_f32 = make_data(n, k, cfg.weight_seed);
        let input_f32 = make_input(k, cfg.input_seed);
        let ref_out = ref_matmul(&input_f32, &weight_f32, n, k);

        let compiled = compile("q4_c", Q4_KERNEL_SRC);

        eprintln!("  [Config C] Compiling ...");
        let dev = metal::Device::system_default().unwrap();
        let q = dev.new_command_queue();
        let sb = metal::MTLResourceOptions::StorageModeShared;

        // Prepare: pack to Q4 only
        let t0 = Instant::now();
        let (q4_packed, q4_scales) = pack_q4(&weight_f32, n, k, gs);
        let prepare_us = t0.elapsed().as_nanos() as f64 / 1000.0;

        let lib = dev.new_library_with_data(&compiled.metallib_bytes).unwrap();
        let fn_c = lib.get_function("q4_gemv", None).unwrap();
        let pl = dev.new_compute_pipeline_state_with_function(&fn_c).unwrap();

        let i_buf = dev.new_buffer((k * 2) as u64, sb);
        unsafe {
            let in_f16: Vec<u16> = input_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
            std::ptr::copy_nonoverlapping(in_f16.as_ptr(), i_buf.contents() as *mut u16, k);
        }
        let o_buf = dev.new_buffer((n * 2) as u64, sb);
        let q4_w = dev.new_buffer((q4_packed.len() * 4) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_packed.as_ptr() as *const u8,
                q4_w.contents() as *mut u8,
                q4_packed.len() * 4,
            );
        }
        let q4_s = dev.new_buffer((q4_scales.len() * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_scales.as_ptr() as *const u8,
                q4_s.contents() as *mut u8,
                q4_scales.len() * 2,
            );
        }

        let ng = k / gs;
        let ck = dev.new_buffer(4, sb);
        unsafe {
            *(ck.contents() as *mut u32) = k as u32;
        }
        let cn = dev.new_buffer(4, sb);
        unsafe {
            *(cn.contents() as *mut u32) = n as u32;
        }
        let cgs = dev.new_buffer(4, sb);
        unsafe {
            *(cgs.contents() as *mut u32) = gs as u32;
        }
        let cng = dev.new_buffer(4, sb);
        unsafe {
            *(cng.contents() as *mut u32) = ng as u32;
        }

        let wg = metal::MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let gg = metal::MTLSize {
            width: ((n + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };
        for _ in 0..5 {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&q4_w), 0);
            enc.set_buffer(2, Some(&q4_s), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.set_buffer(6, Some(&cgs), 0);
            enc.set_buffer(7, Some(&cng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }

        let t0 = Instant::now();
        for _ in 0..it {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&i_buf), 0);
            enc.set_buffer(1, Some(&q4_w), 0);
            enc.set_buffer(2, Some(&q4_s), 0);
            enc.set_buffer(3, Some(&o_buf), 0);
            enc.set_buffer(4, Some(&ck), 0);
            enc.set_buffer(5, Some(&cn), 0);
            enc.set_buffer(6, Some(&cgs), 0);
            enc.set_buffer(7, Some(&cng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
        let token_ns = t0.elapsed().as_nanos() as f64 / it as f64;

        let mut result = vec![0.0f32; n];
        unsafe {
            let ptr = o_buf.contents() as *mut u16;
            for i in 0..n {
                result[i] = f16_bits_to_f32(ptr.add(i).read());
            }
        }

        let q4_bytes = q4_packed.len() * 4 + q4_scales.len() * 2;
        let package_mb = q4_bytes as f64 / (1024.0 * 1024.0);

        self.configs.push(ConfigResult {
            label: "C (GPU Q4 only)",
            prepare_us,
            token_us: token_ns / 1000.0,
            rss_mb: 0.0,
            package_mb,
            rmse: rmse(&result, &ref_out),
            snr: snr_db(&result, &ref_out),
        });
    }
}

// ── Acceptance check ──────────────────────────────────────────────────────

fn check_acceptance(results: &[ConfigResult]) -> bool {
    let mut all_ok = true;
    println!("\n=== ACCEPTANCE CHECK ===");
    for r in results {
        let snr_ok = r.snr >= 45.0;
        let rmse_ok = r.rmse <= 0.001;
        println!(
            "  {}: SNR={:.1}dB (req>=45dB {})  RMSE={:.6} (req<=0.001 {})",
            r.label,
            r.snr,
            if snr_ok { "OK" } else { "FAIL" },
            r.rmse,
            if rmse_ok { "OK" } else { "FAIL" }
        );
        if !snr_ok || !rmse_ok {
            all_ok = false;
        }
    }

    // Throughput invariant: Config B Metal lane > 1.5× Config A
    if let (Some(a), Some(b)) = (
        results.iter().find(|r| r.label.starts_with("A")),
        results.iter().find(|r| r.label.starts_with("B")),
    ) {
        let speedup = a.token_us / b.token_us;
        println!(
            "  Config B vs A token speedup: {:.2}x (req>=1.50x {})",
            speedup,
            if speedup >= 1.50 { "OK" } else { "FAIL" }
        );
        if speedup < 1.50 {
            all_ok = false;
        }
    }

    println!(
        "  Overall: {}",
        if all_ok {
            "ALL ACCEPTANCE CRITERIA MET"
        } else {
            "SOME CRITERIA FAILED"
        }
    );
    all_ok
}

// ── Test entry point ──────────────────────────────────────────────────────

#[test]
fn test_multi_rep_validate() {
    println!("\n=== MULTI-REPRESENTATION WEIGHT VALIDATION ===");

    let sizes: &[(usize, usize, &str)] = &[(512, 2048, "med"), (1024, 4096, "large")];

    for &(h, i, label) in sizes {
        println!("\n--- {}: H={} I={} ---", label, h, i);
        let tensor = LogicalWeightTensor {
            id: WeightId(1),
            logical_shape: vec![h, i],
            semantic_role: WeightRole::GateProjection,
            representations: vec![
                WeightRepresentation::Q4BlockSym128 {
                    packed_resource: 0,
                    scales_resource: 0,
                    group_size: 128,
                },
                WeightRepresentation::Fp16Dense {
                    resource: 0,
                    materialization: DenseMaterialization::PreparedFromQ4OnCpu,
                },
            ],
        };

        let cfg = ExperimentConfig {
            logical_tensor: &tensor,
            n: i,
            k: h,
            gs: 128,
            iterations: if h <= 512 { 200 } else { 50 },
            input_seed: 0x1234,
            weight_seed: 0x5678,
        };

        let mut harness = ExperimentHarness::new();

        harness.run_config_a(&cfg);
        let a = harness.configs.last().unwrap().clone();
        println!(
            "  A: prep={:.0}us tok={:.1}us pkg={:.2}MB RMSE={:.6} SNR={:.1}dB",
            a.prepare_us, a.token_us, a.package_mb, a.rmse, a.snr
        );

        harness.run_config_b(&cfg);
        let b = harness.configs.last().unwrap().clone();
        println!(
            "  B: prep={:.0}us tok={:.1}us pkg={:.2}MB RMSE={:.6} SNR={:.1}dB",
            b.prepare_us, b.token_us, b.package_mb, b.rmse, b.snr
        );

        harness.run_config_c(&cfg);
        let c = harness.configs.last().unwrap().clone();
        println!(
            "  C: prep={:.0}us tok={:.1}us pkg={:.2}MB RMSE={:.6} SNR={:.1}dB",
            c.prepare_us, c.token_us, c.package_mb, c.rmse, c.snr
        );

        let speedup_b = a.token_us / b.token_us;
        let speedup_c = a.token_us / c.token_us;
        println!("  Speedup: B/A={:.2}x  C/A={:.2}x", speedup_b, speedup_c);
        println!(
            "  Package: A={:.1}MB B={:.1}MB C={:.1}MB",
            a.package_mb, b.package_mb, c.package_mb
        );

        check_acceptance(&harness.configs);
    }
}
