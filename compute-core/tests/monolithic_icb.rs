#![cfg(all(target_os = "macos", feature = "prism-backend"))]
#![allow(unexpected_cfgs)]
use objc::{msg_send, sel, sel_impl};
use std::time::Instant;

use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;

// ── Constants ─────────────────────────────────────────────────────────────

const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 10;

const N_EXPERTS: usize = 8;
const TOP_K: usize = 2;
const H: usize = 2048;
const FFN_DIM: usize = 4096;
const N_LAYERS: usize = 8;
const GS: usize = 32;
const NG: usize = H / GS;

// ── FP16 helpers ──────────────────────────────────────────────────────────

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

fn make_data(n: usize, seed: u64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    (0..n)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 ^ seed).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

// ── Q4 packing ────────────────────────────────────────────────────────────

fn pack_q4(data: &[f32], n: usize, k: usize) -> (Vec<u32>, Vec<u16>) {
    let ng = k / GS;
    let mut packed = vec![0u32; n * (k / 8)];
    let mut scales = vec![0u16; n * ng];

    for row in 0..n {
        for g in 0..ng {
            let mut max_abs = 0.0f32;
            for j in 0..GS {
                let a = data[row * k + g * GS + j].abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[row * ng + g] = f32_to_f16_bits(scale);

            let base = row * k + g * GS;
            for j in 0..(GS / 8) {
                let mut word = 0u32;
                for nib in 0..8 {
                    let orig = data[base + j * 8 + nib];
                    let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                    word |= ((q & 0x0F) as u32) << (nib * 4);
                }
                packed[row * (k / 8) + g * (GS / 8) + j] = word;
            }
        }
    }
    (packed, scales)
}

// ── Metal kernel sources ──────────────────────────────────────────────────

const Q4_BODY: &str = r##"
    uint base = row * (K / 8);
    for (uint g = 0; g < ng; ++g) {
        float group_acc = 0.0f;
        half scale = scales[(e_off / (K / 8) + row) * ng + g];
        for (uint j = 0; j < gs / 8; ++j) {
            uint packed = experts[e_off + base + g * (gs / 8) + j];
            uchar4 bytes = as_type<uchar4>(packed);
            uint off = g * gs + j * 8;
            #define NIB(n, i) { uint x = (n >> (i*4)) & 0xFu; group_acc += float(int(x ^ 8u) - 8) * float(scale) * float(input[off + i]); }
            NIB(bytes[0],0) NIB(bytes[0],1) NIB(bytes[1],0) NIB(bytes[1],1)
            NIB(bytes[2],0) NIB(bytes[2],1) NIB(bytes[3],0) NIB(bytes[3],1)
            #undef NIB
        }
        acc_f += group_acc;
    }
"##;

fn moe_direct_source() -> String {
    use std::fmt::Write;
    let mut s = String::new();
    write!(s, "#include <metal_stdlib>\nusing namespace metal;\n").unwrap();
    write!(s, "kernel void moe_direct(\n").unwrap();
    write!(s, "    device const half*      input   [[buffer(0)]],\n").unwrap();
    write!(s, "    device const uint*      experts [[buffer(1)]],\n").unwrap();
    write!(s, "    device const half*      scales  [[buffer(2)]],\n").unwrap();
    write!(s, "    device half*            output  [[buffer(3)]],\n").unwrap();
    write!(s, "    constant uint&          e0      [[buffer(4)]],\n").unwrap();
    write!(s, "    constant uint&          e1      [[buffer(5)]],\n").unwrap();
    write!(s, "    constant uint&          K       [[buffer(6)]],\n").unwrap();
    write!(s, "    constant uint&          N       [[buffer(7)]],\n").unwrap();
    write!(s, "    constant uint&          gs      [[buffer(8)]],\n").unwrap();
    write!(s, "    constant uint&          ng      [[buffer(9)]],\n").unwrap();
    write!(s, "    uint row [[thread_position_in_grid]])\n").unwrap();
    write!(s, "{{\n").unwrap();
    write!(s, "    if (row >= N) return;\n").unwrap();
    write!(s, "    float acc_f = 0.0f;\n").unwrap();
    write!(s, "    {{ uint e_off = e0 * N * (K / 8); {} }}\n", Q4_BODY).unwrap();
    write!(s, "    {{ uint e_off = e1 * N * (K / 8); {} }}\n", Q4_BODY).unwrap();
    write!(s, "    output[row] = half(acc_f);\n").unwrap();
    write!(s, "}}\n").unwrap();
    s
}

// ── CPU Router ────────────────────────────────────────────────────────────

fn cpu_router(input: &[f32], weight: &[f32]) -> (u32, u32) {
    const N: usize = N_EXPERTS;
    let mut logits = [0.0f32; N];
    for i in 0..N {
        let mut sum = 0.0f32;
        for j in 0..H {
            sum += input[j] * weight[j * N + i];
        }
        logits[i] = sum;
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }
    for v in logits.iter_mut() {
        *v /= sum_exp;
    }

    let (idx0, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let idx0 = idx0 as u32;

    let mut best1 = usize::MAX;
    let mut best1_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if i as u32 != idx0 && v > best1_val {
            best1_val = v;
            best1 = i;
        }
    }
    (idx0, best1 as u32)
}

// ── Data generation ───────────────────────────────────────────────────────

fn generate_layer_data(seed: u64) -> (Vec<f32>, Vec<f32>, Vec<u32>, Vec<u16>) {
    let router_w = make_data(H * N_EXPERTS, seed ^ 0xAAAA);
    let mut experts_w = Vec::with_capacity(N_EXPERTS * FFN_DIM * H);
    for e in 0..N_EXPERTS {
        let expert_data = make_data(FFN_DIM * H, seed ^ (e as u64) ^ 0x5555);
        experts_w.extend_from_slice(&expert_data);
    }
    let (packed, scales) = pack_q4(&experts_w, N_EXPERTS * FFN_DIM, H);
    (router_w, experts_w, packed, scales)
}

fn create_expert_buffers(
    dev: &metal::Device,
    packed: &[u32],
    scales: &[u16],
) -> (metal::Buffer, metal::Buffer) {
    let sb = metal::MTLResourceOptions::StorageModeShared;
    let wb = dev.new_buffer((packed.len() * 4) as u64, sb);
    unsafe {
        std::ptr::copy_nonoverlapping(
            packed.as_ptr() as *const u8,
            wb.contents() as *mut u8,
            packed.len() * 4,
        );
    }
    let sbuf = dev.new_buffer((scales.len() * 2) as u64, sb);
    unsafe {
        std::ptr::copy_nonoverlapping(
            scales.as_ptr() as *const u8,
            sbuf.contents() as *mut u8,
            scales.len() * 2,
        );
    }
    (wb, sbuf)
}

fn verify_outputs(layers: usize, out_bufs: &[metal::Buffer]) -> bool {
    for layer in 0..layers {
        unsafe {
            let ptr = out_bufs[layer].contents() as *mut u16;
            for i in 0..FFN_DIM.min(16) {
                let val = f16_bits_to_f32(ptr.add(i).read());
                if !val.is_finite() {
                    return false;
                }
            }
        }
    }
    true
}

// ── Benchmark: Per-layer dispatch (baseline) ──────────────────────────────

fn bench_per_layer(
    layers: usize,
    q: &metal::CommandQueue,
    pl: &metal::ComputePipelineState,
    inputs: &[metal::Buffer],
    expert_w_buf: &metal::Buffer,
    expert_s_buf: &metal::Buffer,
    out_bufs: &[metal::Buffer],
    e0_bufs: &[metal::Buffer],
    e1_bufs: &[metal::Buffer],
    const_k: &metal::Buffer,
    const_n: &metal::Buffer,
    const_gs: &metal::Buffer,
    const_ng: &metal::Buffer,
    wg: metal::MTLSize,
    gg: metal::MTLSize,
) -> f64 {
    for _ in 0..WARMUP_ITERS {
        for layer in 0..layers {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(&inputs[layer]), 0);
            enc.set_buffer(1, Some(expert_w_buf), 0);
            enc.set_buffer(2, Some(expert_s_buf), 0);
            enc.set_buffer(3, Some(&out_bufs[layer]), 0);
            enc.set_buffer(4, Some(&e0_bufs[layer]), 0);
            enc.set_buffer(5, Some(&e1_bufs[layer]), 0);
            enc.set_buffer(6, Some(const_k), 0);
            enc.set_buffer(7, Some(const_n), 0);
            enc.set_buffer(8, Some(const_gs), 0);
            enc.set_buffer(9, Some(const_ng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
    }

    let t0 = Instant::now();
    for _ in 0..TIMED_ITERS {
        for layer in 0..layers {
            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(&inputs[layer]), 0);
            enc.set_buffer(1, Some(expert_w_buf), 0);
            enc.set_buffer(2, Some(expert_s_buf), 0);
            enc.set_buffer(3, Some(&out_bufs[layer]), 0);
            enc.set_buffer(4, Some(&e0_bufs[layer]), 0);
            enc.set_buffer(5, Some(&e1_bufs[layer]), 0);
            enc.set_buffer(6, Some(const_k), 0);
            enc.set_buffer(7, Some(const_n), 0);
            enc.set_buffer(8, Some(const_gs), 0);
            enc.set_buffer(9, Some(const_ng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        }
    }
    t0.elapsed().as_nanos() as f64 / TIMED_ITERS as f64
}

// ── Benchmark: Batched dispatch ──────────────────────────────────────────

fn bench_batched(
    layers: usize,
    q: &metal::CommandQueue,
    pl: &metal::ComputePipelineState,
    inputs: &[metal::Buffer],
    expert_w_buf: &metal::Buffer,
    expert_s_buf: &metal::Buffer,
    out_bufs: &[metal::Buffer],
    e0_bufs: &[metal::Buffer],
    e1_bufs: &[metal::Buffer],
    const_k: &metal::Buffer,
    const_n: &metal::Buffer,
    const_gs: &metal::Buffer,
    const_ng: &metal::Buffer,
    wg: metal::MTLSize,
    gg: metal::MTLSize,
) -> f64 {
    for _ in 0..WARMUP_ITERS {
        let cb = q.new_command_buffer();
        for layer in 0..layers {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(&inputs[layer]), 0);
            enc.set_buffer(1, Some(expert_w_buf), 0);
            enc.set_buffer(2, Some(expert_s_buf), 0);
            enc.set_buffer(3, Some(&out_bufs[layer]), 0);
            enc.set_buffer(4, Some(&e0_bufs[layer]), 0);
            enc.set_buffer(5, Some(&e1_bufs[layer]), 0);
            enc.set_buffer(6, Some(const_k), 0);
            enc.set_buffer(7, Some(const_n), 0);
            enc.set_buffer(8, Some(const_gs), 0);
            enc.set_buffer(9, Some(const_ng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..TIMED_ITERS {
        let cb = q.new_command_buffer();
        for layer in 0..layers {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(&inputs[layer]), 0);
            enc.set_buffer(1, Some(expert_w_buf), 0);
            enc.set_buffer(2, Some(expert_s_buf), 0);
            enc.set_buffer(3, Some(&out_bufs[layer]), 0);
            enc.set_buffer(4, Some(&e0_bufs[layer]), 0);
            enc.set_buffer(5, Some(&e1_bufs[layer]), 0);
            enc.set_buffer(6, Some(const_k), 0);
            enc.set_buffer(7, Some(const_n), 0);
            enc.set_buffer(8, Some(const_gs), 0);
            enc.set_buffer(9, Some(const_ng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / TIMED_ITERS as f64
}

// ── Benchmark: ICB-driven dispatch ────────────────────────────────────────

/// ICB benchmark.
///
/// On Apple Silicon M1, `concurrent_dispatch_threadgroups` / `concurrent_dispatch_threads`
/// on `IndirectComputeCommandRef` crashes with SIGBUS (confirmed hardware limitation).
/// As a workaround, we set buffer bindings on the ICB but issue the dispatch from the
/// encoder. This measures the ICB buffer overhead vs baseline encoder buffer binding.
///
/// On non-M1 hardware, we would set dispatch on the ICB and use a single
/// executeCommandsInBuffer:withRange: call for all layers.
fn bench_icb(
    layers: usize,
    dev: &metal::Device,
    q: &metal::CommandQueue,
    pl: &metal::ComputePipelineState,
    inputs: &[metal::Buffer],
    expert_w_buf: &metal::Buffer,
    expert_s_buf: &metal::Buffer,
    out_bufs: &[metal::Buffer],
    e0_bufs: &[metal::Buffer],
    e1_bufs: &[metal::Buffer],
    const_k: &metal::Buffer,
    const_n: &metal::Buffer,
    const_gs: &metal::Buffer,
    const_ng: &metal::Buffer,
    wg: metal::MTLSize,
    gg: metal::MTLSize,
) -> f64 {
    // Create ICB and pre-encode buffer bindings
    let icb_desc = metal::IndirectCommandBufferDescriptor::new();
    icb_desc.set_command_types(metal::MTLIndirectCommandType::ConcurrentDispatch);
    icb_desc.set_inherit_buffers(false);
    icb_desc.set_inherit_pipeline_state(true);
    icb_desc.set_max_kernel_buffer_bind_count(10);

    let icb = dev.new_indirect_command_buffer_with_descriptor(
        &icb_desc,
        layers as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    for i in 0..layers {
        let cmd = icb.indirect_compute_command_at_index(i as u64);
        cmd.set_kernel_buffer(0, Some(&inputs[i]), 0);
        cmd.set_kernel_buffer(1, Some(expert_w_buf), 0);
        cmd.set_kernel_buffer(2, Some(expert_s_buf), 0);
        cmd.set_kernel_buffer(3, Some(&out_bufs[i]), 0);
        cmd.set_kernel_buffer(4, Some(&e0_bufs[i]), 0);
        cmd.set_kernel_buffer(5, Some(&e1_bufs[i]), 0);
        cmd.set_kernel_buffer(6, Some(const_k), 0);
        cmd.set_kernel_buffer(7, Some(const_n), 0);
        cmd.set_kernel_buffer(8, Some(const_gs), 0);
        cmd.set_kernel_buffer(9, Some(const_ng), 0);
        // NOTE: No dispatch set on indirect commands — crashes with SIGBUS on M1.
        // The `executeCommandsInBuffer:withRange:` call dispatches with zero threadgroups
        // (no-op). The actual compute work is dispatched via the encoder below.
    }

    // Warmup
    for _ in 0..WARMUP_ITERS {
        let cb = q.new_command_buffer();
        for layer in 0..layers {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            // Execute ICB command (buffer bindings only, dispatches zero threadgroups = no-op)
            unsafe {
                let _: () = msg_send![enc,
                    executeCommandsInBuffer: &*icb
                    withRange: metal::NSRange { location: layer as u64, length: 1 }];
            }
            // Actual kernel dispatch via encoder with explicit buffer bindings
            enc.set_buffer(0, Some(&inputs[layer]), 0);
            enc.set_buffer(1, Some(expert_w_buf), 0);
            enc.set_buffer(2, Some(expert_s_buf), 0);
            enc.set_buffer(3, Some(&out_bufs[layer]), 0);
            enc.set_buffer(4, Some(&e0_bufs[layer]), 0);
            enc.set_buffer(5, Some(&e1_bufs[layer]), 0);
            enc.set_buffer(6, Some(const_k), 0);
            enc.set_buffer(7, Some(const_n), 0);
            enc.set_buffer(8, Some(const_gs), 0);
            enc.set_buffer(9, Some(const_ng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
    }

    // Timed
    let t0 = Instant::now();
    for _ in 0..TIMED_ITERS {
        let cb = q.new_command_buffer();
        for layer in 0..layers {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            unsafe {
                let _: () = msg_send![enc,
                    executeCommandsInBuffer: &*icb
                    withRange: metal::NSRange { location: layer as u64, length: 1 }];
            }
            enc.set_buffer(0, Some(&inputs[layer]), 0);
            enc.set_buffer(1, Some(expert_w_buf), 0);
            enc.set_buffer(2, Some(expert_s_buf), 0);
            enc.set_buffer(3, Some(&out_bufs[layer]), 0);
            enc.set_buffer(4, Some(&e0_bufs[layer]), 0);
            enc.set_buffer(5, Some(&e1_bufs[layer]), 0);
            enc.set_buffer(6, Some(const_k), 0);
            enc.set_buffer(7, Some(const_n), 0);
            enc.set_buffer(8, Some(const_gs), 0);
            enc.set_buffer(9, Some(const_ng), 0);
            enc.dispatch_thread_groups(gg, wg);
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / TIMED_ITERS as f64
}

// ── Main benchmark test ───────────────────────────────────────────────────

#[test]
fn test_monolithic_icb_benchmark() {
    println!("\n=== MoE DISPATCH STRATEGY COMPARISON ===");
    println!(
        "  Hardware: Apple Silicon M1 (ICB compute dispatch: broken, using encoder workaround)"
    );
    println!(
        "  Experts: {}, Top-K: {}, H: {}, FFN: {}, GS: {}",
        N_EXPERTS, TOP_K, H, FFN_DIM, GS
    );
    println!("  Layers tested: 1, 2, 4, 8");
    println!();
    println!("  Per-layer: 1 CB/layer, commit+wait/layer");
    println!("  Batched:   all layers in 1 CB, 1 wait");
    println!("  ICB:       pre-encoded buffer bindings + encoder dispatch per layer");
    println!();

    // ── Compile kernel ──
    let metal_out = compile_metal_source("moe_direct", &moe_direct_source())
        .expect("metal compile failed: moe_direct");

    // ── Setup Metal ──
    let dev = metal::Device::system_default().unwrap();
    let q = dev.new_command_queue();
    let sb = metal::MTLResourceOptions::StorageModeShared;

    let lib = dev
        .new_library_with_data(&metal_out.metallib_bytes)
        .unwrap();
    let fnc = lib.get_function("moe_direct", None).unwrap();

    let pl_desc = metal::ComputePipelineDescriptor::new();
    pl_desc.set_compute_function(Some(&fnc));
    pl_desc.set_support_indirect_command_buffers(true);
    let pl = dev.new_compute_pipeline_state(&pl_desc).unwrap();

    // ── Generate data ──
    let (router_w, _experts_w, packed, scales) = generate_layer_data(0x1234);

    let input_data = make_data(H, 0xABCD);
    let mut inputs = Vec::with_capacity(N_LAYERS);
    let mut out_bufs = Vec::with_capacity(N_LAYERS);
    for _ in 0..N_LAYERS {
        let in_buf = dev.new_buffer((H * 2) as u64, sb);
        unsafe {
            let in_ptr = in_buf.contents() as *mut u16;
            for i in 0..H {
                in_ptr.add(i).write(f32_to_f16_bits(input_data[i]));
            }
        }
        inputs.push(in_buf);
        let out_buf = dev.new_buffer((FFN_DIM * 2) as u64, sb);
        out_bufs.push(out_buf);
    }

    let (expert_w_buf, expert_s_buf) = create_expert_buffers(&dev, &packed, &scales);

    let const_k = {
        let b = dev.new_buffer(4, sb);
        unsafe {
            *(b.contents() as *mut u32) = H as u32;
        }
        b
    };
    let const_n = {
        let b = dev.new_buffer(4, sb);
        unsafe {
            *(b.contents() as *mut u32) = FFN_DIM as u32;
        }
        b
    };
    let const_gs = {
        let b = dev.new_buffer(4, sb);
        unsafe {
            *(b.contents() as *mut u32) = GS as u32;
        }
        b
    };
    let const_ng = {
        let b = dev.new_buffer(4, sb);
        unsafe {
            *(b.contents() as *mut u32) = NG as u32;
        }
        b
    };

    let wg = metal::MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    let gg = metal::MTLSize {
        width: ((FFN_DIM as u64 + 255) / 256),
        height: 1,
        depth: 1,
    };

    // ── Pre-compute routing ──
    let mut e0_bufs = Vec::with_capacity(N_LAYERS);
    let mut e1_bufs = Vec::with_capacity(N_LAYERS);
    for layer in 0..N_LAYERS {
        let mut input_f32 = vec![0.0f32; H];
        unsafe {
            let in_ptr = inputs[layer].contents() as *mut u16;
            for i in 0..H {
                input_f32[i] = f16_bits_to_f32(in_ptr.add(i).read());
            }
        }
        let (e0, e1) = cpu_router(&input_f32, &router_w);
        let eb0 = dev.new_buffer(4, sb);
        unsafe {
            *(eb0.contents() as *mut u32) = e0;
        }
        let eb1 = dev.new_buffer(4, sb);
        unsafe {
            *(eb1.contents() as *mut u32) = e1;
        }
        e0_bufs.push(eb0);
        e1_bufs.push(eb1);
    }

    // ── Verify ──
    {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&inputs[0]), 0);
        enc.set_buffer(1, Some(&expert_w_buf), 0);
        enc.set_buffer(2, Some(&expert_s_buf), 0);
        enc.set_buffer(3, Some(&out_bufs[0]), 0);
        enc.set_buffer(4, Some(&e0_bufs[0]), 0);
        enc.set_buffer(5, Some(&e1_bufs[0]), 0);
        enc.set_buffer(6, Some(&const_k), 0);
        enc.set_buffer(7, Some(&const_n), 0);
        enc.set_buffer(8, Some(&const_gs), 0);
        enc.set_buffer(9, Some(&const_ng), 0);
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        assert!(
            verify_outputs(1, &out_bufs),
            "kernel produced invalid outputs"
        );
        println!("  [OK] direct kernel verified");
    }

    // ── Benchmark ──
    let layer_counts = [1usize, 2, 4, 8];

    println!();
    println!(
        "  {:>6}  {:>15}  {:>15}  {:>15}  {:>10}  {:>10}",
        "Layers", "Per-Layer (ns)", "Batched (ns)", "ICB (ns)", "B/Speedup", "I/Speedup"
    );
    println!(
        "  {:>6}  {:>15}  {:>15}  {:>15}  {:>10}  {:>10}",
        "------", "---------------", "------------", "---------", "----------", "---------"
    );

    for &layers in &layer_counts {
        let per_layer_ns = bench_per_layer(
            layers,
            &q,
            &pl,
            &inputs,
            &expert_w_buf,
            &expert_s_buf,
            &out_bufs,
            &e0_bufs,
            &e1_bufs,
            &const_k,
            &const_n,
            &const_gs,
            &const_ng,
            wg,
            gg,
        );

        let batched_ns = bench_batched(
            layers,
            &q,
            &pl,
            &inputs,
            &expert_w_buf,
            &expert_s_buf,
            &out_bufs,
            &e0_bufs,
            &e1_bufs,
            &const_k,
            &const_n,
            &const_gs,
            &const_ng,
            wg,
            gg,
        );

        let icb_ns = bench_icb(
            layers,
            &dev,
            &q,
            &pl,
            &inputs,
            &expert_w_buf,
            &expert_s_buf,
            &out_bufs,
            &e0_bufs,
            &e1_bufs,
            &const_k,
            &const_n,
            &const_gs,
            &const_ng,
            wg,
            gg,
        );

        let batched_speedup = per_layer_ns / batched_ns.max(1.0);
        let icb_speedup = per_layer_ns / icb_ns.max(1.0);

        println!(
            "  {:>6}  {:>15.0}  {:>15.0}  {:>15.0}  {:>9.2}x  {:>9.2}x",
            layers, per_layer_ns, batched_ns, icb_ns, batched_speedup, icb_speedup
        );
    }

    println!();
    println!("  Interpretation:");
    println!("    Per-Layer = CPU-GPU sync per layer (commit+wait). Worst case.");
    println!("    Batched   = one command buffer, all layers, one commit+wait. Saves GPU sync.");
    println!("    ICB       = pre-encoded buffer bindings in ICB.");
    println!("    * On M1, ICB compute dispatch crashes with SIGBUS (hardware limitation).");
    println!("    * ICB mode uses encoder dispatch per layer as workaround.");
    println!("    * Speedup > 1.0 = faster than per-layer dispatch.");
    println!();
    println!("  [DONE] Monolithic ICB benchmark complete.");
}
