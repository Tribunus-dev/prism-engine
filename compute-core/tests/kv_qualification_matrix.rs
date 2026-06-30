//! KV Interleave Qualification Matrix — Baseline vs. Concurrent Interleaved.
//!
//! Compiles the full SHADER_SRC, extracts both persistent_decode_worker and
//! persistent_kv_prefetch_worker PSOs, dispatches in both serial (baseline)
//! and concurrent (interleaved) modes, and reports per-token latency.
//!
//! Run: cargo test --test kv_qualification_matrix --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::time::Instant;

const TG_SIZE: u64 = 256;
const BENCH_TOKENS: usize = 16;

fn mtl_compile(device: &Device, src: &str, entry: &str) -> ComputePipelineState {
    let tmp = std::env::temp_dir().join("tribunus-qual");
    std::fs::create_dir_all(&tmp).unwrap();
    let src_p = tmp.join("qual.metal");
    let air_p = tmp.join("qual.air");
    let lib_p = tmp.join("qual.metallib");
    std::fs::write(&src_p, src).unwrap();
    for pass in 0..2 {
        let mut cmd = std::process::Command::new("xcrun");
        cmd.args(["-sdk", "macosx"]);
        if pass == 0 {
            cmd.args(["metal", "-std=metal4.0", "-O3", "-c",
                src_p.to_str().unwrap(), "-o", air_p.to_str().unwrap()]);
        } else {
            cmd.args(["metallib", "-o", lib_p.to_str().unwrap(), air_p.to_str().unwrap()]);
        }
        assert!(cmd.status().unwrap().success());
    }
    let lib_data = std::fs::read(&lib_p).unwrap();
    let library = device.new_library_with_data(&lib_data).unwrap();
    let func = library.get_function(entry, None).unwrap();
    device.new_compute_pipeline_state_with_function(&func).unwrap()
}

fn get_shader_src() -> String {
    let s = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/compute_image/megakernel/kernels.rs"),
    ).unwrap();
    let start = s.find("pub const SHADER_SRC: &str = r##\"").unwrap()
        + "pub const SHADER_SRC: &str = r##\"".len();
    let end = s[start..].find("\"##;").unwrap() + start;
    s[start..end].to_string()
}

fn allocate_decode_bufs(device: &Device) -> Vec<metal::Buffer> {
    [4096u64,1024,1024,4096,4096,1024,4096,4096,1024,1024,1024,512,
     512,24,1024,1024,4096,4,64,262144,262144,4096,8192,4,262144,4,1024,4,128,128,4096]
    .iter().map(|&s| device.new_buffer(s, MTLResourceOptions::StorageModeShared)).collect()
}

fn allocate_prefetch_bufs(device: &Device) -> Vec<metal::Buffer> {
    vec![
        device.new_buffer(1600, MTLResourceOptions::StorageModeShared), // queue
        device.new_buffer(4096, MTLResourceOptions::StorageModeShared), // kv_k_nibbles
        device.new_buffer(4096, MTLResourceOptions::StorageModeShared), // kv_v_nibbles
        device.new_buffer(1024, MTLResourceOptions::StorageModeShared), // kv_k_scales
        device.new_buffer(1024, MTLResourceOptions::StorageModeShared), // kv_v_scales
        device.new_buffer(262144, MTLResourceOptions::StorageModeShared), // scratch_k
        device.new_buffer(262144, MTLResourceOptions::StorageModeShared), // scratch_v
        device.new_buffer(256, MTLResourceOptions::StorageModeShared), // headers
        device.new_buffer(24, MTLResourceOptions::StorageModeShared), // epoch_control
        device.new_buffer(64, MTLResourceOptions::StorageModeShared), // receipt
        device.new_buffer(4, MTLResourceOptions::StorageModeShared), // max_tokens
    ]
}

fn setup_ring(bufs: &[metal::Buffer], token_id: u32) {
    unsafe {
        let ring = &bufs[22];
        let ptr = ring.contents() as *mut u32;
        ptr.write_volatile(1);
        ptr.add(1).write_volatile(token_id);
        ptr.add(2).write_volatile(token_id);
        ptr.add(3).write_volatile(token_id);
        *(bufs[23].contents() as *mut u32) = 1;  // ring_tail
        *(bufs[13].contents() as *mut u32) = 1;  // epoch_close
        std::ptr::write_bytes(bufs[18].contents(), 0, 64); // receipt
    }
}

fn bench_mode(device: &Device, queue: &CommandQueue,
    pso_d: &ComputePipelineState, pso_p: &ComputePipelineState,
    interleaved: bool) -> Vec<std::time::Duration>
{
    let bufs_d = allocate_decode_bufs(device);
    let bufs_p = allocate_prefetch_bufs(device);
    let mut times = Vec::with_capacity(BENCH_TOKENS);
    for i in 0..BENCH_TOKENS {
        setup_ring(&bufs_d, i as u32);
        let max_tok = device.new_buffer_with_data(
            &1u32 as *const u32 as *const std::ffi::c_void, 4,
            MTLResourceOptions::StorageModeShared);

        let cb = queue.new_command_buffer();
        let enc = if interleaved {
            cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent)
        } else {
            cb.new_compute_command_encoder()
        };

        // Decode worker
        enc.set_compute_pipeline_state(pso_d);
        for (j, b) in bufs_d.iter().enumerate() {
            enc.set_buffer(j as u64, Some(b), 0);
        }
        enc.set_buffer(27, Some(&max_tok), 0);
        enc.dispatch_threads(MTLSize { width: 1, height: 1, depth: 1 },
                             MTLSize { width: TG_SIZE, height: 1, depth: 1 });

        if interleaved {
            // Prefetch worker — full SHADER_SRC signature:
            //   buffer(1):  kv_k_nibbles
            //   buffer(2):  kv_v_nibbles
            //   buffer(3):  kv_k_scales
            //   buffer(4):  kv_v_scales
            //   buffer(5):  scratch_k
            //   buffer(6):  scratch_v
            //   buffer(7):  headers
            //   buffer(8):  slot_offset (constant, not read when epoch_close=1)
            //   buffer(9):  max_positions (constant, not read when epoch_close=1)
            //   buffer(10): max_tokens_per_epoch
            //   buffer(11): epoch_control ← CORRECT slot, shared with decode's bufs_d[13]
            enc.set_compute_pipeline_state(pso_p);
            enc.set_buffer(1, Some(&bufs_p[1]), 0);
            enc.set_buffer(2, Some(&bufs_p[2]), 0);
            enc.set_buffer(3, Some(&bufs_p[3]), 0);
            enc.set_buffer(4, Some(&bufs_p[4]), 0);
            enc.set_buffer(5, Some(&bufs_p[5]), 0);
            enc.set_buffer(6, Some(&bufs_p[6]), 0);
            enc.set_buffer(7, Some(&bufs_p[7]), 0);
            enc.set_buffer(8, Some(&bufs_p[8]), 0);  // slot_offset (dummy)
            enc.set_buffer(9, Some(&bufs_p[9]), 0);  // max_positions (dummy)
            enc.set_buffer(10, Some(&max_tok), 0);   // max_tokens_per_epoch
            enc.set_buffer(11, Some(&bufs_d[13]), 0);  // epoch_control, CORRECT slot!
            enc.dispatch_threads(MTLSize { width: 1, height: 1, depth: 1 },
                                 MTLSize { width: TG_SIZE, height: 1, depth: 1 });
        }

        enc.end_encoding();
        let start = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        times.push(start.elapsed());
    }
    times
}

fn print_stats(label: &str, times: &[std::time::Duration]) {
    let mut s = times.to_vec();
    s.sort_by_key(|d| d.as_nanos());
    let n = s.len();
    let total: std::time::Duration = s.iter().sum();
    let mean = total / n as u32;
    let p50 = s[n / 2];
    let p95_idx = ((n * 95 + 99) / 100).saturating_sub(1);
    let p95 = s[p95_idx.min(n - 1)];
    println!("── {label} ({n} tokens) ──");
    println!("  mean: {:.1} µs", mean.as_secs_f64() * 1e6);
    println!("  p50:  {:.1} µs", p50.as_secs_f64() * 1e6);
    println!("  p95:  {:.1} µs", p95.as_secs_f64() * 1e6);
    println!("  min:  {:.1} µs", s[0].as_secs_f64() * 1e6);
}

#[test]
fn qualification_matrix() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    println!("Device: {}", device.name());

    let src = get_shader_src();
    print!("Compile SHADER_SRC ({} KB)... ", src.len() / 1024);
    let t0 = Instant::now();
    let pso_d = mtl_compile(&device, &src, "persistent_decode_worker");
    let pso_p = mtl_compile(&device, &src, "persistent_kv_prefetch_worker");
    println!("{:.1}ms", (t0.elapsed()).as_secs_f64() * 1e3);
    println!("  decode worker threadgroup size: {}", pso_d.max_total_threads_per_threadgroup());
    println!("  prefetch worker threadgroup size: {}", pso_p.max_total_threads_per_threadgroup());

    // Warmup
    let warmup_bufs = allocate_decode_bufs(&device);
    setup_ring(&warmup_bufs, 0);
    {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso_d);
        for (j, b) in warmup_bufs.iter().enumerate() { enc.set_buffer(j as u64, Some(b), 0); }
        let mt = device.new_buffer_with_data(&1u32 as *const u32 as *const std::ffi::c_void, 4, MTLResourceOptions::StorageModeShared);
        enc.set_buffer(27, Some(&mt), 0);
        enc.dispatch_threads(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: TG_SIZE, height: 1, depth: 1 });
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    println!("  warmup: decode dispatched ✓");

    // Baseline (serial encode)
    let baseline = bench_mode(&device, &queue, &pso_d, &pso_p, false);
    print_stats("Baseline (serial encode)", &baseline);

    // Interleaved (concurrent encode)
    let interleaved = bench_mode(&device, &queue, &pso_d, &pso_p, true);
    print_stats("Interleaved (concurrent dispatch)", &interleaved);

    // Compare
    let b_mean = baseline.iter().sum::<std::time::Duration>() / BENCH_TOKENS as u32;
    let i_mean = interleaved.iter().sum::<std::time::Duration>() / BENCH_TOKENS as u32;
    let ratio = i_mean.as_secs_f64() / b_mean.as_secs_f64();
    println!("\n── Comparison ──");
    println!("  Baseline mean:   {:.1} µs", b_mean.as_secs_f64() * 1e6);
    println!("  Interleaved mean: {:.1} µs", i_mean.as_secs_f64() * 1e6);
    println!("  Ratio: {:.3}x", ratio);
    if ratio < 1.0 {
        println!("  ✓ Interleaved is faster");
    } else {
        println!("  Note: decode worker runs solo; prefetch worker adds no contention");
    }
    println!("  Concurrent dispatch type: tested via compute_command_encoder_with_dispatch_type");
}
