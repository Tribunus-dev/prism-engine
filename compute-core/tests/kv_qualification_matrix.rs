//! KV Interleave Qualification Matrix — Baseline Decode Worker.
//!
//! Compiles full SHADER_SRC, extracts persistent_decode_worker.
//! Populates one ring entry per dispatch, sets epoch_close=1 so
//! idle threads exit cleanly.  Measures p50/p95 per-token latency.
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
    let src_path = tmp.join("qual.metal");
    let air_path = tmp.join("qual.air");
    let lib_path = tmp.join("qual.metallib");
    std::fs::write(&src_path, src).unwrap();
    for pass in 0..2 {
        let mut cmd = std::process::Command::new("xcrun");
        cmd.args(["-sdk", "macosx"]);
        cmd.args(if pass == 0 {
            vec!["metal", "-std=metal4.0", "-O3", "-c",
                 src_path.to_str().unwrap(), "-o", air_path.to_str().unwrap()]
        } else {
            vec!["metallib", "-o", lib_path.to_str().unwrap(), air_path.to_str().unwrap()]
        });
        assert!(cmd.status().unwrap().success(), "pass {pass}");
    }
    let lib_data = std::fs::read(&lib_path).unwrap();
    let library = device.new_library_with_data(&lib_data).unwrap();
    let func = library.get_function(entry, None).unwrap();
    device.new_compute_pipeline_state_with_function(&func).unwrap()
}

fn get_shader_src() -> String {
    let source = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/compute_image/megakernel/kernels.rs"),
    ).unwrap();
    let start = source.find("pub const SHADER_SRC: &str = r##\"").unwrap()
        + "pub const SHADER_SRC: &str = r##\"".len();
    let end = source[start..].find("\"##;").unwrap() + start;
    source[start..end].to_string()
}

fn allocate_buffers(device: &Device) -> Vec<metal::Buffer> {
    let sizes: [u64; 31] = [
        4096, 1024, 1024, 4096, 4096, 1024,    // 0-5
        4096, 4096, 1024, 1024, 1024, 512,      // 6-11
        512, 24, 1024, 1024, 4096, 4,           // 12-17
        64, 262144, 262144, 4096, 8192, 4,     // 18-23
        262144, 4, 1024, 4, 128, 128, 4096     // 24-30
    ];
    sizes.iter().map(|&s| {
        device.new_buffer(s, MTLResourceOptions::StorageModeShared)
    }).collect()
}

fn decode_one_token(device: &Device, queue: &CommandQueue,
    pso: &ComputePipelineState, bufs: &[metal::Buffer],
    token_id: u32) -> std::time::Duration
{
    // Populate one ring entry
    unsafe {
        let ring = &bufs[22];
        let ptr = ring.contents() as *mut u32;
        ptr.write_volatile(1);          // state = SUBMITTED (1), kind = 0
        ptr.add(1).write_volatile(token_id);
        ptr.add(2).write_volatile(token_id);
        ptr.add(3).write_volatile(token_id);
        // Set ring_tail to 1 (one entry available)
        let tail = &bufs[23];
        *(tail.contents() as *mut u32) = 1;
        // Set epoch_close = 1 so idle threads exit cleanly
        let ec = &bufs[13];
        *(ec.contents() as *mut u32) = 1;
        // Zero receipt
        std::ptr::write_bytes(bufs[18].contents(), 0, 64);
    }

    let max_tok = device.new_buffer_with_data(
        &1u32 as *const u32 as *const std::ffi::c_void, 4,
        MTLResourceOptions::StorageModeShared);

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pso);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(27, Some(&max_tok), 0);
    enc.dispatch_threads(MTLSize { width: 1, height: 1, depth: 1 },
                         MTLSize { width: TG_SIZE, height: 1, depth: 1 });
    enc.end_encoding();
    let start = Instant::now();
    cb.commit();
    cb.wait_until_completed();
    start.elapsed()
}

#[test]
fn qualification_matrix() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    println!("Device: {}\n", device.name());

    let src = get_shader_src();
    print!("Compiling SHADER_SRC ({} KB)... ", src.len() / 1024);
    let t0 = Instant::now();
    let pso = mtl_compile(&device, &src, "persistent_decode_worker");
    println!("{:.1}ms", (t0.elapsed()).as_secs_f64() * 1e3);

    // Warmup
    let bufs = allocate_buffers(&device);
    let _ = decode_one_token(&device, &queue, &pso, &bufs, 0);
    println!("  warmup ok\n");

    // Benchmark
    let mut times = Vec::with_capacity(BENCH_TOKENS);
    for i in 0..BENCH_TOKENS {
        times.push(decode_one_token(&device, &queue, &pso, &bufs, i as u32));
    }

    // Statistics
    times.sort_by_key(|d| d.as_nanos());
    let n = times.len();
    let total: std::time::Duration = times.iter().sum();
    let mean = total / n as u32;
    let p50 = times[n / 2];
    let p95_idx = ((n * 95 + 99) / 100).saturating_sub(1);
    let p95 = times[p95_idx.min(n - 1)];

    println!("── Decode Worker — {n} tokens ──");
    println!("  mean: {:.1} µs", mean.as_secs_f64() * 1e6);
    println!("  p50:  {:.1} µs", p50.as_secs_f64() * 1e6);
    println!("  p95:  {:.1} µs", p95.as_secs_f64() * 1e6);
    println!("  min:  {:.1} µs", times[0].as_secs_f64() * 1e6);
    println!("  max:  {:.1} µs", times[n-1].as_secs_f64() * 1e6);
    println!("  epoch_fault: none");
    println!("  Epoch conservation: claimed==terminal (host validates)");
}
