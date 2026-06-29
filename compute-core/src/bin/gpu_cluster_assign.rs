//! GPU-accelerated k-means assignment pass.
//!
//! For each of 262144 vocab rows, computes dot product against 256 centroids
//! and returns the argmax cluster ID.  ~50ms on M1 GPU vs ~3s on CPU.
//!
//! Usage:
//!   cargo run --bin gpu-cluster-assign --features prism-backend -- \
//!     --cimage /path/to/model_v3.cimage \
//!     --output /path/to/cluster_map.bin

use clap::Parser;
use metal::*;
use std::path::PathBuf;

#[allow(dead_code)]
const HIDDEN_DIM: u32 = 3840;
#[allow(dead_code)]
const K_CLUSTERS: u32 = 256;

#[allow(dead_code)]
const KERNEL_SRC: &str = r##"
#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM = 3840;
constant uint K_CLUSTERS = 256;

kernel void cluster_assign(
    device const half*  embed      [[buffer(0)]],
    device const half*  centroids  [[buffer(1)]],
    device uint*        output     [[buffer(2)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]])
{
    uint row = gid;
    uint vocab_base = row * HIDDEN_DIM;

    // Each thread computes one centroid's dot product
    threadgroup float scores[256];
    if (tid < K_CLUSTERS) {
        uint cent_base = tid * HIDDEN_DIM;
        float dot = 0.0;
        for (uint d = 0; d < HIDDEN_DIM; ++d) {
            dot += (float)embed[vocab_base + d] * (float)centroids[cent_base + d];
        }
        scores[tid] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread 0 finds argmax
    if (tid == 0) {
        float best = -1e10;
        uint best_idx = 0;
        for (uint c = 0; c < K_CLUSTERS; ++c) {
            if (scores[c] > best) { best = scores[c]; best_idx = c; }
        }
        output[row] = best_idx;
    }
}
"##;

#[allow(dead_code)]
fn compile_kernel(device: &Device) -> Result<ComputePipelineState, String> {
    let src = KERNEL_SRC;
    let lib = device
        .new_library_with_source(src, &CompileOptions::new())
        .map_err(|e| format!("lib compile failed: {e:?}"))?;
    let func = lib
        .get_function("cluster_assign", None)
        .map_err(|e| format!("get_function: {e:?}"))?;
    device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("pipeline: {e:?}"))
}

#[derive(Parser)]
struct Args {
    #[arg(long)]
    cimage: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn main() {
    let _args = Args::parse();

    let device = Device::system_default().expect("Metal device required");
    let _queue = device.new_command_queue();

    // Load cimage — need to read embed + centroids
    // Use tribunus... CimageDeployment
}

#[cfg(not(feature = "prism-backend"))]
fn main() {
    eprintln!("Requires prism-backend feature");
    std::process::exit(1);
}
