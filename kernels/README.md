# Prism fused GEMV kernels — cross-vendor ports

Interleaved-fused palettized (LUT) and ternary GEMV kernels for NVIDIA, AMD, and
Intel GPUs — the cross-platform counterparts of the Apple Metal kernels in
`templates/` and `compute-core/src/compute_image/templates/`.

**Interleaved-fused** means the compressed weight (a 4-bit LUT index, or a 2-bit
ternary code) is unpacked and dequantized **in registers** and multiplied-
accumulated immediately. The dequantized weights never touch global memory — the
only weight traffic is the compressed bytes. This is what makes the format's
bandwidth win real on every backend, exactly as it is on the ANE/Metal path.

## Files

| Target | Palettized (LUT) | Ternary | iGPU + NPU variant |
|--------|------------------|---------|--------------------|
| NVIDIA (CUDA) | `cuda/palettized_gemv.cu` (+ batched prefill) | `cuda/ternary_gemv.cu` | — (no unified iGPU+NPU class part) |
| AMD (ROCm/HIP) | `hip/palettized_gemv.hip` (+ batched prefill) | `hip/ternary_gemv.hip` | `hip/palettized_gemv_igpu_npu.hip` |
| Intel (SYCL) | `sycl/palettized_gemv.cpp` (+ batched prefill) | `sycl/ternary_gemv.cpp` | `sycl/palettized_gemv_igpu_npu.cpp` |
| Shared | `common/palettized_abi.h` | | |
| Verification | `oracle/palettized_oracle.cpp` (CPU reference + self-test) | | |

## The ABI is the contract

All ports implement the byte layout in `common/palettized_abi.h`, which mirrors
the Metal kernels so a `.cimage` compiled once runs bit-compatibly everywhere:

- **Palettized:** per-row 16-entry fp16 codebook + 4-bit indices (2/byte). Eight
  nibbles per 32-bit word, little-endian; element `8w+j` uses nibble
  `(word >> 4j) & 0xF`. `in_dim % 8 == 0`.
- **Ternary:** 2-bit codes, 4/byte; `00=0, 01=+1, 10=-1, 11=0`. `in_dim % 4 == 0`.
- **Numerics:** fp32 accumulation (more accurate than the Metal fp16 accumulator),
  fp16 store. Switch the accumulator to fp16 if you need Metal bit-parity instead.

`oracle/palettized_oracle.cpp` is a dependency-free CPU implementation of the
exact unpack/LUT/MAC that a straightforward reference is checked against — it is
the admission gate. Run it in CI on every change to the format:

```bash
g++ -O2 -std=c++17 oracle/palettized_oracle.cpp -o oracle && ./oracle
# PASS: unpack/LUT/MAC convention verified
```

Every GPU kernel is a structural transliteration of that verified logic, so a GPU
port that diverges from the oracle is a bug in the port, not the spec.

## Discrete GPU vs iGPU-with-NPU — why two kernels

The discrete-GPU kernels assume **dedicated VRAM**: the backend stages the
`.cimage` weight segments into device memory once, and launches decode (batch=1)
or batched **prefill** on the GPU.

The `*_igpu_npu` kernels target **APUs where a GPU and an NPU share one memory
pool** (AMD Strix Halo: RDNA3.5 iGPU + XDNA NPU; Intel Lunar Lake: Xe2-LPG iGPU +
Intel NPU). They differ deliberately:

1. **Zero-copy unified memory.** Pointers are into the shared pool (HIP:
   `hipHostMalloc`/`hipMallocManaged`; SYCL: `malloc_shared`). No H2D copy — the
   mmap'd `.cimage` weights are read in place by both engines.
2. **Decode-only.** Batched prefill (compute-bound) runs on the **NPU**; these
   kernels are the latency-bound decode step, so they have no batched form.
3. **Right-sized occupancy.** wave32 / SIMD16 sub-groups and small work-groups
   fit the modest iGPU without starving the NPU/CPU on the shared bus.
4. **Bandwidth-frugal.** Codebook pinned in on-chip memory; the AMD variant reads
   indices 128 bits at a time (uint4) to cut DRAM transactions.

### NPU handoff contract

The NPU writes prefill activations/KV into the shared pool; the scheduler passes
that pointer straight in as the decode kernel's `input` — no copy, no repack.
Weights live once (`SegmentKind::TernaryWeights` / palettized indices) and are
read by the NPU for prefill and the iGPU for decode. This is the concrete
realization of the roadmap's "results-only boundary crossing" on a unified die.

## Building

```bash
# NVIDIA
nvcc -arch=sm_80 -O3 --ptx cuda/palettized_gemv.cu -o palettized_gemv.ptx
# AMD (MI300 / RDNA3)
hipcc -O3 --offload-arch=gfx942  -c hip/palettized_gemv.hip
hipcc -O3 --offload-arch=gfx1151 -c hip/palettized_gemv_igpu_npu.hip
# Intel
icpx -fsycl -O3 -fsycl-targets=spir64 sycl/palettized_gemv.cpp -c
```

> **Status:** authored and grounded in the oracle-verified ABI, but **not yet
> compiled on GPU hardware** — they were written in a Linux sandbox without the
> CUDA/ROCm/oneAPI toolchains. Compile + run the parity harness on real devices
> before wiring them into a release. The scalar unpack/LUT/MAC logic they share
> *is* verified by the oracle.

## Integration with `ComputeBackend`

Each backend crate (`backend-cuda`, `backend-rocm`, `backend-l0`) implements the
`ComputeBackend` trait from `src/compute_backend.rs` and calls the `extern "C"`
launchers here (`prism_cuda_palettized_gemv`, `prism_hip_palettized_gemv`,
`prism_hip_igpu_palettized_gemv_decode`, `prism_sycl_palettized_gemv`, …) from
its `fused_ternary_gemv` / palettized entry points. The router selects the
decode vs. batched-prefill launcher and the discrete vs. iGPU variant from
`BackendCaps` (`mem_kind`, `has_planar_lut`) — never from `cfg!(target_os)`.
Each backend must pass the same oracle parity test before it is admitted.
