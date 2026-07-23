//! MI300X acceleration for progressive ternary candidate scoring.
//!
//! The evolutionary policy remains in Prism. This module only accelerates the
//! dense reference/candidate error reduction on a ROCm host.

use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

const KERNEL: &str = r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstdlib>
__global__ void rmse(const float* a, const float* b, double* out, size_t n) {
  __shared__ double scratch[256];
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  double v = i < n ? (double)a[i] - (double)b[i] : 0.0;
  scratch[threadIdx.x] = v * v;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s; s >>= 1) {
    if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, scratch[0]);
}
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream fa(argv[1], std::ios::binary), fb(argv[2], std::ios::binary);
  fa.seekg(0, std::ios::end); size_t bytes = (size_t)fa.tellg(); fa.seekg(0);
  std::vector<float> a(bytes / 4), b(bytes / 4); fa.read((char*)a.data(), bytes); fb.read((char*)b.data(), bytes);
  float *da, *db; double *do_; double zero = 0.0;
  hipMalloc(&da, bytes); hipMalloc(&db, bytes); hipMalloc(&do_, sizeof(double)); hipMemcpy(da,a.data(),bytes,hipMemcpyHostToDevice); hipMemcpy(db,b.data(),bytes,hipMemcpyHostToDevice); hipMemcpy(do_,&zero,sizeof(double),hipMemcpyHostToDevice);
  rmse<<<(a.size()+255)/256,256>>>(da,db,do_,a.size()); hipDeviceSynchronize(); double sum; hipMemcpy(&sum,do_,sizeof(double),hipMemcpyDeviceToHost);
  std::printf("%.17g\n", sum / (double)a.size()); hipFree(da); hipFree(db); hipFree(do_); return 0;
}
"#;

const PACKED_KERNEL: &str = r#"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <fstream>
#include <vector>
__global__ void packed_mse(const float* ref, const unsigned char* packed, const float* scales, double* out, size_t n, size_t group) {
  __shared__ double scratch[256];
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  double d = 0.0;
  if (i < n) {
    unsigned char byte = packed[i >> 2];
    unsigned code = (byte >> ((i & 3) * 2)) & 3;
    float ternary = code == 1 ? 1.0f : (code == 2 ? -1.0f : 0.0f);
    d = (double)ref[i] - (double)ternary * (double)scales[i / group];
    d *= d;
  }
  scratch[threadIdx.x] = d;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s; s >>= 1) { if (threadIdx.x < s) scratch[threadIdx.x] += scratch[threadIdx.x + s]; __syncthreads(); }
  if (threadIdx.x == 0) atomicAdd(out, scratch[0]);
}
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  std::ifstream fr(argv[1], std::ios::binary), fp(argv[2], std::ios::binary), fs(argv[3], std::ios::binary);
  fr.seekg(0, std::ios::end); size_t n = (size_t)fr.tellg() / 4; fr.seekg(0);
  fp.seekg(0, std::ios::end); size_t packed_bytes = (size_t)fp.tellg(); fp.seekg(0);
  fs.seekg(0, std::ios::end); size_t groups = (size_t)fs.tellg() / 4; fs.seekg(0);
  std::vector<float> ref(n), scales(groups); std::vector<unsigned char> packed(packed_bytes);
  fr.read((char*)ref.data(), n * 4); fp.read((char*)packed.data(), packed_bytes); fs.read((char*)scales.data(), groups * 4);
  size_t group = (size_t)std::strtoull(argv[4], nullptr, 10); if (!group || packed_bytes * 4 < n || groups * group < n) return 3;
  float* dr; unsigned char* dp; float* ds; double* out; double zero = 0.0;
  hipMalloc(&dr, n*4); hipMalloc(&dp, packed_bytes); hipMalloc(&ds, groups*4); hipMalloc(&out, 8);
  hipMemcpy(dr,ref.data(),n*4,hipMemcpyHostToDevice); hipMemcpy(dp,packed.data(),packed_bytes,hipMemcpyHostToDevice); hipMemcpy(ds,scales.data(),groups*4,hipMemcpyHostToDevice); hipMemcpy(out,&zero,8,hipMemcpyHostToDevice);
  packed_mse<<<(n+255)/256,256>>>(dr,dp,ds,out,n,group); hipDeviceSynchronize(); double sum; hipMemcpy(&sum,out,8,hipMemcpyDeviceToHost); std::printf("%.17g\n", sum/(double)n);
  hipFree(dr); hipFree(dp); hipFree(ds); hipFree(out); return 0;
}
"#;

pub struct Mi300xTernaryScorer {
    _dir: TempDir,
    exe: std::path::PathBuf,
    packed_exe: std::path::PathBuf,
}

impl Mi300xTernaryScorer {
    pub fn from_env() -> Result<Option<Self>, String> {
        if std::env::var("PRISM_MI300X_GPU").ok().as_deref() != Some("1") {
            return Ok(None);
        }
        let dir = tempfile::tempdir().map_err(|e| e.to_string())?;
        let src = dir.path().join("ternary_rmse.hip");
        let exe = dir.path().join("ternary_rmse");
        let packed_src = dir.path().join("ternary_packed_mse.hip");
        let packed_exe = dir.path().join("ternary_packed_mse");
        std::fs::write(&src, KERNEL).map_err(|e| e.to_string())?;
        std::fs::write(&packed_src, PACKED_KERNEL).map_err(|e| e.to_string())?;
        let status = Command::new("hipcc")
            .args([
                "--offload-arch=gfx942",
                "-O3",
                src.to_str().unwrap(),
                "-o",
                exe.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| format!("hipcc unavailable: {e}"))?;
        if !status.success() {
            return Err("hipcc failed to build MI300X ternary scorer".into());
        }
        let packed_status = Command::new("hipcc")
            .args([
                "--offload-arch=gfx942",
                "-O3",
                packed_src.to_str().unwrap(),
                "-o",
                packed_exe.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| format!("hipcc unavailable: {e}"))?;
        if !packed_status.success() {
            return Err("hipcc failed to build packed MI300X ternary scorer".into());
        }
        Ok(Some(Self {
            _dir: dir,
            exe,
            packed_exe,
        }))
    }

    pub fn mean_squared_error(&self, reference: &[f32], candidate: &[f32]) -> Result<f64, String> {
        if reference.is_empty() || reference.len() != candidate.len() {
            return Err("invalid ternary scorer buffers".into());
        }
        let a = self.exe.with_extension("reference.bin");
        let b = self.exe.with_extension("candidate.bin");
        write_f32(&a, reference)?;
        write_f32(&b, candidate)?;
        let output = Command::new(&self.exe)
            .env("LD_LIBRARY_PATH", rocm_library_path())
            .args([a.to_str().unwrap(), b.to_str().unwrap()])
            .output()
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .map_err(|e| format!("invalid GPU scorer output: {e}"))
    }

    /// Score Prism 2-bit ternary values (0=zero, 1=+1, 2=-1) with one scale
    /// per group. Decoding and reconstruction happen inside the MI300X kernel.
    pub fn packed_mean_squared_error(
        &self,
        reference: &[f32],
        packed: &[u8],
        scales: &[f32],
        group_size: usize,
    ) -> Result<f64, String> {
        if reference.is_empty()
            || group_size == 0
            || packed.len() * 4 < reference.len()
            || scales.len() * group_size < reference.len()
        {
            return Err("invalid packed ternary scorer buffers".into());
        }
        let a = self.packed_exe.with_extension("reference.bin");
        let b = self.packed_exe.with_extension("packed.bin");
        let c = self.packed_exe.with_extension("scales.bin");
        write_f32(&a, reference)?;
        std::fs::write(&b, packed).map_err(|e| e.to_string())?;
        write_f32(&c, scales)?;
        let output = Command::new(&self.packed_exe)
            .env("LD_LIBRARY_PATH", rocm_library_path())
            .args([
                a.to_str().unwrap(),
                b.to_str().unwrap(),
                c.to_str().unwrap(),
                &group_size.to_string(),
            ])
            .output()
            .map_err(|e| e.to_string())?;
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&b);
        let _ = std::fs::remove_file(&c);
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .map_err(|e| format!("invalid packed GPU scorer output: {e}"))
    }
}

fn rocm_library_path() -> String {
    let existing = std::env::var_os("LD_LIBRARY_PATH")
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_default();
    ["/opt/rocm/lib", "/opt/rocm/lib64", existing.as_str()]
        .into_iter()
        .filter(|path| !path.is_empty())
        .collect::<Vec<_>>()
        .join(":")
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::Mi300xTernaryScorer;

    #[test]
    fn mi300x_scorer_smoke_when_enabled() {
        let Ok(Some(scorer)) = Mi300xTernaryScorer::from_env() else {
            return;
        };
        let error = scorer
            .mean_squared_error(&[1.0, 2.0, 3.0, 4.0], &[1.0, 1.0, 4.0, 2.0])
            .expect("MI300X scorer should execute");
        assert!((error - 1.5).abs() < 1e-9, "unexpected GPU MSE: {error}");
    }
}
