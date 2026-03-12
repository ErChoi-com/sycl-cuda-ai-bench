# SYCL/CUDA AI Kernel Benchmark Suite

Side-by-side CUDA and SYCL/DPC++ implementations for AI kernels, with unified metrics, profiling hooks, and portability analysis artifacts.

## Kernels (7)

1. MatMul (tiled GEMM)
2. Conv2D (direct NCHW)
3. Softmax (row-wise)
4. LayerNorm (row-wise)
5. Scaled dot-product attention
6. GELU activation
7. Matrix transpose (tiled)

## Build

Prereqs: CMake 3.23+, C++17 compiler. Optional toolchains: CUDA Toolkit, Intel oneAPI DPC++.

```powershell
cmake -S . -B build
cmake --build build --config Release -j
```

Notes:

- `benchmark_cpu` always builds (validation backend).
- `benchmark_cuda` and `benchmark_sycl` are auto-skipped if toolchains are unavailable.

## Run

```powershell
./scripts/run_benchmarks.ps1 -Quick
./scripts/run_benchmarks.ps1
```

Outputs:

- `results/cpu_latest.csv`
- `results/cuda_latest.csv` (if CUDA available)
- `results/sycl_latest.csv` (if SYCL available)
- `results/merged_latest.csv` (auto-created when CUDA+SYCL are both present)

## Profiling

```powershell
./scripts/profile_nvidia.ps1 -Quick
./scripts/profile_intel.ps1 -Quick
```

## Analysis Docs

- `docs/perf_analysis.md`
- `docs/writeup.md`
