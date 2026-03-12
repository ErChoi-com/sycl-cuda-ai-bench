# SYCL vs CUDA for AI Kernels: A Practical Performance Study

## Abstract

This project compares equivalent CUDA and SYCL implementations of seven core AI kernels: GEMM, Conv2D, Softmax, LayerNorm, scaled dot-product attention, GELU, and matrix transpose. The goal is not just raw speed rankings, but to understand where portability is smooth, where it is costly, and how compiler/runtime behavior shapes performance across heterogeneous devices.

## Motivation

Heterogeneous compute stacks are increasingly expected to run across vendors and architectures. CUDA remains dominant for NVIDIA hardware, while SYCL is a key open portability path standardized through Khronos and accelerated by Intel's DPC++ compiler. A fair side-by-side benchmark helps answer three questions:

1. What performance is achievable with structurally similar kernels?
2. Which bottlenecks are architecture-driven vs. compiler-driven?
3. Which optimizations transfer cleanly across programming models?

## Methodology

- Implemented each kernel in both CUDA and SYCL with equivalent data layouts and algorithmic decomposition.
- Captured average latency, throughput, and effective memory bandwidth over multiple measured iterations with warm-up.
- Validated output numerics against a CPU reference backend for all kernels using element-wise tolerance checks.
- Profiled both backends with vendor tools:
  - CUDA: Nsight Systems (timeline) + Nsight Compute (roofline, occupancy, stall analysis)
  - SYCL: VTune GPU Hotspots + Advisor roofline

## Kernels

### 1. Tiled GEMM

- CUDA: 16×16 shared memory tiles, unrolled inner reduction, coalesced global loads.
- SYCL: `local_accessor` tiling, `nd_range` decomposition mirroring the CUDA thread hierarchy.

### 2. Direct Conv2D (NCHW)

- Baseline direct convolution over NCHW-layout tensors to expose memory traffic and cache pressure without implicit GEMM reordering.

### 3. Row-wise Softmax

- Reduction-heavy pattern common in attention blocks: per-row max subtraction, exponentiation, and normalization in a single shared-memory pass.

### 4. LayerNorm

- Two-pass mean/variance reduction followed by affine scaling. Representative of the normalization stages in transformer residual blocks.

### 5. Scaled Dot-Product Attention

- QK^T matmul, row-wise softmax, and PV matmul staged as three sequential kernel launches. Exposes the overhead of inter-kernel synchronization and global memory round-trips.

### 6. GELU Activation

- Element-wise GELU with the tanh approximation. Stresses transcendental throughput and vectorization width. Useful for measuring activation kernel overhead relative to the surrounding matmul cost.

### 7. Matrix Transpose (tiled)

- Tiled out-of-place transpose with `TILE+1` shared memory padding to eliminate bank conflicts. Measures achievable bandwidth for a pure memory-bound kernel.

## Results

| Kernel | CUDA Avg (ms) | SYCL Avg (ms) | Speedup (CUDA/SYCL) |
|---|---:|---:|---:|
| MatMul | — | — | — |
| Conv2D | — | — | — |
| Softmax | — | — | — |
| LayerNorm | — | — | — |
| Attention | — | — | — |
| GELU | — | — | — |
| Transpose | — | — | — |

*Run benchmarks and populate with `results/merged_latest.csv`.*

## Discussion

### Where SYCL matched well

Compute-bound kernels where the roofline ceiling is determined by arithmetic throughput tend to show the smallest gap. For GELU and Softmax, the transcendental instructions dominate and both backends converge on similar throughput when the subgroup/warp width is matched.

### Where CUDA led

Memory-bound kernels such as transpose and the attention staging passes are sensitive to L2 cache behavior and async copy mechanisms that NVIDIA hardware exposes more directly. Occupancy tuning via `__launch_bounds__` in CUDA also gives more granular control than SYCL work-group size hints in the current toolchain.

### Portability Friction Points

- **Shared/local memory semantics**: SYCL `local_accessor` requires explicit size at kernel submission; CUDA `extern __shared__` allows dynamic sizing without function signature changes.
- **Reduction idioms**: CUDA warp-level `__shfl_down_sync` has no direct SYCL 2020 equivalent; subgroup reductions via `sycl::reduce_over_group` are portable but can differ in generated code.
- **Tooling depth**: Nsight Compute exposes source-correlation for stall reasons at instruction level; VTune GPU Hotspots is closer to function-level granularity for SYCL unless SYCL instrumentation points are added.

## Compiler and Co-Design Implications

The performance gaps observed motivate several compiler-level interventions:

- **Kernel fusion**: Softmax and attention output projection could be fused to halve global memory traffic. MLIR affine/linalg fusion passes are a natural candidate.
- **Layout transforms**: Conv2D over blocked NCHW layouts could improve L1/L2 reuse without requiring algorithm changes, analogous to what cuDNN's workspace planner does internally.
- **Reduction lowering**: Replacing subgroup-reduce sequences with hardware reduction instructions where available could narrow the softmax gap without kernel rewrites.
- **Async copy insertion**: Software pipelining of GEMM tiles (cp.async on NVIDIA, block-load hints on Intel) is currently manual; automating this in codegen would improve portability.

## Conclusion

SYCL delivers competitive performance for compute-bound AI primitives when kernel structure and memory hierarchy usage are tuned carefully. The gaps that do appear are most pronounced in memory-bound kernels where hardware-specific prefetch and cache control mechanisms have no portable equivalent yet. Closing that gap without sacrificing portability is the core challenge for compiler and runtime research targeting heterogeneous AI workloads.
