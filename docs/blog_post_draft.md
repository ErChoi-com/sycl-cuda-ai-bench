# SYCL vs CUDA for AI Kernels: A Practical Performance Study

## Abstract

This project compares equivalent CUDA and SYCL implementations of seven core AI kernels: GEMM, Conv2D, Softmax, LayerNorm, scaled dot-product attention, GELU, and transpose. The goal is not just raw speed rankings, but to understand where portability is smooth, where it is costly, and how compiler/runtime behavior shapes performance across heterogeneous devices.

## Motivation

Heterogeneous AI software stacks are increasingly expected to run across vendors and architectures. CUDA remains dominant, while SYCL is a key open portability path. A fair side-by-side benchmark helps answer three questions:

1. What performance is achievable with structurally similar kernels?
2. Which bottlenecks are architecture-driven vs compiler-driven?
3. Which optimizations transfer well across programming models?

## Methodology

- Implemented each kernel in both CUDA and SYCL with similar data layouts and algorithmic decomposition.
- Captured average latency, throughput, and effective bandwidth.
- Validated output numerics against CPU references for representative subproblems.
- Profiled both backends with vendor tools:
  - CUDA: Nsight Systems + Nsight Compute
  - SYCL: VTune + Advisor

## Kernels

### 1. Tiled GEMM

- CUDA: shared memory tiles + unrolled reduction.
- SYCL: local accessor tile strategy mirroring CUDA decomposition.

### 2. Direct Conv2D (NCHW)

- Baseline direct convolution to expose memory traffic and cache pressure.

### 3. Row-wise Softmax

- Reduction-heavy pattern common in attention blocks.

### 4. LayerNorm

- Mean/variance reduction + normalization, representative of transformer blocks.

### 5. Scaled Dot-Product Attention

- Decomposed into QK^T, softmax, and PV to isolate stage-wise costs.

## Results (fill from CSV)

| Kernel | CUDA Avg (ms) | SYCL Avg (ms) | Relative Speedup |
|---|---:|---:|---:|
| MatMul |  |  |  |
| Conv2D |  |  |  |
| Softmax |  |  |  |
| LayerNorm |  |  |  |
| Attention |  |  |  |

## Discussion

### Where SYCL matched well

- Add your strongest parity findings here.

### Where CUDA led

- Add kernels where compiler/runtime maturity or hardware mapping gave CUDA a clear advantage.

### Portability Friction Points

- Local/shared memory behavior differences.
- Reduction and synchronization idiom differences.
- Tooling differences for low-level occupancy and stall diagnostics.

## Compiler and Co-Design Implications

- Which optimizations should be automated by compiler passes?
- Which kernels motivate hardware-centric features (e.g., matrix engines, cache controls, async copy)?
- How can backend codegen reduce manual backend-specialized tuning?

## Conclusion

SYCL can deliver competitive performance for several AI primitives when kernel structure and memory behavior are tuned carefully. However, the highest-performance path still often requires backend-aware optimization and deep profiling. Bridging this gap is exactly where compiler research and hardware/software co-design can have outsized impact.
