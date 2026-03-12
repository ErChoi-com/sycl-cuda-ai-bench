# Performance Analysis Worksheet: SYCL vs CUDA for AI Kernels

Use this worksheet after each run to build a rigorous and reproducible comparison.

## 1. Test Matrix

- Date:
- Host CPU:
- Memory:
- OS + Driver versions:
- CUDA version:
- oneAPI version:
- GPU #1 (NVIDIA):
- GPU #2 (Intel):

## 2. Run Configuration

- Warmup iterations:
- Measured iterations:
- Validation enabled:
- Build flags:
- Compiler versions (nvcc, dpcpp/icpx):

## 3. Kernel Summary Table

| Kernel | CUDA Avg (ms) | SYCL Avg (ms) | Speedup (CUDA/SYCL) | CUDA TFLOP/s | SYCL TFLOP/s | Notes |
|---|---:|---:|---:|---:|---:|---|
| MatMul |  |  |  |  |  |  |
| Conv2D |  |  |  |  |  |  |
| Softmax |  |  |  |  |  |  |
| LayerNorm |  |  |  |  |  |  |
| Attention |  |  |  |  |  |  |
| GELU |  |  |  |  |  |  |
| Transpose |  |  |  |  |  |  |

## 4. Profiling Findings

For each kernel, capture:

- Occupancy / EU active / SM active
- Achieved memory bandwidth
- L2/L3/cache behavior
- Register pressure and spills
- Dominant stalls (memory, execution dependency, synchronization)

## 5. Portability and Optimization Observations

- Where identical algorithmic structure still diverged in performance.
- Backend-specific optimizations that were required.
- Compiler behavior differences (vectorization, unrolling, local memory usage).
- Runtime differences (launch overhead, queue behavior, synchronization semantics).

## 6. Co-Design Insights (Intel Compiler R&D angle)

- Which bottlenecks should be addressed by compiler transformations vs kernel rewrite.
- Candidate compiler pass opportunities (fusion, layout transform, loop scheduling).
- Potential hardware feature requests to reduce observed bottlenecks.

## 7. Compiler Pass Hypotheses

- Fusion hypothesis: fuse softmax + attention output stages to reduce global memory traffic.
- Layout hypothesis: transform NCHW convolution inputs to a blocked layout to improve cache reuse.
- Loop schedule hypothesis: tile and reorder LayerNorm/GELU loops to increase vectorization width.
- Reduction hypothesis: warp/subgroup reduction lowering improves softmax max/sum stages.
- Alias/readonly hypothesis: stronger no-alias and readonly metadata improves load/store scheduling.
- Prefetch hypothesis: software pipelining and async copy insertion can reduce memory stalls in GEMM/transpose.

For each hypothesis, record:

- Expected effect (latency, bandwidth, occupancy)
- Candidate LLVM/MLIR pass location
- Evidence from profiler counters
- Outcome after implementation

## 8. Next Experiment Queue

- Sweep tile sizes (8×8, 16×16, 32×32) for tiled GEMM and transpose; find peak occupancy per GPU.
- Fused softmax-attention pass: measure latency reduction vs. three-stage baseline.
- Conv2D with blocked NHWC layout vs. NCHW to isolate layout impact on cache reuse.
