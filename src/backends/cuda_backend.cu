#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backends/backend.hpp"

namespace bench {
namespace {

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err__ = (call);                                                            \
        if (err__ != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
        }                                                                                      \
    } while (0)

constexpr int TILE = 16;

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int m, int n, int k) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    for (int tileK = 0; tileK < (k + TILE - 1) / TILE; ++tileK) {
        const int aCol = tileK * TILE + threadIdx.x;
        const int bRow = tileK * TILE + threadIdx.y;

        as[threadIdx.y][threadIdx.x] = (row < m && aCol < k) ? a[row * k + aCol] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (bRow < k && col < n) ? b[bRow * n + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += as[threadIdx.y][t] * bs[t][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = acc;
    }
}

__global__ void conv2d_nchw_kernel(const float* input,
                                   const float* weight,
                                   const float* bias,
                                   float* output,
                                   int n,
                                   int c,
                                   int h,
                                   int w,
                                   int outC,
                                   int kH,
                                   int kW,
                                   int stride,
                                   int pad,
                                   int outH,
                                   int outW) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * outC * outH * outW;
    if (idx >= total) {
        return;
    }

    const int ow = idx % outW;
    const int oh = (idx / outW) % outH;
    const int oc = (idx / (outW * outH)) % outC;
    const int batch = idx / (outW * outH * outC);

    float sum = bias ? bias[oc] : 0.0f;

    for (int ic = 0; ic < c; ++ic) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                const int ih = oh * stride - pad + kh;
                const int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                    const int inOffset = ((batch * c + ic) * h + ih) * w + iw;
                    const int wOffset = ((oc * c + ic) * kH + kh) * kW + kw;
                    sum += input[inOffset] * weight[wOffset];
                }
            }
        }
    }

    output[idx] = sum;
}

__global__ void softmax_rows_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float shared[];
    float* maxBuf = shared;
    float* sumBuf = shared + blockDim.x;

    const int row = blockIdx.x;
    const int lane = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float localMax = -CUDART_INF_F;
    for (int col = lane; col < cols; col += blockDim.x) {
        localMax = fmaxf(localMax, input[row * cols + col]);
    }
    maxBuf[lane] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            maxBuf[lane] = fmaxf(maxBuf[lane], maxBuf[lane + stride]);
        }
        __syncthreads();
    }

    const float maxValue = maxBuf[0];

    float localSum = 0.0f;
    for (int col = lane; col < cols; col += blockDim.x) {
        localSum += expf(input[row * cols + col] - maxValue);
    }
    sumBuf[lane] = localSum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sumBuf[lane] += sumBuf[lane + stride];
        }
        __syncthreads();
    }

    const float denom = sumBuf[0] + 1e-12f;
    for (int col = lane; col < cols; col += blockDim.x) {
        output[row * cols + col] = expf(input[row * cols + col] - maxValue) / denom;
    }
}

__global__ void layernorm_rows_kernel(const float* input, float* output, int rows, int cols, float epsilon) {
    extern __shared__ float shared[];
    float* sumBuf = shared;
    float* sqBuf = shared + blockDim.x;

    const int row = blockIdx.x;
    const int lane = threadIdx.x;

    if (row >= rows) {
        return;
    }

    float localSum = 0.0f;
    float localSq = 0.0f;

    for (int col = lane; col < cols; col += blockDim.x) {
        const float value = input[row * cols + col];
        localSum += value;
        localSq += value * value;
    }

    sumBuf[lane] = localSum;
    sqBuf[lane] = localSq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sumBuf[lane] += sumBuf[lane + stride];
            sqBuf[lane] += sqBuf[lane + stride];
        }
        __syncthreads();
    }

    const float mean = sumBuf[0] / static_cast<float>(cols);
    const float var = sqBuf[0] / static_cast<float>(cols) - mean * mean;
    const float invStd = rsqrtf(var + epsilon);

    for (int col = lane; col < cols; col += blockDim.x) {
        const float value = input[row * cols + col];
        output[row * cols + col] = (value - mean) * invStd;
    }
}

__global__ void gelu_kernel(const float* input, float* output, int count) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    const float x = input[idx];
    const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
    output[idx] = 0.5f * x * (1.0f + tanhf(u));
}

__global__ void transpose_tiled_kernel(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];

    const int x = blockIdx.x * TILE + threadIdx.x;
    const int y = blockIdx.y * TILE + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }
    __syncthreads();

    const int tx = blockIdx.y * TILE + threadIdx.x;
    const int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < rows && ty < cols) {
        output[ty * rows + tx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void attention_scores_kernel(const float* q,
                                        const float* k,
                                        float* scores,
                                        int batchHeads,
                                        int seq,
                                        int dim,
                                        float scale) {
    const int key = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;

    if (bh >= batchHeads || query >= seq || key >= seq) {
        return;
    }

    const float* qPtr = q + (bh * seq + query) * dim;
    const float* kPtr = k + (bh * seq + key) * dim;

    float dot = 0.0f;
    for (int d = 0; d < dim; ++d) {
        dot += qPtr[d] * kPtr[d];
    }

    scores[(bh * seq + query) * seq + key] = dot * scale;
}

__global__ void attention_output_kernel(const float* probs,
                                        const float* v,
                                        float* out,
                                        int batchHeads,
                                        int seq,
                                        int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    const int query = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;

    if (bh >= batchHeads || query >= seq || d >= dim) {
        return;
    }

    float acc = 0.0f;
    const float* pRow = probs + (bh * seq + query) * seq;
    for (int key = 0; key < seq; ++key) {
        const float value = v[(bh * seq + key) * dim + d];
        acc += pRow[key] * value;
    }

    out[(bh * seq + query) * dim + d] = acc;
}

template <typename LaunchFn>
KernelMetrics run_timed_cuda(const BenchmarkOptions& options, LaunchFn&& launch) {
    std::vector<float> timings;
    timings.reserve(static_cast<std::size_t>(options.measuredIterations));

    for (int i = 0; i < options.warmupIterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < options.measuredIterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        timings.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    KernelMetrics metrics;
    metrics.minMs = *std::min_element(timings.begin(), timings.end());
    metrics.maxMs = *std::max_element(timings.begin(), timings.end());
    metrics.avgMs = std::accumulate(timings.begin(), timings.end(), 0.0) / static_cast<double>(timings.size());
    return metrics;
}

std::vector<float> matmul_cpu(const std::vector<float>& a, const std::vector<float>& b, int m, int n, int k) {
    std::vector<float> out(static_cast<std::size_t>(m) * n, 0.0f);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; ++kk) {
                acc += a[row * k + kk] * b[kk * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    return out;
}

std::vector<float> softmax_cpu(const std::vector<float>& in, int rows, int cols) {
    std::vector<float> out(in.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        float maxV = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < cols; ++col) {
            maxV = std::max(maxV, in[row * cols + col]);
        }
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += std::exp(in[row * cols + col] - maxV);
        }
        for (int col = 0; col < cols; ++col) {
            out[row * cols + col] = std::exp(in[row * cols + col] - maxV) / sum;
        }
    }
    return out;
}

std::vector<float> layernorm_cpu(const std::vector<float>& in, int rows, int cols, float epsilon) {
    std::vector<float> out(in.size(), 0.0f);
    for (int row = 0; row < rows; ++row) {
        float sum = 0.0f;
        float sq = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const float value = in[row * cols + col];
            sum += value;
            sq += value * value;
        }
        const float mean = sum / static_cast<float>(cols);
        const float var = sq / static_cast<float>(cols) - mean * mean;
        const float invStd = 1.0f / std::sqrt(var + epsilon);
        for (int col = 0; col < cols; ++col) {
            out[row * cols + col] = (in[row * cols + col] - mean) * invStd;
        }
    }
    return out;
}

std::vector<float> gelu_cpu(const std::vector<float>& in) {
    std::vector<float> out(in.size(), 0.0f);
    for (std::size_t i = 0; i < in.size(); ++i) {
        const float x = in[i];
        const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
        out[i] = 0.5f * x * (1.0f + std::tanh(u));
    }
    return out;
}

std::vector<float> transpose_cpu(const std::vector<float>& in, int rows, int cols) {
    std::vector<float> out(static_cast<std::size_t>(rows) * cols, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
    return out;
}

class CudaBackend final : public KernelBackend {
public:
    std::string name() const override {
        return "CUDA";
    }

    std::string device_name() const override {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        return prop.name;
    }

    void synchronize() override {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    KernelMetrics run_matmul(const MatmulProblem& problem, const BenchmarkOptions& options) override {
        const int m = problem.m;
        const int n = problem.n;
        const int k = problem.k;

        const auto hA = make_random_vector(static_cast<std::size_t>(m) * k, options.randomSeed);
        const auto hB = make_random_vector(static_cast<std::size_t>(k) * n, options.randomSeed + 1);
        std::vector<float> hC(static_cast<std::size_t>(m) * n, 0.0f);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice));

        const dim3 block(TILE, TILE);
        const dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);

        auto metrics = run_timed_cuda(options, [&]() {
            matmul_tiled_kernel<<<grid, block>>>(dA, dB, dC, m, n, k);
        });

        metrics.tflops = (2.0 * m * n * k) / (metrics.avgMs * 1e9);
        const std::size_t bytes = (hA.size() + hB.size() + hC.size()) * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost));
            const int vm = std::min(m, 128);
            const int vn = std::min(n, 128);
            const int vk = std::min(k, 128);
            std::vector<float> vA(static_cast<std::size_t>(vm) * vk);
            std::vector<float> vB(static_cast<std::size_t>(vk) * vn);
            for (int i = 0; i < vm; ++i) {
                std::copy_n(hA.begin() + static_cast<std::size_t>(i) * k, vk, vA.begin() + static_cast<std::size_t>(i) * vk);
            }
            for (int i = 0; i < vk; ++i) {
                std::copy_n(hB.begin() + static_cast<std::size_t>(i) * n, vn, vB.begin() + static_cast<std::size_t>(i) * vn);
            }
            const auto ref = matmul_cpu(vA, vB, vm, vn, vk);
            std::vector<float> got(static_cast<std::size_t>(vm) * vn);
            for (int i = 0; i < vm; ++i) {
                std::copy_n(hC.begin() + static_cast<std::size_t>(i) * n, vn, got.begin() + static_cast<std::size_t>(i) * vn);
            }
            metrics.valid = allclose(got, ref, 5e-2f, 5e-2f);
        }

        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
        return metrics;
    }

    KernelMetrics run_conv2d(const Conv2dProblem& problem, const BenchmarkOptions& options) override {
        const int outH = (problem.h + 2 * problem.padding - problem.kernelH) / problem.stride + 1;
        const int outW = (problem.w + 2 * problem.padding - problem.kernelW) / problem.stride + 1;

        const std::size_t inSize = static_cast<std::size_t>(problem.n) * problem.c * problem.h * problem.w;
        const std::size_t weightSize = static_cast<std::size_t>(problem.outChannels) * problem.c * problem.kernelH * problem.kernelW;
        const std::size_t outSize = static_cast<std::size_t>(problem.n) * problem.outChannels * outH * outW;

        const auto hIn = make_random_vector(inSize, options.randomSeed + 3);
        const auto hW = make_random_vector(weightSize, options.randomSeed + 4);
        const auto hBias = make_random_vector(problem.outChannels, options.randomSeed + 5);

        float *dIn = nullptr, *dW = nullptr, *dBias = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, inSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dW, weightSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dBias, hBias.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, outSize * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), inSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW, hW.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice));

        const int total = static_cast<int>(outSize);
        const int block = 256;
        const int grid = (total + block - 1) / block;

        auto metrics = run_timed_cuda(options, [&]() {
            conv2d_nchw_kernel<<<grid, block>>>(dIn,
                                                dW,
                                                dBias,
                                                dOut,
                                                problem.n,
                                                problem.c,
                                                problem.h,
                                                problem.w,
                                                problem.outChannels,
                                                problem.kernelH,
                                                problem.kernelW,
                                                problem.stride,
                                                problem.padding,
                                                outH,
                                                outW);
        });

        const double opsPerOutput = 2.0 * problem.c * problem.kernelH * problem.kernelW;
        metrics.tflops = (opsPerOutput * outSize) / (metrics.avgMs * 1e9);
        const std::size_t bytes = (inSize + weightSize + outSize + hBias.size()) * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);
        metrics.note = "Direct NCHW conv kernel";

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dW));
        CUDA_CHECK(cudaFree(dBias));
        CUDA_CHECK(cudaFree(dOut));
        return metrics;
    }

    KernelMetrics run_softmax(const SoftmaxProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 8);

        float *dIn = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), count * sizeof(float), cudaMemcpyHostToDevice));

        const int threads = 256;
        const int blocks = problem.rows;
        const std::size_t shmem = static_cast<std::size_t>(threads) * 2 * sizeof(float);

        auto metrics = run_timed_cuda(options, [&]() {
            softmax_rows_kernel<<<blocks, threads, shmem>>>(dIn, dOut, problem.rows, problem.cols);
        });

        const double ops = 3.0 * count;
        metrics.tflops = ops / (metrics.avgMs * 1e9);
        const std::size_t bytes = 2 * count * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            std::vector<float> hOut(count);
            CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, count * sizeof(float), cudaMemcpyDeviceToHost));
            const int vr = std::min(problem.rows, 256);
            const int vc = std::min(problem.cols, 512);
            std::vector<float> vIn(static_cast<std::size_t>(vr) * vc);
            std::vector<float> vOut(static_cast<std::size_t>(vr) * vc);
            for (int r = 0; r < vr; ++r) {
                std::copy_n(hIn.begin() + static_cast<std::size_t>(r) * problem.cols,
                            vc,
                            vIn.begin() + static_cast<std::size_t>(r) * vc);
                std::copy_n(hOut.begin() + static_cast<std::size_t>(r) * problem.cols,
                            vc,
                            vOut.begin() + static_cast<std::size_t>(r) * vc);
            }
            const auto ref = softmax_cpu(vIn, vr, vc);
            metrics.valid = allclose(vOut, ref, 5e-3f, 5e-3f);
        }

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));
        return metrics;
    }

    KernelMetrics run_layernorm(const LayerNormProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 13);

        float *dIn = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), count * sizeof(float), cudaMemcpyHostToDevice));

        const int threads = 256;
        const int blocks = problem.rows;
        const std::size_t shmem = static_cast<std::size_t>(threads) * 2 * sizeof(float);

        auto metrics = run_timed_cuda(options, [&]() {
            layernorm_rows_kernel<<<blocks, threads, shmem>>>(dIn, dOut, problem.rows, problem.cols, problem.epsilon);
        });

        const double ops = 6.0 * count;
        metrics.tflops = ops / (metrics.avgMs * 1e9);
        const std::size_t bytes = 2 * count * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            std::vector<float> hOut(count);
            CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, count * sizeof(float), cudaMemcpyDeviceToHost));
            const int vr = std::min(problem.rows, 256);
            const int vc = std::min(problem.cols, 512);
            std::vector<float> vIn(static_cast<std::size_t>(vr) * vc);
            std::vector<float> vOut(static_cast<std::size_t>(vr) * vc);
            for (int r = 0; r < vr; ++r) {
                std::copy_n(hIn.begin() + static_cast<std::size_t>(r) * problem.cols,
                            vc,
                            vIn.begin() + static_cast<std::size_t>(r) * vc);
                std::copy_n(hOut.begin() + static_cast<std::size_t>(r) * problem.cols,
                            vc,
                            vOut.begin() + static_cast<std::size_t>(r) * vc);
            }
            const auto ref = layernorm_cpu(vIn, vr, vc, problem.epsilon);
            metrics.valid = allclose(vOut, ref, 8e-3f, 8e-3f);
        }

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));
        return metrics;
    }

    KernelMetrics run_gelu(const GeluProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 30);
        std::vector<float> hOut(count, 0.0f);

        float *dIn = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), count * sizeof(float), cudaMemcpyHostToDevice));

        const int threads = 256;
        const int blocks = static_cast<int>((count + threads - 1) / threads);
        auto metrics = run_timed_cuda(options, [&]() {
            gelu_kernel<<<blocks, threads>>>(dIn, dOut, static_cast<int>(count));
        });

        metrics.tflops = (10.0 * count) / (metrics.avgMs * 1e9);
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, count * sizeof(float), cudaMemcpyDeviceToHost));
            const auto ref = gelu_cpu(hIn);
            metrics.valid = allclose(hOut, ref, 2e-3f, 2e-3f);
        }

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));
        return metrics;
    }

    KernelMetrics run_transpose(const TransposeProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 33);
        std::vector<float> hOut(count, 0.0f);

        float *dIn = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dIn, count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dIn, hIn.data(), count * sizeof(float), cudaMemcpyHostToDevice));

        const dim3 block(TILE, TILE);
        const dim3 grid((problem.cols + TILE - 1) / TILE, (problem.rows + TILE - 1) / TILE);

        auto metrics = run_timed_cuda(options, [&]() {
            transpose_tiled_kernel<<<grid, block>>>(dIn, dOut, problem.rows, problem.cols);
        });

        metrics.tflops = 0.0;
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, count * sizeof(float), cudaMemcpyDeviceToHost));
            const auto ref = transpose_cpu(hIn, problem.rows, problem.cols);
            metrics.valid = allclose(hOut, ref, 1e-5f, 1e-5f);
        }

        CUDA_CHECK(cudaFree(dIn));
        CUDA_CHECK(cudaFree(dOut));
        return metrics;
    }

    KernelMetrics run_attention(const AttentionProblem& problem, const BenchmarkOptions& options) override {
        const int batchHeads = problem.batch * problem.heads;
        const int seq = problem.seqLen;
        const int dim = problem.headDim;

        const std::size_t qkvCount = static_cast<std::size_t>(batchHeads) * seq * dim;
        const std::size_t scoreCount = static_cast<std::size_t>(batchHeads) * seq * seq;

        const auto hQ = make_random_vector(qkvCount, options.randomSeed + 19);
        const auto hK = make_random_vector(qkvCount, options.randomSeed + 20);
        const auto hV = make_random_vector(qkvCount, options.randomSeed + 21);

        float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dScore = nullptr, *dProb = nullptr, *dOut = nullptr;
        CUDA_CHECK(cudaMalloc(&dQ, qkvCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dK, qkvCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dV, qkvCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dScore, scoreCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dProb, scoreCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dOut, qkvCount * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(dQ, hQ.data(), qkvCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK, hK.data(), qkvCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV, hV.data(), qkvCount * sizeof(float), cudaMemcpyHostToDevice));

        const dim3 scoreBlock(16, 16);
        const dim3 scoreGrid((seq + 15) / 16, (seq + 15) / 16, batchHeads);
        const int softmaxThreads = 256;
        const int softmaxRows = batchHeads * seq;
        const std::size_t softmaxShmem = static_cast<std::size_t>(softmaxThreads) * 2 * sizeof(float);
        const dim3 outBlock(16, 16);
        const dim3 outGrid((dim + 15) / 16, (seq + 15) / 16, batchHeads);
        const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

        auto metrics = run_timed_cuda(options, [&]() {
            attention_scores_kernel<<<scoreGrid, scoreBlock>>>(dQ, dK, dScore, batchHeads, seq, dim, scale);
            softmax_rows_kernel<<<softmaxRows, softmaxThreads, softmaxShmem>>>(dScore, dProb, softmaxRows, seq);
            attention_output_kernel<<<outGrid, outBlock>>>(dProb, dV, dOut, batchHeads, seq, dim);
        });

        const double scoreOps = 2.0 * batchHeads * seq * seq * dim;
        const double outputOps = 2.0 * batchHeads * seq * seq * dim;
        metrics.tflops = (scoreOps + outputOps) / (metrics.avgMs * 1e9);
        const std::size_t bytes = (3 * qkvCount + 2 * scoreCount + qkvCount) * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);
        metrics.note = "Scaled dot-product attention (QK^T + softmax + PV)";

        CUDA_CHECK(cudaFree(dQ));
        CUDA_CHECK(cudaFree(dK));
        CUDA_CHECK(cudaFree(dV));
        CUDA_CHECK(cudaFree(dScore));
        CUDA_CHECK(cudaFree(dProb));
        CUDA_CHECK(cudaFree(dOut));
        return metrics;
    }
};

}  // namespace

std::unique_ptr<KernelBackend> create_backend() {
    return std::make_unique<CudaBackend>();
}

}  // namespace bench
