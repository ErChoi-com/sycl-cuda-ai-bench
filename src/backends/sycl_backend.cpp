#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backends/backend.hpp"

namespace bench {
namespace {

constexpr int TILE = 16;

template <typename SubmitFn>
KernelMetrics run_timed_sycl(sycl::queue& queue, const BenchmarkOptions& options, SubmitFn&& submit) {
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(options.measuredIterations));

    for (int i = 0; i < options.warmupIterations; ++i) {
        submit().wait();
    }

    for (int i = 0; i < options.measuredIterations; ++i) {
        const auto start = std::chrono::high_resolution_clock::now();
        sycl::event ev = submit();
        ev.wait();
        const auto end = std::chrono::high_resolution_clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

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

class SyclBackend final : public KernelBackend {
public:
    SyclBackend()
        : queue_(sycl::default_selector_v, sycl::property_list{sycl::property::queue::enable_profiling()}) {}

    std::string name() const override {
        return "SYCL";
    }

    std::string device_name() const override {
        return queue_.get_device().get_info<sycl::info::device::name>();
    }

    void synchronize() override {
        queue_.wait();
    }

    KernelMetrics run_matmul(const MatmulProblem& problem, const BenchmarkOptions& options) override {
        const int m = problem.m;
        const int n = problem.n;
        const int k = problem.k;

        const auto hA = make_random_vector(static_cast<std::size_t>(m) * k, options.randomSeed);
        const auto hB = make_random_vector(static_cast<std::size_t>(k) * n, options.randomSeed + 1);
        std::vector<float> hC(static_cast<std::size_t>(m) * n, 0.0f);

        float* dA = sycl::malloc_device<float>(hA.size(), queue_);
        float* dB = sycl::malloc_device<float>(hB.size(), queue_);
        float* dC = sycl::malloc_device<float>(hC.size(), queue_);

        queue_.memcpy(dA, hA.data(), hA.size() * sizeof(float)).wait();
        queue_.memcpy(dB, hB.data(), hB.size() * sizeof(float)).wait();

        const sycl::range<2> local(TILE, TILE);
        const sycl::range<2> global(
            static_cast<std::size_t>((m + TILE - 1) / TILE) * TILE,
            static_cast<std::size_t>((n + TILE - 1) / TILE) * TILE);

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            return queue_.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> tileA(sycl::range<2>(TILE, TILE), h);
                sycl::local_accessor<float, 2> tileB(sycl::range<2>(TILE, TILE), h);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    const int row = static_cast<int>(item.get_global_id(0));
                    const int col = static_cast<int>(item.get_global_id(1));
                    const int lr = static_cast<int>(item.get_local_id(0));
                    const int lc = static_cast<int>(item.get_local_id(1));

                    float acc = 0.0f;
                    for (int tileK = 0; tileK < (k + TILE - 1) / TILE; ++tileK) {
                        const int aCol = tileK * TILE + lc;
                        const int bRow = tileK * TILE + lr;

                        tileA[lr][lc] = (row < m && aCol < k) ? dA[row * k + aCol] : 0.0f;
                        tileB[lr][lc] = (bRow < k && col < n) ? dB[bRow * n + col] : 0.0f;
                        item.barrier(sycl::access::fence_space::local_space);

                        #pragma unroll
                        for (int t = 0; t < TILE; ++t) {
                            acc += tileA[lr][t] * tileB[t][lc];
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (row < m && col < n) {
                        dC[row * n + col] = acc;
                    }
                });
            });
        });

        metrics.tflops = (2.0 * m * n * k) / (metrics.avgMs * 1e9);
        const std::size_t bytes = (hA.size() + hB.size() + hC.size()) * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            queue_.memcpy(hC.data(), dC, hC.size() * sizeof(float)).wait();
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
            metrics.valid = allclose(got, ref, 6e-2f, 6e-2f);
        }

        sycl::free(dA, queue_);
        sycl::free(dB, queue_);
        sycl::free(dC, queue_);
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

        float* dIn = sycl::malloc_device<float>(inSize, queue_);
        float* dW = sycl::malloc_device<float>(weightSize, queue_);
        float* dBias = sycl::malloc_device<float>(hBias.size(), queue_);
        float* dOut = sycl::malloc_device<float>(outSize, queue_);

        queue_.memcpy(dIn, hIn.data(), inSize * sizeof(float)).wait();
        queue_.memcpy(dW, hW.data(), weightSize * sizeof(float)).wait();
        queue_.memcpy(dBias, hBias.data(), hBias.size() * sizeof(float)).wait();

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            return queue_.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(outSize), [=](sycl::id<1> id) {
                    const int idx = static_cast<int>(id[0]);
                    const int ow = idx % outW;
                    const int oh = (idx / outW) % outH;
                    const int oc = (idx / (outW * outH)) % problem.outChannels;
                    const int batch = idx / (outW * outH * problem.outChannels);

                    float sum = dBias[oc];
                    for (int ic = 0; ic < problem.c; ++ic) {
                        for (int kh = 0; kh < problem.kernelH; ++kh) {
                            for (int kw = 0; kw < problem.kernelW; ++kw) {
                                const int ih = oh * problem.stride - problem.padding + kh;
                                const int iw = ow * problem.stride - problem.padding + kw;
                                if (ih >= 0 && ih < problem.h && iw >= 0 && iw < problem.w) {
                                    const int inOffset = ((batch * problem.c + ic) * problem.h + ih) * problem.w + iw;
                                    const int wOffset = ((oc * problem.c + ic) * problem.kernelH + kh) * problem.kernelW + kw;
                                    sum += dIn[inOffset] * dW[wOffset];
                                }
                            }
                        }
                    }

                    dOut[idx] = sum;
                });
            });
        });

        const double opsPerOutput = 2.0 * problem.c * problem.kernelH * problem.kernelW;
        metrics.tflops = (opsPerOutput * outSize) / (metrics.avgMs * 1e9);
        const std::size_t bytes = (inSize + weightSize + outSize + hBias.size()) * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);
        metrics.note = "Direct NCHW conv kernel";

        sycl::free(dIn, queue_);
        sycl::free(dW, queue_);
        sycl::free(dBias, queue_);
        sycl::free(dOut, queue_);
        return metrics;
    }

    KernelMetrics run_softmax(const SoftmaxProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 8);

        float* dIn = sycl::malloc_device<float>(count, queue_);
        float* dOut = sycl::malloc_device<float>(count, queue_);
        queue_.memcpy(dIn, hIn.data(), count * sizeof(float)).wait();

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            return queue_.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(problem.rows), [=](sycl::id<1> id) {
                    const int row = static_cast<int>(id[0]);
                    float maxV = -INFINITY;
                    for (int col = 0; col < problem.cols; ++col) {
                        maxV = sycl::fmax(maxV, dIn[row * problem.cols + col]);
                    }

                    float sum = 0.0f;
                    for (int col = 0; col < problem.cols; ++col) {
                        sum += sycl::exp(dIn[row * problem.cols + col] - maxV);
                    }

                    const float inv = 1.0f / (sum + 1e-12f);
                    for (int col = 0; col < problem.cols; ++col) {
                        dOut[row * problem.cols + col] = sycl::exp(dIn[row * problem.cols + col] - maxV) * inv;
                    }
                });
            });
        });

        const double ops = 3.0 * count;
        metrics.tflops = ops / (metrics.avgMs * 1e9);
        const std::size_t bytes = 2 * count * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            std::vector<float> hOut(count);
            queue_.memcpy(hOut.data(), dOut, count * sizeof(float)).wait();
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
            metrics.valid = allclose(vOut, ref, 6e-3f, 6e-3f);
        }

        sycl::free(dIn, queue_);
        sycl::free(dOut, queue_);
        return metrics;
    }

    KernelMetrics run_layernorm(const LayerNormProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 13);

        float* dIn = sycl::malloc_device<float>(count, queue_);
        float* dOut = sycl::malloc_device<float>(count, queue_);
        queue_.memcpy(dIn, hIn.data(), count * sizeof(float)).wait();

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            return queue_.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(problem.rows), [=](sycl::id<1> id) {
                    const int row = static_cast<int>(id[0]);
                    float sum = 0.0f;
                    float sq = 0.0f;
                    for (int col = 0; col < problem.cols; ++col) {
                        const float value = dIn[row * problem.cols + col];
                        sum += value;
                        sq += value * value;
                    }
                    const float mean = sum / static_cast<float>(problem.cols);
                    const float var = sq / static_cast<float>(problem.cols) - mean * mean;
                    const float invStd = sycl::rsqrt(var + problem.epsilon);
                    for (int col = 0; col < problem.cols; ++col) {
                        dOut[row * problem.cols + col] = (dIn[row * problem.cols + col] - mean) * invStd;
                    }
                });
            });
        });

        const double ops = 6.0 * count;
        metrics.tflops = ops / (metrics.avgMs * 1e9);
        const std::size_t bytes = 2 * count * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            std::vector<float> hOut(count);
            queue_.memcpy(hOut.data(), dOut, count * sizeof(float)).wait();
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

        sycl::free(dIn, queue_);
        sycl::free(dOut, queue_);
        return metrics;
    }

    KernelMetrics run_gelu(const GeluProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 30);
        std::vector<float> hOut(count, 0.0f);

        float* dIn = sycl::malloc_device<float>(count, queue_);
        float* dOut = sycl::malloc_device<float>(count, queue_);
        queue_.memcpy(dIn, hIn.data(), count * sizeof(float)).wait();

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            return queue_.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
                    const std::size_t i = static_cast<std::size_t>(id[0]);
                    const float x = dIn[i];
                    const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
                    dOut[i] = 0.5f * x * (1.0f + sycl::tanh(u));
                });
            });
        });

        metrics.tflops = (10.0 * count) / (metrics.avgMs * 1e9);
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            queue_.memcpy(hOut.data(), dOut, count * sizeof(float)).wait();
            const auto ref = gelu_cpu(hIn);
            metrics.valid = allclose(hOut, ref, 2e-3f, 2e-3f);
        }

        sycl::free(dIn, queue_);
        sycl::free(dOut, queue_);
        return metrics;
    }

    KernelMetrics run_transpose(const TransposeProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto hIn = make_random_vector(count, options.randomSeed + 33);
        std::vector<float> hOut(count, 0.0f);

        float* dIn = sycl::malloc_device<float>(count, queue_);
        float* dOut = sycl::malloc_device<float>(count, queue_);
        queue_.memcpy(dIn, hIn.data(), count * sizeof(float)).wait();

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            return queue_.submit([&](sycl::handler& h) {
                sycl::local_accessor<float, 2> tile(sycl::range<2>(TILE, TILE + 1), h);
                const sycl::range<2> local(TILE, TILE);
                const sycl::range<2> global(
                    static_cast<std::size_t>((problem.rows + TILE - 1) / TILE) * TILE,
                    static_cast<std::size_t>((problem.cols + TILE - 1) / TILE) * TILE);

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
                    const int y = static_cast<int>(item.get_global_id(0));
                    const int x = static_cast<int>(item.get_global_id(1));
                    const int ly = static_cast<int>(item.get_local_id(0));
                    const int lx = static_cast<int>(item.get_local_id(1));

                    if (x < problem.cols && y < problem.rows) {
                        tile[ly][lx] = dIn[y * problem.cols + x];
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    const int tx = static_cast<int>(item.get_group(0)) * TILE + lx;
                    const int ty = static_cast<int>(item.get_group(1)) * TILE + ly;
                    if (tx < problem.rows && ty < problem.cols) {
                        dOut[ty * problem.rows + tx] = tile[lx][ly];
                    }
                });
            });
        });

        metrics.tflops = 0.0;
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);

        if (options.validate) {
            queue_.memcpy(hOut.data(), dOut, count * sizeof(float)).wait();
            const auto ref = transpose_cpu(hIn, problem.rows, problem.cols);
            metrics.valid = allclose(hOut, ref, 1e-5f, 1e-5f);
        }

        sycl::free(dIn, queue_);
        sycl::free(dOut, queue_);
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

        float* dQ = sycl::malloc_device<float>(qkvCount, queue_);
        float* dK = sycl::malloc_device<float>(qkvCount, queue_);
        float* dV = sycl::malloc_device<float>(qkvCount, queue_);
        float* dScore = sycl::malloc_device<float>(scoreCount, queue_);
        float* dProb = sycl::malloc_device<float>(scoreCount, queue_);
        float* dOut = sycl::malloc_device<float>(qkvCount, queue_);

        queue_.memcpy(dQ, hQ.data(), qkvCount * sizeof(float)).wait();
        queue_.memcpy(dK, hK.data(), qkvCount * sizeof(float)).wait();
        queue_.memcpy(dV, hV.data(), qkvCount * sizeof(float)).wait();

        const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

        auto metrics = run_timed_sycl(queue_, options, [&]() {
            sycl::event ev1 = queue_.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(scoreCount), [=](sycl::id<1> id) {
                    const int idx = static_cast<int>(id[0]);
                    const int key = idx % seq;
                    const int query = (idx / seq) % seq;
                    const int bh = idx / (seq * seq);

                    const float* qPtr = dQ + (bh * seq + query) * dim;
                    const float* kPtr = dK + (bh * seq + key) * dim;
                    float dot = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        dot += qPtr[d] * kPtr[d];
                    }
                    dScore[idx] = dot * scale;
                });
            });

            sycl::event ev2 = queue_.submit([&](sycl::handler& h) {
                h.depends_on(ev1);
                h.parallel_for(sycl::range<1>(batchHeads * seq), [=](sycl::id<1> id) {
                    const int row = static_cast<int>(id[0]);
                    const int base = row * seq;

                    float maxV = -INFINITY;
                    for (int i = 0; i < seq; ++i) {
                        maxV = sycl::fmax(maxV, dScore[base + i]);
                    }

                    float sum = 0.0f;
                    for (int i = 0; i < seq; ++i) {
                        sum += sycl::exp(dScore[base + i] - maxV);
                    }

                    const float inv = 1.0f / (sum + 1e-12f);
                    for (int i = 0; i < seq; ++i) {
                        dProb[base + i] = sycl::exp(dScore[base + i] - maxV) * inv;
                    }
                });
            });

            return queue_.submit([&](sycl::handler& h) {
                h.depends_on(ev2);
                h.parallel_for(sycl::range<1>(qkvCount), [=](sycl::id<1> id) {
                    const int idx = static_cast<int>(id[0]);
                    const int d = idx % dim;
                    const int query = (idx / dim) % seq;
                    const int bh = idx / (seq * dim);

                    const float* pRow = dProb + (bh * seq + query) * seq;
                    float acc = 0.0f;
                    for (int key = 0; key < seq; ++key) {
                        acc += pRow[key] * dV[(bh * seq + key) * dim + d];
                    }
                    dOut[idx] = acc;
                });
            });
        });

        const double scoreOps = 2.0 * batchHeads * seq * seq * dim;
        const double outputOps = 2.0 * batchHeads * seq * seq * dim;
        metrics.tflops = (scoreOps + outputOps) / (metrics.avgMs * 1e9);
        const std::size_t bytes = (3 * qkvCount + 2 * scoreCount + qkvCount) * sizeof(float);
        metrics.gbs = (static_cast<double>(bytes) / 1e9) / (metrics.avgMs / 1e3);
        metrics.note = "Scaled dot-product attention (QK^T + softmax + PV)";

        sycl::free(dQ, queue_);
        sycl::free(dK, queue_);
        sycl::free(dV, queue_);
        sycl::free(dScore, queue_);
        sycl::free(dProb, queue_);
        sycl::free(dOut, queue_);
        return metrics;
    }

private:
    sycl::queue queue_;
};

}  // namespace

std::unique_ptr<KernelBackend> create_backend() {
    return std::make_unique<SyclBackend>();
}

}  // namespace bench
