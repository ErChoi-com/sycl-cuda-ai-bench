#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "backends/backend.hpp"

namespace bench {
namespace {

template <typename Fn>
KernelMetrics run_timed_cpu(const BenchmarkOptions& options, Fn&& fn) {
    using clock = std::chrono::high_resolution_clock;
    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(options.measuredIterations));

    for (int i = 0; i < options.warmupIterations; ++i) {
        fn();
    }

    for (int i = 0; i < options.measuredIterations; ++i) {
        const auto start = clock::now();
        fn();
        const auto end = clock::now();
        timings.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    KernelMetrics metrics;
    metrics.minMs = *std::min_element(timings.begin(), timings.end());
    metrics.maxMs = *std::max_element(timings.begin(), timings.end());
    metrics.avgMs = std::accumulate(timings.begin(), timings.end(), 0.0) / static_cast<double>(timings.size());
    return metrics;
}

class CpuBackend final : public KernelBackend {
public:
    std::string name() const override {
        return "CPU";
    }

    std::string device_name() const override {
        return "Host CPU (validation backend)";
    }

    void synchronize() override {}

    KernelMetrics run_matmul(const MatmulProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t aCount = static_cast<std::size_t>(problem.m) * problem.k;
        const std::size_t bCount = static_cast<std::size_t>(problem.k) * problem.n;
        const std::size_t cCount = static_cast<std::size_t>(problem.m) * problem.n;

        const auto a = make_random_vector(aCount, options.randomSeed);
        const auto b = make_random_vector(bCount, options.randomSeed + 1);
        std::vector<float> c(cCount, 0.0f);

        auto metrics = run_timed_cpu(options, [&]() {
            for (int row = 0; row < problem.m; ++row) {
                for (int col = 0; col < problem.n; ++col) {
                    float acc = 0.0f;
                    for (int k = 0; k < problem.k; ++k) {
                        acc += a[row * problem.k + k] * b[k * problem.n + col];
                    }
                    c[row * problem.n + col] = acc;
                }
            }
        });

        metrics.tflops = (2.0 * problem.m * problem.n * problem.k) / (metrics.avgMs * 1e9);
        metrics.gbs = ((aCount + bCount + cCount) * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        return metrics;
    }

    KernelMetrics run_conv2d(const Conv2dProblem& problem, const BenchmarkOptions& options) override {
        const int outH = (problem.h + 2 * problem.padding - problem.kernelH) / problem.stride + 1;
        const int outW = (problem.w + 2 * problem.padding - problem.kernelW) / problem.stride + 1;

        const std::size_t inSize = static_cast<std::size_t>(problem.n) * problem.c * problem.h * problem.w;
        const std::size_t weightSize = static_cast<std::size_t>(problem.outChannels) * problem.c * problem.kernelH * problem.kernelW;
        const std::size_t outSize = static_cast<std::size_t>(problem.n) * problem.outChannels * outH * outW;

        const auto in = make_random_vector(inSize, options.randomSeed + 3);
        const auto w = make_random_vector(weightSize, options.randomSeed + 4);
        const auto bias = make_random_vector(problem.outChannels, options.randomSeed + 5);
        std::vector<float> out(outSize, 0.0f);

        auto metrics = run_timed_cpu(options, [&]() {
            for (int n = 0; n < problem.n; ++n) {
                for (int oc = 0; oc < problem.outChannels; ++oc) {
                    for (int oh = 0; oh < outH; ++oh) {
                        for (int ow = 0; ow < outW; ++ow) {
                            float sum = bias[oc];
                            for (int ic = 0; ic < problem.c; ++ic) {
                                for (int kh = 0; kh < problem.kernelH; ++kh) {
                                    for (int kw = 0; kw < problem.kernelW; ++kw) {
                                        const int ih = oh * problem.stride - problem.padding + kh;
                                        const int iw = ow * problem.stride - problem.padding + kw;
                                        if (ih >= 0 && ih < problem.h && iw >= 0 && iw < problem.w) {
                                            const int inOff = ((n * problem.c + ic) * problem.h + ih) * problem.w + iw;
                                            const int wOff = ((oc * problem.c + ic) * problem.kernelH + kh) * problem.kernelW + kw;
                                            sum += in[inOff] * w[wOff];
                                        }
                                    }
                                }
                            }
                            const int outOff = ((n * problem.outChannels + oc) * outH + oh) * outW + ow;
                            out[outOff] = sum;
                        }
                    }
                }
            }
        });

        metrics.tflops = (2.0 * problem.c * problem.kernelH * problem.kernelW * outSize) / (metrics.avgMs * 1e9);
        metrics.gbs = ((inSize + weightSize + outSize + bias.size()) * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        metrics.note = "CPU reference conv";
        return metrics;
    }

    KernelMetrics run_softmax(const SoftmaxProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto in = make_random_vector(count, options.randomSeed + 8);
        std::vector<float> out(count, 0.0f);

        auto metrics = run_timed_cpu(options, [&]() {
            for (int row = 0; row < problem.rows; ++row) {
                float maxV = -std::numeric_limits<float>::infinity();
                for (int col = 0; col < problem.cols; ++col) {
                    maxV = std::max(maxV, in[row * problem.cols + col]);
                }
                float sum = 0.0f;
                for (int col = 0; col < problem.cols; ++col) {
                    sum += std::exp(in[row * problem.cols + col] - maxV);
                }
                const float inv = 1.0f / (sum + 1e-12f);
                for (int col = 0; col < problem.cols; ++col) {
                    out[row * problem.cols + col] = std::exp(in[row * problem.cols + col] - maxV) * inv;
                }
            }
        });

        metrics.tflops = (3.0 * count) / (metrics.avgMs * 1e9);
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        return metrics;
    }

    KernelMetrics run_layernorm(const LayerNormProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto in = make_random_vector(count, options.randomSeed + 13);
        std::vector<float> out(count, 0.0f);

        auto metrics = run_timed_cpu(options, [&]() {
            for (int row = 0; row < problem.rows; ++row) {
                float sum = 0.0f;
                float sq = 0.0f;
                for (int col = 0; col < problem.cols; ++col) {
                    const float value = in[row * problem.cols + col];
                    sum += value;
                    sq += value * value;
                }
                const float mean = sum / static_cast<float>(problem.cols);
                const float var = sq / static_cast<float>(problem.cols) - mean * mean;
                const float invStd = 1.0f / std::sqrt(var + problem.epsilon);
                for (int col = 0; col < problem.cols; ++col) {
                    out[row * problem.cols + col] = (in[row * problem.cols + col] - mean) * invStd;
                }
            }
        });

        metrics.tflops = (6.0 * count) / (metrics.avgMs * 1e9);
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        return metrics;
    }

    KernelMetrics run_attention(const AttentionProblem& problem, const BenchmarkOptions& options) override {
        const int batchHeads = problem.batch * problem.heads;
        const int seq = problem.seqLen;
        const int dim = problem.headDim;

        const std::size_t qkvCount = static_cast<std::size_t>(batchHeads) * seq * dim;
        const std::size_t scoreCount = static_cast<std::size_t>(batchHeads) * seq * seq;

        const auto q = make_random_vector(qkvCount, options.randomSeed + 19);
        const auto k = make_random_vector(qkvCount, options.randomSeed + 20);
        const auto v = make_random_vector(qkvCount, options.randomSeed + 21);
        std::vector<float> score(scoreCount, 0.0f);
        std::vector<float> prob(scoreCount, 0.0f);
        std::vector<float> out(qkvCount, 0.0f);

        const float scale = 1.0f / std::sqrt(static_cast<float>(dim));

        auto metrics = run_timed_cpu(options, [&]() {
            for (int bh = 0; bh < batchHeads; ++bh) {
                for (int query = 0; query < seq; ++query) {
                    for (int key = 0; key < seq; ++key) {
                        float dot = 0.0f;
                        for (int d = 0; d < dim; ++d) {
                            dot += q[(bh * seq + query) * dim + d] * k[(bh * seq + key) * dim + d];
                        }
                        score[(bh * seq + query) * seq + key] = dot * scale;
                    }
                }
            }

            for (int row = 0; row < batchHeads * seq; ++row) {
                const int base = row * seq;
                float maxV = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < seq; ++i) {
                    maxV = std::max(maxV, score[base + i]);
                }
                float sum = 0.0f;
                for (int i = 0; i < seq; ++i) {
                    sum += std::exp(score[base + i] - maxV);
                }
                const float inv = 1.0f / (sum + 1e-12f);
                for (int i = 0; i < seq; ++i) {
                    prob[base + i] = std::exp(score[base + i] - maxV) * inv;
                }
            }

            for (int bh = 0; bh < batchHeads; ++bh) {
                for (int query = 0; query < seq; ++query) {
                    for (int d = 0; d < dim; ++d) {
                        float acc = 0.0f;
                        for (int key = 0; key < seq; ++key) {
                            acc += prob[(bh * seq + query) * seq + key] * v[(bh * seq + key) * dim + d];
                        }
                        out[(bh * seq + query) * dim + d] = acc;
                    }
                }
            }
        });

        const double scoreOps = 2.0 * batchHeads * seq * seq * dim;
        const double outputOps = 2.0 * batchHeads * seq * seq * dim;
        metrics.tflops = (scoreOps + outputOps) / (metrics.avgMs * 1e9);
        metrics.gbs = ((4.0 * qkvCount + 2.0 * scoreCount) * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        metrics.note = "CPU reference attention";
        return metrics;
    }

    KernelMetrics run_gelu(const GeluProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto in = make_random_vector(count, options.randomSeed + 30);
        std::vector<float> out(count, 0.0f);

        auto metrics = run_timed_cpu(options, [&]() {
            for (std::size_t i = 0; i < count; ++i) {
                const float x = in[i];
                const float u = 0.7978845608f * (x + 0.044715f * x * x * x);
                out[i] = 0.5f * x * (1.0f + std::tanh(u));
            }
        });

        metrics.tflops = (10.0 * count) / (metrics.avgMs * 1e9);
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        return metrics;
    }

    KernelMetrics run_transpose(const TransposeProblem& problem, const BenchmarkOptions& options) override {
        const std::size_t count = static_cast<std::size_t>(problem.rows) * problem.cols;
        const auto in = make_random_vector(count, options.randomSeed + 33);
        std::vector<float> out(count, 0.0f);

        auto metrics = run_timed_cpu(options, [&]() {
            for (int r = 0; r < problem.rows; ++r) {
                for (int c = 0; c < problem.cols; ++c) {
                    out[c * problem.rows + r] = in[r * problem.cols + c];
                }
            }
        });

        metrics.tflops = 0.0;
        metrics.gbs = (2.0 * count * sizeof(float) / 1e9) / (metrics.avgMs / 1e3);
        return metrics;
    }
};

}  // namespace

std::unique_ptr<KernelBackend> create_backend() {
    return std::make_unique<CpuBackend>();
}

}  // namespace bench
