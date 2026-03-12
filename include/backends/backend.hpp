#pragma once

#include <memory>
#include <string>

#include "common/benchmark.hpp"

namespace bench {

class KernelBackend {
public:
    virtual ~KernelBackend() = default;

    virtual std::string name() const = 0;
    virtual std::string device_name() const = 0;
    virtual void synchronize() = 0;

    virtual KernelMetrics run_matmul(const MatmulProblem& problem, const BenchmarkOptions& options) = 0;
    virtual KernelMetrics run_conv2d(const Conv2dProblem& problem, const BenchmarkOptions& options) = 0;
    virtual KernelMetrics run_softmax(const SoftmaxProblem& problem, const BenchmarkOptions& options) = 0;
    virtual KernelMetrics run_layernorm(const LayerNormProblem& problem, const BenchmarkOptions& options) = 0;
    virtual KernelMetrics run_attention(const AttentionProblem& problem, const BenchmarkOptions& options) = 0;
    virtual KernelMetrics run_gelu(const GeluProblem& problem, const BenchmarkOptions& options) = 0;
    virtual KernelMetrics run_transpose(const TransposeProblem& problem, const BenchmarkOptions& options) = 0;
};

std::unique_ptr<KernelBackend> create_backend();

}  // namespace bench
