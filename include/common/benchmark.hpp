#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bench {

struct BenchmarkOptions {
    int warmupIterations = 10;
    int measuredIterations = 50;
    bool validate = true;
    int randomSeed = 42;
};

struct KernelMetrics {
    double avgMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
    double tflops = 0.0;
    double gbs = 0.0;
    bool valid = true;
    std::string note;
};

struct MatmulProblem {
    int m = 4096;
    int n = 4096;
    int k = 4096;
};

struct Conv2dProblem {
    int n = 1;
    int c = 64;
    int h = 224;
    int w = 224;
    int outChannels = 64;
    int kernelH = 3;
    int kernelW = 3;
    int stride = 1;
    int padding = 1;
};

struct SoftmaxProblem {
    int rows = 4096;
    int cols = 1024;
};

struct LayerNormProblem {
    int rows = 8192;
    int cols = 1024;
    float epsilon = 1e-5f;
};

struct AttentionProblem {
    int batch = 1;
    int heads = 8;
    int seqLen = 512;
    int headDim = 64;
};

struct GeluProblem {
    int rows = 8192;
    int cols = 1024;
};

struct TransposeProblem {
    int rows = 4096;
    int cols = 4096;
};

struct BenchmarkRecord {
    std::string backend;
    std::string kernel;
    KernelMetrics metrics;
};

std::string csv_header();
std::string csv_row(const BenchmarkRecord& record);
std::string pretty_line(const BenchmarkRecord& record);

std::vector<float> make_random_vector(std::size_t count, int seed, float minValue = -1.0f, float maxValue = 1.0f);
bool allclose(const std::vector<float>& a, const std::vector<float>& b, float atol = 1e-3f, float rtol = 1e-2f);

}  // namespace bench
