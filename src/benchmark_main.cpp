#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "backends/backend.hpp"

namespace {

void apply_quick_profile(bench::MatmulProblem& matmul,
                         bench::Conv2dProblem& conv,
                         bench::SoftmaxProblem& softmax,
                         bench::LayerNormProblem& layernorm,
                         bench::AttentionProblem& attention,
                         bench::GeluProblem& gelu,
                         bench::TransposeProblem& transpose,
                         bench::BenchmarkOptions& options) {
    matmul = {384, 384, 384};
    conv = {1, 16, 64, 64, 16, 3, 3, 1, 1};
    softmax = {512, 256};
    layernorm = {1024, 256, 1e-5f};
    attention = {1, 4, 128, 32};
    gelu = {1024, 256};
    transpose = {1024, 1024};
    options.warmupIterations = 2;
    options.measuredIterations = 5;
}

void apply_full_profile(bench::MatmulProblem& matmul,
                        bench::Conv2dProblem& conv,
                        bench::SoftmaxProblem& softmax,
                        bench::LayerNormProblem& layernorm,
                        bench::AttentionProblem& attention,
                        bench::GeluProblem& gelu,
                        bench::TransposeProblem& transpose,
                        bench::BenchmarkOptions& options) {
    matmul = {2048, 2048, 2048};
    conv = {1, 64, 224, 224, 64, 3, 3, 1, 1};
    softmax = {4096, 1024};
    layernorm = {8192, 1024, 1e-5f};
    attention = {1, 16, 512, 64};
    gelu = {8192, 1024};
    transpose = {4096, 4096};
    options.warmupIterations = 10;
    options.measuredIterations = 40;
}

}  // namespace

int main(int argc, char** argv) {
    std::string csvPath = "results/latest.csv";
    bool quick = false;

    bench::BenchmarkOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--csv" && i + 1 < argc) {
            csvPath = argv[++i];
        } else if (arg == "--quick") {
            quick = true;
        } else if (arg == "--no-validate") {
            options.validate = false;
        } else if (arg == "--iters" && i + 1 < argc) {
            options.measuredIterations = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            options.warmupIterations = std::stoi(argv[++i]);
        }
    }

    bench::MatmulProblem matmul;
    bench::Conv2dProblem conv;
    bench::SoftmaxProblem softmax;
    bench::LayerNormProblem layernorm;
    bench::AttentionProblem attention;
    bench::GeluProblem gelu;
    bench::TransposeProblem transpose;

    if (quick) {
        apply_quick_profile(matmul, conv, softmax, layernorm, attention, gelu, transpose, options);
    } else {
        apply_full_profile(matmul, conv, softmax, layernorm, attention, gelu, transpose, options);
    }

    auto backend = bench::create_backend();
    std::cout << "Backend: " << backend->name() << "\n";
    std::cout << "Device : " << backend->device_name() << "\n";
    std::cout << "Warmup=" << options.warmupIterations
              << " Measured=" << options.measuredIterations
              << " Validate=" << (options.validate ? "true" : "false")
              << "\n\n";

    std::vector<bench::BenchmarkRecord> records;

    records.push_back({backend->name(), "matmul", backend->run_matmul(matmul, options)});
    records.push_back({backend->name(), "conv2d", backend->run_conv2d(conv, options)});
    records.push_back({backend->name(), "softmax", backend->run_softmax(softmax, options)});
    records.push_back({backend->name(), "layernorm", backend->run_layernorm(layernorm, options)});
    records.push_back({backend->name(), "attention", backend->run_attention(attention, options)});
    records.push_back({backend->name(), "gelu", backend->run_gelu(gelu, options)});
    records.push_back({backend->name(), "transpose", backend->run_transpose(transpose, options)});

    std::cout << "Backend  | Kernel       |    Avg (ms) | Throughput | Bandwidth | Validation\n";
    std::cout << "---------+--------------+-------------+------------+-----------+-----------\n";
    for (const auto& record : records) {
        std::cout << bench::pretty_line(record) << "\n";
    }

    std::filesystem::create_directories(std::filesystem::path(csvPath).parent_path());
    std::ofstream csv(csvPath, std::ios::out | std::ios::trunc);
    csv << bench::csv_header() << "\n";
    for (const auto& record : records) {
        csv << bench::csv_row(record) << "\n";
    }
    std::cout << "\nSaved CSV to: " << csvPath << "\n";

    return 0;
}
