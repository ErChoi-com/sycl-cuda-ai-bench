#include "common/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <sstream>

namespace bench {

std::string csv_header() {
    return "backend,kernel,avg_ms,min_ms,max_ms,tflops,gbs,valid,note";
}

std::string csv_row(const BenchmarkRecord& record) {
    std::ostringstream out;
    out << record.backend << ","
        << record.kernel << ","
        << std::fixed << std::setprecision(6)
        << record.metrics.avgMs << ","
        << record.metrics.minMs << ","
        << record.metrics.maxMs << ","
        << record.metrics.tflops << ","
        << record.metrics.gbs << ","
        << (record.metrics.valid ? "true" : "false") << ","
        << '"' << record.metrics.note << '"';
    return out.str();
}

std::string pretty_line(const BenchmarkRecord& record) {
    std::ostringstream out;
    out << std::left << std::setw(8) << record.backend << " | "
        << std::setw(12) << record.kernel << " | "
        << std::right << std::fixed << std::setprecision(3)
        << std::setw(10) << record.metrics.avgMs << " ms | "
        << std::setw(8) << record.metrics.tflops << " TFLOP/s | "
        << std::setw(8) << record.metrics.gbs << " GB/s | "
        << (record.metrics.valid ? "valid" : "invalid");
    return out.str();
}

std::vector<float> make_random_vector(std::size_t count, int seed, float minValue, float maxValue) {
    std::vector<float> data(count);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::uniform_real_distribution<float> dist(minValue, maxValue);
    for (float& value : data) {
        value = dist(rng);
    }
    return data;
}

bool allclose(const std::vector<float>& a, const std::vector<float>& b, float atol, float rtol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float lhs = a[i];
        const float rhs = b[i];
        const float diff = std::fabs(lhs - rhs);
        const float tol = atol + rtol * std::fabs(rhs);
        if (diff > tol || std::isnan(lhs) || std::isnan(rhs)) {
            return false;
        }
    }
    return true;
}

}  // namespace bench
