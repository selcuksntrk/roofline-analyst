#include <arm_neon.h>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <system_error>

#include "benchmark_guard.hpp"

namespace {

struct BenchmarkConfig {
    std::uint64_t buffer_size_mib = 256;
    std::uint64_t memory_iterations = 200;
    std::uint64_t fma_iterations = 500'000'000ULL;
};

bool parse_positive_uint64(
    const std::string_view text,
    std::uint64_t& value
) {
    std::uint64_t parsed = 0;
    const char* const begin = text.data();
    const char* const end = begin + text.size();

    const auto [parsed_end, error] =
        std::from_chars(begin, end, parsed, 10);

    if (error != std::errc{} || parsed_end != end || parsed == 0) {
        return false;
    }

    value = parsed;
    return true;
}

bool parse_arguments(
    const int argc,
    char* argv[],
    BenchmarkConfig& config
) {
    bool saw_buffer_size = false;
    bool saw_memory_iterations = false;
    bool saw_fma_iterations = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view flag = argv[index];

        const bool known_flag =
            flag == "--buffer-mib" ||
            flag == "--memory-iterations" ||
            flag == "--fma-iterations";

        if (!known_flag) {
            std::cerr << "unknown argument: " << flag << '\n';
            return false;
        }

        if (index + 1 >= argc) {
            std::cerr << "missing value for " << flag << '\n';
            return false;
        }

        const std::string_view value_text = argv[++index];
        std::uint64_t value = 0;

        if (!parse_positive_uint64(value_text, value)) {
            std::cerr << "invalid positive integer for "
                      << flag << ": " << value_text << '\n';
            return false;
        }

        if (flag == "--buffer-mib") {
            if (saw_buffer_size) {
                std::cerr << "--buffer-mib specified more than once\n";
                return false;
            }

            config.buffer_size_mib = value;
            saw_buffer_size = true;
        } else if (flag == "--memory-iterations") {
            if (saw_memory_iterations) {
                std::cerr << "--memory-iterations specified more than once\n";
                return false;
            }

            config.memory_iterations = value;
            saw_memory_iterations = true;
        } else {
            if (saw_fma_iterations) {
                std::cerr << "--fma-iterations specified more than once\n";
                return false;
            }

            config.fma_iterations = value;
            saw_fma_iterations = true;
        }
    }

    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    constexpr std::size_t bytes_per_mib = 1024ULL * 1024ULL;
    constexpr std::size_t bytes_per_gb = 1000ULL * 1000ULL * 1000ULL;
    constexpr double flops_per_fma = 8.0;
    constexpr double fmas_per_iteration = 8.0;

    BenchmarkConfig config;
    if (!parse_arguments(argc, argv, config)) {
        return 1;
    }

    const std::uint64_t max_buffer_size_mib =
        static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max() / bytes_per_mib
        );

    if (config.buffer_size_mib > max_buffer_size_mib) {
        std::cerr << "--buffer-mib is too large for this platform\n";
        return 1;
    }

    const std::size_t buffer_size_bytes =
        static_cast<std::size_t>(config.buffer_size_mib) * bytes_per_mib;

    void* source_a = std::malloc(buffer_size_bytes);
    void* source_b = std::malloc(buffer_size_bytes);
    void* destination = std::malloc(buffer_size_bytes);

    if (source_a == nullptr || source_b == nullptr || destination == nullptr) {
        std::free(source_a);
        std::free(source_b);
        std::free(destination);
        std::cerr << "allocation failed\n";
        return 1;
    }

    std::memset(source_a, 1, buffer_size_bytes);
    std::memset(source_b, 2, buffer_size_bytes);
    std::memset(destination, 0, buffer_size_bytes);

    const auto memory_start = std::chrono::steady_clock::now();

    for (std::uint64_t i = 0; i < config.memory_iterations; ++i) {
        void* const current_source =
            (i % 2 == 0) ? source_a : source_b;

        std::memcpy(destination, current_source, buffer_size_bytes);

        consume_copy_result(
            static_cast<const unsigned char*>(destination),
            buffer_size_bytes
        );
    }

    const auto memory_end = std::chrono::steady_clock::now();

    const auto memory_elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            memory_end - memory_start
        ).count();
    const double memory_elapsed_seconds =
        static_cast<double>(memory_elapsed_ns) / 1e9;
    const double memory_total_bytes =
        static_cast<double>(config.memory_iterations) * 2.0 *
        static_cast<double>(buffer_size_bytes);
    const double bandwidth_gbps =
        memory_total_bytes / memory_elapsed_seconds /
        static_cast<double>(bytes_per_gb);

    const auto* const destination_bytes =
        static_cast<const unsigned char*>(destination);
    unsigned long long memory_checksum = 0;
    for (std::size_t i = 0; i < buffer_size_bytes; i += 4096) {
        memory_checksum += destination_bytes[i];
    }

    const unsigned long long expected_memory_checksum =
        static_cast<unsigned long long>(buffer_size_bytes / 4096) *
        (config.memory_iterations % 2 == 0 ? 2ULL : 1ULL);

    std::free(source_a);
    std::free(source_b);
    std::free(destination);

    if (memory_checksum != expected_memory_checksum) {
        std::cerr << "memory checksum validation failed\n";
        return 1;
    }

    const float32x4_t a = vdupq_n_f32(1.0001F);
    const float32x4_t b = vdupq_n_f32(1.0002F);

    float32x4_t acc0 = vdupq_n_f32(1.0F);
    float32x4_t acc1 = vdupq_n_f32(2.0F);
    float32x4_t acc2 = vdupq_n_f32(3.0F);
    float32x4_t acc3 = vdupq_n_f32(4.0F);
    float32x4_t acc4 = vdupq_n_f32(5.0F);
    float32x4_t acc5 = vdupq_n_f32(6.0F);
    float32x4_t acc6 = vdupq_n_f32(7.0F);
    float32x4_t acc7 = vdupq_n_f32(8.0F);

    const auto flops_start = std::chrono::steady_clock::now();

    for (std::uint64_t i = 0; i < config.fma_iterations; ++i) {
        acc0 = vfmaq_f32(acc0, a, b);
        acc1 = vfmaq_f32(acc1, a, b);
        acc2 = vfmaq_f32(acc2, a, b);
        acc3 = vfmaq_f32(acc3, a, b);
        acc4 = vfmaq_f32(acc4, a, b);
        acc5 = vfmaq_f32(acc5, a, b);
        acc6 = vfmaq_f32(acc6, a, b);
        acc7 = vfmaq_f32(acc7, a, b);
    }

    const auto flops_end = std::chrono::steady_clock::now();

    const float32x4_t sum01 = vaddq_f32(acc0, acc1);
    const float32x4_t sum23 = vaddq_f32(acc2, acc3);
    const float32x4_t sum45 = vaddq_f32(acc4, acc5);
    const float32x4_t sum67 = vaddq_f32(acc6, acc7);
    const float32x4_t sum0123 = vaddq_f32(sum01, sum23);
    const float32x4_t sum4567 = vaddq_f32(sum45, sum67);
    volatile float flops_sink = vaddvq_f32(vaddq_f32(sum0123, sum4567));

    const auto flops_elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            flops_end - flops_start
        ).count();
    const double flops_elapsed_seconds =
        static_cast<double>(flops_elapsed_ns) / 1e9;
    const double total_flops =
        static_cast<double>(config.fma_iterations) * fmas_per_iteration *
        flops_per_fma;
    const double peak_gflops =
        total_flops / flops_elapsed_seconds / 1e9;

    if (!std::isfinite(bandwidth_gbps) || !std::isfinite(peak_gflops)) {
        std::cerr << "benchmark produced a non-finite metric\n";
        return 1;
    }

    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "{\"memory_bandwidth_gbps\":" << bandwidth_gbps
              << ",\"cpu_neon_fp32_gflops\":" << peak_gflops
              << "}\n";

    return 0;
}