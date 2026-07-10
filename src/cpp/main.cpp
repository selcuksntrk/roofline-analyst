#include <arm_neon.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <cmath>
#include <iomanip>
#include <limits>

#include "benchmark_guard.hpp"

int main() {
    constexpr std::size_t bytes_per_mib = 1024ULL * 1024ULL;
    constexpr std::size_t bytes_per_gib = 1024ULL * bytes_per_mib;
    constexpr std::size_t buffer_size_mib = 256ULL;
    constexpr std::size_t buffer_size_bytes = buffer_size_mib * bytes_per_mib;
    constexpr std::size_t bytes_per_gb = 1000ULL * 1000ULL * 1000ULL;
    constexpr int memory_iterations = 200;
    constexpr std::uint64_t fma_iterations = 500'000'000ULL;
    constexpr double flops_per_fma = 8.0;
    constexpr double fmas_per_iteration = 8.0;


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

    for (int i = 0; i < memory_iterations; ++i) {
        void* current_source =
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
        static_cast<double>(memory_iterations) * 2.0 *
        static_cast<double>(buffer_size_bytes);
    const double bandwidth_gib_per_second =
        memory_total_bytes / memory_elapsed_seconds /
        static_cast<double>(bytes_per_gib);

    const double bandwidth_gbps =
    memory_total_bytes / memory_elapsed_seconds /
    static_cast<double>(bytes_per_gb);

    const auto* destination_bytes =
        static_cast<const unsigned char*>(destination);
    unsigned long long memory_checksum = 0;
    for (std::size_t i = 0; i < buffer_size_bytes; i += 4096) {
        memory_checksum += destination_bytes[i];
    }

    const unsigned long long expected_memory_checksum =
    static_cast<unsigned long long>(buffer_size_bytes / 4096) *
    (memory_iterations % 2 == 0 ? 2ULL : 1ULL);

    if (memory_checksum != expected_memory_checksum) {
        std::cerr << "memory checksum validation failed\n";
        std::free(source_a);
        std::free(source_b);
        std::free(destination);
        return 1;
    }

    std::free(source_a);
    std::free(source_b);
    std::free(destination);

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

    for (std::uint64_t i = 0; i < fma_iterations; ++i) {
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
        static_cast<double>(fma_iterations) * fmas_per_iteration *
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