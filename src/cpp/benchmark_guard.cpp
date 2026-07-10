//
// Created by Selcuk Senturk on 10.07.2026.
//

#include "benchmark_guard.hpp"

#include <cstdint>

void consume_copy_result(
    const unsigned char* buffer,
    std::size_t buffer_size_bytes
) {
    static volatile std::uint64_t sink = 0;

    const std::uint64_t previous = sink;
    const std::size_t index =
        static_cast<std::size_t>(previous % buffer_size_bytes);

    sink = previous + buffer[index];
}