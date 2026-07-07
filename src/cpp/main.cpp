#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

int main() {
    constexpr std::size_t bytes_per_mib = 1024ULL * 1024ULL;
    constexpr std::size_t buffer_size_mib = 256ULL;
    constexpr std::size_t buffer_size_bytes = buffer_size_mib * bytes_per_mib;

    void* source = std::malloc(buffer_size_bytes);
    void* destination = std::malloc(buffer_size_bytes);

    if (source == nullptr || destination == nullptr) {
        std::free(source);
        std::free(destination);
        std::cerr << "allocation failed\n";
        return 1;
    }

    std::memset(source, 1, buffer_size_bytes);
    std::memset(destination, 0, buffer_size_bytes);

    const auto start = std::chrono::steady_clock::now();

    std::memcpy(destination, source, buffer_size_bytes);

    const auto end = std::chrono::steady_clock::now();

    const auto elapsed_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    const auto* destination_bytes = static_cast<const unsigned char*>(destination);
    unsigned long long checksum = 0;
    for (std::size_t i = 0; i < buffer_size_bytes; i += 4096) {
        checksum += destination_bytes[i];
    }

    std::cout << "elapsed_ns=" << elapsed_ns << '\n';
    std::cout << "checksum=" << checksum << '\n';

    std::free(source);
    std::free(destination);

    return 0;
}