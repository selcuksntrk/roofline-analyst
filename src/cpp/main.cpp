#include <cstdlib>
#include <iostream>

int main() {
    constexpr std::size_t buffer_size_bytes = 256ULL * 1024ULL * 1024ULL;

    void* source = std::malloc(buffer_size_bytes);
    void* destination = std::malloc(buffer_size_bytes);

    if (source == nullptr || destination == nullptr) {
        std::free(source);
        std::free(destination);
        std::cerr << "allocation failed\n";
        return 1;
    }

    std::cout << "allocated two 256 MiB buffers\n";

    std::free(source);
    std::free(destination);

    return 0;
}