// memcpy_bench.cpp — Bài tập 1.1
// Đo memory bandwidth thực tế theo từng cấp cache.
// Build: g++ -O3 memcpy_bench.cpp -o memcpy_bench   (hoặc dùng CMake)
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    const size_t sizes[] = {
        16 * 1024,        // 16 KB  - vừa L1
        256 * 1024,       // 256 KB - vừa L2
        8 * 1024 * 1024,  // 8 MB   - vừa L3
        512 * 1024 * 1024 // 512 MB - phải vào DRAM
    };

    for (size_t size : sizes) {
        std::vector<char> src(size, 1);
        std::vector<char> dst(size);

        auto start = std::chrono::high_resolution_clock::now();
        const int iters = 100;
        for (int i = 0; i < iters; i++) {
            std::memcpy(dst.data(), src.data(), size);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double gb_moved = (double)size * iters / 1e9;
        std::cout << "Size: " << size / 1024 << " KB, BW: "
                  << gb_moved / seconds << " GB/s\n";
    }
}
