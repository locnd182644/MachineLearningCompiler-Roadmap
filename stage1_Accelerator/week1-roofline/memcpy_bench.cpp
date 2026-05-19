// memcpy_bench.cpp — Bài tập 1.1
// Đo memory bandwidth thực tế theo từng cấp cache.
// Build: g++ -O3 -march=native memcpy_bench.cpp -o memcpy_bench (hoặc dùng CMake)
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

// Vòng copy thủ công thay cho std::memcpy.
// Lý do: memcpy của glibc dùng non-temporal store (movnt) cho buffer lớn ->
// ghi thẳng ra DRAM, bỏ qua cache, nên không bao giờ đo được cache.
// Phép "+ 1" giữ cho compiler không thay vòng lặp này bằng memcpy.
static void copy_loop(uint64_t* dst, const uint64_t* src, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i] + 1;
    }
}

int main() {
    struct Case {
        const char* name;
        size_t bytes;  // working set thực = src + dst = 2 * bytes
    };
    // Check cache sizes of your machine (e.g., using lscpu) and adjust bytes accordingly.
    // `lscpu | grep cache`
    // Mục tiêu: đo băng thông khi dữ liệu nằm gọn trong L1, L2, L3, và DRAM.
    // Cache của máy: L1d 48 KB, L2 1.25 MB, L3 24 MB (mỗi core / dùng chung).
    // Chọn bytes sao cho 2*bytes nằm gọn trong cấp cache cần đo.
    const Case cases[] = {
        {"L1  ", 16 * 1024},           // 2x = 32 KB   < 48 KB
        {"L2  ", 256 * 1024},          // 2x = 512 KB  < 1.25 MB
        {"L3  ", 4 * 1024 * 1024},     // 2x = 8 MB    < 24 MB
        {"DRAM", 256 * 1024 * 1024},   // 2x = 512 MB  >> L3
    };

    for (const Case& c : cases) {
        size_t n = c.bytes / sizeof(uint64_t);
        std::vector<uint64_t> src(n, 1), dst(n, 0);

        // Số vòng lặp: buffer càng nhỏ càng phải lặp nhiều để đo đủ lâu.
        int iters = (int)(1e9 / c.bytes);
        if (iters < 20) iters = 20;

        // Warm-up: nạp dữ liệu vào cache + để CPU lên turbo frequency.
        for (int i = 0; i < iters; i++) copy_loop(dst.data(), src.data(), n);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; i++) copy_loop(dst.data(), src.data(), n);
        auto end = std::chrono::high_resolution_clock::now();

        // Chống dead-code elimination.
        volatile uint64_t sink = dst[n - 1];
        (void)sink;

        double seconds = std::chrono::duration<double>(end - start).count();
        // Mỗi vòng chuyển 2 * bytes (đọc src + ghi dst).
        double gb_moved = 2.0 * c.bytes * iters / 1e9;
        std::cout << c.name << " | size=" << c.bytes / 1024 << " KB"
                  << " | working set=" << 2 * c.bytes / 1024 << " KB"
                  << " | BW: " << gb_moved / seconds << " GB/s\n";
    }
}
