# Tuần 2 — CPU SIMD & GPU SIMT: Baseline để so sánh

> **Câu hỏi central:** GPU làm AI nhanh hơn CPU vì sao? Cụ thể, không chung chung.

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `avx_matmul.cpp` | Đầy đủ | 2.1 — Matmul thủ công với AVX2 intrinsics |
| `matmul.cu` | Đầy đủ (Naive, Tiled, cuBLAS) | 2.2 — 3 phiên bản CUDA matmul |
| `analysis.md` | Đầy đủ | 2.3 — Phân tích Nsight Compute |

## Build & Run

```bash
cmake -B build -G Ninja && cmake --build build

./build/avx_matmul          # CPU
./build/matmul              # GPU (cần CUDA)
```

Nếu không có CUDA, CMake sẽ chỉ build phần CPU và bỏ qua `matmul.cu`.

## Bài tập 2.3 — Profile

```bash
ncu --set full -o profile_naive ./build/matmul   # chỉnh args để chạy từng version
ncu-ui profile_naive.ncu-rep
```

Xem: SM Throughput %, Memory Throughput %, Achieved Occupancy, L1/L2 hit rate.

## Cần tự làm

- [x] `matmul.cu`: hoàn thiện **V3 — cuBLAS** (`cublasSgemm`) để có mốc peak so sánh.
- [x] `analysis.md`: điền số đo và giải thích từng bottleneck với metric Nsight cụ thể.

## Checklist output cuối tuần

- [x] AVX matmul nhanh hơn naive ~4-8x
- [x] 3 phiên bản CUDA chạy, có số GFLOPS cho N=2048
- [x] Phân tích lý thuyết & giải thích bottleneck dựa trên Nsight Compute
- [x] `analysis.md` hoàn chỉnh
