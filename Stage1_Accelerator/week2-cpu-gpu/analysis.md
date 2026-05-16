# Tuần 2 — Phân tích bottleneck (Bài tập 2.3)

## Kết quả đo

| Kernel | Thời gian (ms) | GFLOPS | % so với cuBLAS |
|--------|----------------|--------|-----------------|
| AVX (CPU, N=512) | | | — |
| V1 Naive (N=2048) | | | |
| V2 Tiled (N=2048) | | | |
| V3 cuBLAS (N=2048) | | | 100% |

## Metric Nsight Compute

| Metric | V1 Naive | V2 Tiled | Nhận xét |
|--------|----------|----------|----------|
| SM Throughput % | | | |
| Memory Throughput % | | | |
| Achieved Occupancy | | | |
| L1 hit rate | | | |
| L2 hit rate | | | |

## Giải thích bottleneck

### V1 Naive — vì sao chậm?

> (Gợi ý: mỗi phần tử của A và B bị load lại bao nhiêu lần từ global memory?
>  Liên hệ coalesced access và roofline.)

### V2 Tiled — cải thiện ở đâu?

> (Shared memory reuse: mỗi tile load 1 lần, dùng TILE lần. AI tăng thế nào?)

### Khoảng cách Tiled → cuBLAS

> (Tensor core, register tiling, software pipelining, double buffering...)

## Liên hệ HW-SW

> Coalesced access, bank conflicts, tile size selection — compiler phải biết
> microarchitecture cụ thể (Ampere vs Hopper), không chỉ "GPU" chung chung.
