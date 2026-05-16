# Tuần 3 — Phân tích systolic array

## Bài tập 3.3 — So sánh với GPU

Matmul 256×256×256:

| Chỉ số | Systolic 16×16 (sim) | GPU thực tế (PyTorch) |
|--------|----------------------|------------------------|
| Số cycle | | — |
| HBM traffic (byte) | | |
| Utilization | | |

## Vì sao TPU dùng systolic array? — 5 điểm so với GPU SIMT

1.
2.
3.
4.
5.

> Gợi ý các trục so sánh: control granularity (mỗi PE vs mỗi thread),
> data reuse (chảy qua PE vs shared memory), năng lượng/diện tích,
> độ phức tạp scheduler phần cứng, trách nhiệm chuyển sang compiler.

## Liên hệ HW-SW

TPU MXU không có scheduler, cache, branch. Hệ quả cho compiler:

1. Phải biết kích thước MXU → tile matmul thành chunk 256×256.
2. Phải schedule DMA ahead-of-time → sai 1 cycle = stall.
3. Phải lo padding khi tensor không chia hết MXU.
4. Phải lo numerical precision (INT8/BF16, scale factors).

> "Cho AI accelerator, compiler không *tối ưu* code — compiler **viết** code."
