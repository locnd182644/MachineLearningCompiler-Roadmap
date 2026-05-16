# Tuần 6 — Memory Hierarchy & Project tích hợp + Luckfox NPU

> **Câu hỏi central:** Toàn bộ kiến thức tuần 1-5 áp dụng thế nào lên 1 chip
> AI thật?

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `matmul_tiled_3level.cu` | **Stub** | 6.1 — Tiled matmul 3 cấp (CUTLASS-style) |
| `flash_attention.py` | **Stub** | 6.2 — Flash Attention bằng Triton |
| `luckfox/convert_rknn.py` | Đầy đủ (cần ONNX) | 6.3 — Convert model sang RKNN |
| `luckfox/README.md` | Hướng dẫn | 6.3 — Deploy + benchmark trên NPU thật |
| `summary.md` | Template | Tổng kết Giai đoạn 1 — 10 insights |

## Build & Run

```bash
cmake -B build -G Ninja && cmake --build build   # CUDA
./build/matmul_3level

python flash_attention.py                        # cần GPU + triton
```

## Cần tự làm

- `matmul_tiled_3level.cu`: 3 cấp tiling — block (128×128) → warp (32×32)
  → thread (8×8). Mục tiêu ~70-80% cuBLAS.
- `flash_attention.py`: Flash Attention naive bằng Triton, verify vs PyTorch.
- Luckfox NPU: xem `luckfox/README.md` — đây là **portfolio piece**.

## Checklist output cuối tuần

- [ ] Tiled matmul 3 cấp + Nsight profile
- [ ] Flash Attention Triton + verify correctness + đo memory
- [ ] Luckfox: model → convert → benchmark → analysis
- [ ] `summary.md`: 10 insights chính của Giai đoạn 1
