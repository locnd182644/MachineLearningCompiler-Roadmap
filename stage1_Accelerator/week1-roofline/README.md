# Tuần 1 — Performance Modeling: Roofline & Memory Wall

> **Câu hỏi central:** Tại sao một matmul 1024×1024 chạy nhanh hơn một vector add
> 10 triệu phần tử, dù vector add ít FLOPs hơn?

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `memcpy_bench.cpp` | Đầy đủ | 1.1 — Đo memory bandwidth CPU theo cache level |
| `roofline_theory.py` | Đầy đủ + 1 TODO (plot) | 1.2 — Tính AI lý thuyết cho 5 kernel |
| `roofline_measured.py` | Đầy đủ | 1.3 — Đo thực tế trên GPU, so với lý thuyết |

## Build & Run

```bash
# C++
cmake -B build -G Ninja && cmake --build build
./build/memcpy_bench

# Python
python roofline_theory.py     # cần sửa PEAK_FLOPS / PEAK_BW theo GPU của bạn
python roofline_measured.py   # cần GPU NVIDIA
```

## Cần điền trước khi chạy

- `roofline_theory.py`: `PEAK_FLOPS`, `PEAK_BW` — lấy từ spec GPU của bạn
  (xem `nvidia-smi`, datasheet). RTX 3060 ví dụ: ~13 TFLOPS FP32, ~360 GB/s.

## Checklist output cuối tuần

- [x] `memcpy_bench` chạy, thấy BW giảm rõ khi size vượt L3
- [x] `roofline_theory.py` in AI + bound cho 5 kernel + vẽ plot roofline (`roofline.png`)
- [x] `roofline_measured.py` cho % peak của matmul và vecadd
- [x] Viết phần "chỗ chênh lệch lý thuyết vs thực tế" vào README này bên dưới

## Phân tích (tự viết)

GPU đo: **NVIDIA GeForce RTX 4050 Laptop** — peak tham chiếu ~9 TFLOPS FP32, ~192 GB/s.

Kết quả đo (`roofline_measured.py`):

| Kernel | Thời gian | Đo được | % peak |
|--------|-----------|---------|--------|
| Matmul 4K | 21.07 ms | 6.52 TFLOPS | **72.4%** peak FLOPS |
| VecAdd 10M | 0.76 ms | 157.5 GB/s | **82.0%** peak BW |

**Vì sao Matmul gần peak compute còn VecAdd gần peak BW — giải thích bằng roofline:**

Roofline cho biết hiệu năng đạt được = `min(peak_flops, AI × peak_bw)`, trong đó
`AI` (arithmetic intensity) = FLOPs / byte. Ranh giới là `AI_crit ≈ 46.9 FLOPs/byte`.

- **Matmul 4K — compute-bound.** AI ≈ 683 FLOPs/byte, *cao hơn nhiều* `AI_crit`.
  Mỗi phần tử nạp từ bộ nhớ được tái sử dụng O(N) lần (data reuse qua tiling/cache),
  nên bộ nhớ không phải nút thắt — các đơn vị FP32 ALU mới là giới hạn. Kernel chạm
  vào "mái ngang" (compute roof). Đạt 72% chứ chưa 100% vì đây là matmul FP32 thường
  (không dùng Tensor Core), cộng overhead lập lịch và kích thước chưa tối ưu cho GPU.

- **VecAdd 10M — memory-bound.** AI ≈ 0.083 FLOPs/byte, *thấp hơn rất nhiều* `AI_crit`.
  Mỗi phần tử chỉ làm 1 phép cộng nhưng phải đọc 2 + ghi 1 (12 byte) — gần như không
  có tái sử dụng dữ liệu. ALU ngồi chờ DRAM, nên kernel chạm vào "mái nghiêng"
  (memory roof). Đạt 82% peak BW.

**Trả lời câu hỏi central:** matmul "nhanh hơn" không phải vì ít việc — nó làm nhiều
FLOPs hơn hẳn — mà vì AI cao giúp nó *tận dụng được* sức tính của GPU. VecAdd ít FLOPs
nhưng AI quá thấp nên bị memory wall chặn: thêm ALU cũng vô ích, chỉ băng thông bộ nhớ
mới quyết định. Đó là lý do tối ưu kernel ML thường xoay quanh việc *tăng AI* (fusion,
tiling, tái sử dụng dữ liệu) chứ không chỉ giảm số phép tính.
