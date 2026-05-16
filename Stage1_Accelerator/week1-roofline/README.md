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

- [ ] `memcpy_bench` chạy, thấy BW giảm rõ khi size vượt L3
- [ ] `roofline_theory.py` in AI + bound cho 5 kernel; **TODO**: vẽ plot roofline
- [ ] `roofline_measured.py` cho % peak của matmul và vecadd
- [ ] Viết phần "chỗ chênh lệch lý thuyết vs thực tế" vào README này bên dưới

## Phân tích (tự viết)

> Matmul đạt ... % peak FLOPS. VecAdd đạt ... % peak BW.
> Giải thích bằng roofline: ...
