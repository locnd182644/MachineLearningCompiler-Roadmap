# Tuần 3 — TPU & Systolic Array

> **Câu hỏi central:** Tại sao Google thiết kế TPU dùng systolic array thay vì
> SIMD như GPU? Compiler cho TPU phải làm gì khác compiler GPU?

Đây là **project lớn nhất Giai đoạn 1**. Code systolic simulator sẽ được tái dùng
làm target backend cho compiler bạn xây ở Giai đoạn 2.

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `systolic_sim.py` | Đầy đủ | 3.1 — Output-stationary systolic array simulator |
| `systolic_extend.py` | **Stub** | 3.2 — Weight-stationary + tiling + đếm byte + plot |
| `analysis.md` | Template | 3.3 — So sánh với GPU + phân tích TPU |

## Run

```bash
python systolic_sim.py        # test matmul 4x4, in utilization
python systolic_extend.py     # sau khi bạn cài đặt phần TODO
```

## Cần tự làm (`systolic_extend.py`)

1. Chế độ **weight-stationary** (B ở yên trong PE, A và partial sums chảy qua)
2. **Tiling**: matmul lớn hơn array → chia tile, compute lần lượt
3. Đếm byte đọc/ghi "DRAM" (mỗi tile load)
4. Plot utilization theo cycle — quan sát ramp-up / ramp-down

## Checklist output cuối tuần

- [ ] `systolic_sim.py` chạy đúng (max error ~0), in utilization
- [ ] `systolic_extend.py`: weight-stationary chạy đúng
- [ ] Test matmul 4x4, 16x16, 64x64 (cần tiling)
- [ ] Plot utilization có ramp-up/ramp-down
- [ ] `analysis.md`: "vì sao TPU dùng systolic? 5 điểm so với GPU SIMT"
