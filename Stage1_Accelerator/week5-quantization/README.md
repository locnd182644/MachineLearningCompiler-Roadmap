# Tuần 5 — Numerical Formats & Quantization

> **Câu hỏi central:** Tại sao chip AI không dùng FP64? Khi nào INT8 đủ,
> khi nào không?

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `quantize.py` | Đầy đủ | 5.1 — Quantization tự cài (symmetric INT8) |
| `quantize_resnet18.py` | **Stub** | 5.2 — PTQ ResNet18, đo accuracy drop + speedup |
| `precision_stability.py` | **Stub** | 5.3 — FP32 vs FP16 vs BF16 training stability |

## Run

```bash
python quantize.py               # chạy được ngay
python quantize_resnet18.py      # sau khi cài TODO; cần ImageNet val subset
python precision_stability.py    # sau khi cài TODO; tải MNIST tự động
```

## Cần tự làm

- `quantize_resnet18.py`: PTQ ResNet18 bằng `torch.quantization`, so accuracy
  FP32 vs INT8, đo speedup inference trên CPU.
- `precision_stability.py`: train MLP trên MNIST với FP32/FP16/BF16, log
  gradient magnitude — quan sát FP16 dễ NaN.

## Checklist output cuối tuần

- [ ] `quantize.py`: mean relative error của INT8 matmul
- [ ] ResNet18 PTQ: accuracy drop + speedup
- [ ] Biểu đồ gradient magnitude 3 precision
- [ ] Note 1 trang: "precision strategies in modern AI chips"
