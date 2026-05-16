# Tuần 4 — Dataflow & Spatial Architectures: Vượt khỏi TPU

> **Câu hỏi central:** Có những cách thiết kế AI chip nào khác ngoài systolic?
> Tradeoff là gì?

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `eyeriss_sim.py` | **Stub** | 4.1 — Mô phỏng Eyeriss row-stationary cho conv |
| `tt_metalium_notes.md` | Template | 4.2 — Build tt-metalium + đọc example |
| `comparison.md` | Template (có sẵn bảng) | 4.3 — So sánh 4 kiến trúc |

## Cần tự làm

- `eyeriss_sim.py`: cài đặt row-stationary dataflow cho convolution 2D.
  Mở rộng tư duy từ systolic simulator tuần 3.
- `tt_metalium_notes.md`: ghi lại quá trình build `tenstorrent/tt-metal` và
  đọc 1-2 example (`eltwise_binary`, `matmul_single_core`).
- `comparison.md`: hoàn thiện bảng so sánh + viết phân tích 3 trang.

## Checklist output cuối tuần

- [ ] `eyeriss_sim.py` chạy conv đúng với row-stationary
- [ ] Build tt-metalium thành công, ghi notes về example
- [ ] `comparison.md`: bảng đầy đủ + 3-page analysis về tradeoffs
