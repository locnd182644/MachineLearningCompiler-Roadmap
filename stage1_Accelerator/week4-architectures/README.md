# Tuần 4 — Dataflow & Spatial Architectures: Vượt khỏi TPU

> **Câu hỏi central:** Có những cách thiết kế AI chip nào khác ngoài systolic?
> Tradeoff là gì?

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `eyeriss_sim.py` | ✅ **Hoàn thành** | 4.1 — Mô phỏng Eyeriss row-stationary cho conv |
| `eyeriss_sim_cycle.py` | ✅ **Bonus: Cycle-accurate** | 4.1+ — Version cycle-accurate với utilization tracking |
| `tt_metalium_notes.md` | ✅ **Hoàn thành** | 4.2 — Build tt-metalium + đọc example |
| `comparison.md` | ✅ **Hoàn thành** | 4.3 — So sánh 4 kiến trúc (phân tích 3+ trang) |

## Cần tự làm

- `eyeriss_sim.py`: cài đặt row-stationary dataflow cho convolution 2D.
  Mở rộng tư duy từ systolic simulator tuần 3.
- `tt_metalium_notes.md`: ghi lại quá trình build `tenstorrent/tt-metal` và
  đọc 1-2 example (`eltwise_binary`, `matmul_single_core`).
- `comparison.md`: hoàn thiện bảng so sánh + viết phân tích 3 trang.

## Checklist output cuối tuần

- [x] `eyeriss_sim.py` chạy conv đúng với row-stationary
- [x] `eyeriss_sim_cycle.py` (bonus): cycle-accurate simulator với utilization metrics
- [x] Build tt-metalium thành công, ghi notes về example
- [x] `comparison.md`: bảng đầy đủ + 3-page analysis về tradeoffs

## Tóm tắt nội dung đã hoàn thành

### 4.1 Eyeriss Simulator
- **`eyeriss_sim.py`**: Mô phỏng row-stationary dataflow, tính đúng convolution 2D
  - Đếm data movement (DRAM vs on-chip)
  - Energy model: so sánh Eyeriss vs naive (giảm ~10-50x truy cập DRAM)
  - Stats chi tiết về weight reuse, input streaming, psum accumulation
  
- **`eyeriss_sim_cycle.py`**: Cycle-accurate version
  - 1 MAC/cycle per PE
  - Track utilization, active MACs per cycle
  - Skew mechanism cho psum accumulation
  - Tiling theo output rows

### 4.2 Tenstorrent tt-metal Notes
- Build process đầy đủ với dependencies
- Phân tích `eltwise_binary` example:
  - 3 kernels: reader, compute, writer
  - Circular buffer mechanism
  - Host vs device code separation
  
- Phân tích `matmul_single_core`:
  - Tiling strategy 32×32
  - Dispatch flow từ host → core
  - So sánh systolic array: explicit tiling vs implicit dataflow

### 4.3 Architecture Comparison
Chi tiết 3+ trang phân tích 4 kiến trúc:

1. **Compute organization**: 
   - TPU: monolithic MXU (throughput cao, kém flexible)
   - Tenstorrent: mesh of tiles (flexible, compiler phức tạp)
   - Groq: deterministic lanes (latency thấp, compile lâu)
   - Cerebras: 850K cores (throughput đỉnh, giá cực cao)

2. **Memory hierarchy**:
   - HBM-centric (TPU): capacity lớn, bandwidth bottleneck
   - SRAM-centric (Groq/Cerebras): latency thấp, capacity giới hạn
   - Hybrid (Tenstorrent): balance nhưng NoC overhead

3. **Programming model**:
   - Static (TPU/Groq): predictable, inflexible
   - Dataflow (Tenstorrent): flexible, runtime overhead
   - Streaming (Cerebras): zero DRAM, sync phức tạp

4. **Kết luận thực tế**: chọn kiến trúc theo workload
   - Latency-critical → Groq
   - Throughput batch lớn → TPU
   - Training khổng lồ → Cerebras
   - Research flexible → Tenstorrent

## Key Insights

1. **Không có kiến trúc "tốt nhất"**: mỗi thiết kế optimize cho 1 điểm trong design space
2. **Compiler complexity tăng theo flexibility**: TPU (khó) → Groq (extreme) → Tenstorrent (very hard) → Cerebras (nightmare)
3. **Memory hierarchy quyết định scalability**: on-chip SRAM cho latency, HBM cho capacity
4. **Dataflow vs static scheduling**: tradeoff giữa flexibility và predictability
