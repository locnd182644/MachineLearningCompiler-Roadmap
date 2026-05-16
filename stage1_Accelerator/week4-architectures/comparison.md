# Tuần 4 — So sánh kiến trúc AI accelerator (Bài tập 4.3)

## Bảng so sánh

| Aspect | TPU v4 | Tenstorrent | Groq TSP | Cerebras WSE-2 |
|--------|--------|-------------|----------|----------------|
| Compute organization | 1 lớn MXU/core | Mesh of tiles | Vector lanes | 850K cores |
| On-chip memory | HBM + UB | Per-tile SRAM | 220 MB SRAM | 40 GB SRAM |
| Off-chip memory | HBM | DDR | None | None |
| Programming model | Static schedule (XLA) | Dataflow (Metalium) | Static (Groq compiler) | Streaming (CSL) |
| Compiler complexity | High | Very high | Extreme | Extreme |
| Best workload | Large dense matmul | Flexible | Low latency inference | Massive parallel |

## Mỗi kiến trúc → mỗi loại compiler

- **TPU (1 big MXU)**: compiler lo tiling cho MXU + memory scheduling.
- **Tenstorrent (mesh)**: compiler lo placement + routing qua NoC.
- **Groq (deterministic)**: compiler phải biết *exact cycle* mọi instruction.
- **Cerebras (wafer-scale)**: compiler lo dataflow trên 850K core.

## Phân tích tradeoff (tự viết — mục tiêu ~3 trang)

### 1. Compute organization: tập trung vs phân tán

> ...

### 2. Memory: on-chip lớn vs HBM

> ...

### 3. Programming model: static vs dataflow vs streaming

> ...

### Kết luận: chọn kiến trúc nào cho LLM inference?

> ...
