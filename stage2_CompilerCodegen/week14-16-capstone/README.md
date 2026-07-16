# Tuần 14-16 — Capstone: Mini Compiler PyTorch → Systolic Simulator

> **Câu hỏi central:** *Bạn đã thấy 4 hệ compiler của người khác. Giờ tự xây một cái — mọi mảnh kiến thức 15 tuần qua có khớp thành một hệ thống chạy được không?*
>
> 📖 Đặc tả đầy đủ: [README giai đoạn 2](../README.md#tuần-14-16--capstone-mini-compiler-pytorch--systolic-simulator)
>
> ⭐ **Portfolio piece quan trọng nhất của cả lộ trình.** Tái dùng systolic simulator tuần 3 stage 1 làm target backend.

## Pipeline mục tiêu

```
PyTorch MLP/CNN → torch.export/FX → Graph IR (tự thiết kế)
  → graph passes (fusion, const-fold, DCE)
  → Tensor IR (ops + tiles) → tiling + memory planning
  → Instruction stream (ISA tự định nghĩa)
  → Systolic simulator (upgraded: scratchpad + DMA + cycle count)
```

**Scope levels** — khuyến nghị mức 2:
- Mức 1: MLP, fusion matmul+relu, tiling 1 cấp, greedy memory planner
- Mức 2: + conv2d (im2col), double buffering, cost report fused vs unfused
- Mức 3: + INT8 quantization pass (tái dùng tuần 5 stage 1) hoặc viết bằng MLIR out-of-tree dialect

## Cấu trúc dự kiến

```
week14-16-capstone/
├── README.md              # File này
├── design.md              # Thiết kế IR + ISA TRÊN GIẤY TRƯỚC
├── tinycc/
│   ├── frontend/          # torch.export/FX → Graph IR
│   ├── graph_ir/          # IR định nghĩa + text printer
│   ├── passes/
│   │   ├── shape_inference.py
│   │   ├── const_fold.py
│   │   ├── dce.py
│   │   └── fusion.py      # matmul+bias+relu → fused op
│   ├── tensor_ir/         # Tensor IR + tiling + memory planner
│   ├── codegen/           # Tensor IR → instruction stream
│   ├── interpreter.py     # Golden reference (NumPy)
│   └── tests/
├── simulator/             # Systolic sim tuần 3, upgraded
│   ├── systolic_sim.py    # + scratchpad, DMA, cycle counter
│   └── isa.md             # đặc tả instruction set
├── evaluation/
│   ├── run_matrix.py
│   └── results.md         # ⭐ evaluation matrix
└── writeup.md             # ⭐ 2000+ từ, blog-quality
```

## TODO

### Tuần 14 — Frontend + Graph IR + graph passes

- [ ] **Thiết kế trước khi code** (`design.md`): Graph IR node/edge/shape/dtype, danh sách op tối thiểu, ISA sketch
- [ ] `torch.export` (hoặc `torch.fx`) một MLP nhỏ, duyệt FX graph, hiểu từng node
- [ ] Translator FX → Graph IR tự thiết kế
- [ ] Text printer cho Graph IR (dump được là debug được — bài học tuần 7)
- [ ] Pass: shape inference
- [ ] Pass: constant folding
- [ ] Pass: DCE
- [ ] Pass: fusion `matmul+bias+relu` → 1 fused op (pattern matcher)
- [ ] Interpreter tham chiếu chạy Graph IR bằng NumPy — **golden reference**
- [ ] Property test: mọi pass giữ nguyên output (so trước/sau pass vs golden)

### Tuần 15 — Tensor IR + tiling + memory planning

- [ ] Nâng cấp simulator: scratchpad SRAM kích thước cấu hình được (mặc định 256KB)
- [ ] Thêm DMA engine HBM↔scratchpad với cycle cost model
- [ ] Định nghĩa ISA: `DMA_LOAD`, `LOAD_WEIGHTS`, `MATMUL`, `DMA_STORE`, (mức 2: `ACT_RELU` fused) — viết `isa.md`
- [ ] Cycle counter + utilization + HBM traffic report trong simulator
- [ ] Tensor IR: biểu diễn op theo tiles
- [ ] Tiling pass: matmul lớn → tiles vừa array (16x16) + vừa scratchpad
- [ ] Xử lý padding khi shape không chia hết (bài học TPU tuần 3 stage 1)
- [ ] (Mức 2) Conv2d → im2col → matmul
- [ ] Memory planner: gán offset scratchpad, liveness đơn giản, phát hiện conflict
- [ ] (Mức 2) Double buffering: DMA tile kế tiếp trong khi compute tile hiện tại

### Tuần 16 — Codegen + evaluation + writeup

- [ ] Codegen: Tensor IR → instruction stream
- [ ] End-to-end test: PyTorch MLP → simulator, output khớp golden (sai số < 1e-4)
- [ ] (Mức 2) End-to-end CNN nhỏ
- [ ] **Evaluation matrix** (`results.md`):
  - [ ] Baseline: không fusion, không double-buffer → cycles / utilization / HBM traffic
  - [ ] + Fusion → đo lại
  - [ ] + Double buffering → đo lại
  - [ ] Tile size sweep: 8 / 16 / 32 / 64
  - [ ] (Mức 3) FP32 vs INT8
- [ ] Phân tích: fusion giảm HBM traffic bao nhiêu %? double buffering tăng utilization bao nhiêu? tile size tối ưu là gì và vì sao?
- [ ] **Writeup 2000+ từ**: kiến trúc compiler, mỗi pass ↔ ràng buộc hardware nào, số liệu, 3 điều sẽ làm khác đi
- [ ] Cập nhật README gốc của repo: đánh dấu Giai đoạn 2 hoàn thành

### Definition of Done

- [ ] `pytest` toàn bộ pass
- [ ] 1 lệnh duy nhất chạy demo end-to-end: `python -m tinycc examples/mlp.py --report`
- [ ] Evaluation matrix có số liệu thật, không placeholder
- [ ] Writeup đọc được như blog post
