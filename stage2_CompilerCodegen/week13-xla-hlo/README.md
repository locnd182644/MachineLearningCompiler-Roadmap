# Tuần 13 — XLA & HLO IR

> **Câu hỏi central:** *Compiler production phục vụ hàng triệu TPU-hours mỗi ngày trông như thế nào — và nó khác đồ chơi của mình chỗ nào?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-13--xla--hlo-ir)
>
> ⚠️ Tuần "nhẹ" — nếu trễ tiến độ, nén xuống 2-3 ngày để bảo toàn capstone.

## Cấu trúc dự kiến

```
week13-xla-hlo/
├── README.md               # File này
├── dumps/
│   ├── mlp_stablehlo.txt   # trước optimize
│   ├── mlp_hlo_opt.txt     # sau optimize
│   └── pattern_*.txt       # 5 patterns fusion analysis
├── fusion_analysis.md      # ⭐ bảng pattern → fuse hay không → lý do
├── xla_pass_notes.md       # note đọc algebraic_simplifier
└── hlo_vs_mlir.md          # op set nhỏ chặt vs dialect mở
```

## TODO

### Lý thuyết (6h)
- [ ] Đọc XLA architecture docs (openxla.org): HLO, pipeline, backends
- [ ] Hiểu StableHLO vs HLO — vai trò serialization/interchange format
- [ ] Đọc về XLA fusion: producer-consumer, multi-output, horizontal
- [ ] Note: vì sao XLA fusion là heuristic (không search)? — ràng buộc JIT compile time

### Bài tập 13.1 — JAX → HLO dump
- [ ] Cài JAX, viết MLP 2 lớp (matmul → relu → matmul → softmax)
- [ ] Dump StableHLO: `jax.jit(f).lower(...).as_text()`
- [ ] Dump HLO optimized: `.compile().as_text()`
- [ ] Chú thích: op nào bị gộp vào `fusion` op? Layout nào được gán?

### Bài tập 13.2 — Phân tích fusion decisions ⭐
- [ ] Viết 5 hàm JAX: (1) elementwise chain, (2) matmul+bias+relu, (3) reduce sau matmul, (4) reshape ở giữa chain, (5) dynamic slice
- [ ] Dump HLO optimized cho từng cái
- [ ] Lập bảng `fusion_analysis.md`: pattern → fuse được không → suy luận lý do
- [ ] Xác nhận giả thuyết: reshape/dynamic shape phá fusion thế nào

### Bài tập 13.3 — (Optional) Colab TPU
- [ ] Chạy cùng code trên Colab TPU, dump HLO cho backend TPU
- [ ] So layout assignment CPU vs TPU — tìm dấu vết tiled layout theo MXU + padding

### Bài tập 13.4 — Đọc 1 pass XLA thật
- [ ] Đọc `algebraic_simplifier.cc` trong openxla/xla (~30-60 phút, đọc chọn lọc)
- [ ] Note 5 rewrite rules thú vị vào `xla_pass_notes.md`
- [ ] Đối chiếu với canonicalization patterns MLIR tuần 9 — cùng khái niệm, khác codebase

### Output cuối tuần
- [ ] HLO dumps + chú thích
- [ ] `fusion_analysis.md` (5 patterns)
- [ ] `hlo_vs_mlir.md`
- [ ] (Optional) Blog post tuần 13
