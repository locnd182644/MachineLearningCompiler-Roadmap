# Tuần 10 — MLIR ML Dialects: linalg, tensor, memref, affine

> **Câu hỏi central:** *Một `linalg.matmul` đi xuống loop nest cụ thể qua những bước nào, và tiling/fusion xảy ra ở đâu trong hành trình đó?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-10--mlir-ml-dialects-linalg-tensor-memref-affine)

## Cấu trúc dự kiến

```
week10-ml-dialects/
├── README.md                    # File này
├── lowering_walkthrough/
│   ├── matmul.mlir              # linalg.matmul trên tensor
│   ├── step1_tiled.mlir         # sau tiling
│   ├── step2_bufferized.mlir    # sau bufferization
│   ├── step3_loops.mlir         # sau convert-linalg-to-loops
│   ├── step4_llvm.mlir          # sau lower xuống LLVM dialect
│   ├── pipeline.sh              # script chạy toàn bộ
│   └── walkthrough.md           # chú thích từng bước
├── transform_scripts/
│   ├── tile_2level.mlir         # transform dialect: tile 64x64 → 8x8
│   └── fuse_matmul_relu.mlir    # fusion demo
├── standalone-pass/             # Out-of-tree MLIR pass project (C++)
│   ├── CMakeLists.txt
│   ├── lib/MulToAddPass.cpp     # mulf(x, 2.0) → addf(x, x)
│   └── test/mul_to_add.mlir     # FileCheck test
└── notes/
    ├── linalg_notes.md
    ├── tensor_vs_memref.md
    └── transform_dialect_notes.md
```

## TODO

### Lý thuyết (8h)
- [ ] Đọc Linalg Dialect Rationale (docs) — hiểu structured ops philosophy
- [ ] Hiểu `linalg.generic`: indexing maps + iterator types + scalar body — tự viết tay 1 `linalg.generic` tương đương matmul
- [ ] Đọc Bufferization docs — tensor vs memref, one-shot-bufferize
- [ ] Note `tensor_vs_memref.md`: ai có địa chỉ, ai immutable, bufferization quyết định gì
- [ ] Học `affine.for` vs `scf.for` — vì sao affine phân tích dependence được
- [ ] Làm Transform Dialect tutorial (docs chính thức)

### Bài tập 10.1 — Matmul lowering pipeline tay ⭐ project chính
- [ ] Viết `matmul.mlir` với `linalg.matmul` trên `tensor<128x128xf32>`
- [ ] Tile 32x32x32 (transform dialect hoặc test pass), lưu `step1_tiled.mlir`
- [ ] Bufferize với `--one-shot-bufferize`, lưu `step2_bufferized.mlir`
- [ ] `--convert-linalg-to-loops`, lưu `step3_loops.mlir`
- [ ] Lower xuống LLVM dialect đầy đủ, lưu `step4_llvm.mlir`
- [ ] Chạy qua `mlir-runner`, verify kết quả đúng vs NumPy
- [ ] Viết `walkthrough.md` — chú thích cái gì thay đổi ở mỗi bước, thông tin gì mất đi

### Bài tập 10.2 — Tiling 2 cấp bằng Transform dialect
- [ ] Transform script tile matmul 64x64 → 8x8
- [ ] Dump IR, đối chiếu với CUDA tiled matmul tuần 6 stage 1 — note điểm tương đồng

### Bài tập 10.3 — Fusion quan sát được
- [ ] Viết matmul + ReLU (`linalg.generic`) liên tiếp
- [ ] Fuse bằng transform dialect
- [ ] Chứng minh bằng IR: intermediate tensor biến mất
- [ ] Note: liên hệ với op fusion / HBM traffic (tuần 1 stage 1)

### Bài tập 10.4 — Standalone C++ pass
- [ ] Setup out-of-tree project từ template `mlir/examples/standalone`
- [ ] Viết pass: `arith.mulf(x, 2.0)` → `arith.addf(x, x)` bằng `RewritePattern`
- [ ] Đăng ký pass, chạy được qua opt tool riêng
- [ ] Viết FileCheck test
- [ ] Note quy trình: đây là skeleton cho mọi pass sau này

### Output cuối tuần
- [ ] Lowering walkthrough hoàn chỉnh (5 IR dumps + chú thích)
- [ ] 2 transform scripts hoạt động
- [ ] Standalone pass + FileCheck test pass
- [ ] (Optional) Blog post tuần 10
