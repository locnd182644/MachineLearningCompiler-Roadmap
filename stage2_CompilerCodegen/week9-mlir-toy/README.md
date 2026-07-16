# Tuần 9 — MLIR Toy Tutorial (7 chương)

> **Câu hỏi central:** *Xây một compiler MLIR từ đầu gồm những mảnh nào — dialect, lowering, codegen ghép với nhau ra sao?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-9--mlir-toy-tutorial-7-chương)
> 🔗 Tutorial chính thức: https://mlir.llvm.org/docs/Tutorials/Toy/

## Cấu trúc dự kiến

```
week9-mlir-toy/
├── README.md                # File này
├── notes/
│   ├── ch1_ch2_notes.md     # AST → MLIR, ODS/TableGen
│   ├── ch3_notes.md         # Canonicalization
│   ├── ch4_notes.md         # Interfaces (ShapeInference)
│   ├── ch5_notes.md         # Partial lowering — QUAN TRỌNG NHẤT
│   ├── ch6_notes.md         # LLVM lowering + JIT
│   └── ch7_notes.md         # Struct types
├── extensions/              # Bài tập mở rộng (code out-of-tree hoặc patch)
│   ├── toy_sub/             # Op mới toy.sub
│   ├── canon_x_minus_x/     # x - x → zeros pattern
│   └── op_counter_pass/     # Pass thống kê op
└── lowering_trace.md        # IR walkthrough qua từng pass
```

## TODO

### Ch1-2 — AST & định nghĩa Toy dialect (ngày 1)
- [ ] Build và chạy `toyc-ch1`, `toyc-ch2` từ `mlir/examples/toy/`
- [ ] Đọc kỹ `Ops.td` — hiểu ODS: arguments, results, builders, traits
- [ ] Gõ lại (không copy) định nghĩa 2 op trong ODS, build lại thành công
- [ ] Note: TableGen sinh ra những file C++ nào? (xem trong build dir)

### Ch3 — High-level optimization (ngày 2)
- [ ] Chạy ví dụ `transpose(transpose(x)) = x`
- [ ] Đọc `ToyCombine.cpp` + `ToyCombine.td` — 2 cách viết pattern (C++ vs DRR)
- [ ] Hiểu `getCanonicalizationPatterns` được gọi khi nào

### Ch4 — Interfaces (ngày 3)
- [ ] Hiểu vì sao inlining + shape inference cần interface thay vì hardcode
- [ ] Trace ShapeInferencePass: nó lan truyền shape qua call graph thế nào
- [ ] Note: `OpInterface` vs `DialectInterface` khác nhau gì

### Ch5 — Partial lowering → affine (ngày 4) ⭐ QUAN TRỌNG NHẤT
- [ ] Hiểu `ConversionTarget`: legal/illegal ops, partial vs full conversion
- [ ] Trace `toy.mul` → `affine.for` lồng nhau + `arith.mulf`
- [ ] Vì sao `toy.print` được giữ lại (partial)? Điều này cho phép gì?
- [ ] Note: `TypeConverter` làm gì khi tensor → memref

### Ch6 — Lowering → LLVM + JIT (ngày 5)
- [ ] Chạy end-to-end: Toy source → JIT execute ra kết quả
- [ ] Trace pipeline pass đầy đủ trong `toyc-ch6`
- [ ] Note: `ExecutionEngine` hoạt động thế nào

### Ch7 — Struct types (ngày 6, đọc nhanh)
- [ ] Đọc hiểu custom type definition, không cần làm sâu

### Bài tập mở rộng (ngày 6-7) — BẮT BUỘC
- [ ] **`toy.sub`**: define trong ODS + verifier + lower xuống affine + test end-to-end — làm không nhìn guide
- [ ] **Canonicalization `x - x → zeros`**: viết `RewritePattern`, verify bằng `--canonicalize`
- [ ] **Op counter pass**: pass in "module có N matmul, M transpose" — học pass manager + `walk()`
- [ ] **Lowering trace**: chạy 1 chương trình Toy với `--mlir-print-ir-after-all`, viết `lowering_trace.md` giải thích từng bước biến đổi

### Output cuối tuần
- [ ] Toy compiler cả 7 chương build & chạy
- [ ] 3 extension tự viết hoạt động
- [ ] `lowering_trace.md`
- [ ] (Optional) Blog post tuần 9
