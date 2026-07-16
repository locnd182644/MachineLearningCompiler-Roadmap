# Tuần 8 — Giới thiệu MLIR + Build từ source

> **Câu hỏi central:** *Vì sao LLVM IR không đủ cho ML, đến mức phải xây cả một infrastructure mới (MLIR)?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-8--giới-thiệu-mlir--build-từ-source)

## Cấu trúc dự kiến

```
week8-mlir-intro/
├── README.md                # File này
├── build_notes.md           # Log quá trình build LLVM/MLIR + troubleshooting
├── handwritten_mlir/
│   ├── 01_arith_basic.mlir
│   ├── 02_scf_loop.mlir
│   ├── 03_dot_product.mlir  # chạy được qua mlir-cpu-runner
│   └── run_pipelines.sh     # các pass pipeline đã thử
├── mlir_anatomy_cheatsheet.md
└── paper_notes_mlir.md      # Note paper Lattner 2021
```

## TODO

### Lý thuyết (8h)
- [ ] Đọc MLIR paper (Lattner et al. 2021, CGO) — lần 1: big picture
- [ ] Đọc MLIR paper lần 2 — tập trung: dialects, regions, progressive lowering → viết `paper_notes_mlir.md`
- [ ] Đọc MLIR LangRef: operation / value / type / attribute / region / block
- [ ] Đọc tutorial "Understanding the IR Structure"
- [ ] Hiểu block arguments thay cho phi nodes (so sánh với SSA tuần 7)
- [ ] Tour các dialect: `func`, `arith`, `tensor`, `memref`, `linalg`, `affine`, `scf`, `vector`, `llvm` — note mỗi dialect ở mức trừu tượng nào
- [ ] Xem talk MLIR Tutorial (Mehdi Amini, LLVM Dev Meeting)

### Bài tập 8.1 — Build LLVM/MLIR từ source
- [ ] Clone llvm-project
- [ ] CMake configure với `-DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON` (xem lệnh đầy đủ trong README giai đoạn)
- [ ] `ninja check-mlir` pass toàn bộ
- [ ] Note lại RAM/disk/thời gian build + lỗi gặp phải vào `build_notes.md`
- [ ] Thêm `build/bin` vào PATH, verify `mlir-opt --version`

### Bài tập 8.2 — mlir-opt hands-on
- [ ] Viết tay `01_arith_basic.mlir` (hàm cộng nhân đơn giản), parse được bằng `mlir-opt`
- [ ] Viết tay `02_scf_loop.mlir` dùng `scf.for` + `iter_args`
- [ ] Viết tay `03_dot_product.mlir` — chạy đúng kết quả qua `mlir-cpu-runner`
- [ ] Chạy `--canonicalize`, quan sát khác biệt
- [ ] Chạy pipeline lower xuống LLVM dialect: `--convert-scf-to-cf --convert-arith-to-llvm ...`
- [ ] Dùng `--mlir-print-ir-after-all` xem IR sau từng pass, lưu output

### Bài tập 8.3 — Đọc code MLIR thật
- [ ] Đọc `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` — cách define op + folder + canonicalization
- [ ] Đọc file `.td` (TableGen/ODS) tương ứng `ArithOps.td` — hiểu ODS sinh gì
- [ ] Note cấu trúc vào cheatsheet

### Output cuối tuần
- [ ] LLVM/MLIR build thành công (`check-mlir` pass)
- [ ] 3+ file `.mlir` viết tay + `run_pipelines.sh`
- [ ] `mlir_anatomy_cheatsheet.md`
- [ ] (Optional) Blog post tuần 8
