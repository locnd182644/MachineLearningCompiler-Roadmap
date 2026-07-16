# Tuần 7 — Compiler Fundamentals Refresher

> **Câu hỏi central:** *Compiler truyền thống (C/C++) làm gì qua từng phase, và AI compiler kế thừa/vứt bỏ những gì?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-7--compiler-fundamentals-refresher)

## Cấu trúc dự kiến

```
week7-compiler-basics/
├── README.md              # File này
├── toycalc/               # Toy calculator compiler
│   ├── lexer.py
│   ├── parser.py          # → AST
│   ├── ast_printer.py
│   ├── ir.py              # 3-address code / SSA-ish IR
│   ├── passes/
│   │   ├── const_fold.py
│   │   └── dce.py
│   ├── codegen.py         # → stack machine instructions
│   ├── vm.py              # stack machine interpreter
│   └── tests/
├── llvm_ir/
│   ├── examples/          # add.c, loop_sum.c, branchy_max.c + .ll dumps
│   └── llvm_ir_notes.md   # phân tích -O0 vs -O2
└── ai_vs_traditional_compiler.md
```

## TODO

### Lý thuyết (8h)
- [ ] Đọc Cooper & Torczon ch.1 (Overview) — note pipeline: lexer → parser → AST → IR → opt → codegen
- [ ] Đọc Cooper & Torczon ch.5 (Intermediate Representations)
- [ ] Học SSA form (ch.9 hoặc tài liệu online): def-use chains, phi nodes — **bắt buộc nắm chắc, MLIR/LLVM đều là SSA**
- [ ] Đọc LLVM LangRef phần instructions cơ bản (`br`, `phi`, `getelementptr`, `alloca`)
- [ ] Viết note so sánh AI compiler vs traditional compiler (bảng trong README giai đoạn)

### Bài tập 7.1 — Toy calculator compiler
- [ ] Lexer: tokenize `let x = 3 + 4 * 2; print(x)`
- [ ] Parser → AST (precedence đúng: `*` trước `+`)
- [ ] AST pretty printer (dump được là debug được!)
- [ ] Lower AST → IR tuyến tính (3-address code, mỗi temp gán 1 lần)
- [ ] IR printer — in IR trước/sau mỗi pass
- [ ] Pass 1: Constant folding (`3 + 4 * 2` → `11` tại compile time)
- [ ] Pass 2: Dead code elimination (biến không dùng bị xóa)
- [ ] Codegen → stack machine instructions tự định nghĩa
- [ ] VM/interpreter chạy stack machine
- [ ] Test suite: 5+ chương trình, so kết quả với `eval()` Python

### Bài tập 7.2 — Đọc LLVM IR
- [ ] Viết `add.c`, `loop_sum.c`, `branchy_max.c`
- [ ] `clang -S -emit-llvm -O0` và `-O2` cho cả 3
- [ ] Diff từng cặp, giải thích mỗi khác biệt trong `llvm_ir_notes.md`
- [ ] Tìm được: loop bị vectorize chưa? branch bị chuyển thành `select` không?

### Output cuối tuần
- [ ] Toy calculator compiler chạy được, có test
- [ ] `llvm_ir_notes.md`
- [ ] `ai_vs_traditional_compiler.md` (1 trang)
- [ ] (Optional) Blog post tuần 7
