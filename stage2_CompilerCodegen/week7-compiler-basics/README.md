# Tuần 7 — Compiler Fundamentals Refresher

> **Câu hỏi central:** *Compiler truyền thống (C/C++) làm gì qua từng phase, và AI compiler kế thừa/vứt bỏ những gì?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-7--compiler-fundamentals-refresher)

## Cấu trúc thư mục

```
week7-compiler-basics/
├── README.md                       # File này
├── ai_vs_traditional_compiler.md   # ✅ So sánh AI compiler vs Traditional compiler
├── toycalc/                        # ✅ Toy calculator compiler
│   ├── __init__.py
│   ├── lexer.py                    # ✅ Phase 1: Tokenization (DFA-based)
│   ├── parser.py                   # ✅ Phase 2: Recursive descent parser → AST
│   ├── ast_printer.py              # ✅ Pretty printer (tree + compact + DOT)
│   ├── ir.py                       # ✅ Phase 3: AST → 3-address code (SSA)
│   ├── passes/
│   │   ├── const_fold.py           # ✅ Pass 1: Constant folding (fixpoint)
│   │   └── dce.py                  # ✅ Pass 2: Dead code elimination
│   ├── codegen.py                  # ✅ Phase 5: IR → stack machine instructions
│   ├── vm.py                       # ✅ Phase 6: Stack machine interpreter
│   ├── main.py                     # ✅ Entry point (CLI + demo mode)
│   └── tests/
│       └── test_toycalc.py         # ✅ 53 tests, tất cả pass
└── llvm_ir/
    ├── examples/                   # ✅ add.c, loop_sum.c, branchy_max.c + .ll dumps
    │   ├── add.c / add_O0.ll / add_O2.ll
    │   ├── loop_sum.c / loop_sum_O0.ll / loop_sum_O2.ll
    │   └── branchy_max.c / branchy_max_O0.ll / branchy_max_O2.ll
    └── llvm_ir_notes.md            # ✅ Phân tích -O0 vs -O2
```

## TODO

### Lý thuyết (8h)
- [ ] Đọc Cooper & Torczon ch.1 (Overview) — note pipeline: lexer → parser → AST → IR → opt → codegen
- [ ] Đọc Cooper & Torczon ch.5 (Intermediate Representations)
- [ ] Học SSA form (ch.9 hoặc tài liệu online): def-use chains, phi nodes — **bắt buộc nắm chắc, MLIR/LLVM đều là SSA**
- [ ] Đọc LLVM LangRef phần instructions cơ bản (`br`, `phi`, `getelementptr`, `alloca`)
- [x] Viết note so sánh AI compiler vs traditional compiler → [`ai_vs_traditional_compiler.md`](./ai_vs_traditional_compiler.md)

### Bài tập 7.1 — Toy calculator compiler
- [x] Lexer: tokenize `let x = 3 + 4 * 2; print(x)` → [`lexer.py`](./toycalc/lexer.py)
- [x] Parser → AST (precedence đúng: `*` trước `+`) → [`parser.py`](./toycalc/parser.py)
- [x] AST pretty printer (dump được là debug được!) → [`ast_printer.py`](./toycalc/ast_printer.py)
- [x] Lower AST → IR tuyến tính (3-address code, mỗi temp gán 1 lần) → [`ir.py`](./toycalc/ir.py)
- [x] IR printer — in IR trước/sau mỗi pass
- [x] Pass 1: Constant folding (`3 + 4 * 2` → `11` tại compile time) → [`const_fold.py`](./toycalc/passes/const_fold.py)
- [x] Pass 2: Dead code elimination (biến không dùng bị xóa) → [`dce.py`](./toycalc/passes/dce.py)
- [x] Codegen → stack machine instructions tự định nghĩa → [`codegen.py`](./toycalc/codegen.py)
- [x] VM/interpreter chạy stack machine → [`vm.py`](./toycalc/vm.py)
- [x] Test suite: 53 tests, 13+ full pipeline tests → [`test_toycalc.py`](./toycalc/tests/test_toycalc.py)

### Bài tập 7.2 — Đọc LLVM IR
- [x] Viết `add.c`, `loop_sum.c`, `branchy_max.c` → [`llvm_ir/examples/`](./llvm_ir/examples/)
- [x] `clang -S -emit-llvm -O0` và `-O2` cho cả 3 → 6 file .ll
- [x] Diff từng cặp, giải thích mỗi khác biệt trong [`llvm_ir_notes.md`](./llvm_ir/llvm_ir_notes.md)
- [x] Tìm được: loop bị vectorize chưa? branch bị chuyển thành `select` không? → ✅

### Output cuối tuần
- [x] Toy calculator compiler chạy được, có test (53 tests all pass ✅)
- [x] `llvm_ir_notes.md` — phân tích -O0 vs -O2
- [x] `ai_vs_traditional_compiler.md` (so sánh chi tiết)
- [ ] (Optional) Blog post tuần 7

## Chạy nhanh

```bash
# Chạy demo
python -m toycalc.main --demo

# Compile một chương trình cụ thể
python -m toycalc.main 'let x = 3 + 4 * 2; print(x)'

# Xem chi tiết pipeline
python -m toycalc.main --verbose 'let x = 3 + 4 * 2; print(x)'

# Xem IR trước/sau optimization
python -m toycalc.main --dump-ir 'let x = 3 + 4 * 2; print(x)'

# Chạy tests
python -m pytest toycalc/tests/ -v
```
