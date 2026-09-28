# Tuần 9 — MLIR Toy Tutorial (7 chương)

> **Câu hỏi central:** *Xây một compiler MLIR từ đầu gồm những mảnh nào — dialect, lowering, codegen ghép với nhau ra sao?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-9--mlir-toy-tutorial-7-chương)
> 🔗 Tutorial chính thức: https://mlir.llvm.org/docs/Tutorials/Toy/
>
> ⏱ Thời lượng: ~18h (toàn tuần là thực hành, lý thuyết đan xen trong từng chương)

---

## Mục lục

- [Tổng quan tuần 9](#tổng-quan-tuần-9)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Chuẩn bị trước khi bắt đầu](#chuẩn-bị-trước-khi-bắt-đầu)
- [Ngày 1: Ch1-2 — Toy Language, AST & Toy Dialect](#ngày-1-ch1-2--toy-language-ast--toy-dialect)
- [Ngày 2: Ch3 — High-level Optimization](#ngày-2-ch3--high-level-optimization-canonicalization)
- [Ngày 3: Ch4 — Interfaces (Inlining & Shape Inference)](#ngày-3-ch4--interfaces-inlining--shape-inference)
- [Ngày 4: Ch5 — Partial Lowering → Affine ⭐](#ngày-4-ch5--partial-lowering--affine-)
- [Ngày 5: Ch6 — Full Lowering → LLVM + JIT](#ngày-5-ch6--full-lowering--llvm--jit)
- [Ngày 6: Ch7 — Struct Types](#ngày-6-ch7--struct-types-custom-type)
- [Ngày 6-7: Bài tập mở rộng (BẮT BUỘC)](#ngày-6-7-bài-tập-mở-rộng-bắt-buộc)
- [Liên hệ HW-SW](#liên-hệ-hw-sw)
- [TODO Checklist](#todo-checklist)
- [Output cuối tuần](#output-cuối-tuần)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Tổng quan tuần 9

Tuần này là **tuần nặng nhất** của Giai đoạn 2 (cùng tuần 10). Bạn sẽ đi qua toàn bộ 7 chương của [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) — xây một compiler hoàn chỉnh từ language → MLIR → optimization → lowering → JIT execution.

**Tại sao tuần này quan trọng?** Vì nó là **mô hình thu nhỏ hoàn chỉnh** của mọi compiler MLIR-based (IREE, Triton, tt-mlir, Mojo...):

```
Toy tutorial (tuần này):               Compiler thật (production):

Ch1-2: Toy source → Toy dialect        torch.export → StableHLO/Linalg
Ch3:   Canonicalization patterns        Op fusion, algebraic simplification
Ch4:   Interfaces (inlining, shapes)    Call graph analysis, type inference
Ch5:   Partial lowering → Affine        Linalg → Affine/SCF → target dialect
Ch6:   Full lowering → LLVM + JIT       → LLVM → machine code/PTX
Ch7:   Custom types                     Custom tensor types cho hardware
```

Mọi khái niệm học ở đây sẽ **xuất hiện lại** ở tuần 10-16 và trong bất kỳ dự án MLIR thật nào.

### Kết nối với tuần trước

| Tuần 7 (bạn tự xây) | Tuần 8 (MLIR infrastructure) | Tuần 9 (tuần này) |
|---------------------|------------------------------|-------------------|
| Lexer/parser tay | Đọc hiểu MLIR Language Ref | Frontend cho ngôn ngữ Toy |
| 3-address code IR | Viết tay `.mlir`, dùng `mlir-opt` | Define dialect + ODS |
| Constant folding pass | Quan sát `--canonicalize` | Viết `RewritePattern` |
| DCE pass | Quan sát pass pipeline | Implement interface pass |
| Stack machine codegen | Pipeline lower xuống LLVM | Dialect conversion framework |

**Tuần 7** bạn tự xây mọi thứ từ đầu để hiểu nguyên lý. **Tuần 8** bạn thấy MLIR có sẵn infrastructure mạnh hơn. **Tuần 9** bạn dùng infrastructure đó xây compiler hoàn chỉnh — và hiểu vì sao có nó.

### Nguyên tắc tuần này

1. **KHÔNG copy-paste code từ tutorial.** Đọc tutorial → hiểu → gõ lại phần core. Build → test → thử sửa đổi. Copy-paste thì 2 ngày xong nhưng không hiểu gì.

2. **Dump IR mọi lúc.** Kỹ năng nghề nghiệp số 1 của compiler engineer. Dùng `--mlir-print-ir-after-all` để xem IR sau từng pass. Khi gặp bug, dump IR trước/sau pass — 90% lỗi nhìn thấy ngay.

3. **Mỗi chương, tự trả lời:** "Pass/component này giải quyết ràng buộc nào? Nếu bỏ nó đi, điều gì xảy ra?"

---

## Cấu trúc thư mục

```
week9-mlir-toy/
├── README.md                    # File này — lesson plan chi tiết
├── notes/
│   ├── ch1_ch2_notes.md         # AST → MLIR, ODS/TableGen
│   ├── ch3_notes.md             # Canonicalization patterns
│   ├── ch4_notes.md             # Interfaces (Inlining + ShapeInference)
│   ├── ch5_notes.md             # Partial lowering — QUAN TRỌNG NHẤT
│   ├── ch6_notes.md             # LLVM lowering + JIT
│   └── ch7_notes.md             # Struct types
├── extensions/                  # Bài tập mở rộng (code out-of-tree hoặc patch)
│   ├── toy_sub/                 # Op mới toy.sub
│   ├── canon_x_minus_x/         # x - x → zeros pattern
│   └── op_counter_pass/         # Pass thống kê op
├── lowering_trace.md            # IR walkthrough qua từng pass
└── toy_test_programs/           # Chương trình Toy tự viết để test
    ├── basic_matmul.toy         # Matmul đơn giản
    ├── nested_transpose.toy     # Test transpose(transpose(x))
    ├── multi_function.toy       # Test inlining
    └── shape_inference.toy      # Test shape propagation
```

---

## Chuẩn bị trước khi bắt đầu

### Prerequisite check

Trước khi bắt đầu, verify bạn đã có:

```bash
# 1. LLVM/MLIR đã build từ tuần 8 (với LLVM_BUILD_EXAMPLES=ON!)
mlir-opt --version
which toyc-ch1   # Nếu không có → rebuild với LLVM_BUILD_EXAMPLES=ON

# 2. Kiểm tra Toy Tutorial code
ls $LLVM_BUILD_DIR/../mlir/examples/toy/
# Phải thấy: Ch1/ Ch2/ Ch3/ Ch4/ Ch5/ Ch6/ Ch7/

# 3. Các binary toy phải build được
ls $LLVM_BUILD_DIR/bin/toyc-ch*
# Phải thấy: toyc-ch1, toyc-ch2, toyc-ch3, toyc-ch4, toyc-ch5, toyc-ch6, toyc-ch7
```

Nếu thiếu `toyc-ch*`, rebuild:

```bash
cd $LLVM_BUILD_DIR
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
ninja toyc-ch1 toyc-ch2 toyc-ch3 toyc-ch4 toyc-ch5 toyc-ch6 toyc-ch7
```

### File test chuẩn

Tạo file test chuẩn để dùng xuyên suốt 7 chương:

```python
# test/basic.toy — chương trình cơ bản
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
}

def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

```python
# test/nested_transpose.toy — test double transpose elimination
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b = transpose(transpose(a));
  print(b);
}
```

---

## Ngày 1: Ch1-2 — Toy Language, AST & Toy Dialect

> **Mục tiêu:** Hiểu pipeline frontend Toy → AST → MLIR, và cách define dialect bằng ODS/TableGen.
>
> ⏱ ~3h
> 📚 Đọc: [Ch1](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/) + [Ch2](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
> 📁 Code: `mlir/examples/toy/Ch1/` + `mlir/examples/toy/Ch2/`

### Ch1 — Toy Language & AST

#### 1.1. Ngôn ngữ Toy

Toy là ngôn ngữ đồ chơi tensor-based, hỗ trợ:
- **Kiểu dữ liệu**: chỉ có tensor double (f64), rank ≤ 2
- **Biến**: `var` với shape tùy chọn `var x<2, 3> = ...`
- **Hàm**: `def` — tất cả đều trả về 1 giá trị
- **Phép tính**: `+` (cộng element-wise), `*` (nhân element-wise)
- **Built-in**: `transpose()`, `print()`
- **Type inference**: shapes suy luận từ context

```python
# Toy source — giống Python nhưng đơn giản hơn rất nhiều
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

#### 1.2. Frontend: Lexer → Parser → AST

Frontend của Toy là compiler cổ điển (giống tuần 7 của bạn):

```
Source code → Lexer (tokenize) → Parser (recursive descent) → AST
```

AST nodes:
- `NumberExprAST`, `LiteralExprAST` — hằng số
- `VariableExprAST` — biến
- `BinaryExprAST` — phép toán `+`, `*`
- `CallExprAST` — gọi hàm
- `PrintExprAST` — print
- `ReturnExprAST` — return
- `PrototypeAST`, `FunctionAST`, `ModuleAST` — khai báo hàm/module

> 💡 **Liên hệ tuần 7**: AST này tương tự cái bạn xây cho calculator — nhưng hỗ trợ functions, tensors, và type inference.

#### 1.3. Thực hành Ch1

```bash
# Build (nếu chưa)
cd $LLVM_BUILD_DIR && ninja toyc-ch1

# Chạy — dump AST
toyc-ch1 test/basic.toy -emit=ast
```

**Quan sát output AST:**
- Mỗi function → `FunctionAST` node
- Tensor literal `[[1,2,3],[4,5,6]]` → nested `LiteralExprAST`
- `transpose(a) * transpose(b)` → `BinaryExprAST { CallExprAST, *, CallExprAST }`

**Câu hỏi tự trả lời:**
1. AST biết shape của `a` (2×3) không? (Đáp án: biết nếu khai báo `var a<2,3>`, không biết nếu là tham số hàm)
2. AST phân biệt được `+` và `*`? (Đáp án: có, qua op field trong `BinaryExprAST`)
3. AST hoàn toàn đủ làm compiler Toy không? Cần IR vì sao? (Đáp án: cần IR để optimize — AST là cây, khó traverse lại; IR là SSA tuyến tính, dễ phân tích dataflow)

---

### Ch2 — Emitting Basic MLIR (Toy Dialect)

> **Đây là bước nhảy quan trọng:** từ AST → MLIR. Bạn sẽ hiểu tại sao cần MLIR thay vì AST.

#### 2.1. Tại sao cần MLIR IR thay vì dùng AST trực tiếp?

| | AST | MLIR IR |
|---|-----|---------|
| Cấu trúc | Cây (tree) | SSA graph tuyến tính |
| Traverse | Khó (phải recursive) | Dễ (duyệt op tuần tự) |
| Optimization | Rất khó (phải sửa cây) | Dễ (pattern match + rewrite) |
| Analysis | Manual | SSA tự cho def-use chains |
| Shared infra | Không | Pass manager, verifier, printer... |
| Testing | Ad-hoc | FileCheck — framework chuẩn |

#### 2.2. Define Toy Dialect bằng ODS/TableGen

ODS (Operation Definition Specification) là DSL khai báo ops. Từ file `.td`, TableGen sinh C++ tự động.

File chính: `Ops.td` — define tất cả ops của Toy dialect:

```tablegen
// Định nghĩa ConstantOp — op hằng số tensor
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant operation";
  let description = [{
    Constant operation produces a constant tensor value.
  }];

  // Đầu vào: 1 attribute (tensor literal — compile-time constant)
  let arguments = (ins F64ElementsAttr:$value);
  // Đầu ra: 1 tensor f64
  let results = (outs F64Tensor);

  // Custom builders — cách tạo op từ C++
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{...}]>,
    OpBuilder<(ins "double":$value), [{...}]>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}
```

**TableGen sinh ra những gì?** (kiểm tra trong build directory)
```bash
# Tìm generated files
ls $LLVM_BUILD_DIR/tools/mlir/examples/toy/Ch2/*.inc
# → Ops.h.inc, Ops.cpp.inc — class C++ cho mỗi op
# → Dialect.h.inc, Dialect.cpp.inc — class cho dialect
```

| Khai báo trong `.td` | TableGen sinh |
|---------------------|---------------|
| `def ConstantOp` | Class `ConstantOp` kế thừa `Op<>` |
| `let arguments` | `getOperands()`, `getValue()` methods |
| `let results` | `getResult()` method |
| `let builders` | `build()` static methods |
| `let hasVerifier` | Skeleton `verify()` — bạn implement |
| `let hasCustomAssemblyFormat` | Skeleton `parse()/print()` |

#### 2.3. MLIRGen — AST → MLIR

`MLIRGen.cpp` traverse AST và tạo MLIR ops bằng `OpBuilder`:

```cpp
// Pseudocode đơn giản hóa
mlir::Value mlirGen(BinaryExprAST &binOp) {
  mlir::Value lhs = mlirGen(*binOp.getLHS());
  mlir::Value rhs = mlirGen(*binOp.getRHS());

  if (binOp.getOp() == '+')
    return builder.create<AddOp>(loc, lhs, rhs);
  if (binOp.getOp() == '*')
    return builder.create<MulOp>(loc, lhs, rhs);
}
```

**Location tracking**: Mọi op đều có `loc` — source location cho diagnostics. Đây là lý do MLIR error messages luôn chỉ đúng dòng lỗi.

#### 2.4. Toy MLIR output

```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
      -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
         : tensor<2x3xf64>
    %1 = toy.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]>
         : tensor<6xf64>
    %2 = toy.reshape(%1 : tensor<6xf64>) to tensor<2x3xf64>
    %3 = toy.generic_call @multiply_transpose(%0, %2)
         : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %4 = toy.generic_call @multiply_transpose(%2, %0)
         : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %4 : tensor<*xf64>
  }
}
```

**Quan sát:**
- `tensor<*xf64>` = unranked tensor (shape chưa biết — tham số hàm)
- `tensor<2x3xf64>` = ranked tensor (shape đã biết — literal)
- `toy.reshape` tự chèn khi shape khai báo khác shape literal
- Mỗi op SSA: `%0`, `%1`, `%2` — gán đúng 1 lần
- `toy.generic_call` gọi hàm — chưa inline

#### 2.5. Verifier — tại sao cần?

```cpp
// ConstantOp verifier — kiểm tra invariants
mlir::LogicalResult ConstantOp::verify() {
  // Kiểm tra result type là tensor f64
  auto resultType = llvm::dyn_cast<RankedTensorType>(getResult().getType());
  if (!resultType) return success();  // unranked thì ok

  // Kiểm tra shape khớp attribute value
  auto attrType = llvm::cast<RankedTensorType>(getValue().getType());
  if (attrType.getShape() != resultType.getShape())
    return emitOpError("shape mismatch between attribute and result");

  return success();
}
```

**Tại sao verifier quan trọng?** Mỗi pass giả định IR đúng invariant. Nếu pass trước sinh IR sai → pass sau crash bí ẩn. Verifier chạy **sau mỗi pass** (khi enable assertions) → catch lỗi sớm nhất có thể. **Đây là bài học đắt giá: compiler mà không verify = debug nightmare.**

#### 2.6. Thực hành Ch2

```bash
# Build
ninja toyc-ch2

# AST dump (giống ch1)
toyc-ch2 test/basic.toy -emit=ast

# MLIR dump — SO SÁNH VỚI AST!
toyc-ch2 test/basic.toy -emit=mlir

# Parse lại bằng mlir-opt (verify IR hợp lệ)
toyc-ch2 test/basic.toy -emit=mlir 2>/dev/null | mlir-opt
```

**Bài tập thực hành Ch1-2:**

1. **Đọc `Ops.td`** — hiểu ODS: `arguments`, `results`, `builders`, `traits` cho mỗi op
2. **Gõ lại 2 op**: chọn `ConstantOp` và `TransposeOp`, gõ lại định nghĩa ODS từ đầu (không nhìn source), build lại — nếu build pass, bạn hiểu ODS
3. **Kiểm tra TableGen output**: tìm file `.inc` trong build dir, mở đọc — hiểu TableGen sinh code gì
4. **Viết chương trình Toy mới**: tạo `nested_transpose.toy` với `transpose(transpose(a))`, dump MLIR, quan sát output

> 📝 Output: [`notes/ch1_ch2_notes.md`](./notes/ch1_ch2_notes.md) — ghi lại: ODS syntax chính, TableGen sinh file gì, cấu trúc MLIRGen

---

## Ngày 2: Ch3 — High-level Optimization (Canonicalization)

> **Mục tiêu:** Hiểu canonicalization pattern — cách compiler "dọn dẹp" IR bằng pattern matching.
>
> ⏱ ~2.5h
> 📚 Đọc: [Ch3](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
> 📁 Code: `mlir/examples/toy/Ch3/`

### 3.1. Vấn đề: IR sau MLIRGen chưa tối ưu

Sau Ch2, MLIR sinh ra trực tiếp từ AST — chưa có optimization nào:

```mlir
// transpose(transpose(x)) — hiển nhiên thừa!
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>  // = %arg0!

// reshape không cần thiết khi shape đã đúng
%0 = toy.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf64>
%1 = toy.reshape(%0 : tensor<2x2xf64>) to tensor<2x2xf64>  // no-op!
```

### 3.2. Canonicalization Framework

MLIR có sẵn canonicalization pass (`--canonicalize`) hoạt động theo nguyên tắc:
1. Mỗi op **đăng ký** canonicalization patterns trong `getCanonicalizationPatterns()`
2. Pass chạy greedy pattern rewriting cho đến fixpoint (không còn pattern nào match)

Có **2 cách** viết pattern: C++ trực tiếp và DRR (Declarative Rewrite Rules).

### 3.3. Cách 1: C++ RewritePattern

```cpp
// Pattern: transpose(transpose(x)) → x
struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      TransposeOp op, mlir::PatternRewriter &rewriter) const override {
    // Nhìn xem operand của transpose này có phải là transpose không
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    if (!transposeInputOp)
      return failure();  // Operand không phải transpose → không match

    // transpose(transpose(x)) = x → thay thế op này bằng x
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

// Đăng ký pattern vào TransposeOp
void TransposeOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<SimplifyRedundantTranspose>(context);
}
```

**Phân tích từng dòng:**
- `matchAndRewrite` — hàm core: nhận op cần xét, trả `success()` nếu match + đã rewrite, `failure()` nếu không match
- `op.getOperand()` — lấy input SSA value
- `transposeInput.getDefiningOp<TransposeOp>()` — "ai tạo ra value này?" → nếu là TransposeOp thì match
- `rewriter.replaceOp(op, {transposeInputOp.getOperand()})` — thay thế toàn bộ op bằng kết quả khác
- **Rewriter**: không được sửa IR trực tiếp — phải thông qua `PatternRewriter` để MLIR theo dõi modifications

> 💡 **Liên hệ tuần 7**: constant folding bạn viết cho calculator duyệt qua IR list, tìm `CONST + CONST` → fold. Ở đây cũng cùng ý tưởng — nhưng infrastructure mạnh hơn: pattern matching tự động, greedy fixpoint, conflict resolution.

### 3.4. Cách 2: DRR (Declarative Rewrite Rules) bằng TableGen

```tablegen
// Cùng pattern, viết bằng TableGen — ngắn hơn nhiều!
def TransposeTransposeOptPattern : Pat<
  (TransposeOp(TransposeOp $arg)),        // match: transpose(transpose($arg))
  (replaceWithValue $arg)>;               // rewrite: → $arg
```

Ưu điểm DRR:
- Ngắn gọn, ít sai
- Sinh C++ code tự động — nhanh, nhất quán

Nhược điểm DRR:
- Pattern đơn giản mới viết được — complex logic cần C++
- Debug khó hơn (phải đọc generated code)

### 3.5. Reshape canonicalization patterns

```tablegen
// Pattern 1: reshape(reshape(x)) → reshape(x) với shape mới
def ReshapeReshapeOptPattern : Pat<
  (ReshapeOp(ReshapeOp $arg)),
  (ReshapeOp $arg)>;

// Pattern 2: reshape(x) khi shape x == shape output → xóa reshape
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg),
  (replaceWithValue $arg),
  [(HasNoEffect $res, $arg)]>;  // constraint: shape phải giống nhau
```

### 3.6. ODS khai báo — kích hoạt canonicalization

Trong `Ops.td`, mỗi op cần khai báo:

```tablegen
def TransposeOp : Toy_Op<"transpose", [Pure]> {
  // ...
  let hasCanonicalizer = 1;  // ← Báo cho MLIR: op này có canonicalization patterns
}
```

`hasCanonicalizer = 1` tạo khai báo hàm `getCanonicalizationPatterns()` — bạn implement trong `.cpp`.

### 3.7. Thực hành Ch3

```bash
ninja toyc-ch3

# Không optimize — thấy transpose(transpose(x))
toyc-ch3 test/nested_transpose.toy -emit=mlir

# Có optimize — transpose biến mất!
toyc-ch3 test/nested_transpose.toy -emit=mlir -opt

# Diff để thấy rõ
diff <(toyc-ch3 test/nested_transpose.toy -emit=mlir 2>/dev/null) \
     <(toyc-ch3 test/nested_transpose.toy -emit=mlir -opt 2>/dev/null)
```

**Bài tập thực hành Ch3:**

1. **Đọc `ToyCombine.cpp` + `ToyCombine.td`** — hiểu 2 cách viết pattern
2. **Tìm hiểu `getCanonicalizationPatterns` được gọi khi nào** — trace trong MLIR source: `Canonicalizer` pass → `GreedyPatternRewriteDriver` → apply patterns
3. **Viết chương trình Toy**: tạo test case có cả `transpose(transpose(x))` và reshape dư → verify cả 2 pattern đều fire

> 📝 Output: [`notes/ch3_notes.md`](./notes/ch3_notes.md)

---

## Ngày 3: Ch4 — Interfaces (Inlining & Shape Inference)

> **Mục tiêu:** Hiểu tại sao interface > hardcode — và cách MLIR cho phép generic transformations hoạt động trên nhiều dialects.
>
> ⏱ ~3h
> 📚 Đọc: [Ch4](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)
> 📁 Code: `mlir/examples/toy/Ch4/`

### 4.1. Vấn đề: generic transformation cần giao diện chung

Sau Ch3, pipeline là: parse → MLIR → canonicalize. Nhưng 2 vấn đề:

**1) Inlining**: `multiply_transpose()` được gọi 2 lần → nếu inline, compiler thấy toàn bộ computation → optimize tốt hơn.

**2) Shape inference**: Tham số hàm có type `tensor<*xf64>` (unranked — shape không biết). Nhưng tại call site, ta truyền `tensor<2x3xf64>` → shape suy luận được → **cần propagate shape thông tin qua call graph**.

**Tại sao cần interface?** Inlining pass của MLIR là **generic** — nó hoạt động cho mọi dialect, không chỉ Toy. Để nó biết "op nào gọi hàm? op nào là hàm? op nào hợp lệ để inline?", mỗi dialect phải implement **interface** — hợp đồng giữa pass và dialect.

### 4.2. Inlining — DialectInlinerInterface

```cpp
// Toy dialect phải implement DialectInlinerInterface
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // "Có hợp lệ inline op này vào caller không?"
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;  // Mọi op Toy đều inline được
  }

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  // Type mismatch handling — khi caller có tensor<2x3xf64>
  // nhưng callee expect tensor<*xf64> → cần cast
  Operation *materializeCallConversion(
      OpBuilder &builder, Value input, Type resultType,
      Location loc) const final {
    return builder.create<CastOp>(loc, resultType, input);
  }
};
```

Thêm vào dialect initialization:
```cpp
void ToyDialect::initialize() {
  addInterfaces<ToyInlinerInterface>();
  // ...
}
```

**Và ops cần implement CallOpInterface / CallableOpInterface:**

```tablegen
// generic_call là "call op" — inliner biết nó gọi hàm
def GenericCallOp : Toy_Op<"generic_call", [/*...*/]> {
  // DeclareOpInterfaceMethods cho CallOpInterface
}

// toy.func là "callable" — inliner biết nó là hàm có thể inline
def FuncOp : Toy_Op<"func", [/*...*/]> {
  // DeclareOpInterfaceMethods cho CallableOpInterface
}
```

### 4.3. Shape Inference — Custom OpInterface

Inlining xong, tất cả code nằm trong `main()`. Nhưng vẫn còn `tensor<*xf64>` (unranked) — cần suy luận shape.

**Define interface bằng ODS:**

```tablegen
def ShapeInference : OpInterface<"ShapeInference"> {
  let description = [{
    Interface for ops that can infer their output shapes.
  }];
  let methods = [
    InterfaceMethod<
      "Infer and set the output shape for the current operation.",
      "void", "inferShapes"
    >
  ];
}
```

**Mỗi Toy op implement `inferShapes()`:**

```cpp
// TransposeOp: output shape = reversed input shape
void TransposeOp::inferShapes() {
  auto inputType = llvm::cast<RankedTensorType>(getOperand().getType());
  SmallVector<int64_t, 2> dims(llvm::reverse(inputType.getShape()));
  getResult().setType(RankedTensorType::get(dims, inputType.getElementType()));
}

// MulOp: output shape = input shape (element-wise)
void MulOp::inferShapes() {
  getResult().setType(getOperand(0).getType());
}
```

### 4.4. ShapeInferencePass — Worklist algorithm

```cpp
// Pass chạy shape inference bằng worklist algorithm:
// 1. Thu thập mọi op có ShapeInference interface
// 2. Nếu input shapes đã biết → gọi inferShapes()
// 3. Lặp lại đến khi mọi shape đều đã biết (fixpoint)

struct ShapeInferencePass : public mlir::PassWrapper<...> {
  void runOnOperation() override {
    auto f = getOperation();  // Đang xử lý function

    // Worklist: tất cả ops cần infer shape
    llvm::SmallPtrSet<Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // Fixpoint loop
    while (!opWorklist.empty()) {
      auto nextop = findReadyOp(opWorklist);  // op có input đã known
      if (!nextop) break;

      // Gọi interface method
      if (auto shapeOp = dyn_cast<ShapeInference>(nextop))
        shapeOp.inferShapes();

      opWorklist.erase(nextop);
    }

    // Verify: không còn dynamic shape nào
    if (!opWorklist.empty())
      signalPassFailure();
  }
};
```

### 4.5. Kết quả sau Ch4

**Trước inlining + shape inference:**
```mlir
toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
    -> tensor<*xf64> {
  %0 = toy.transpose(%arg0) to tensor<*xf64>       // shape không biết
  %1 = toy.transpose(%arg1) to tensor<*xf64>
  %2 = toy.mul %0, %1 : tensor<*xf64>
  toy.return %2 : tensor<*xf64>
}
```

**Sau inlining + shape inference + canonicalize:**
```mlir
toy.func @main() {
  %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
       : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>   // shape BIẾT!
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %1, %2 : tensor<3x2xf64>                          // shape BIẾT!
  toy.print %3 : tensor<3x2xf64>
}
```

**Thay đổi chính:**
- Functions inline → tất cả code trong `main()`
- `tensor<*xf64>` → `tensor<3x2xf64>` — shapes đã concrete
- Dead functions (multiply_transpose) bị DCE loại bỏ

### 4.6. Khái niệm quan trọng: OpInterface vs DialectInterface

| | `OpInterface` | `DialectInterface` |
|---|-------------|-------------------|
| Phạm vi | Per-op | Per-dialect |
| Ai implement | Từng op (TransposeOp, MulOp...) | Dialect (ToyDialect) |
| Ví dụ | `ShapeInference::inferShapes()` | `DialectInlinerInterface::isLegalToInline()` |
| Khi nào dùng | Behavior khác nhau per-op | Policy chung cho cả dialect |

### 4.7. Thực hành Ch4

```bash
ninja toyc-ch4

# Không optimize — thấy generic_call, unranked types
toyc-ch4 test/basic.toy -emit=mlir

# Có optimize — inline, shapes concrete
toyc-ch4 test/basic.toy -emit=mlir -opt

# So sánh trước/sau — quan sát inline + shape inference
diff <(toyc-ch4 test/basic.toy -emit=mlir 2>/dev/null) \
     <(toyc-ch4 test/basic.toy -emit=mlir -opt 2>/dev/null)
```

**Bài tập thực hành Ch4:**

1. **Trace ShapeInferencePass**: đọc code pass, vẽ ra trên giấy worklist algorithm chạy cho `basic.toy` — op nào infer trước, op nào sau
2. **Hiểu `OpInterface` vs `DialectInterface`**: viết note phân biệt — khi nào dùng cái nào
3. **Viết chương trình Toy**: tạo case có 3 functions gọi lồng nhau, dump MLIR trước/sau inlining, verify tất cả đều inline thành công

> 📝 Output: [`notes/ch4_notes.md`](./notes/ch4_notes.md)

---

## Ngày 4: Ch5 — Partial Lowering → Affine ⭐

> **⭐ CHƯƠNG QUAN TRỌNG NHẤT — dành nhiều thời gian nhất ở đây.**
>
> **Mục tiêu:** Hiểu Dialect Conversion Framework — cách lower từ Toy dialect xuống Affine/Arith/MemRef dialects.
>
> ⏱ ~3.5h
> 📚 Đọc: [Ch5](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)
> 📁 Code: `mlir/examples/toy/Ch5/`

### 5.1. Tại sao cần lowering?

Sau Ch4, IR vẫn ở mức Toy dialect — compiler không biết cách chạy `toy.mul` trên CPU. Cần lower xuống dialects mà MLIR biết cách compile tiếp:

```
toy.mul %a, %b : tensor<3x2xf64>
    ↓ lowering
affine.for %i = 0 to 3 {
  affine.for %j = 0 to 2 {
    %a_ij = affine.load %A[%i, %j] : memref<3x2xf64>
    %b_ij = affine.load %B[%i, %j] : memref<3x2xf64>
    %prod = arith.mulf %a_ij, %b_ij : f64
    affine.store %prod, %C[%i, %j] : memref<3x2xf64>
  }
}
```

**Tại sao "partial"?** Vì `toy.print` **không lower** ở bước này — nó cần logic riêng (gọi `printf` qua LLVM), sẽ xử lý ở Ch6. Các op khác lower hết.

### 5.2. Dialect Conversion Framework — 3 thành phần chính

#### Thành phần 1: ConversionTarget — "op nào hợp lệ?"

```cpp
mlir::ConversionTarget target(getContext());

// Affine, Arith, Func, MemRef → tất cả ops trong các dialect này đều legal
target.addLegalDialect<mlir::affine::AffineDialect,
                       mlir::arith::ArithDialect,
                       mlir::func::FuncDialect,
                       mlir::memref::MemRefDialect>();

// Toy dialect → TẤT CẢ ops illegal (phải convert hết)
target.addIllegalDialect<ToyDialect>();

// NGOẠI TRỪ toy.print — legal NẾU operand đã là memref
target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
  return llvm::none_of(op->getOperandTypes(),
    [](Type type) { return llvm::isa<TensorType>(type); });
});
```

**Giải thích `addDynamicallyLegalOp`:**
- `toy.print %x : tensor<3x2xf64>` → **illegal** (operand vẫn là tensor)
- `toy.print %x : memref<3x2xf64>` → **legal** (operand đã convert thành memref)
- Framework sẽ convert operand trước, rồi update `toy.print` để nhận memref

#### Thành phần 2: TypeConverter — "tensor → memref"

```cpp
ToyTypeConverter typeConverter;
typeConverter.addConversion([](TensorType type) -> Type {
  return MemRefType::get(type.getShape(), type.getElementType());
});
// tensor<3x2xf64> → memref<3x2xf64>
```

**Đây là ranh giới functional → imperative:**
- `tensor` = immutable value, no side effects → dễ optimize
- `memref` = mutable buffer, có địa chỉ → sát hardware
- **Thời điểm convert tensor → memref = thời điểm compiler quyết định memory allocation**

> 💡 **Liên hệ Stage 1:** Systolic array (tuần 3) có scratchpad SRAM — software phải tự quản lý. Chuyển từ tensor → memref chính là bước compiler bắt đầu "nghĩ về memory" — ở bước nào alloc, ở bước nào dealloc, có reuse được không.

#### Thành phần 3: ConversionPattern — "lower op cụ thể thế nào?"

Ví dụ lower `toy.mul`:

```cpp
struct MulOpLowering : public ConversionPattern {
  MulOpLowering(MLIRContext *ctx)
      : ConversionPattern(toy::MulOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    auto tensorType = llvm::cast<RankedTensorType>(
        (*op->result_type_begin()));

    // Alloc output memref
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = rewriter.create<memref::AllocOp>(loc, memRefType);

    // Build nested affine.for loops
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    SmallVector<int64_t, 4> upperBounds(tensorType.getShape());
    SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);

    buildAffineLoopNest(
        rewriter, loc, lowerBounds, upperBounds, steps,
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
          // Load a[i,j] and b[i,j]
          auto loadedLhs = nestedBuilder.create<affine::AffineLoadOp>(
              loc, operands[0], ivs);
          auto loadedRhs = nestedBuilder.create<affine::AffineLoadOp>(
              loc, operands[1], ivs);
          // Multiply element-wise
          auto mulResult = nestedBuilder.create<arith::MulFOp>(
              loc, loadedLhs, loadedRhs);
          // Store result[i,j]
          nestedBuilder.create<affine::AffineStoreOp>(
              loc, mulResult, alloc, ivs);
        });

    rewriter.replaceOp(op, alloc);
    return success();
  }
};
```

**Quan sát:**
- Input: `toy.mul %a, %b : tensor<3x2xf64>` (1 op)
- Output: `memref.alloc` + 2 `affine.for` lồng nhau + `affine.load` × 2 + `arith.mulf` + `affine.store` (nhiều ops!)
- Đây là **lowering = mở rộng trừu tượng**: 1 op cao → nhiều op thấp hơn
- `operands[0]` đã là `memref<3x2xf64>` — TypeConverter đã convert từ tensor

#### Lowering cho các ops khác

| Toy op | Lower thành | Ghi chú |
|--------|------------|---------|
| `toy.constant` | `arith.constant` + `memref.alloc` + `affine.store` | Constant → alloc buffer → store values |
| `toy.add` | `affine.for` loops + `arith.addf` | Giống mul nhưng dùng addf |
| `toy.mul` | `affine.for` loops + `arith.mulf` | Element-wise multiply |
| `toy.transpose` | `affine.for` loops + swapped indices | `load [i,j]` → `store [j,i]` |
| `toy.return` | `func.return` | Đơn giản replace |
| `toy.func` | `func.func` | Đơn giản replace + type convert |
| `toy.print` | **Giữ nguyên** (partial!) | Operand type convert: tensor → memref |

### 5.3. Chạy conversion

```cpp
void ToyToAffineLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  // ... setup target + type converter (ở trên)

  RewritePatternSet patterns(&getContext());
  patterns.add<AddOpLowering, ConstantOpLowering, MulOpLowering,
               TransposeOpLowering, PrintOpLowering, ReturnOpLowering,
               FuncOpLowering>(typeConverter, &getContext());

  // Partial conversion — cho phép toy.print tồn tại
  if (failed(applyPartialConversion(getOperation(), target,
                                     std::move(patterns))))
    signalPassFailure();
}
```

**`applyPartialConversion` vs `applyFullConversion`:**
- **Partial**: một số op legal ban đầu (toy.print) — được giữ lại
- **Full**: mọi op phải convert — nếu còn sót → fail

### 5.4. Tại sao lower xuống Affine dialect?

`affine.for` không phải loop bình thường — nó là **polyhedral loop**: bounds và access patterns là biểu thức affine. MLIR có thể:

- **Phân tích dependence chính xác** → biết loop nào có thể interchange, parallelize
- **Tiling tự động** → chia loop thành tiles vừa cache/scratchpad
- **Fusion** → gộp 2 loop nests liền kề nếu không có dependence conflict

**Nếu lower thẳng xuống `scf.for`** (generic loop) — mất khả năng phân tích này. **Đây là lý do progressive lowering tồn tại: optimize TRƯỚC khi mất thông tin.**

### 5.5. IR sau Ch5 lowering

```mlir
func.func @main() {
  // toy.constant → alloc + store
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  // ...
  %alloc = memref.alloc() : memref<2x3xf64>
  affine.store %cst, %alloc[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %alloc[0, 1] : memref<2x3xf64>
  // ...

  // toy.transpose → affine.for + swapped indices
  %alloc_1 = memref.alloc() : memref<3x2xf64>
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %val = affine.load %alloc[%j, %i] : memref<2x3xf64>
      affine.store %val, %alloc_1[%i, %j] : memref<3x2xf64>
    }
  }

  // toy.mul → affine.for + arith.mulf
  %alloc_2 = memref.alloc() : memref<3x2xf64>
  affine.for %i = 0 to 3 {
    affine.for %j = 0 to 2 {
      %a = affine.load %alloc_1[%i, %j] : memref<3x2xf64>
      %b = affine.load %alloc_1[%i, %j] : memref<3x2xf64>
      %prod = arith.mulf %a, %b : f64
      affine.store %prod, %alloc_2[%i, %j] : memref<3x2xf64>
    }
  }

  // toy.print — VẪN CÒN! (partial lowering)
  toy.print %alloc_2 : memref<3x2xf64>

  // Memory cleanup
  memref.dealloc %alloc_2 : memref<3x2xf64>
  memref.dealloc %alloc_1 : memref<3x2xf64>
  memref.dealloc %alloc : memref<2x3xf64>
}
```

**Quan sát quan trọng:**
1. **`toy.print` vẫn còn** — partial lowering cho phép mix dialects
2. **Memory management xuất hiện**: `memref.alloc`, `memref.dealloc`
3. **Đã có loops** — affine.for biết bounds → có thể phân tích tiling
4. **`tensor` → `memref`** — thế giới functional → imperative

### 5.6. Thực hành Ch5

```bash
ninja toyc-ch5

# Lowered output — quan sát affine loops!
toyc-ch5 test/basic.toy -emit=mlir-affine

# So sánh Toy dialect vs Affine
toyc-ch5 test/basic.toy -emit=mlir 2>/dev/null > /tmp/toy_before.mlir
toyc-ch5 test/basic.toy -emit=mlir-affine 2>/dev/null > /tmp/toy_after.mlir
diff /tmp/toy_before.mlir /tmp/toy_after.mlir

# Đặc biệt: dùng --mlir-print-ir-after-all để thấy từng bước
toyc-ch5 test/basic.toy -emit=mlir-affine \
  -mlir-print-ir-after-all 2>ch5_ir_trace.log
less ch5_ir_trace.log
```

**Bài tập thực hành Ch5 (QUAN TRỌNG):**

1. **Hiểu ConversionTarget**: trả lời — nếu không khai báo `addDynamicallyLegalOp` cho `toy.print`, chuyện gì xảy ra? (Thử bỏ đi, build, chạy → lỗi gì?)
2. **Trace `toy.mul` lowering**: vẽ trên giấy — input IR (1 op) → output IR (bao nhiêu ops, gì?). Đếm: lowering "expand" bao nhiêu lần?
3. **TypeConverter**: `tensor<3x2xf64>` convert thành gì? Layout memref mặc định là gì? (row-major hay column-major?)
4. **Đọc IR dump** từ `--mlir-print-ir-after-all`: pass nào chạy trước, pass nào sau? Ghi lại thứ tự vào note.

> 📝 Output: [`notes/ch5_notes.md`](./notes/ch5_notes.md) — **ghi chi tiết nhất**, đây là kiến thức nền tảng

---

## Ngày 5: Ch6 — Full Lowering → LLVM + JIT

> **Mục tiêu:** Lower toàn bộ IR (kể cả `toy.print`) xuống LLVM dialect → chạy thật bằng JIT.
>
> ⏱ ~2.5h
> 📚 Đọc: [Ch6](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)
> 📁 Code: `mlir/examples/toy/Ch6/`

### 6.1. Hoàn thành lowering — `toy.print` cuối cùng phải biến mất

Ch5 để `toy.print` lại vì nó cần logic đặc biệt: in tensor ra console. Ch6 lower nó thành:

```
toy.print %memref : memref<3x2xf64>
    ↓ lowering
nested affine.for loops {
  %val = affine.load %memref[%i, %j]
  llvm.call @printf(%format_str, %val)  // gọi C printf
}
```

Cần declare `printf` trong LLVM dialect:
```cpp
// Declare printf nếu chưa có
auto printfRef = getOrInsertPrintf(rewriter, module);
// Create format string "%f " hoặc "\n"
auto formatStr = getOrCreateGlobalString(loc, builder, "format", "%f ");
```

### 6.2. Full Conversion Pipeline

```cpp
void ToyToLLVMLoweringPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVMDialect>();
  target.addLegalOp<ModuleOp>();  // Module không convert

  // Type converter: mọi type → LLVM types
  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());

  // 1. Lower toy.print → llvm.call @printf
  patterns.add<PrintOpLowering>(&getContext());

  // 2. Lower affine → scf → cf (built-in passes)
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowPatterns(patterns);

  // 3. Lower arith, memref, func → LLVM (built-in passes)
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Full conversion — KHÔNG ĐƯỢC CÒN op ngoài LLVM dialect
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```

**Quan sát:**
- Chỉ `PrintOpLowering` là tự viết — tất cả conversion khác dùng **built-in patterns** MLIR có sẵn!
- Thứ tự pipeline: `affine → scf → cf → arith/memref/func → llvm`
- `applyFullConversion` — mọi op phải legal (LLVM dialect only)

### 6.3. Chuỗi lowering hoàn chỉnh

```
Toy source
    │ [Lexer/Parser]
    ↓
AST
    │ [MLIRGen]
    ↓
Toy MLIR (toy dialect)             ← Ch2
    │ [Canonicalize]
    ↓
Optimized Toy MLIR                 ← Ch3
    │ [Inline + ShapeInference]
    ↓
Fully-shaped Toy MLIR              ← Ch4
    │ [ToyToAffineLowering]
    ↓
Affine + Arith + MemRef MLIR       ← Ch5 (partial — toy.print còn)
    │ [ToyToLLVMLowering]
    ↓
LLVM dialect MLIR                  ← Ch6
    │ [mlir-translate / JIT]
    ↓
LLVM IR → Machine code → Execute!
```

### 6.4. JIT Execution — ExecutionEngine

```cpp
// Trong toyc-ch6.cpp
auto engine = mlir::ExecutionEngine::create(module);
if (!engine) {
  llvm::errs() << "Failed to create ExecutionEngine\n";
  return;
}

// Chạy hàm main()!
auto result = engine->invoke("main");
```

`ExecutionEngine` là wrapper của LLVM ORC JIT:
1. MLIR LLVM dialect → `mlir-translate` → LLVM IR
2. LLVM IR → ORC JIT → compile to machine code
3. Execute in-process

### 6.5. Thực hành Ch6

```bash
ninja toyc-ch6

# JIT execute — IN RA KẾT QUẢ THẬT!
toyc-ch6 test/basic.toy -emit=jit

# Xem LLVM IR (trước JIT)
toyc-ch6 test/basic.toy -emit=llvm

# Xem full pipeline: Toy → Affine → LLVM
toyc-ch6 test/basic.toy -emit=mlir        # Toy MLIR (optimized)
toyc-ch6 test/basic.toy -emit=mlir-affine  # Affine MLIR
toyc-ch6 test/basic.toy -emit=llvm         # LLVM IR

# IR trace đầy đủ — QUAN TRỌNG
toyc-ch6 test/basic.toy -emit=jit \
  -mlir-print-ir-after-all 2>ch6_full_trace.log
```

**Bài tập thực hành Ch6:**

1. **Chạy end-to-end**: `toyc-ch6 test/basic.toy -emit=jit` — xác nhận output đúng
2. **Trace pipeline đầy đủ**: mở `ch6_full_trace.log`, đếm bao nhiêu pass, ghi lại tên từng pass theo thứ tự
3. **Đọc LLVM IR output**: so sánh LLVM IR do MLIR sinh ra với LLVM IR bạn đọc ở tuần 7 (clang output) — cấu trúc giống/khác gì?
4. **Hiểu ExecutionEngine**: nó wrap cái gì? flow là gì? (MLIR → LLVM IR → JIT → execute)

> 📝 Output: [`notes/ch6_notes.md`](./notes/ch6_notes.md)

---

## Ngày 6: Ch7 — Struct Types (Custom Type)

> **Mục tiêu:** Hiểu cách thêm custom type vào dialect — đọc nhanh, ít quan trọng hơn Ch5-6.
>
> ⏱ ~1h
> 📚 Đọc: [Ch7](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)
> 📁 Code: `mlir/examples/toy/Ch7/`

### 7.1. Mở rộng Toy language với struct

```python
# Toy với struct type
struct MyStruct {
  var a;
  var b;
}

def multiply_transpose(MyStruct value) {
  return transpose(value.a) * transpose(value.b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var c = MyStruct{a, b};
  var d = multiply_transpose(c);
  print(d);
}
```

### 7.2. Custom Type Definition

```tablegen
def Toy_StructType : Toy_Type<"Struct", "struct"> {
  let summary = "Toy struct type";
  let description = [{
    A struct type composed of named element types.
  }];
  let parameters = (ins
    ArrayRefParameter<"mlir::Type", "element types">:$elementTypes
  );
  let hasCustomAssemblyFormat = 1;
}
```

MLIR in ra: `!toy.struct<tensor<2x3xf64>, tensor<2x3xf64>>`

### 7.3. Ops mới cho struct

- `toy.struct_constant` — tạo struct value từ constant tensors
- `toy.struct_access` — truy cập member bằng index

```mlir
%0 = toy.struct_constant dense<[[1.0, 2.0, 3.0], ...]> : !toy.struct<...>
%1 = toy.struct_access %0[0] : !toy.struct<...> -> tensor<2x3xf64>
```

### 7.4. Cập nhật pipeline cho struct

- **Inliner**: type materialization cho struct cast
- **Shape inference**: propagate qua struct access
- **Lowering**: struct_constant → individual tensor constants; struct_access → extract trực tiếp

### 7.5. Thực hành Ch7

```bash
ninja toyc-ch7

# JIT execute với struct
toyc-ch7 test/struct.toy -emit=jit

# Quan sát MLIR với struct type
toyc-ch7 test/struct.toy -emit=mlir
```

**Bài tập thực hành Ch7 (đọc nhanh):**

1. Đọc hiểu custom type definition — note lại cách ODS define type
2. So sánh: thêm type mới vs thêm op mới — bước nào giống, bước nào khác?
3. Khi nào cần custom type? (Ví dụ thực tế: quantized tensor type, sparse tensor type, layout-annotated tensor)

> 📝 Output: [`notes/ch7_notes.md`](./notes/ch7_notes.md)

---

## Ngày 6-7: Bài tập mở rộng (BẮT BUỘC)

> **Đây là phần biến "làm theo tutorial" thành "HIỂU". Nếu chỉ follow tutorial mà không làm extension, bạn chưa thật sự hiểu.**

### Extension 1: Thêm op `toy.sub` (trừ element-wise)

> 📁 Output: [`extensions/toy_sub/`](./extensions/toy_sub/)
> ⏱ ~2h

**Yêu cầu: làm KHÔNG NHÌN GUIDE — tự hoàn thành mọi bước.**

Checklist:

- [ ] **ODS**: Thêm `SubOp` vào `Ops.td` — copy pattern từ `AddOp`, đổi tên
- [ ] **Parser**: Cập nhật parser/lexer recognize `-` operator → `toy.sub`
- [ ] **MLIRGen**: Cập nhật `BinaryExprAST` xử lý `-`
- [ ] **Verifier**: Type check cho SubOp (giống AddOp)
- [ ] **Lowering Ch5**: Pattern lower `toy.sub` → `affine.for` + `arith.subf`
- [ ] **Test end-to-end**: Viết chương trình Toy dùng `-`, chạy JIT ra kết quả đúng

```python
# Test program cho toy.sub
def main() {
  var a<2, 2> = [[5, 6], [7, 8]];
  var b<2, 2> = [[1, 2], [3, 4]];
  var c = a - b;   # [[4, 4], [4, 4]]
  print(c);
}
```

**Nếu bạn làm được mà không cần nhìn guide → bạn đã hiểu Ch1-6.**

### Extension 2: Canonicalization `x - x → zeros`

> 📁 Output: [`extensions/canon_x_minus_x/`](./extensions/canon_x_minus_x/)
> ⏱ ~1h

Viết `RewritePattern` cho SubOp:

```cpp
// Pattern: x - x → zeros (constant tensor toàn 0)
struct SimplifyXMinusX : public mlir::OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      SubOp op, mlir::PatternRewriter &rewriter) const override {
    // Kiểm tra: 2 operands có phải cùng 1 SSA value?
    if (op.getLhs() != op.getRhs())
      return failure();

    // Tạo constant tensor toàn 0
    auto resultType = op.getResult().getType();
    // ... tạo DenseElementsAttr toàn 0 ...
    auto zeroConst = rewriter.create<ConstantOp>(op.getLoc(), zeroAttr);
    rewriter.replaceOp(op, zeroConst);
    return success();
  }
};
```

Đăng ký trong `getCanonicalizationPatterns()` của SubOp. Verify:

```bash
# Viết test program
# def main() { var a<2,2> = ...; var b = a - a; print(b); }
# Chạy với --canonicalize → b phải thành constant 0
toyc-ch6 test/x_minus_x.toy -emit=mlir -opt
# Kiểm tra: thấy constant zeros thay vì sub op?
```

### Extension 3: Op Counter Pass

> 📁 Output: [`extensions/op_counter_pass/`](./extensions/op_counter_pass/)
> ⏱ ~1h

Viết pass in ra thống kê ops:

```cpp
struct OpCounterPass : public PassWrapper<OpCounterPass,
                                          OperationPass<ModuleOp>> {
  void runOnOperation() override {
    llvm::StringMap<unsigned> opCounts;

    // Walk mọi op trong module
    getOperation()->walk([&](Operation *op) {
      opCounts[op->getName().getStringRef()]++;
    });

    // In kết quả
    llvm::outs() << "=== Op Statistics ===\n";
    for (auto &entry : opCounts) {
      llvm::outs() << entry.first() << ": " << entry.second << "\n";
    }
  }

  StringRef getArgument() const final { return "toy-op-counter"; }
  StringRef getDescription() const final { return "Count ops in module"; }
};
```

**Mục tiêu học:** cách đăng ký pass, cách dùng `walk()`, cách tương tác với pass manager.

### Extension 4: Lowering Trace (BẮT BUỘC)

> 📁 Output: [`lowering_trace.md`](./lowering_trace.md)
> ⏱ ~1.5h

Chạy 1 chương trình Toy với `--mlir-print-ir-after-all`, chụp IR sau **từng pass**, viết giải thích:

```bash
toyc-ch6 test/basic.toy -emit=jit -mlir-print-ir-after-all 2>full_trace.log
```

Trong `lowering_trace.md`, ghi lại:

```markdown
## Pass 1: Inline
**IR trước:** (copy 10 dòng quan trọng nhất)
**IR sau:** (copy 10 dòng quan trọng nhất)
**Gì thay đổi:** Functions inline vào main, generic_call biến mất

## Pass 2: ShapeInference
**IR trước:** tensor<*xf64> (unranked)
**IR sau:** tensor<3x2xf64> (concrete shapes)
**Gì thay đổi:** Mọi type bây giờ có shape cụ thể

## Pass 3: Canonicalize
...

## Pass 4: ToyToAffine lowering
...

## Pass 5: ToyToLLVM lowering
...
```

Đây là kỹ năng nghề nghiệp **quan trọng nhất**: nhìn IR trước/sau pass, giải thích biến đổi.

---

## Liên hệ HW-SW

> 🔑 Phần này kết nối kiến thức tuần 9 với hardware đã học ở stage 1.

### Toy tutorial = mô hình thu nhỏ của compiler thật

| Toy tutorial (tuần 9) | Production compiler | Ràng buộc HW (stage 1) |
|----------------------|--------------------|-----------------------|
| Ch2: Toy dialect | StableHLO, TOSA, Linalg | Giữ semantic cao nhất có thể |
| Ch3: Canonicalize | XLA algebraic simplifier | Tiết kiệm compute + memory |
| Ch4: Shape inference | Type inference + analysis | Static scheduling cho NPU/TPU (tuần 3-4) |
| Ch5: Partial lowering | Linalg → Affine/Loops | Tiling, memory planning cho scratchpad |
| Ch6: Full lowering + JIT | → LLVM/PTX → execute | ISA-specific codegen |
| Ch5 TypeConverter (tensor→memref) | Bufferization | Scratchpad software-managed (tuần 3) |

### Progressive lowering — tại sao?

```
toy.mul        → thông tin: "đây là element-wise multiply"
                  Compiler CÓ THỂ: fuse với op trước/sau
                  Compiler CÓ THỂ: chọn tiling strategy
                  Compiler CÓ THỂ: chọn hardware accelerator

affine.for     → thông tin: "đây là loop lồng, bounds affine"
                  Compiler CÓ THỂ: interchange, tile, parallelize
                  Compiler KHÔNG THỂ: biết đây là multiply nữa

llvm.fmul      → thông tin: "đây là 1 phép nhân scalar"
                  Compiler KHÔNG THỂ: tile (đã là scalar)
                  Compiler KHÔNG THỂ: fuse (chỉ thấy instruction)
```

**Thông tin bị MẤT VĨNH VIỄN qua mỗi bước lowering. Phải optimize TRƯỚC khi lower.**

Trong capstone tuần 14-16, bạn sẽ xây compiler tương tự nhưng thay tầng cuối bằng **instruction cho systolic simulator** (tuần 3 stage 1). Pipeline sẽ giống hệt:

```
PyTorch model → Graph IR → fuse/tile → Tensor IR → ISA cho systolic array
                    ↑           ↑           ↑              ↑
                  Ch2         Ch3-4       Ch5            Ch6
```

### Câu hỏi suy ngẫm cuối tuần

1. **Ch5 partial lowering giữ `toy.print` lại — trong compiler thật, op nào tương tự?** (Gợi ý: runtime calls, debug ops, profiling hooks)

2. **Tại sao lower thành `affine.for` thay vì `scf.for`?** (Đáp án: affine cho phép polyhedral analysis → tiling chính xác. scf.for là generic → mất khả năng phân tích bounds. Đối với TPU/NPU cần static scheduling → affine bắt buộc.)

3. **Nếu thay systolic array 16×16 bằng 64×64, pass nào trong pipeline phải thay đổi?** (Đáp án: tiling pass — tile size phải match array size. Phần còn lại pipeline giữ nguyên. Đây là sức mạnh progressive lowering: thay 1 pass, không sửa toàn bộ.)

4. **ShapeInferencePass (Ch4) tương đương với gì trong XLA?** (Đáp án: HLO shape inference — XLA yêu cầu shapes tĩnh vì TPU cần static scheduling.)

---

## TODO Checklist

### Chuẩn bị
- [ ] LLVM/MLIR build với `LLVM_BUILD_EXAMPLES=ON`, `toyc-ch*` binaries có mặt
- [ ] Tạo file test programs (`basic.toy`, `nested_transpose.toy`, ...)

### Ch1-2 — AST & Toy Dialect (Ngày 1, ~3h)
- [ ] Build và chạy `toyc-ch1 test.toy -emit=ast`
- [ ] Build và chạy `toyc-ch2 test.toy -emit=mlir`
- [ ] Đọc kỹ `Ops.td` — hiểu ODS: arguments, results, builders, traits
- [ ] Gõ lại (không copy) 2 op trong ODS (ConstantOp, TransposeOp), build lại thành công
- [ ] Kiểm tra: TableGen sinh file `.inc` gì trong build dir
- [ ] Đọc `MLIRGen.cpp` — hiểu AST traversal → MLIR ops
- [ ] Hiểu ConstantOp verifier — tại sao cần verify
- [ ] Viết [`notes/ch1_ch2_notes.md`](./notes/ch1_ch2_notes.md)

### Ch3 — Canonicalization (Ngày 2, ~2.5h)
- [ ] Chạy `toyc-ch3 nested_transpose.toy -emit=mlir` (không opt) vs `-opt` — diff
- [ ] Đọc `ToyCombine.cpp` — C++ RewritePattern cho transpose(transpose)
- [ ] Đọc `ToyCombine.td` — DRR pattern cho reshape
- [ ] Hiểu: `getCanonicalizationPatterns` được gọi khi nào, bởi pass nào
- [ ] Hiểu: `hasCanonicalizer = 1` trong ODS kích hoạt gì
- [ ] Viết [`notes/ch3_notes.md`](./notes/ch3_notes.md)

### Ch4 — Interfaces (Ngày 3, ~3h)
- [ ] Hiểu vì sao inlining + shape inference cần interface thay vì hardcode
- [ ] Đọc `ToyInlinerInterface` — `isLegalToInline`, `materializeCallConversion`
- [ ] Đọc `CallOpInterface` / `CallableOpInterface` trên `GenericCallOp` / `FuncOp`
- [ ] Trace ShapeInferencePass: worklist algorithm chạy thế nào, fixpoint khi nào
- [ ] So sánh trước/sau inlining: `toyc-ch4 test.toy -emit=mlir` vs `-opt`
- [ ] Hiểu `OpInterface` vs `DialectInterface`
- [ ] Viết [`notes/ch4_notes.md`](./notes/ch4_notes.md)

### Ch5 — Partial Lowering (Ngày 4, ~3.5h) ⭐
- [ ] Hiểu `ConversionTarget`: legal/illegal ops, `addDynamicallyLegalOp`
- [ ] Hiểu `TypeConverter`: tensor → memref conversion
- [ ] Trace lowering `toy.mul` → `affine.for` + `arith.mulf` — đếm ops expand
- [ ] Trace lowering `toy.transpose` → `affine.for` + swapped indices
- [ ] Hiểu vì sao `toy.print` giữ lại (partial conversion)
- [ ] Hiểu `applyPartialConversion` vs `applyFullConversion`
- [ ] Chạy `toyc-ch5 -emit=mlir-affine` + dump IR trace
- [ ] Viết [`notes/ch5_notes.md`](./notes/ch5_notes.md) — **chi tiết nhất**

### Ch6 — Full Lowering + JIT (Ngày 5, ~2.5h)
- [ ] Chạy `toyc-ch6 test.toy -emit=jit` — verify output đúng
- [ ] Xem `toyc-ch6 test.toy -emit=llvm` — đọc LLVM IR output
- [ ] Hiểu full conversion pipeline: affine → scf → cf → arith/memref/func → llvm
- [ ] Hiểu `ExecutionEngine` — LLVM ORC JIT wrapper
- [ ] Trace full pipeline với `--mlir-print-ir-after-all`
- [ ] Viết [`notes/ch6_notes.md`](./notes/ch6_notes.md)

### Ch7 — Struct Types (Ngày 6, ~1h)
- [ ] Đọc hiểu custom type definition trong ODS
- [ ] Chạy `toyc-ch7 test/struct.toy -emit=jit`
- [ ] Note: khi nào cần custom type trong compiler thật
- [ ] Viết [`notes/ch7_notes.md`](./notes/ch7_notes.md)

### Bài tập mở rộng (Ngày 6-7, ~5.5h) — BẮT BUỘC
- [ ] **`toy.sub`**: ODS + parser + MLIRGen + verifier + lowering + test end-to-end
- [ ] **Canonicalization `x - x → zeros`**: RewritePattern + verify bằng `--canonicalize`
- [ ] **Op counter pass**: implement + đăng ký + chạy qua `-toy-op-counter`
- [ ] **Lowering trace**: `--mlir-print-ir-after-all` → viết [`lowering_trace.md`](./lowering_trace.md) giải thích từng pass

---

## Output cuối tuần

- [ ] Toy compiler cả 7 chương build & chạy (`toyc-ch1` đến `toyc-ch7`)
- [ ] Notes cho từng chương (`notes/ch1_ch2_notes.md` → `ch7_notes.md`)
- [ ] 3 extension tự viết hoạt động (`toy.sub`, `x-x→zeros`, op counter)
- [ ] [`lowering_trace.md`](./lowering_trace.md) — IR walkthrough qua từng pass
- [ ] (Optional) Blog post tuần 9

---

## Tổng kết: Compiler Toy = template cho mọi compiler MLIR

Sau tuần 9, bạn đã thấy **template pattern** mà mọi compiler MLIR-based đều theo:

```
1. DEFINE DIALECT          — ops, types, attributes cho domain
2. FRONTEND                — source/model → dialect IR
3. HIGH-LEVEL TRANSFORMS   — canonicalize, fold, simplify
4. INTERFACES              — generic passes (inline, infer types)
5. LOWERING                — dialect conversion → lower-level dialects
6. CODEGEN                 — → LLVM / target ISA → execute
```

Tuần 10, bạn sẽ dùng dialects có sẵn (`linalg`, `tensor`, `memref`, `affine`) — không cần define dialect riêng, nhưng **transform pipeline giống y hệt**: tiling pass, bufferization, lower linalg → loops → LLVM.

Tuần 14-16 (capstone), bạn sẽ xây pipeline này cho systolic simulator — template giống hệt, chỉ target backend khác.

---

## Tài liệu tham khảo

### Tutorial chính thức (bắt buộc)
- [MLIR Toy Tutorial Overview](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [Ch1 — Toy Language and AST](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/)
- [Ch2 — Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
- [Ch3 — High-level Transformations](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/)
- [Ch4 — Interfaces](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)
- [Ch5 — Partial Lowering](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/)
- [Ch6 — Lowering to LLVM](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/)
- [Ch7 — Struct Types](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/)

### MLIR Docs tham chiếu
- [ODS — Operation Definition Specification](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [Interfaces](https://mlir.llvm.org/docs/Interfaces/)
- [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [TableGen Overview](https://mlir.llvm.org/docs/DefiningDialects/)

### Blog / External
- [Jeremy Kun — MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) — blog series bổ trợ tốt
- [MLIR — Getting Started](https://mlir.llvm.org/getting_started/) — build guide chính thức

### Video
- 🎥 [MLIR Tutorial — Mehdi Amini](https://www.youtube.com/results?search_query=MLIR+tutorial+Mehdi+Amini) (LLVM Dev Meeting)
- 🎥 [MLIR Toy Tutorial Walkthrough](https://www.youtube.com/results?search_query=MLIR+Toy+tutorial+walkthrough)

### Source code
- 📁 `$LLVM_SRC/mlir/examples/toy/` — Toy tutorial source code
- 📁 `$LLVM_SRC/mlir/test/Examples/Toy/` — FileCheck tests cho Toy

### Cộng đồng
- 💬 [LLVM Discourse — MLIR](https://discourse.llvm.org/c/mlir/) — hỏi đáp
- 💬 [LLVM Discord](https://discord.gg/xS7Z362) — channel #mlir

---

*Tuần trước: [Tuần 8 — Giới thiệu MLIR](../week8-mlir-intro/) · Tuần sau: [Tuần 10 — ML Dialects](../week10-ml-dialects/)*
