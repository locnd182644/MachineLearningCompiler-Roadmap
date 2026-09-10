# Notes: Đọc source code ArithOps (MLIR)

> Bài tập 8.3 — Hiểu cách MLIR define operations trong thực tế
> 
> Files đọc:
> - `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` — Implementation
> - `mlir/include/mlir/Dialect/Arith/IR/ArithOps.td` — ODS Definition

---

## 1. Tổng quan cấu trúc Arith Dialect

Arith dialect chứa các operations số học cơ bản: cộng, trừ, nhân, chia, so sánh, casting, constants. Đây là dialect **nền tảng** — hầu hết mọi MLIR program đều dùng.

### Kiến trúc files

```
mlir/include/mlir/Dialect/Arith/IR/
├── ArithOps.td          # ODS definitions (TableGen) — khai báo ops
├── ArithBase.td         # Base classes, enums (CmpIPredicate, etc.)
└── ArithOps.h           # Generated C++ header (sinh từ .td)

mlir/lib/Dialect/Arith/IR/
├── ArithOps.cpp         # Implementation — fold, canonicalize, verify
├── ArithDialect.cpp     # Dialect registration
└── CMakeLists.txt       # Build rules
```

**Key insight:** MLIR tách khai báo (`.td`) và implementation (`.cpp`). TableGen sinh boilerplate C++ code, developer chỉ viết logic thực sự (fold, canonicalize, verify).

---

## 2. ODS Definition (ArithOps.td)

ODS = Operation Definition Specification — DSL để khai báo ops.

### Ví dụ: arith.addi

```tablegen
// Đơn giản hóa từ source thật
def Arith_AddIOp : Arith_IntBinaryOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    The `addi` operation takes two integer operands and returns their sum.
    
    Example:
    ```mlir
    %result = arith.addi %a, %b : i32
    ```
  }];
  
  let hasFolder = 1;           // Có fold function → constant folding
  let hasCanonicalizer = 1;    // Có canonicalization patterns
}
```

### Breakdown cấu trúc ODS

| Thành phần | Ý nghĩa | Ví dụ |
|-----------|---------|-------|
| `Arith_IntBinaryOp` | Base class — 2 operands cùng kiểu integer, 1 result | Tái dùng cho addi, subi, muli, etc. |
| `"addi"` | Tên op (sẽ thành `arith.addi`) | |
| `[Commutative]` | Traits — op có tính chất gì | Commutative → `a + b = b + a` |
| `hasFolder = 1` | Khai báo op có fold function | Cho phép constant folding |
| `hasCanonicalizer = 1` | Khai báo op có canonicalization patterns | Simplification rules |

### TableGen sinh gì?

Từ khai báo trên, TableGen tự sinh:

1. **C++ class** `arith::AddIOp` với:
   - Builder methods: `AddIOp::build(builder, result, lhs, rhs)`
   - Accessor methods: `getResult()`, `getLhs()`, `getRhs()`
   - Type checking
   
2. **Parser/Printer**: custom assembly format `arith.addi %a, %b : i32`

3. **Verifier skeleton**: kiểm tra types match, operands hợp lệ

4. **Interface implementations**: `Commutative` trait cho phép optimizer biết phép cộng giao hoán

---

## 3. Fold Functions (ArithOps.cpp)

### Fold = Constant Folding ở mức MLIR

Fold function nhận operands (có thể là constants), trả về kết quả đã fold nếu có thể. MLIR tự gọi fold trong canonicalization.

### Ví dụ: arith.addi fold

```cpp
// Đơn giản hóa từ source thật
OpFoldResult arith::AddIOp::fold(FoldAdaptor adaptor) {
  // addi(x, 0) → x
  if (matchPattern(getRhs(), m_Zero()))
    return getLhs();
  
  // addi(const_a, const_b) → const(a + b)  
  return constFoldBinaryOp<IntegerAttr>(
    adaptor.getOperands(),
    [](APInt a, const APInt &b) { return std::move(a) + b; }
  );
}
```

**Giải thích:**
1. Kiểm tra special case: nếu cộng với 0, trả luôn operand kia → xóa op
2. Nếu cả 2 operands là constants (IntegerAttr), tính kết quả tại compile time
3. Nếu không fold được, trả `nullptr` → giữ op nguyên

### Ví dụ khác: arith.muli fold

```cpp
OpFoldResult arith::MulIOp::fold(FoldAdaptor adaptor) {
  // muli(x, 0) → 0
  if (matchPattern(getRhs(), m_Zero()))
    return getRhs();
  
  // muli(x, 1) → x
  if (matchPattern(getRhs(), m_One()))
    return getLhs();
  
  // muli(const_a, const_b) → const(a * b)
  return constFoldBinaryOp<IntegerAttr>(
    adaptor.getOperands(),
    [](APInt a, const APInt &b) { return std::move(a) * b; }
  );
}
```

### So sánh với tuần 7

| Tuần 7 (Toy calculator) | MLIR Arith |
|--------------------------|------------|
| `if isinstance(instr, BinOp) and all_const:` | `constFoldBinaryOp<IntegerAttr>(...)` |
| `result = eval(op, val1, val2)` | Lambda `[](APInt a, APInt b) { return a + b; }` |
| Hardcoded cho mỗi op | Generic framework, mỗi op chỉ cung cấp lambda |
| Chạy thủ công trong pass loop | MLIR tự gọi fold trong canonicalization |

**Key insight:** Cùng nguyên lý (fold constants tại compile time), nhưng MLIR infrastructure làm nó scalable cho hàng trăm ops.

---

## 4. Canonicalization Patterns

### Pattern = RewritePattern

Canonicalization đi xa hơn fold: thay đổi CẤU TRÚC IR, không chỉ fold constants.

### Ví dụ: arith.addi canonicalization

```cpp
// Đơn giản hóa
void arith::AddIOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  // addi(subi(a, b), b) → a
  patterns.add<AddSubSimplify>(context);
  
  // addi(x, const) → addi(const, x) nếu cần normalize
  // (sắp xếp constants sang phải cho canonical form)
}
```

### Cấu trúc RewritePattern

```cpp
class AddSubSimplify : public OpRewritePattern<arith::AddIOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  
  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // 1. Match: kiểm tra pattern
    auto subOp = op.getLhs().getDefiningOp<arith::SubIOp>();
    if (!subOp || subOp.getRhs() != op.getRhs())
      return failure();  // Không match → bỏ qua
    
    // 2. Rewrite: thay thế
    rewriter.replaceOp(op, subOp.getLhs());
    return success();
  }
};
```

**Giải thích pattern `(a - b) + b → a`:**
1. **Match**: Kiểm tra operand trái của `addi` có phải `subi`, và operand phải của `addi` có giống operand phải của `subi` không
2. **Rewrite**: Nếu match, thay toàn bộ `addi` bằng operand trái của `subi` (tức `a`)

### So sánh với tuần 7

Tuần 7 bạn viết DCE = "xóa instruction không ai dùng". Canonicalization rộng hơn:
- DCE: xóa dead code
- Constant folding: fold constants
- Algebraic simplification: `x + 0 → x`, `x * 1 → x`
- Pattern rewriting: `(a - b) + b → a`

Tất cả đều là **pass thay đổi IR** — khái niệm giống hệt tuần 7, infrastructure mạnh hơn gấp bội.

---

## 5. Verifier

Mỗi op có thể có verify function kiểm tra invariants:

```cpp
LogicalResult arith::AddIOp::verify() {
  // Type checking tự động (ODS sinh) — chỉ cần verify bổ sung
  // Ví dụ: kiểm tra operands cùng type, result type match
  
  // Với arith binary ops: ODS đã handle phần lớn verification
  // chỉ cần custom verify cho trường hợp đặc biệt
  return success();
}
```

**Verify chạy khi nào?**
- Sau mỗi parse (đọc file `.mlir`)
- Sau mỗi pass (kiểm tra pass không phá invariants)
- Khi explicitly gọi `mlir-opt --verify-diagnostics`

---

## 6. Key Takeaways

### Cách MLIR define 1 op (workflow)

```
1. Khai báo trong .td (ODS)
   └── TableGen sinh: C++ class, parser, printer, builder, verifier skeleton

2. Implement trong .cpp
   ├── fold(): constant folding + algebraic identity
   ├── getCanonicalizationPatterns(): structural rewrites
   └── verify(): custom invariant checks (nếu cần)

3. Register trong dialect .cpp
   └── Dialect::initialize() thêm op vào registry
```

### Nguyên tắc thiết kế

1. **Separation of concerns**: khai báo (.td) tách implementation (.cpp) tách registration
2. **Code generation**: TableGen sinh boilerplate → developer focus logic
3. **Generic infrastructure**: `constFoldBinaryOp` tái dùng cho mọi binary op
4. **Composable patterns**: mỗi canonicalization pattern độc lập, compose bằng PatternSet
5. **Verify-by-construction**: ODS tự sinh type checking → ít bug hơn

### Câu hỏi tự kiểm tra

- [ ] Fold function trả về gì khi KHÔNG fold được? → `nullptr` (hoặc `OpFoldResult()`)
- [ ] Tại sao `Commutative` trait quan trọng? → Optimizer biết `a + b = b + a`, có thể normalize
- [ ] Canonicalization khác fold ở đâu? → Fold chỉ fold constants/identity, canonicalization rewrite cấu trúc
- [ ] ODS tự sinh những gì? → Class, parser, printer, builder, verifier skeleton, trait impls
- [ ] Khi nào MLIR gọi fold? → Trong canonicalize pass, MLIR tự gọi fold cho mọi op

---

*Đọc source code MLIR là kỹ năng quan trọng — tuần 9 (Toy Tutorial) sẽ yêu cầu bạn ĐỌC + SỬA code MLIR C++. Nắm chắc cấu trúc ở đây sẽ giúp rất nhiều.*
