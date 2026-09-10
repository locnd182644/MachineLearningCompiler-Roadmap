# Tuần 8 — Giới thiệu MLIR + Build từ source

> **Câu hỏi central:** *Vì sao LLVM IR không đủ cho ML, đến mức phải xây cả một infrastructure mới (MLIR)?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-8--giới-thiệu-mlir--build-từ-source)
> 
> ⏱ Thời lượng: ~20h (8h lý thuyết + 12h thực hành)

---

## Mục lục

- [Tổng quan tuần 8](#tổng-quan-tuần-8)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Phần 1: Lý thuyết (8h)](#phần-1-lý-thuyết-8h)
  - [Ngày 1: MLIR Paper](#ngày-1-mlir-paper-lattner-et-al-2021)
  - [Ngày 2-3: Ngôn ngữ MLIR](#ngày-2-3-ngôn-ngữ-mlir--cấu-trúc-ir)
  - [Ngày 4-5: Dialect Ecosystem Tour](#ngày-4-5-dialect-ecosystem-tour)
- [Phần 2: Thực hành (12h)](#phần-2-thực-hành-12h)
  - [Bài tập 8.1: Build LLVM/MLIR](#bài-tập-81--build-llvmmlir-từ-source)
  - [Bài tập 8.2: mlir-opt hands-on](#bài-tập-82--mlir-opt-hands-on)
  - [Bài tập 8.3: Đọc code MLIR thật](#bài-tập-83--đọc-code-mlir-thật)
- [Liên hệ HW-SW](#liên-hệ-hw-sw)
- [TODO Checklist](#todo-checklist)
- [Output cuối tuần](#output-cuối-tuần)
- [Tài liệu tham khảo](#tài-liệu-tham-khảo)

---

## Tổng quan tuần 8

Tuần này là **bước chuyển lớn**: từ compiler truyền thống (tuần 7) sang **compiler infrastructure hiện đại** cho AI/ML. Bạn sẽ:

1. **Hiểu TẠI SAO** MLIR được tạo ra — vấn đề gì của LLVM IR và các framework ML IRs mà MLIR giải quyết
2. **Nắm cấu trúc** cốt lõi của MLIR: Operation, Value, Type, Attribute, Region, Block, Dialect
3. **Build LLVM/MLIR** từ source — công cụ đầu tiên trong workshop
4. **Thực hành viết tay** file `.mlir` và dùng `mlir-opt` — kỹ năng nền tảng cho toàn bộ giai đoạn 2

```
Tuần 7 (đã làm):                    Tuần 8 (tuần này):
Toy calculator compiler              MLIR infrastructure
• Lexer → Parser → AST → IR         • Hiểu meta-IR framework
• Constant folding + DCE             • Dialects + progressive lowering
• Stack machine codegen              • mlir-opt + viết .mlir tay
```

### Kết nối với tuần trước

Tuần 7 bạn tự xây compiler nhỏ: lexer, parser, IR, 2 pass, codegen. **Mọi thứ bạn xây tay, MLIR đã có sẵn infrastructure** — và mạnh hơn gấp bội:

| Bạn xây (tuần 7) | MLIR có sẵn |
|-------------------|-------------|
| IR tuyến tính tự thiết kế | SSA-based IR với type system phong phú |
| Constant folding pass | Canonicalization framework (tổng quát hơn) |
| DCE pass | DCE + CSE + hàng chục pass built-in |
| IR printer | Printer/Parser tự sinh từ ODS |
| Test bằng pytest | FileCheck — test framework cho compiler |

---

## Cấu trúc thư mục

```
week8-mlir-intro/
├── README.md                    # File này — lesson plan chi tiết
├── build_notes.md               # Log quá trình build LLVM/MLIR + troubleshooting
├── paper_notes_mlir.md          # Note MLIR paper (Lattner et al. 2021, CGO)
├── mlir_anatomy_cheatsheet.md   # Quick reference: op/value/type/attr/region/block
├── handwritten_mlir/
│   ├── 01_arith_basic.mlir      # Arithmetic cơ bản (func + arith dialects)
│   ├── 02_scf_loop.mlir         # Structured control flow (scf dialect)
│   ├── 03_dot_product.mlir      # Dot product chạy được qua mlir-cpu-runner
│   └── run_pipelines.sh         # Các pass pipeline đã thử + output
└── arith_source_notes.md        # Note từ đọc ArithOps.cpp/ArithOps.td
```

---

## Phần 1: Lý thuyết (8h)

### Ngày 1: MLIR Paper (Lattner et al. 2021)

> 📄 Lattner et al. (2021), *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, CGO.
> Link: https://arxiv.org/abs/2002.11054
>
> **Paper quan trọng nhất giai đoạn này — đọc 2 lần.**

#### Lần đọc 1 — Big picture (~2h)

Đọc lướt toàn bộ paper, tập trung trả lời 3 câu hỏi:

1. **Vấn đề gì?** — Mỗi ML framework tự chế IR riêng (TF Graph, XLA HLO, TorchScript, Glow, nGraph, ONNX...), dẫn đến:
   - Trùng lặp infrastructure: mỗi IR phải tự viết parser, printer, verifier, pass manager, diagnostics, location tracking
   - Pass không tái dùng được: constant folding viết cho TF Graph không dùng được cho XLA HLO
   - Lowering giữa các tầng = full rewrite, đắt đỏ và mất thông tin

2. **Giải pháp gì?** — MLIR không phải "thêm 1 IR nữa" mà là **framework để xây IR**:
   - Dialect system: mỗi dialect = 1 namespace ops + types ở cùng mức trừu tượng
   - Nhiều dialect cùng tồn tại trong 1 module → progressive lowering
   - Dùng chung infrastructure: pass manager, verifier, rewrite framework

3. **Tại sao LLVM IR không đủ?** — LLVM IR chỉ có scalar + vector + pointer:
   - `matmul(A, B)` bị lower thành 3 loop lồng ngay lập tức → mất thông tin "đây là matmul"
   - Không có tensor type, không có structured ops → mọi high-level optimization bất khả thi
   - Fixed instruction set → không extensible cho domain-specific ops

#### Lần đọc 2 — Chi tiết kỹ thuật (~2h)

Đọc kỹ các sections:

- **Section 3 (IR Design)**: Operation, Region, Block — hiểu tại sao "everything is an op"
- **Section 4 (Dialect Ecosystem)**: cách dialect được define, register, interact
- **Section 5 (Evaluation)**: case studies thực tế
- **Block arguments vs phi nodes**: MLIR dùng block arguments thay phi nodes — thanh lịch hơn vì predecessor tự cung cấp argument, không cần op đặc biệt tham chiếu predecessor

> 📝 Output: [`paper_notes_mlir.md`](./paper_notes_mlir.md)

---

### Ngày 2-3: Ngôn ngữ MLIR — Cấu trúc IR

> 📚 Đọc:
> - [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
> - [Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)

#### Anatomy của 1 Operation

Operation là đơn vị cơ bản **vạn năng** trong MLIR. Mọi thứ đều là op:

```mlir
// Generic format — mọi op đều có thể viết dạng này
%result = "dialect.opname"(%operand1, %operand2) {attr = 42 : i32}
          : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>

// Custom format — cùng ngữ nghĩa, dễ đọc hơn
%result = dialect.opname %operand1, %operand2 {attr = 42 : i32}
          : tensor<4x8xf32>, tensor<8x16xf32>
```

Thành phần của operation:

| Thành phần | Ý nghĩa | Ví dụ |
|-----------|---------|-------|
| **Op name** | `dialect.opname` — dialect nào, op nào | `arith.addf`, `linalg.matmul` |
| **Operands** | SSA values đầu vào | `%a`, `%b` |
| **Results** | SSA values đầu ra | `%result` |
| **Attributes** | Hằng số compile-time gắn vào op | `{value = 42 : i32}` |
| **Regions** | Code lồng bên trong op | `func.func` chứa body |
| **Successors** | Block tiếp theo (cho terminators) | `cf.br ^bb1` |
| **Type signature** | Kiểu input → output | `(f32, f32) -> f32` |

#### 6 khái niệm cốt lõi phải nắm

1. **Operation** — Đơn vị vạn năng. Function là op (`func.func`), module là op (`builtin.module`), constant là op (`arith.constant`), loop cũng là op (`scf.for`).

2. **Value (SSA)** — `%x`. Mỗi value được gán đúng 1 lần. Hai nguồn: kết quả của op hoặc block argument. Bạn đã hiểu SSA từ tuần 7.

3. **Type** — Phong phú hơn LLVM:
   ```
   Scalars:  i1, i8, i32, i64, f16, bf16, f32, f64, index
   Shaped:   tensor<4x8xf32>     — immutable value (functional world)
             memref<4x8xf32>     — mutable buffer (imperative world)  
             vector<4xf32>       — SIMD-like fixed size
   ```

4. **Attribute** — Hằng số compile-time. `{value = 42 : i32}`, `dense<[[1.0, 2.0]]> : tensor<1x2xf32>`. Gắn vào op, không phải runtime value.

5. **Region & Block** — Op có thể chứa regions, region chứa blocks, block chứa ops → **cấu trúc cây lồng nhau**:
   ```
   Module (op)
     └── Region
          └── Block
               └── Function (op)
                    └── Region
                         ├── Block ^entry(args...)
                         │    ├── arith.addi (op)
                         │    └── cf.br ^next (terminator op)
                         └── Block ^next(args...)
                              └── return (terminator op)
   ```

6. **Dialect** — Namespace chứa ops + types + attributes cùng mức trừu tượng:
   - `linalg.matmul` — biết đây là matmul, tile/fuse được
   - `affine.for` — biết đây là loop affine, phân tích dependence được
   - `llvm.fadd` — sát machine code, sẵn sàng emit
   - **Nhiều dialect sống chung trong 1 module** — progressive lowering dần dần

#### Block Arguments thay Phi Nodes

Đây là design decision quan trọng, so sánh:

```
// LLVM IR dùng phi nodes:
bb1:
  %x = phi i32 [%a, %bb0], [%b, %bb2]
  // Phi phải biết predecessor nào → dễ sai khi transform

// MLIR dùng block arguments:
^bb1(%x: i32):           // Block nhận argument
  ...
cf.br ^bb1(%new_val: i32) // Predecessor cung cấp argument
// Không cần phi op → verify dễ hơn, transform an toàn hơn
```

> 📝 Output: [`mlir_anatomy_cheatsheet.md`](./mlir_anatomy_cheatsheet.md)

---

### Ngày 4-5: Dialect Ecosystem Tour

> 📚 Đọc: [mlir.llvm.org/docs/Dialects](https://mlir.llvm.org/docs/Dialects/)
> 🎥 Xem: Talk "MLIR Tutorial" — Mehdi Amini (LLVM Dev Meeting, YouTube)

#### Bản đồ các dialect theo mức trừu tượng

```
CAO (gần người)                         THẤP (gần máy)
────────────────────────────────────────────────────────►

 linalg          tensor       affine       scf        cf        llvm
 (structured     (value       (polyhedral  (for,      (br,      (LLVM IR
  ops: matmul,    semantics,   loops,       while,     cond_br)   mapping)
  conv, generic)  no address)  analyzable)  if)

                 memref                    arith      math
                 (buffers,                 (add, mul, (exp, sin,
                  addresses)               cmp)       log)
                                           
                 vector                    func       builtin
                 (SIMD-like,              (functions, (module,
                  fixed size)              calls)     types)
```

#### 10 dialect cần biết

| Dialect | Mức | Vai trò | Op chính | Khi nào gặp |
|---------|-----|---------|----------|-------------|
| `builtin` | Meta | Module container | `module` | Mọi file MLIR |
| `func` | Cao | Khai báo/gọi hàm | `func.func`, `func.call`, `func.return` | Mọi file MLIR |
| `arith` | Trung | Số học scalar | `addi`, `addf`, `muli`, `mulf`, `cmpi`, `constant` | Mọi nơi có tính toán |
| `math` | Trung | Hàm toán học | `exp`, `log`, `sin`, `sqrt` | Activation functions |
| `tensor` | Cao | Tensor (value) | `extract`, `insert`, `generate`, `empty` | Trước bufferization |
| `memref` | Trung-thấp | Memory buffers | `alloc`, `load`, `store`, `dealloc` | Sau bufferization |
| `linalg` | Cao | Structured ops | `matmul`, `conv_2d`, `generic`, `map` | ML core computation |
| `affine` | Trung | Polyhedral loops | `affine.for`, `affine.if`, `affine.load` | Sau lower linalg |
| `scf` | Trung | Control flow cấu trúc | `scf.for`, `scf.while`, `scf.if` | Loops tổng quát |
| `vector` | Trung-thấp | SIMD operations | `transfer_read`, `transfer_write`, `contract` | Vectorization |
| `cf` | Thấp | Branch/jump | `br`, `cond_br` | Sau lower scf |
| `llvm` | Thấp nhất | LLVM IR mapping | `llvm.add`, `llvm.call`, `llvm.getelementptr` | Bước cuối → codegen |

#### Progressive lowering path điển hình cho ML

```
linalg.matmul (tensor<M×K×f32>, tensor<K×N×f32>) → tensor<M×N×f32>
    │
    │ tiling (transform dialect hoặc pass)
    ↓
linalg.matmul trên tile nhỏ hơn (e.g., 32×32×32)
    │
    │ bufferization (tensor → memref)
    ↓
linalg.matmul trên memref<32×32×f32>
    │
    │ convert-linalg-to-loops
    ↓
scf.for / affine.for (loop nests) + memref.load/store
    │
    │ convert-scf-to-cf
    ↓
cf.br / cf.cond_br + block arguments
    │
    │ convert-*-to-llvm
    ↓
llvm dialect (llvm.add, llvm.load, llvm.br...)
    │
    │ mlir-translate → LLVM IR → llc
    ↓
Machine code (x86, ARM, GPU PTX...)
```

**Điểm mấu chốt:** Mỗi bước lowering chỉ commit MỘT PHẦN quyết định. `linalg.matmul` biết nó là matmul → tile được theo cấu trúc. Sau khi lower thành loop lồng → thông tin "đây là matmul" **mất vĩnh viễn**. **Phải optimize TRƯỚC khi mất thông tin.**

> 📝 Sau khi đọc xong, cập nhật vào [`mlir_anatomy_cheatsheet.md`](./mlir_anatomy_cheatsheet.md)

---

## Phần 2: Thực hành (12h)

### Bài tập 8.1 — Build LLVM/MLIR từ source

> ⏱ ~4h (chủ yếu chờ build)
> 📝 Chi tiết: [`build_notes.md`](./build_notes.md)

#### Các bước

```bash
# 1. Clone
git clone --depth 1 https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build

# 2. Configure (Release + Assertions — đủ cho học, nhẹ hơn Debug)
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_CCACHE_BUILD=ON \
  -DLLVM_USE_LINKER=lld

# 3. Build
ninja -j$(nproc)

# 4. Test
ninja check-mlir

# 5. Verify
export PATH=$PWD/bin:$PATH
mlir-opt --version
echo 'func.func @test() { return }' | mlir-opt
```

#### Checklist verify

- [ ] `mlir-opt --version` chạy được
- [ ] `echo 'func.func @test() { return }' | mlir-opt` parse thành công
- [ ] `ninja check-mlir` đạt >95% test pass
- [ ] `mlir-cpu-runner --help` hiện help text
- [ ] Lưu build log vào [`build_notes.md`](./build_notes.md)

---

### Bài tập 8.2 — mlir-opt hands-on

> ⏱ ~6h
> 📁 Output: [`handwritten_mlir/`](./handwritten_mlir/)

**Nguyên tắc vàng: VIẾT TAY, không copy-paste.** Gõ từng dòng giúp build muscle memory cho cú pháp MLIR.

#### 8.2a — Arithmetic cơ bản (`01_arith_basic.mlir`)

Viết file MLIR dùng `func` + `arith` dialect:

```mlir
// Hàm cộng 2 số nguyên
func.func @add_integers(%a: i32, %b: i32) -> i32 {
  %result = arith.addi %a, %b : i32
  return %result : i32
}

// Hàm có constants — thử canonicalize xem fold không?
func.func @with_constants() -> i32 {
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c2 = arith.constant 2 : i32
  %mul = arith.muli %c4, %c2 : i32
  %add = arith.addi %c3, %mul : i32
  return %add : i32
}
```

Chạy thử:
```bash
mlir-opt 01_arith_basic.mlir                  # parse + verify
mlir-opt 01_arith_basic.mlir --canonicalize   # constant folding!
# So sánh: with_constants() trước/sau canonicalize
```

> 💡 **Liên hệ tuần 7:** Constant folding bạn viết tay cho calculator — đây là **cùng 1 pass** nhưng general hơn, áp dụng cho mọi dialect.

#### 8.2b — Structured Control Flow (`02_scf_loop.mlir`)

Viết file MLIR dùng `scf.for` với `iter_args`:

```mlir
// Tính tổng 0 + 1 + 2 + ... + (n-1)
func.func @sum_0_to_n(%n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32
  
  %result = scf.for %i = %c0 to %n step %c1 
      iter_args(%acc = %init) -> (i32) {
    %i_i32 = arith.index_cast %i : index to i32
    %new_acc = arith.addi %acc, %i_i32 : i32
    scf.yield %new_acc : i32
  }
  return %result : i32
}
```

Chạy pipeline lowering:
```bash
# Bước 1: scf → cf (structured → unstructured)
mlir-opt 02_scf_loop.mlir --convert-scf-to-cf

# Bước 2: full lowering xuống LLVM
mlir-opt 02_scf_loop.mlir \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

# Bước 3: xem IR sau TỪNG pass
mlir-opt 02_scf_loop.mlir \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --convert-func-to-llvm \
  --mlir-print-ir-after-all 2>&1 | less
```

> ⚠️ **`--mlir-print-ir-after-all` output ra stderr!** Redirect bằng `2>&1` hoặc `2>ir_dump.log`.

#### 8.2c — Dot Product chạy thật (`03_dot_product.mlir`)

**Bài tập quan trọng nhất tuần**: viết dot product dùng `memref` + `scf.for`, chạy qua `mlir-cpu-runner`:

```mlir
func.func private @printMemrefF32(memref<*xf32>)

func.func @dot_product(%A: memref<4xf32>, %B: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %f0 = arith.constant 0.0 : f32
  
  %result = scf.for %i = %c0 to %c4 step %c1 
      iter_args(%acc = %f0) -> (f32) {
    %a_i = memref.load %A[%i] : memref<4xf32>
    %b_i = memref.load %B[%i] : memref<4xf32>
    %prod = arith.mulf %a_i, %b_i : f32
    %new_acc = arith.addf %acc, %prod : f32
    scf.yield %new_acc : f32
  }
  return %result : f32
}

func.func @main() {
  // Allocate, initialize, call, print, dealloc
  // ... (xem file đầy đủ)
}
```

Pipeline lower + chạy:
```bash
mlir-opt 03_dot_product.mlir \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-cf-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts \
| mlir-cpu-runner -e main -entry-point-result=void \
    --shared-libs=libmlir_runner_utils.so,libmlir_c_runner_utils.so
# Expected: 30.0 (= 1*1 + 2*2 + 3*3 + 4*4)
```

> 💡 **Nếu chạy đúng, bạn vừa làm xong pipeline mini:** viết IR tay → lower qua nhiều tầng → chạy trên CPU thật. Đây là mô hình thu nhỏ của mọi AI compiler.

#### 8.2d — Pass Pipeline Explorer (`run_pipelines.sh`)

Tạo script chạy nhiều pipelines, lưu output, so sánh. Xem [`handwritten_mlir/run_pipelines.sh`](./handwritten_mlir/run_pipelines.sh).

---

### Bài tập 8.3 — Đọc code MLIR thật

> ⏱ ~2h
> 📝 Output: `arith_source_notes.md`

#### 8.3a — Đọc ArithOps.cpp

Mở `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` trong LLVM source:

```bash
# Tìm file
find llvm-project -name "ArithOps.cpp" -path "*/Arith/*"

# Đọc — tập trung vào:
# 1. Canonicalization patterns (fold functions)
# 2. Verify functions  
# 3. Cách op interact với type system
```

**Điều cần note:**

1. **Fold functions** — mỗi op có thể có `fold()` method: nhận operands, trả về folded result nếu có thể. Ví dụ `arith.addi(const_3, const_4)` → fold thành `const_7`. **Đây chính là constant folding của tuần 7, nhưng infrastructure MLIR làm tự động cho mọi op.**

2. **Canonicalization patterns** — `getCanonicalizationPatterns()` đăng ký các RewritePatterns. Ví dụ: `x + 0 → x`, `x * 1 → x`. Mỗi pattern là 1 hàm match-and-rewrite.

3. **Verifier** — `verify()` kiểm tra invariants: types phải match, attributes phải hợp lệ.

#### 8.3b — Đọc ArithOps.td (TableGen/ODS)

```bash
find llvm-project -name "ArithOps.td" -path "*/Arith/*"
```

ODS (Operation Definition Specification) là DSL khai báo ops. Từ file `.td`, TableGen sinh:
- C++ class cho op
- Parser/printer
- Builder methods
- Verifier skeleton

Ví dụ đơn giản hóa:
```tablegen
def Arith_AddIOp : ArithBinaryOp<"addi", [Commutative]> {
  let summary = "integer addition operation";
  let hasFolder = 1;  // Có fold function → constant folding
}
```

**Câu hỏi tự trả lời:** `hasFolder = 1` khai báo ở `.td`, implementation ở `.cpp` — tại sao tách? (Đáp án: separation of concerns — khai báo interface vs implementation, tái dùng code generation.)

---

## Liên hệ HW-SW

> 🔑 Phần này kết nối kiến thức tuần 8 với hardware đã học ở stage 1.

### Tại sao LLVM IR không đủ cho AI chip?

```
LLVM IR biểu diễn matmul:     MLIR biểu diễn matmul:
                                
for i = 0 to M:                linalg.matmul
  for j = 0 to N:               ins(%A, %B)
    for k = 0 to K:              outs(%C)
      %a = load A[i,k]          : tensor<MxKxf32>,
      %b = load B[k,j]            tensor<KxNxf32>
      %c = load C[i,j]          -> tensor<MxNxf32>
      %p = fmul %a, %b
      %s = fadd %c, %p         // Compiler VẪN BIẾT đây là matmul!
      store %s, C[i,j]         // → Có thể tile theo MXU size
                                // → Có thể fuse với ReLU phía sau
// Compiler KHÔNG BIẾT đây       // → Có thể map lên tensor core
// là matmul → không thể tile    // → Có thể quyết định layout
// theo MXU, không fuse được
```

### Bảng ánh xạ MLIR concept → Hardware constraint

| MLIR Concept | Ràng buộc hardware (stage 1) |
|-------------|------------------------------|
| Progressive lowering | Mỗi tầng optimize theo 1 ràng buộc: bandwidth (graph), tile size (loop), instruction (ISA) |
| `tensor` vs `memref` | Functional (no alias) vs imperative (scratchpad management) |
| `linalg.matmul` (structured op) | Giữ info "đây là matmul" → map lên MXU/tensor core |
| Dialect extensibility | Mỗi chip vendor tạo dialect riêng cho hardware |
| Canonicalization | `broadcast(2.0) * ones(1024)` fold → tiết kiệm 1 kernel + 4KB HBM |

### Câu hỏi suy ngẫm cuối tuần

1. **Tại sao `linalg.matmul` tốt hơn 3 loop lồng?** Vì nó giữ semantic "matmul" → tiling, fusion, hardware mapping đều dễ hơn. Khi lower thành loops, info mất vĩnh viễn.

2. **Tại sao MLIR cho nhiều dialect cùng sống trong 1 module?** Vì progressive lowering: bạn có thể lower `linalg.matmul` thành `affine.for` trong khi `arith.addi` vẫn ở nguyên — partial lowering.

3. **Chip startup dùng MLIR thế nào?** Chỉ cần viết dialect riêng cho hardware + backend lowering. Toàn bộ infrastructure (pass manager, verifier, printer...) **miễn phí**.

---

## TODO Checklist

### Lý thuyết (8h)
- [ ] Đọc MLIR paper (Lattner et al. 2021, CGO) — lần 1: big picture
- [ ] Đọc MLIR paper lần 2 — tập trung: dialects, regions, progressive lowering → viết [`paper_notes_mlir.md`](./paper_notes_mlir.md)
- [ ] Đọc MLIR LangRef: operation / value / type / attribute / region / block
- [ ] Đọc tutorial "Understanding the IR Structure"
- [ ] Hiểu block arguments thay cho phi nodes (so sánh với SSA tuần 7)
- [ ] Tour các dialect: `func`, `arith`, `tensor`, `memref`, `linalg`, `affine`, `scf`, `vector`, `llvm` — note mỗi dialect ở mức trừu tượng nào
- [ ] Xem talk MLIR Tutorial (Mehdi Amini, LLVM Dev Meeting)

### Bài tập 8.1 — Build LLVM/MLIR từ source
- [ ] Clone llvm-project
- [ ] CMake configure với `-DLLVM_ENABLE_PROJECTS=mlir -DLLVM_BUILD_EXAMPLES=ON` (xem lệnh đầy đủ ở trên)
- [ ] `ninja check-mlir` pass toàn bộ
- [ ] Note lại RAM/disk/thời gian build + lỗi gặp phải vào [`build_notes.md`](./build_notes.md)
- [ ] Thêm `build/bin` vào PATH, verify `mlir-opt --version`

### Bài tập 8.2 — mlir-opt hands-on
- [ ] Viết tay `01_arith_basic.mlir` (hàm cộng nhân đơn giản), parse được bằng `mlir-opt`
- [ ] Viết tay `02_scf_loop.mlir` dùng `scf.for` + `iter_args`
- [ ] Viết tay `03_dot_product.mlir` — chạy đúng kết quả qua `mlir-cpu-runner`
- [ ] Chạy `--canonicalize`, quan sát khác biệt (constant folding!)
- [ ] Chạy pipeline lower xuống LLVM dialect: `--convert-scf-to-cf --convert-arith-to-llvm ...`
- [ ] Dùng `--mlir-print-ir-after-all` xem IR sau từng pass, lưu output
- [ ] Tạo `run_pipelines.sh` lưu lại các pipeline đã thử

### Bài tập 8.3 — Đọc code MLIR thật
- [ ] Đọc `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` — cách define op + folder + canonicalization
- [ ] Đọc file `.td` (TableGen/ODS) tương ứng `ArithOps.td` — hiểu ODS sinh gì
- [ ] Note cấu trúc vào `arith_source_notes.md`

---

## Output cuối tuần

- [ ] LLVM/MLIR build thành công (`check-mlir` pass)
- [ ] 3+ file `.mlir` viết tay + `run_pipelines.sh`
- [ ] [`paper_notes_mlir.md`](./paper_notes_mlir.md) — notes MLIR paper
- [ ] [`mlir_anatomy_cheatsheet.md`](./mlir_anatomy_cheatsheet.md) — quick reference
- [ ] [`build_notes.md`](./build_notes.md) — build log + troubleshooting
- [ ] (Optional) Blog post tuần 8

---

## Tài liệu tham khảo

### Paper
- **[Bắt buộc]** Lattner et al. (2021), *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, CGO — https://arxiv.org/abs/2002.11054

### Docs chính thức
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)
- [Dialects Overview](https://mlir.llvm.org/docs/Dialects/)
- [Operation Definition Specification (ODS)](https://mlir.llvm.org/docs/DefiningDialects/Operations/)
- [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)

### Blog / Tutorial
- [Jeremy Kun — MLIR for Beginners](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) — blog series thực hành rất tốt
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054) — cùng paper, title khác

### Video
- 🎥 [MLIR Tutorial — Mehdi Amini](https://www.youtube.com/results?search_query=MLIR+tutorial+Mehdi+Amini+LLVM) (LLVM Dev Meeting)
- 🎥 [MLIR: Multi-Level IR Compiler Framework — Chris Lattner](https://www.youtube.com/results?search_query=MLIR+Chris+Lattner) (Google, 2019)

### Cộng đồng
- 💬 [LLVM Discourse — MLIR category](https://discourse.llvm.org/c/mlir/) — hỏi đáp, RFC
- 💻 [llvm/llvm-project](https://github.com/llvm/llvm-project) — source code

---

*Tuần trước: [Tuần 7 — Compiler Fundamentals](../week7-compiler-basics/) · Tuần sau: [Tuần 9 — MLIR Toy Tutorial](../week9-mlir-toy/)*
