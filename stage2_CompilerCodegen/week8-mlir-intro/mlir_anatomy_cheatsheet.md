# MLIR Anatomy Cheatsheet

> Quick reference cho cấu trúc MLIR — dùng tra cứu nhanh khi đọc/viết MLIR code, giúp bạn hiểu rõ bản chất và cách thiết kế của hạ tầng compiler này.

## 1. Tổng quan cấu trúc phân cấp (Hierarchy)

MLIR được thiết kế theo cấu trúc cây phân cấp (tree-like hierarchy) rất chặt chẽ. Mỗi thành phần đều được chứa (nested) bên trong một thành phần khác lớn hơn:

```
Module (builtin.module)               <-- Gốc của một file MLIR
  └── Function (func.func)            <-- Chức năng logic/unit of code
       └── Region                     <-- Container chứa các blocks (ví dụ: thân hàm)
            └── Block                 <-- Basic block (label + block arguments)
                 └── Operation        <-- Lệnh thực thi (đơn vị cơ bản nhất)
                      ├── Results     <-- SSA values sinh ra từ operation
                      ├── Operands    <-- SSA values truyền vào operation
                      ├── Attributes  <-- Compile-time constants & metadata
                      ├── Regions     <-- Nested regions (ví dụ: thân của vòng lặp scf.for)
                      └── Successors  <-- Block đích đến (dành cho terminator ops như nhánh rẽ)
```

## 2. Operation — Đơn vị cơ bản

Trong MLIR, *mọi thứ* đều là một Operation. Từ định nghĩa hàm, module, đến phép cộng hai số.

### Cấu trúc chi tiết (Anatomy)
```mlir
%result:2 = "dialect.opname"(%arg0, %arg1) <{inherent_attr = 42}> ({...region...}) {discardable_attr = "hello"}
           : (tensor<4x8xf32>, tensor<8x16xf32>) -> (tensor<4x16xf32>, index)
```

Giải thích các thành phần:
- **`%result:2`** (Results): Các giá trị SSA mới được định nghĩa. Ở đây có 2 kết quả trả về.
- **`"dialect.opname"`**: Tên đầy đủ của operation, bắt đầu bằng tên dialect (ví dụ: `"arith.addf"` hoặc `"linalg.matmul"`).
- **`(%arg0, %arg1)`** (Operands): Các giá trị SSA đầu vào mà operation này tiêu thụ.
- **`<{inherent_attr = 42}>`**: Inherent attributes. Các thuộc tính này nằm trong định nghĩa bắt buộc của op (thường liên quan chặt chẽ tới semantics, như giá trị so sánh trong `cmp`).
- **`({...region...})`**: Regions. Các block code lồng bên trong op này. Ví dụ: vòng lặp (`scf.for`) sẽ chứa region định nghĩa các phép toán lặp.
- **`{discardable_attr = "hello"}`**: Discardable attributes. Metadata phụ (như pragma, debug flags) mà compiler có thể drop một cách an toàn mà không làm sai semantics.
- **` : (...) -> (...)`** (Type signature): Kiểu dữ liệu của operands (trước `->`) và kết quả (sau `->`).

### Generic Format vs Custom Assembly Format

MLIR có hai cách viết IR:

```mlir
// 1. Generic format (luôn hợp lệ, compiler luôn sinh ra được)
%0 = "arith.addf"(%a, %b) : (f32, f32) -> f32

// 2. Custom format (Dễ đọc hơn, ngắn gọn hơn - do dialect tự định nghĩa cách parse/print)
%0 = arith.addf %a, %b : f32
```
> **Tip:** Khi bị lỗi syntax hoặc không hiểu custom format, hãy yêu cầu mlir-opt in ra generic format bằng cờ `--mlir-print-op-generic`.

## 3. Value — SSA Value

- **Tính chất SSA (Static Single Assignment):** Mỗi value được gán (`define`) đúng 1 lần. Giá trị của nó không bao giờ thay đổi sau khi khởi tạo.
- **Nguồn gốc:** Chỉ có 2 nơi sinh ra value:
  1. Là **Result** của một Operation.
  2. Là **Block Argument** truyền vào một Block.
- **Syntax:** Luôn bắt đầu bằng `%` (ví dụ: `%name`).
- **Naming convention:** 
  - Đặt tên tự động: `%0, %1, %2,...`
  - Đặt tên gợi nhớ (descriptive): `%sum, %idx,...` (Rất hữu ích khi tự viết tay hoặc debug).

## 4. Type System

MLIR là một typed IR. Mọi value và biến đều phải có kiểu dữ liệu rõ ràng.

### Built-in types:
```mlir
// Số nguyên (Integers) - Hỗ trợ bit-width tùy ý
i1, i8, i16, i32, i64, i128

// Số thực (Floats)
f16, bf16, f32, f64, f80, f128

// Index
index  // Số nguyên phụ thuộc vào platform (như size_t trong C), dùng làm chỉ số mảng, số đếm vòng lặp.

// Function type
(i32, f32) -> (i64)

// None type
none
```

### Shaped types (Cấu trúc dữ liệu có hình học):
```mlir
// 1. Tensor (Bất biến - Immutable, Value semantics)
tensor<4x8xf32>           // Ranked, static shape (Biết chính xác kích thước 4x8)
tensor<4x?xf32>           // Ranked, dynamic dimension (Không biết trước cột, runtime mới biết)
tensor<*xf32>             // Unranked (Không biết số chiều)

// 2. MemRef (Có thể thay đổi - Mutable, Có địa chỉ/Layout)
memref<4x8xf32>           // Vùng nhớ liên tục (contiguous), default layout
memref<4x8xf32, strided<[8,1]>>  // Explicit strides (Layout bộ nhớ tùy chỉnh)
memref<4x8xf32, affine_map<...>> // Affine layout cho các phép chiếu phức tạp
memref<4x8xf32, 1>        // Memory space = 1 (Ví dụ: map vào shared memory trên GPU)

// 3. Vector (SIMD-like, Fixed size)
vector<4xf32>             // Vector 1D (Dùng trực tiếp cho hardware SIMD registers)
vector<4x8xf32>           // Vector multi-dimensional
```

### Quan trọng: Sự khác biệt giữa `tensor` và `memref`
Điều này là khái niệm cốt lõi khi làm việc với Machine Learning Compilers:

- **`tensor` (Functional World):** 
  - Hoạt động với Value semantics (như các biến cơ bản `i32`).
  - Là **Immutable** (không thể sửa đổi tại chỗ). Một phép gán/cập nhật tensor về logic sẽ tạo ra một tensor mới.
  - Không có alias (hai tensor khác nhau không bao giờ trỏ chung dữ liệu).
  - Mục đích: Giúp compiler thực hiện các High-level optimizations (fusion, tiling) cực kỳ dễ dàng vì không phải bận tâm việc đọc/ghi chéo gây data hazard.
  
- **`memref` (Imperative World):**
  - Hoạt động với Reference semantics.
  - Trỏ đến một vùng nhớ vật lý thực sự (RAM, VRAM, Cache).
  - Có thể read/write, thay đổi trực tiếp (Mutable). Phải đối mặt với vấn đề aliasing (hai `memref` có thể trỏ cùng một vùng nhớ).
  - Mục đích: Phục vụ Codegen sinh ra lệnh cho phần cứng. Phần cứng luôn hoạt động với con trỏ vùng nhớ.

- **Bufferization:** Quá trình chuyển đổi IR từ thế giới `tensor` sang `memref`. Ở bước này, compiler sẽ phân bổ bộ nhớ (memory allocation) và quản lý tái sử dụng buffer (in-place updates).

## 5. Attributes

Attributes là các hằng số (constants) tại thời điểm biên dịch, cung cấp metadata hoặc cấu hình cho operations. Attributes không phải là SSA values.

```mlir
// Integer attribute
{value = 42 : i32}

// Float attribute  
{value = 3.14 : f64}

// String attribute
{sym_name = "my_function"}

// Array attribute
{values = [1, 2, 3]}

// Dense elements (Cực kỳ phổ biến để khởi tạo Tensor hằng số)
{value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>}

// Affine map (Cấu hình layout/truy xuất cho vòng lặp, memref)
{map = affine_map<(d0, d1) -> (d0, d1)>}

// Unit attribute (Một dạng cờ/flag - chỉ cần có tên là "true")
{sym_visibility = "private"}
```

## 6. Region & Block

### Region
- Là danh sách các Blocks.
- Có 2 loại chính:
  1. **SSACFG Region:** Có luồng điều khiển (Control Flow Graph), thực thi từ block đầu đến cuối, tuần tự, có rẽ nhánh.
  2. **Graph Region:** Các op bên trong thực thi không theo thứ tự từ trên xuống dưới mà theo luồng dữ liệu (Dataflow) — ví dụ như thiết kế mạch (hardware circuits).
- Một Region luôn bị "bao bọc" (enclosed) bởi một operation cha (như `func.func`, `scf.for`).

### Block
- Gồm: `Label` (ví dụ `^bb0`), danh sách **Block Arguments**, và danh sách tuần tự các Operations.
- Mọi Block hợp lệ đều phải kết thúc bằng một **Terminator Operation** (như `func.return`, `cf.br`, `cf.cond_br`, `scf.yield`). Terminator điều hướng flow đi tới block khác hoặc trả luồng điều khiển về op cha.

### Block Arguments thay thế Phi Nodes
Nếu bạn đã học LLVM IR, bạn sẽ biết tới Phi Node (cơ chế merge data path từ nhiều nhánh). MLIR loại bỏ hoàn toàn Phi Nodes và thay bằng **Block Arguments**.

**So sánh:**
```mlir
// LLVM phi node style (Dùng biến ảo để minh họa):
bb1:
  %x = phi [%a, bb0], [%b, bb2]

// MLIR block arguments (Rất giống việc gọi hàm):
^bb1(%x: i32):     // %x nhận giá trị tương ứng từ bất kỳ predecessor nào nhảy tới đây
  ...
// Nhảy từ bb0 tới bb1:
cf.br ^bb1(%a : i32)  // Truyền %a vào làm tham số %x
// Nhảy từ bb2 tới bb1:
cf.br ^bb1(%b : i32)  // Truyền %b vào làm tham số %x
```

**Ưu điểm của Block Arguments:**
- Dễ đọc, sạch hơn: Không cần tra cứu lặp lại tên của predecessor block bên trong hàm con.
- Đơn giản hóa quá trình verify: Mỗi lần gọi lệnh `br` (branch), compiler chỉ cần kiểm tra xem kiểu tham số truyền đi có khớp với chữ ký của block đích không.

## 7. Dialect Quick Reference

MLIR sử dụng khái niệm "Dialect" để đóng gói và nhóm các operation, types, attributes có chung một ngữ nghĩa/mục đích. Dưới đây là các Dialects phổ biến:

| Dialect | Mức trừu tượng | Ops tiêu biểu | Ví dụ cú pháp |
|---------|---------------|-----------|-------|
| `builtin` | Hạ tầng gốc | module, func, tensor | `builtin.module { ... }` |
| `func` | Hàm cơ bản | func.func, call, return | `func.func @add(%a: i32) -> i32` |
| `arith` | Toán vô hướng | addi, addf, muli, cmpf | `%c = arith.addi %a, %b : i32` |
| `math` | Hàm toán học | exp, log, sin, cos | `%y = math.exp %x : f32` |
| `tensor` | Thao tác trên tensor | extract, insert, pad | `tensor.extract %t[%i]` |
| `memref` | Thao tác trên buffer | alloc, dealloc, load, store | `memref.load %m[%i, %j]` |
| `linalg` | Toán tử Tensor có cấu trúc | matmul, conv_2d, generic | `linalg.matmul ins(...) outs(...)` |
| `affine` | Vòng lặp Polyhedral | for, if, load (có affine map) | `affine.for %i = 0 to 128` |
| `scf` | CFG có cấu trúc (Structured CF)| for, while, if, yield | `scf.for %i = 0 to %n step 1` |
| `cf` | Nhánh rẽ tự do (Unstructured CF)| br, cond_br, switch | `cf.br ^bb1(%val : i32)` |
| `vector` | SIMD Hardware Vector | transfer_read, fma, broadcast| `vector.transfer_read %m[...]` |
| `llvm` | LLVM IR Mapping 1:1 | llvm.add, llvm.getelementptr | `llvm.add %a, %b : i64` |

## 8. Progressive Lowering Path (Hành trình biên dịch)

Compiler dịch một model Deep Learning xuống mã máy bằng cách hạ cấp (lower) dần dần qua nhiều dialects từ mức cao (high-level) xuống mức thấp (low-level). Một chuỗi điển hình:

```text
  linalg.matmul  (High-level, structured, tensor semantics)
         ↓ Tiling + Fusion (Optimizations)
  linalg.generic (Tiled version)
         ↓ Convert linalg-to-loops
  scf.for / affine.for (Vòng lặp cơ bản)
         ↓ Bufferization (Phân bổ vùng nhớ thực: tensor → memref)
  memref ops lồng trong scf.for (Imperative semantics)
         ↓ Convert scf-to-cf
  cf.br / cf.cond_br (Kiến trúc khối lệnh nhảy tự do, dọn dẹp loops)
         ↓ Convert arith/memref/cf to llvm
  llvm dialect (Mô phỏng LLVM IR bên trong MLIR)
         ↓ mlir-translate
  LLVM IR (Thoát khỏi hệ sinh thái MLIR, bước vào LLVM backend)
         ↓ llc (LLVM Compiler)
  Machine code (Assembly/Object file)
```

## 9. mlir-opt Quick Reference

Công cụ `mlir-opt` là bạn đồng hành không thể thiếu khi debug compiler passes.

```bash
# Parse và verify file (Kiểm tra xem IR có hợp lệ không)
mlir-opt input.mlir

# Chạy pass Canonicalize (Rút gọn, tối ưu logic thừa)
mlir-opt input.mlir --canonicalize

# Theo dõi sự biến đổi của IR sau mỗi pass (Vô cùng hữu ích để debug)
mlir-opt input.mlir --mlir-print-ir-after-all

# Chạy một chuỗi pipeline hạ cấp (Full lowering pipeline)
mlir-opt input.mlir \
  --convert-linalg-to-loops \
  --convert-scf-to-cf \
  --convert-arith-to-llvm \
  --finalize-memref-to-llvm \
  --convert-func-to-llvm \
  --reconcile-unrealized-casts

# Tìm kiếm tất cả các pass có sẵn
mlir-opt --help | grep '\-\-'

# Chạy pass trên cụ thể từng function/module với cú pháp pipeline chi tiết
mlir-opt input.mlir --pass-pipeline='builtin.module(func.func(canonicalize,cse))'
```

## 10. Cheat Sheet: Mẹo đọc hiểu MLIR IR

Khi nhìn vào một đoạn code MLIR dài dằng dặc, đừng hoảng, hãy tiếp cận theo trình tự sau:

1. **Tìm điểm bắt đầu:** Bắt đầu từ cấp độ module `builtin.module`, rồi khoanh vùng hàm cần phân tích `func.func`. Đọc qua signature của hàm để biết nó nhận gì và trả về gì.
2. **Lần theo dấu vết SSA Values:** Nhìn vào các toán hạng (operands) của một lệnh. Ví dụ thấy `%3 = ...`, hãy dùng tìm kiếm (Ctrl+F) tìm `%3` hoặc truy ngược lên trên để xem `%3` được sinh ra từ lệnh nào.
3. **Luôn chú ý Type Signatures:** Đặc biệt với tensor và memref. Hãy nhìn vào phần sau dấu `:` của lệnh. Type signature cho biết kích thước, số chiều (shape), kiểu dữ liệu (i32, f32) và memory layout. Điều này giúp bạn mường tượng data được tổ chức như thế nào.
4. **Phân biệt Dialects:** Nhìn vào tiền tố của op (như `arith.`, `scf.`, `linalg.`). Điều này lập tức cho bạn manh mối về "tầng trừu tượng" (abstraction layer) hiện hành và mục đích của lệnh đó.
5. **Đọc vòng lặp qua Block Arguments:** Nếu gặp vòng lặp `scf.for` hoặc label `^bb1(%x: i32)`, hãy nhớ biến `%x` đóng vai trò "carry loop variable" (biến mang trạng thái truyền qua từng lần lặp). Dò xem giá trị nào được `yield` hoặc `br` ở cuối block để cập nhật lại `%x`.
6. **Không bỏ qua Attributes:** Chúng thường chứa các cấu hình quan trọng (vd: kích thước kernel của phép convolution, hoặc affine_map định tuyến truy xuất mảng). 
7. **Đừng sợ Custom Formatting:** Nếu một dialect in ra định dạng quá khó hiểu, dùng `mlir-opt --mlir-print-op-generic` để xem dưới dạng generic thuần túy, mọi bí mật (types, attributes ẩn) sẽ phơi bày rõ ràng.
