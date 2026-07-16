# GIAI ĐOẠN 2 — COMPILER & CODE GENERATION CHO AI

> **Lộ trình ML Compiler Engineer cho AI chip**
> Thời lượng: 10 tuần (tuần 7-16) · ~15-20h/tuần
> Trọng tâm: **MLIR sâu** + Triton + TVM + XLA, kết thúc bằng capstone mini compiler end-to-end
> Prerequisite: Hoàn thành Giai đoạn 1 (đặc biệt systolic simulator tuần 3 — sẽ là **target backend** của capstone)

---

## Mục lục

- [Phần 0 — Bức tranh toàn cảnh: Từ hardware sang compiler](#phần-0--bức-tranh-toàn-cảnh-từ-hardware-sang-compiler)
- [Tổng quan giai đoạn 2](#tổng-quan-giai-đoạn-2)
- [Tuần 7 — Compiler Fundamentals Refresher](#tuần-7--compiler-fundamentals-refresher)
- [Tuần 8 — Giới thiệu MLIR + Build từ source](#tuần-8--giới-thiệu-mlir--build-từ-source)
- [Tuần 9 — MLIR Toy Tutorial (7 chương)](#tuần-9--mlir-toy-tutorial-7-chương)
- [Tuần 10 — MLIR ML Dialects: linalg, tensor, memref, affine](#tuần-10--mlir-ml-dialects-linalg-tensor-memref-affine)
- [Tuần 11 — Triton: DSL viết kernel GPU](#tuần-11--triton-dsl-viết-kernel-gpu)
- [Tuần 12 — TVM: Tensor Expressions & Auto-tuning](#tuần-12--tvm-tensor-expressions--auto-tuning)
- [Tuần 13 — XLA & HLO IR](#tuần-13--xla--hlo-ir)
- [Tuần 14-16 — Capstone: Mini Compiler PyTorch → Systolic Simulator](#tuần-14-16--capstone-mini-compiler-pytorch--systolic-simulator)
- [Tổng kết Giai đoạn 2](#tổng-kết-giai-đoạn-2)
- [Tài liệu reference](#tài-liệu-reference-cho-toàn-giai-đoạn-2)

---

## PHẦN 0 — BỨC TRANH TOÀN CẢNH: TỪ HARDWARE SANG COMPILER

### 0.1. Giai đoạn 1 dạy bạn "tại sao" — Giai đoạn 2 dạy bạn "làm thế nào"

Ở Giai đoạn 1, bạn đã nắm intuition: mọi optimization compiler là phản ứng với một ràng buộc hardware. Bây giờ bạn học **cách compiler thực sự làm điều đó** — bằng IR, pass, và codegen.

```
Giai đoạn 1 (đã học):                Giai đoạn 2 (sẽ học):

"HBM chậm hơn compute 50x"     →     Op fusion PASS trên Graph IR
"SRAM chỉ 100KB"               →     Tiling TRANSFORM trên Tensor IR
"MXU là 256x256"               →     Lowering + padding trong CODEGEN
"INT8 4x nhanh hơn FP32"       →     Quantization REWRITE PATTERN
"DMA và compute độc lập"       →     Double buffering SCHEDULING PASS
```

### 0.2. Tại sao MLIR là trung tâm của giai đoạn này

Nhìn vào compiler stack của các công ty AI chip (2026):

| Công ty / Project | Compiler stack | Dựa trên |
|-------------------|----------------|----------|
| Google (TPU) | XLA → OpenXLA/StableHLO | MLIR (dần dần) |
| NVIDIA | Triton, TensorRT, cuDNN | Triton dùng MLIR |
| AMD | ROCm, MIGraphX, IREE | IREE = MLIR |
| Modular | Mojo, MAX | MLIR (Lattner sáng lập cả hai) |
| Tenstorrent | tt-mlir, tt-forge | MLIR |
| AWS Trainium | Neuron compiler | MLIR (NKI) |
| Intel/Habana | oneAPI, Gaudi SW | MLIR (một phần) |
| Microsoft Maia | Maia SDK | Triton → MLIR |

**MLIR không phải "một compiler" — nó là framework để xây compiler.** Học MLIR nghĩa là học được cách mọi công ty trên xây stack của họ. TVM học sau để có góc nhìn so sánh (auto-tuning based thay vì pass based).

### 0.3. Khái niệm cốt lõi phải nắm trước khi bắt đầu

**IR (Intermediate Representation)** — cách compiler biểu diễn chương trình giữa source và machine code. Mỗi mức IR trade-off giữa "gần người" (dễ optimize theo ngữ nghĩa) và "gần máy" (dễ sinh code).

**Progressive lowering** — triết lý MLIR: thay vì 1 bước nhảy từ graph xuống assembly, đi qua nhiều tầng, mỗi tầng làm đúng việc của nó:

```
PyTorch model
    ↓ (capture)
Graph IR          — "tính gì": op fusion, constant folding, CSE
    ↓ (lower)
Tensor IR         — "tính thế nào, trừu tượng": tiling, vectorize, layout
    ↓ (lower)
Loop/Affine IR    — "loop cụ thể + memory": unroll, pipeline, alloc
    ↓ (lower)
Hardware ISA      — "instruction cụ thể": systolic commands, DMA, sync
```

**Pass** — một transformation trên IR: nhận IR vào, trả IR ra (đã optimize hoặc đã lower). Compiler = chuỗi pass (pass pipeline). Nghề compiler engineer phần lớn là **viết và debug pass**.

**Dialect (MLIR-specific)** — một "namespace" chứa ops + types cùng mức trừu tượng. `linalg.matmul` và `arith.addf` và `llvm.fadd` sống chung trong 1 file MLIR — đây là điểm thiên tài của MLIR: nhiều mức trừu tượng cùng tồn tại, lowering dần dần.

### 0.4. Câu hỏi xuyên suốt giai đoạn 2

> 🔑 *"Pass này tồn tại vì ràng buộc hardware nào?"*
> 🔑 *"Thông tin gì bị mất khi lower từ tầng này xuống tầng kia? Vì sao phải optimize trước khi mất nó?"*

Ví dụ: `linalg.matmul` biết nó là matmul → có thể tile theo cấu trúc. Sau khi lower xuống loop lồng nhau, thông tin "đây là matmul" **mất vĩnh viễn** → không thể map lên MXU nữa. **Đây là lý do IR nhiều tầng tồn tại.**

---

## TỔNG QUAN GIAI ĐOẠN 2

### Mục tiêu khi kết thúc

Sau 10 tuần, bạn phải:

1. **Đọc và viết MLIR thành thạo** — dialect, op, type, attribute, region, pass.
2. **Implement pass MLIR** từ đầu: fusion, tiling, lowering giữa các dialect.
3. **Viết kernel Triton** hiệu năng cao — Flash Attention level, hiểu Triton compile xuống PTX thế nào.
4. **Hiểu TVM pipeline** Relay/Relax → TE → TIR → codegen, chạy được MetaSchedule auto-tuning.
5. **Đọc HLO dump** của JAX/XLA và chỉ ra được fusion decisions.
6. **Build mini compiler end-to-end**: PyTorch model → graph IR → tiling/fusion → codegen → chạy trên systolic simulator của bạn (tuần 3). **Đây là portfolio piece quan trọng nhất của cả lộ trình.**

### Bản đồ khái niệm

```
                     COMPILER CHO AI CHIP
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   IR DESIGN           TRANSFORMS            CODEGEN
   (biểu diễn)         (biến đổi)           (sinh code)
        │                   │                   │
   • SSA, CFG          • Fusion             • Instruction sel
   • Dialects          • Tiling             • Register alloc
   • Regions           • Vectorization      • Scheduling
   • Type system       • Layout transform   • Runtime calls
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                 4 HỆ SINH THÁI THỰC TẾ
        MLIR (tuần 8-10) · Triton (tuần 11)
        TVM (tuần 12)    · XLA (tuần 13)
                            ↓
                    CAPSTONE (tuần 14-16)
              tự xây compiler cho systolic sim
```

### Phân bổ thời gian

| Tuần | Chủ đề | Trọng số | Output chính |
|------|--------|----------|--------------|
| 7 | Compiler fundamentals | Nhẹ (ôn tập) | Toy calculator compiler |
| 8 | MLIR intro + build | Trung bình | LLVM/MLIR build + mlir-opt thành thạo |
| 9 | MLIR Toy tutorial | **Nặng** | Toy compiler 7 chương hoàn chỉnh |
| 10 | ML dialects | **Nặng** | Matmul lowering pipeline tay |
| 11 | Triton | Trung bình | Flash Attention bằng Triton |
| 12 | TVM | Trung bình | Model compile + MetaSchedule |
| 13 | XLA/HLO | Nhẹ | HLO analysis notes |
| 14-16 | **Capstone** | **Rất nặng** | Mini compiler end-to-end |

### Cấu trúc thư mục

```
stage2_CompilerCodegen/
├── README.md                  # File này
├── week7-compiler-basics/     # Toy calculator compiler
├── week8-mlir-intro/          # Build LLVM/MLIR + IR exploration
├── week9-mlir-toy/            # Toy tutorial 7 chương
├── week10-ml-dialects/        # linalg/tensor/memref/affine lowering
├── week11-triton/             # Triton kernels + Flash Attention
├── week12-tvm/                # TVM TE/TIR + auto-tuning
├── week13-xla-hlo/            # JAX → HLO phân tích
└── week14-16-capstone/        # Mini compiler PyTorch → systolic sim
```

Mỗi thư mục tuần có `README.md` riêng với TODO checklist chi tiết.

---

## TUẦN 7 — Compiler Fundamentals Refresher

> **Câu hỏi central:** *Compiler truyền thống (C/C++) làm gì qua từng phase, và AI compiler kế thừa/vứt bỏ những gì?*

📁 Chi tiết & TODO: [`week7-compiler-basics/`](./week7-compiler-basics/)

### Lý thuyết (8h)

#### Ngày 1-2: Compiler pipeline cổ điển

Đọc **Cooper & Torczon, *Engineering a Compiler*** (3rd ed.) — chương 1 (Overview) + chương 5 (IR). Nếu đã học compiler ở đại học, đọc lướt. Nếu chưa, đọc kỹ chương 1.

Nắm pipeline:

```
Source → Lexer → Parser → AST → Semantic Analysis → IR
       → Optimization passes → Instruction Selection
       → Register Allocation → Scheduling → Machine code
```

#### Ngày 3: SSA form — khái niệm quan trọng nhất

Đọc chương SSA trong Cooper & Torczon (ch. 9) hoặc bài "Static Single Assignment" bất kỳ.

- SSA = mỗi biến gán đúng 1 lần → def-use chains hiển nhiên → mọi dataflow analysis dễ đi
- Phi nodes tại join points
- **MLIR/LLVM đều là SSA** — không hiểu SSA thì không đọc được MLIR

#### Ngày 4: LLVM IR mức đọc hiểu

- Đọc [LLVM Language Reference](https://llvm.org/docs/LangRef.html) — phần instructions cơ bản
- Chạy `clang -S -emit-llvm foo.c` với vài hàm C đơn giản, đọc output
- Nhận diện: function, basic block, `br`, `phi`, `getelementptr`

#### Ngày 5: AI compiler khác compiler truyền thống chỗ nào

Tự viết note trả lời:

| | C compiler | AI compiler |
|---|-----------|-------------|
| Đơn vị tối ưu | Scalar, loop | Tensor op, graph |
| Analysis khó nhất | Alias analysis | Layout/memory planning |
| Codegen target | 1 ISA cố định | Nhiều accelerator |
| Autotuning | Ít (PGO) | Nhiều (search-based) |
| Ai quyết định memory | Hardware (cache) | Compiler (scratchpad) |

### Thực hành (10h)

**Bài tập 7.1 — Toy calculator compiler** (project chính của tuần):

Viết compiler cho ngôn ngữ biểu thức số học, bằng Python hoặc C++:

```
Input:  "let x = 3 + 4 * 2; let y = x * x; print(y - 1)"
Pipeline: Lexer → Parser (AST) → IR tuyến tính (3-address code, SSA-ish)
        → 2 optimization passes (constant folding + dead code elimination)
        → Codegen ra Python bytecode-like hoặc stack machine tự định nghĩa
        → Interpreter chạy stack machine đó
```

Yêu cầu bắt buộc:
- AST in ra được (pretty printer)
- IR trước/sau optimization in ra được — **thói quen dump IR là kỹ năng nghề nghiệp số 1 của compiler engineer**
- Constant folding: `3 + 4 * 2` → `11` tại compile time
- DCE: biến không dùng bị xóa

**Bài tập 7.2 — Đọc LLVM IR:** compile 3 hàm C (`add`, loop sum, branchy max) với `-O0` và `-O2`, diff 2 output, giải thích từng khác biệt.

### Liên hệ HW-SW

Constant folding trong calculator của bạn là **cùng một pass** với const folding trong XLA — chỉ khác đơn vị: scalar vs tensor. Khi XLA fold `broadcast(2.0) * ones(1024x1024)`, nó tiết kiệm cả một kernel launch + 4MB HBM traffic. Nguyên lý một, tác động khuếch đại theo kích thước tensor.

### Output cuối tuần

- Toy calculator compiler với 2 pass, có test
- `llvm_ir_notes.md` — phân tích -O0 vs -O2
- Note 1 trang: "AI compiler vs traditional compiler"

---

## TUẦN 8 — Giới thiệu MLIR + Build từ source

> **Câu hỏi central:** *Vì sao LLVM IR không đủ cho ML, đến mức phải xây cả một infrastructure mới (MLIR)?*

📁 Chi tiết & TODO: [`week8-mlir-intro/`](./week8-mlir-intro/)

### Lý thuyết (8h)

#### Ngày 1: MLIR paper

Đọc Lattner et al. (2021), *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*, CGO. **Paper quan trọng nhất giai đoạn này — đọc 2 lần.**

Điểm phải nắm:
- Vấn đề: mỗi framework tự chế IR (TF Graph, XLA HLO, TorchScript, Glow...) → trùng lặp infrastructure, không tái dùng pass
- Giải pháp: 1 meta-IR cho phép define nhiều dialect, dùng chung pass infrastructure, type system, printer/parser
- Region-based IR: op chứa region chứa block chứa op — biểu diễn được cả graph lẫn control flow

#### Ngày 2-3: Ngôn ngữ MLIR

Đọc [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/) + [Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/).

Nắm chắc anatomy của 1 op:

```mlir
%result = "dialect.opname"(%operand1, %operand2) {attr = 42 : i32}
          : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
```

- **Operation** — đơn vị vạn năng (function cũng là op, module cũng là op)
- **Value** (SSA) — `%x`, kết quả của op hoặc block argument
- **Type** — `tensor<4x8xf32>`, `memref<4x8xf32>`, `i32`, `f32`
- **Attribute** — hằng số compile-time gắn vào op
- **Region & Block** — op có thể chứa cả "body" (như `func`, `scf.for`)
- Không có phi node — MLIR dùng **block arguments** (thanh lịch hơn)

#### Ngày 4-5: Dialect ecosystem tour

Đọc overview các dialect chính tại [mlir.llvm.org/docs/Dialects](https://mlir.llvm.org/docs/Dialects/): `func`, `arith`, `tensor`, `memref`, `linalg`, `affine`, `scf`, `vector`, `llvm`. Chưa cần sâu — tuần 10 sẽ đào. Mục tiêu: biết dialect nào ở mức trừu tượng nào.

🎥 Xem talk "MLIR Tutorial" của Mehdi Amini (LLVM Dev Meeting, YouTube).

### Thực hành (12h)

**Bài tập 8.1 — Build LLVM/MLIR từ source** (nửa ngày, chủ yếu chờ):

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && mkdir build && cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_CCACHE_BUILD=ON
ninja check-mlir   # build + chạy test suite
```

Lưu ý: cần ~16GB RAM (nếu thiếu, thêm `-DLLVM_PARALLEL_LINK_JOBS=2`), ~100GB disk cho build debug (Release nhẹ hơn nhiều).

**Bài tập 8.2 — mlir-opt là bạn thân mới:**

Viết tay file `.mlir` nhỏ (dùng `arith` + `func` + `scf`), rồi:

```bash
mlir-opt input.mlir --canonicalize
mlir-opt input.mlir --convert-scf-to-cf --convert-arith-to-llvm
mlir-opt input.mlir --mlir-print-ir-after-all  # xem IR sau từng pass
```

Yêu cầu: viết được tay (không copy) 1 hàm MLIR tính `dot product` bằng `scf.for` + `arith`, chạy qua `mlir-cpu-runner` ra kết quả đúng.

**Bài tập 8.3 — Đọc code MLIR thật:** đọc source `mlir/lib/Dialect/Arith/IR/ArithOps.cpp` — xem 1 op được define + canonicalize pattern thế nào. Note lại cấu trúc.

### Liên hệ HW-SW

LLVM IR chỉ có scalar + vector + pointer → biểu diễn matmul thành loop lồng 3 tầng **ngay lập tức**, mất thông tin cấu trúc. MLIR cho phép giữ `linalg.matmul` như một op nguyên tử đến tận lúc quyết định tiling — **thời điểm lower chính là thời điểm compiler cam kết với một chiến lược hardware**. Chip có MXU cần biết "đây là matmul" càng lâu càng tốt.

### Output cuối tuần

- LLVM/MLIR build thành công, `check-mlir` pass
- 3+ file `.mlir` viết tay + script các pass pipeline đã thử
- Note: "MLIR anatomy cheatsheet" (op/value/type/attr/region)

---

## TUẦN 9 — MLIR Toy Tutorial (7 chương)

> **Câu hỏi central:** *Xây một compiler MLIR từ đầu gồm những mảnh nào — dialect, lowering, codegen ghép với nhau ra sao?*

📁 Chi tiết & TODO: [`week9-mlir-toy/`](./week9-mlir-toy/)

### Kế hoạch (toàn tuần là thực hành, ~18h)

Làm [Toy Tutorial chính thức](https://mlir.llvm.org/docs/Tutorials/Toy/) — 7 chương. Code có sẵn trong `mlir/examples/toy/`, **nhưng không copy-paste**: với mỗi chương, đọc → gõ lại phần core → build → thử nghiệm sửa đổi.

| Ngày | Chương | Nội dung | Điều phải hiểu sâu |
|------|--------|----------|--------------------|
| 1 | Ch1-2 | AST → MLIR, define Toy dialect | ODS/TableGen — op được sinh code thế nào |
| 2 | Ch3 | High-level optimization | Canonicalization pattern, `transpose(transpose(x)) = x` |
| 3 | Ch4 | Interfaces | ShapeInference qua OpInterface — vì sao interface > hardcode |
| 4 | Ch5 | Partial lowering → affine | **Chương quan trọng nhất** — dialect conversion framework, legal/illegal ops |
| 5 | Ch6 | Lowering → LLVM + JIT | Full conversion, chạy được thật qua ExecutionEngine |
| 6 | Ch7 | Struct types | Custom type — đọc nhanh, ít quan trọng hơn |
| 7 | Buffer | Bài tập mở rộng (dưới) | Tự kiểm tra |

### Bài tập mở rộng (bắt buộc — đây là phần biến "làm theo tutorial" thành "hiểu")

1. **Thêm op mới `toy.sub`** (trừ element-wise): define trong ODS, thêm parser/verifier, lower xuống affine. Nếu làm được không nhìn guide → bạn đã hiểu ch1-5.
2. **Thêm canonicalization**: `x - x → zeros`. Viết `RewritePattern`.
3. **Thêm 1 pass đếm op**: pass thống kê in ra "module có N matmul, M transpose" — làm quen với pass manager + walker.
4. **Trace 1 lowering bằng `--mlir-print-ir-after-all`**: chụp lại IR sau từng pass cho 1 chương trình Toy, viết note giải thích từng bước biến đổi.

### Liên hệ HW-SW

Ch5 (partial lowering) là mô hình thu nhỏ của mọi AI compiler: `toy.matmul` (không biết gì về loop) → `affine.for` lồng nhau (biết loop nhưng chưa biết ISA) → LLVM (biết máy). Trong capstone tuần 14-16, bạn sẽ làm y hệt nhưng thay tầng cuối bằng **instruction cho systolic simulator**.

### Output cuối tuần

- Toy compiler build & chạy được cả 7 chương
- `toy.sub` + canonicalization + counting pass tự viết
- `lowering_trace.md` — walkthrough IR qua từng pass

---

## TUẦN 10 — MLIR ML Dialects: linalg, tensor, memref, affine

> **Câu hỏi central:** *Một `linalg.matmul` đi xuống loop nest cụ thể qua những bước nào, và tiling/fusion xảy ra ở đâu trong hành trình đó?*

📁 Chi tiết & TODO: [`week10-ml-dialects/`](./week10-ml-dialects/)

### Lý thuyết (8h)

#### Ngày 1-2: Linalg dialect — trái tim của ML codegen trong MLIR

Đọc [Linalg Dialect Rationale](https://mlir.llvm.org/docs/Dialects/Linalg/) — dài nhưng là tài liệu quan trọng nhất tuần.

- `linalg.generic` — mọi op ML biểu diễn được bằng: indexing maps (affine maps) + iterator types (parallel/reduction) + body scalar
- `linalg.matmul`, `linalg.conv_2d` = named ops, đường tắt của generic
- **Structured ops philosophy**: op mang theo đủ metadata để transform (tile, fuse, vectorize) mà không cần phân tích lại loop

#### Ngày 3: tensor vs memref — ranh giới quan trọng nhất

- `tensor<4x8xf32>` — immutable value, không có địa chỉ (thế giới functional)
- `memref<4x8xf32>` — mutable buffer, có địa chỉ + layout (thế giới imperative)
- **Bufferization** = pass chuyển tensor → memref = thời điểm compiler quyết định memory allocation. Đọc [Bufferization docs](https://mlir.llvm.org/docs/Bufferization/).

#### Ngày 4: affine & scf

- `affine.for` / `affine.if` — loop có ràng buộc affine → phân tích dependence chính xác → hợp pháp hóa tiling/interchange tự động
- `scf.for` / `scf.while` — loop tổng quát, ít phân tích được hơn
- Đọc lại note polyhedral từ tuần 6 stage 1 — giờ bạn thấy nó trong thực tế

#### Ngày 5: Transform dialect (mới & quan trọng)

Đọc [Transform Dialect tutorial](https://mlir.llvm.org/docs/Tutorials/transform/) — điều khiển tiling/fusion bằng chính MLIR script thay vì hardcode C++ pass. Đây là hướng hiện đại (IREE dùng nặng).

### Thực hành (12h)

**Bài tập 10.1 — Matmul lowering pipeline tay** (project chính):

Viết `matmul.mlir` dùng `linalg.matmul` trên tensor, rồi tự dựng pipeline lower dần xuống LLVM, quan sát IR từng bước:

```bash
# Bước 1: tile 32x32x32
mlir-opt matmul.mlir --transform-interpreter  # hoặc test pass tiling
# Bước 2: bufferize
mlir-opt --one-shot-bufferize="bufferize-function-boundaries"
# Bước 3: lower linalg → loops
mlir-opt --convert-linalg-to-loops
# Bước 4: xuống LLVM
mlir-opt --convert-scf-to-cf --convert-arith-to-llvm \
         --finalize-memref-to-llvm --convert-func-to-llvm \
         --reconcile-unrealized-casts
# Bước 5: chạy
mlir-runner -e main --entry-point-result=void \
  --shared-libs=libmlir_runner_utils.so
```

Yêu cầu: lưu IR dump sau **từng** bước vào repo + viết chú thích cái gì thay đổi.

**Bài tập 10.2 — Tiling bằng Transform dialect:** viết transform script tile `linalg.matmul` thành 2 cấp (ví dụ 64x64 rồi 8x8), dump IR, đối chiếu với tiled matmul CUDA bạn viết ở tuần 6 stage 1 — **cùng một transformation, một cái tay một cái tự động**.

**Bài tập 10.3 — Fusion quan sát được:** viết `linalg.matmul` + `linalg.generic` (ReLU) liên tiếp, dùng transform dialect fuse chúng, chứng minh bằng IR rằng intermediate tensor biến mất. **Đây chính là op fusion mà stage 1 nói suốt — giờ bạn tự tay làm.**

**Bài tập 10.4 — Viết pass C++ thật:** viết standalone MLIR pass (out-of-tree project, có CMake template chính thức `mlir/examples/standalone`) đơn giản: đổi mọi `arith.mulf` có hằng 2.0 thành `arith.addf x, x`. Mục tiêu là học được **cả tooling** (đăng ký pass, viết RewritePattern, FileCheck test).

### Liên hệ HW-SW

Chuỗi transformations tuần này ánh xạ 1:1 với ràng buộc hardware stage 1:

| Transform (tuần này) | Ràng buộc HW (stage 1) |
|----------------------|------------------------|
| Tiling `linalg.matmul` | SRAM/shared memory nhỏ (tuần 6) |
| Fusion matmul+relu | HBM bandwidth wall (tuần 1) |
| Bufferization + alloc | Scratchpad software-managed (tuần 3) |
| Vectorize → `vector` dialect | SIMD/tensor core width (tuần 2) |

### Output cuối tuần

- Matmul lowering walkthrough hoàn chỉnh (IR dump từng bước + chú thích)
- Transform script tiling 2 cấp + fusion demo
- 1 standalone C++ pass có FileCheck test

---

## TUẦN 11 — Triton: DSL viết kernel GPU

> **Câu hỏi central:** *Triton tự động hóa được gì mà CUDA bắt bạn làm tay — và cái giá phải trả là gì?*

📁 Chi tiết & TODO: [`week11-triton/`](./week11-triton/)

### Lý thuyết (6h)

#### Ngày 1: Triton paper + philosophy

Đọc Tillet et al. (2019), *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*.

Insight cốt lõi: lập trình ở mức **block/tile** thay vì thread. Bạn viết "load tile 128x64, nhân, cộng", compiler lo: thread mapping, shared memory, coalescing, bank conflict, pipelining. So sánh với tuần 2 stage 1 — mọi thứ bạn làm tay trong CUDA tiled matmul, Triton làm hộ.

#### Ngày 2: Triton compile pipeline (góc nhìn compiler engineer)

Triton không chỉ là DSL — nó là MLIR-based compiler:

```
@triton.jit Python AST → Triton IR (MLIR dialect)
  → TritonGPU IR (layout: blocked/mma/dot-operand encodings)
  → LLVM IR → PTX → SASS
```

Dùng `TRITON_KERNEL_DUMP=1` hoặc `triton.compile(...)` artifacts để xem từng tầng IR. **Bạn vừa học MLIR 3 tuần — giờ đọc được IR của Triton.**

### Thực hành (14h)

Làm [tutorials chính thức](https://triton-lang.org/main/getting-started/tutorials/) theo thứ tự, mỗi bài đều benchmark + so với PyTorch:

**Bài tập 11.1 — Vector add + softmax (tutorial 1-2):** nắm `tl.load/store`, mask, `tl.program_id`. Với softmax: hiểu vì sao fused 1-pass nhanh hơn PyTorch eager nhiều lần (đáp án: HBM round-trips — roofline tuần 1!).

**Bài tập 11.2 — Matmul (tutorial 3):** block-level matmul với autotuning configs. So sánh đạt bao nhiêu % cuBLAS. Xem PTX sinh ra có `mma.sync` (tensor core) không.

**Bài tập 11.3 — Flash Attention (project chính):** implement Flash Attention forward bằng Triton. Tuần 6 stage 1 bạn đã làm bản đơn giản — tuần này làm tử tế:

- Online softmax (rescaling trick) đúng chuẩn
- Causal masking
- Benchmark vs `torch.nn.functional.scaled_dot_product_attention` trên seq_len 512→8K
- Đo HBM traffic bằng Nsight → chứng minh O(N) memory thay vì O(N²)

**Bài tập 11.4 — Mổ xẻ IR:** dump Triton IR / TritonGPU IR / PTX của matmul kernel. Viết note: layout encoding trong TritonGPU IR nghĩa là gì, shared memory được chèn ở pass nào.

### Liên hệ HW-SW

Triton là minh chứng cho luận điểm trung tâm của cả lộ trình: **nâng mức trừu tượng đúng cách không làm mất hiệu năng, nếu compiler đủ thông minh và đủ hiểu hardware**. Triton biết Ampere/Hopper microarchitecture (tensor core shapes, shared memory banks, async copy) — nên nó dám hứa "bạn viết tile, tôi lo phần còn lại". Một DSL tương tự cho systolic array sẽ là cảm hứng cho capstone.

### Output cuối tuần

- 4+ Triton kernels có benchmark
- Flash Attention đạt tốc độ cạnh tranh với SDPA + memory analysis
- `triton_ir_notes.md` — pipeline Triton IR → PTX

---

## TUẦN 12 — TVM: Tensor Expressions & Auto-tuning

> **Câu hỏi central:** *Thay vì con người viết schedule (Triton) hay pass cố định (MLIR), có thể để máy TÌM KIẾM ra schedule tối ưu không?*

📁 Chi tiết & TODO: [`week12-tvm/`](./week12-tvm/)

### Lý thuyết (6h)

#### Ngày 1: TVM paper + kiến trúc

Đọc Chen et al. (2018), *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*, OSDI.

Khái niệm trung tâm — **tách compute khỏi schedule** (kế thừa Halide):

```python
# COMPUTE — tính gì (bất biến)
C = te.compute((M, N), lambda i, j: te.sum(A[i,k] * B[k,j], axis=k))
# SCHEDULE — tính thế nào (tìm kiếm được)
s[C].tile(...); s[C].vectorize(...); s[C].parallel(...)
```

Cùng compute + schedule khác nhau = cùng kết quả, hiệu năng chênh 100x. **Không gian schedule là không gian tìm kiếm** → auto-tuning.

#### Ngày 2: Pipeline TVM hiện đại

- Relax (graph IR, thay Relay) → TE → **TensorIR (TIR)** → codegen (LLVM/CUDA/C)
- MetaSchedule: thay AutoTVM/Ansor cũ — sinh candidate schedules, đo trên hardware thật, cost model học dần
- 🎓 Xem CMU 10-414 (Tianqi Chen) các bài về ML compilation — chính tác giả giảng

### Thực hành (14h)

**Bài tập 12.1 — Cài TVM + first compile:** cài từ pip (`apache-tvm`) hoặc build source. Import ResNet18 từ PyTorch qua `relax.frontend.torch`, compile với `target="llvm"` và `"cuda"`, benchmark vs PyTorch eager.

**Bài tập 12.2 — TE/TIR schedule tay:** viết matmul bằng TE, apply thủ công: `tile`, `reorder`, `vectorize`, `parallel`, `cache_read` (shared memory). Đo từng bước một trên GPU — **tự tái hiện lại hành trình naive→tiled của tuần 2 stage 1, nhưng bằng schedule primitives thay vì viết CUDA tay**. Lưu bảng: schedule → GFLOPS.

**Bài tập 12.3 — MetaSchedule auto-tuning:** tune matmul + conv2d với MetaSchedule (~1000 trials), so sánh: naive TIR / schedule tay của bạn / MetaSchedule / cuBLAS-cuDNN. Câu hỏi phải trả lời: **máy tìm ra trick nào mà bạn không nghĩ tới?** (đọc best schedule được export).

**Bài tập 12.4 — So sánh 3 triết lý (essay ngắn nhưng quan trọng):** viết 2 trang so sánh cách đạt performance:

| | MLIR | Triton | TVM |
|---|------|--------|-----|
| Ai quyết định schedule | Pass author (C++/transform) | Kernel author (Python) + autotuner nhỏ | Search + cost model |
| Ưu | Kiểm soát, tái dùng infrastructure | Productivity, đọc được | Đỡ tốn công người, portable |
| Nhược | Tốn công engineering | Chỉ GPU-like targets | Search tốn giờ, khó debug |
| Ai dùng | IREE, XLA-next, chip vendors | OpenAI, PyTorch inductor | Có chỗ đứng ở edge/embedded |

### Liên hệ HW-SW

Auto-tuning tồn tại vì một sự thật khó chịu: **cost model tĩnh không dự đoán nổi hardware hiện đại** (cache, contention, instruction scheduling tương tác phi tuyến). Đo trên silicon thật là ground truth duy nhất. Nhưng chú ý tradeoff: Groq/TPU static scheduling lại **cần** cost model chính xác tuyệt đối — đó là lý do hardware của họ deterministic để cost model có thể đúng. Compiler philosophy và hardware philosophy là một.

### Output cuối tuần

- ResNet18 compile + benchmark notebook
- Bảng schedule-tay vs MetaSchedule vs vendor libs
- Essay "3 triết lý codegen: MLIR vs Triton vs TVM"

---

## TUẦN 13 — XLA & HLO IR

> **Câu hỏi central:** *Compiler production phục vụ hàng triệu TPU-hours mỗi ngày trông như thế nào — và nó khác đồ chơi của mình chỗ nào?*

📁 Chi tiết & TODO: [`week13-xla-hlo/`](./week13-xla-hlo/)

### Lý thuyết (6h)

#### Ngày 1-2: XLA architecture

Đọc docs tại [openxla.org](https://openxla.org/xla/architecture):

- HLO (High Level Operations) — op set nhỏ (~100 ops), ngữ nghĩa chặt, functional
- StableHLO — serialization format chung cho hệ sinh thái (PyTorch/JAX/TF đều emit được)
- Pipeline: HLO optimizations (fusion, algebraic simplification, layout assignment) → backend emitters (TPU: MXU instructions; GPU: Triton/cuDNN calls)

#### Ngày 3: Fusion trong XLA — kỹ nghệ nhất ngành

Đọc về XLA fusion passes (docs + code `xla/service/`): producer-consumer fusion, multi-output fusion, horizontal fusion. XLA fusion là **heuristic-based** (không search như TVM) — đánh giá chi phí bằng analytical model. Vì sao? Compile time của JIT phải tính bằng giây, không phải giờ.

### Thực hành (10h)

**Bài tập 13.1 — JAX → HLO dump:**

```python
import jax, jax.numpy as jnp

def f(x, w1, w2):
    h = jax.nn.relu(x @ w1)
    return jax.nn.softmax(h @ w2)

print(jax.jit(f).lower(x, w1, w2).as_text())          # StableHLO
print(jax.jit(f).lower(x, w1, w2).compile().as_text()) # HLO sau optimize
```

So sánh trước/sau optimize: op nào bị fuse vào `fusion` op? Layout nào bị gán?

**Bài tập 13.2 — Phân tích fusion decisions:** viết 5 hàm JAX với pattern khác nhau (elementwise chain, matmul+bias+relu, reduce theo sau matmul, reshape ở giữa, dynamic slice), dump HLO optimized, lập bảng: pattern nào fuse được, pattern nào không, **suy luận lý do** (gợi ý: reshape và dynamic shapes là kẻ phá fusion).

**Bài tập 13.3 — (Colab TPU, optional) Chạy trên TPU thật:** cùng code, so sánh HLO cho backend CPU vs TPU — thấy layout assignment khác nhau (TPU thích tiled layouts theo MXU 128x128, padding hiện rõ).

**Bài tập 13.4 — Đọc 1 pass XLA thật:** chọn `algebraic_simplifier.cc` (openxla/xla repo), đọc ~30 phút, note 5 rewrite rules thú vị. Đối chiếu với canonicalization patterns của MLIR tuần 9 — cùng một khái niệm.

### Liên hệ HW-SW

HLO cố tình **nhỏ và chặt** (op set ~100, không side effect, shape tĩnh) vì TPU cần static scheduling toàn bộ (tuần 3-4 stage 1: no cache, no speculation). So sánh: PyTorch có ~2000 ops vì eager mode không cần compile. **Op set size là quyết định compiler-hardware co-design, không phải sở thích API.**

### Output cuối tuần

- HLO dumps trước/sau optimize + chú thích
- Bảng phân tích fusion cho 5 patterns
- Note "XLA vs MLIR-generic: op set nhỏ chặt vs dialect mở"

---

## TUẦN 14-16 — Capstone: Mini Compiler PyTorch → Systolic Simulator

> **Câu hỏi central:** *Bạn đã thấy 4 hệ compiler của người khác. Giờ tự xây một cái — mọi mảnh kiến thức 15 tuần qua có khớp thành một hệ thống chạy được không?*

📁 Chi tiết & TODO: [`week14-16-capstone/`](./week14-16-capstone/)

### Đặc tả project

Xây compiler `tinycc` (tên tùy bạn) biên dịch model PyTorch nhỏ → chạy đúng + đo cycle trên **systolic simulator tuần 3 (stage 1)**:

```
PyTorch MLP/CNN nhỏ
    ↓  torch.export / torch.fx        [Frontend]
Graph IR (tự thiết kế, Python)
    ↓  fusion, const-fold, DCE        [Mid-end: graph passes]
Tensor IR (tự thiết kế: ops + tiles)
    ↓  tiling theo array size,        [Mid-end: tensor passes]
    ↓  memory planning cho scratchpad
Instruction stream (ISA tự định nghĩa)
    ↓                                  [Backend: codegen]
Systolic simulator (tuần 3) — upgraded: scratchpad + DMA + cycle count
```

**Chọn scope theo sức** (khuyến nghị mức 2):

- **Mức 1 (tối thiểu):** MLP (matmul + relu + bias), fusion matmul+relu, tiling 1 cấp, greedy memory planner
- **Mức 2 (chuẩn):** + conv2d (lower thành im2col matmul), double buffering DMA/compute overlap, cost report so sánh fused vs unfused
- **Mức 3 (tham vọng):** + quantization pass INT8 (tái dùng tuần 5 stage 1), hoặc viết bằng MLIR out-of-tree dialect thay vì Python thuần

### Tuần 14 — Frontend + Graph IR + graph passes

- Ngày 1-2: `torch.export` một MLP → duyệt FX graph → dịch sang Graph IR tự thiết kế (node = op, edge = tensor, có shape + dtype). **Thiết kế IR trên giấy trước khi code** — in ra được dạng text (bài học tuần 7: dump được là debug được).
- Ngày 3-4: 3 graph passes: shape inference, constant folding, **fusion matmul+bias+relu thành 1 fused op** (pattern matcher đơn giản).
- Ngày 5: Interpreter tham chiếu cho Graph IR (chạy bằng NumPy) — **golden reference để test mọi pass về sau**. Property test: mọi pass phải giữ nguyên output.

### Tuần 15 — Tensor IR + tiling + memory planning

- Ngày 1-2: Nâng cấp systolic simulator: thêm scratchpad SRAM (kích thước cấu hình được, ví dụ 256KB), DMA engine (HBM↔scratchpad, có cycle cost), ISA dạng: `DMA_LOAD tile → LOAD_WEIGHTS → MATMUL → DMA_STORE`, cycle counter + utilization report.
- Ngày 3-4: Tiling pass: matmul lớn → tile vừa array (ví dụ 16x16) + vừa scratchpad. Xử lý padding khi shape không chia hết (bài học TPU tuần 3). Conv2d → im2col → matmul (mức 2).
- Ngày 5: Memory planner: gán offset scratchpad cho từng tile buffer, phát hiện khi nào 2 buffer sống cùng lúc (liveness đơn giản), double buffering cho DMA overlap (mức 2).

### Tuần 16 — Codegen + evaluation + writeup

- Ngày 1-2: Codegen: Tensor IR → instruction stream → chạy trên simulator → so output vs golden reference (sai số < 1e-4).
- Ngày 3: **Evaluation matrix** — phần giá trị nhất của portfolio:

| Config | Cycles | Utilization | HBM traffic |
|--------|--------|-------------|-------------|
| Không fusion, không double-buffer | ? | ? | ? |
| + Fusion | ? | ? | ? |
| + Double buffering | ? | ? | ? |
| Tile size sweep (8/16/32/64) | ? | ? | ? |

- Ngày 4-5: Writeup dài (blog-quality, 2000+ từ): kiến trúc compiler, mỗi pass giải quyết ràng buộc hardware nào, số liệu, 3 điều sẽ làm khác đi. **Đây là tài liệu bạn mang đi phỏng vấn.**

### Liên hệ HW-SW

Capstone là nơi mọi intuition 4 giai đoạn giao nhau: fusion của bạn tồn tại vì HBM traffic (tuần 1), tiling vì scratchpad (tuần 6), padding vì array cố định (tuần 3), double buffering vì DMA độc lập compute (tuần 6), và toàn bộ cấu trúc pass pipeline là những gì tuần 7-13 dạy. **Nếu evaluation matrix cho thấy fusion giảm HBM traffic và double buffering tăng utilization — bạn đã chứng minh được bằng số liệu điều mà cả lộ trình khẳng định bằng lời.**

### Output capstone

- Repo compiler hoàn chỉnh: frontend / passes / codegen / simulator / tests
- Evaluation matrix với số liệu thật
- Writeup 2000+ từ — portfolio piece chính của cả lộ trình

---

## TỔNG KẾT GIAI ĐOẠN 2

### Kiểm tra kiến thức

Trả lời không nhìn note:

1. Vì sao MLIR dùng nhiều dialect thay vì 1 IR to? Thông tin gì mất đi khi lower sớm?
2. `tensor` khác `memref` thế nào? Bufferization quyết định gì?
3. Mô tả dialect conversion framework: legal/illegal op, partial vs full conversion.
4. `linalg.generic` biểu diễn op bằng những thành phần nào? Vì sao "structured" giúp transform?
5. Triton che giấu những quyết định nào khỏi người viết kernel? Nó vẫn bắt bạn quyết định gì?
6. Compute/schedule separation của TVM là gì? Vì sao nó biến optimization thành search?
7. Vì sao XLA fusion dùng heuristic còn TVM dùng search? Ràng buộc nào tạo khác biệt?
8. Trong capstone của bạn: nếu scratchpad giảm 1 nửa, pass nào phải thay đổi và thay đổi thế nào?
9. Op set nhỏ (HLO ~100) vs dialect mở (MLIR) — tradeoff?
10. Fusion matmul+relu tiết kiệm chính xác bao nhiêu bytes HBM cho shape [M,N]? (viết công thức)

### Câu hỏi phỏng vấn mẫu (Giai đoạn 2 level)

- "Walk me through what happens when torch.compile compiles a model" → Dynamo capture → FX → Inductor → Triton
- "How would you add a new op to an MLIR-based compiler?" → ODS, verifier, lowering pattern, test
- "When would you NOT fuse two ops?" → register pressure, recompute cost, shape mismatch, fusion phá tiling tốt hơn
- "Design a compiler for a chip with 1MB scratchpad and a 128x128 MXU" → chính là capstone của bạn, kể lại với số liệu

### Output tổng cộng giai đoạn 2

- Toy calculator + Toy MLIR compiler (7 chương + extensions)
- Matmul lowering walkthrough + standalone MLIR pass C++
- Flash Attention Triton cạnh tranh SDPA
- TVM auto-tuning study + essay 3 triết lý
- HLO fusion analysis
- **Mini compiler end-to-end với evaluation matrix** — portfolio chính

---

## Tài liệu reference cho toàn giai đoạn 2

### Sách

- Cooper & Torczon, *Engineering a Compiler* (3rd ed.) — tuần 7 + tra cứu
- (Optional) *SSA-based Compiler Design* (Rastello) — nếu muốn sâu SSA

### Paper bắt buộc

- Lattner et al. 2021 — MLIR (CGO)
- Tillet et al. 2019 — Triton (MAPL)
- Chen et al. 2018 — TVM (OSDI)
- Ragan-Kelley et al. 2013 — Halide (PLDI) — nguồn gốc compute/schedule separation
- (Optional) Vasilache et al. 2022 — Structured Ops / Linalg design

### Docs & tutorials

- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) — xương sống tuần 9
- [MLIR Transform Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/transform/)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [TVM docs](https://tvm.apache.org/docs/) + MetaSchedule
- [OpenXLA architecture](https://openxla.org/xla/architecture)
- [Jeremy Kun — MLIR for Beginners series](https://www.jeremykun.com/2023/08/10/mlir-getting-started/) — blog series thực hành rất tốt

### Khóa học

- 🎓 CMU 10-414/714 *Deep Learning Systems* (Tianqi Chen) — nửa sau khóa về compilation
- 🎥 MLIR talks — LLVM Dev Meeting playlist (Mehdi Amini, Chris Lattner, River Riddle)

### Cộng đồng

- 💬 LLVM Discourse (discourse.llvm.org) — mục MLIR, đọc RFC là cách học design thinking
- 💬 Triton Discord, TVM Discuss
- 📅 Hội nghị: CGO, PLDI, MLSys — đọc proceedings gần nhất

---

## Lời khuyên trước khi đi tiếp

**Tuần 9-10 là dốc nhất.** MLIR có learning curve gắt (TableGen, C++ templates, CMake). Kẹt 1-2 ngày là bình thường — hỏi trên LLVM Discourse, người ta trả lời tử tế. Đừng bỏ qua để "học sau".

**Capstone quan trọng hơn hoàn hảo từng tuần.** Nếu trễ tiến độ, cắt tuần 13 xuống 2 ngày (XLA chỉ cần đọc-hiểu) để bảo toàn 3 tuần capstone. Một compiler chạy được end-to-end + số liệu > mọi thứ khác trong CV.

**Dump IR mọi lúc.** Kỹ năng nghề nghiệp số 1 của compiler engineer là nhìn IR trước/sau pass và giải thích khác biệt. `--mlir-print-ir-after-all` là bạn thân.

**Khi sang Giai đoạn 3**, compiler của bạn sinh ra instruction stream — nhưng ai cấp phát memory thật, ai đẩy lệnh xuống queue, ai sync? Đó là runtime. Capstone sẽ được tái dùng tiếp: Giai đoạn 3 xây mini runtime cho chính simulator này.

---

*File này thuộc series lộ trình ML Compiler Engineer. Giai đoạn trước: [Giai đoạn 1 — Kiến trúc accelerator](../stage1_Accelerator/README.md). Giai đoạn tiếp theo: **Giai đoạn 3 — Runtime, driver, kernel programming** (6 tuần).*
