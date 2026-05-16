# 🧠 Machine Learning Compiler Engineer Roadmap

> **Lộ trình tự học từ con số không đến Machine Learning Compiler Engineer cho AI accelerator chips (TPU/NPU/ASIC).**
> Hành trình cá nhân, mở để bất kỳ ai cùng quan tâm tham khảo, fork, hoặc đồng hành.

[![Status](https://img.shields.io/badge/status-in_progress-yellow)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Language](https://img.shields.io/badge/lang-VN%2FEN-red)](https://github.com)

---

## 📌 Tổng quan

Machine Learning Compiler (ML Compiler) là **cây cầu giữa mô hình AI (PyTorch/JAX) và silicon chuyên dụng (TPU/NPU/ASIC)**. Khi mỗi chip AI mới ra đời cần toàn bộ software stack riêng, ML Compiler Engineer là vai trò chuyên biệt, lương cao, và nhu cầu tuyển dụng đang tăng mạnh (Google, NVIDIA, Tenstorrent, Groq, Cerebras, AWS Trainium, Apple, Huawei, Modular, v.v.).

Repo này là **lộ trình 4 giai đoạn** mình tự thiết kế để đi từ nền tảng kiến trúc đến trở thành compiler engineer chuyên về AI chip. Toàn bộ project, code, note, và blog post sẽ được public ở đây.

### 🎯 Mục tiêu cuối cùng

Sau khi hoàn thành lộ trình, mình có thể:

- Đọc và hiểu paper kiến trúc của bất kỳ AI chip nào (TPU, Tenstorrent, Groq, Cerebras, Ascend...)
- Đóng góp code cho LLVM/MLIR, IREE, TVM, hoặc Triton ở mức meaningful
- Thiết kế và implement một backend compiler cho một AI accelerator giả lập từ đầu đến cuối
- Apply được vào vị trí ML Compiler Engineer ở các công ty chip AI

### 🛠 Hardware tận dụng

- **NVIDIA GPU** — học CUDA, Triton, profiling với Nsight
- **Luckfox Pico Plus (RV1103 NPU 0.5 TOPS)** — chạm tay vào NPU thật, dùng RKNN toolkit
- (Optional) Google Colab TPU — thử nghiệm XLA/JAX

---

## 🗺️ Bản đồ lộ trình

```
┌────────────────────────────────────────────────────────────────┐
│                  ML COMPILER ENGINEER ROADMAP                  │
└────────────────────────────────────────────────────────────────┘

Giai đoạn 1: Kiến trúc accelerator         (6 tuần)
     ↓        Hiểu HARDWARE — vì sao AI chip thiết kế khác CPU/GPU

Giai đoạn 2: Compiler & code generation    (10 tuần)
     ↓        Học MLIR sâu + TVM — IR, passes, lowering, codegen

Giai đoạn 3: Runtime, driver, kernel       (6 tuần)
     ↓        Code sát silicon — CUDA, runtime, memory, driver

Giai đoạn 4: Framework integration         (4 tuần)
              Đưa compiler ra production — PyTorch, ONNX, serving

Tổng: ~26 tuần (~6 tháng full-time hoặc 10-12 tháng part-time)
```

---

## 📖 Triết lý xuyên suốt

Mỗi giai đoạn, mỗi bài tập đều xoay quanh **một câu hỏi**:

> 🔑 *"Phần cứng có đặc tính gì khiến compiler phải làm thế này?"*
> 🔑 *"Nếu compiler không làm thế này, phần cứng sẽ bị lãng phí ở đâu?"*

Mọi optimization của compiler là phản ứng với một ràng buộc hardware. Không hiểu hardware → optimize sai. Không hiểu framework → developer không dùng được. ML compiler engineer phải đứng giữa, hiểu cả hai.

---

## 📚 Giai đoạn 1 — Nền tảng kiến trúc accelerator (6 tuần)

> Hiểu **tại sao** AI chip được thiết kế khác CPU/GPU, và compiler phải làm gì để khai thác hardware đó.

### Nội dung chính

| Tuần | Chủ đề | Output |
|------|--------|--------|
| 1 | Roofline model & memory wall | Roofline plots cho GPU của mình |
| 2 | CPU SIMD vs GPU SIMT | AVX matmul + 3 versions CUDA matmul + Nsight profile |
| 3 | TPU & Systolic Array | Systolic array simulator từ đầu (Python) |
| 4 | Dataflow architectures: Tenstorrent, Groq, Cerebras | Eyeriss row-stationary simulator + comparison analysis |
| 5 | Numerical formats & quantization | INT8 quantization từ đầu + ResNet18 PTQ |
| 6 | Memory hierarchy + **Luckfox NPU end-to-end** | Tiled matmul 3 cấp + Flash Attention + RKNN deployment |

### Kỹ năng đạt được

- ✅ Đọc paper kiến trúc AI chip (TPU, Eyeriss, Groq) không bị ngợp
- ✅ Phân tích roofline cho bất kỳ kernel nào, dự đoán bottleneck
- ✅ Profile CUDA kernel với Nsight Compute, hiểu metric
- ✅ Viết được systolic array simulator có thể tile matmul lớn
- ✅ Chạy được model thật trên NPU (Rockchip RV1103) qua RKNN compiler

### Tài liệu

- 📘 *Computer Architecture: A Quantitative Approach* (Hennessy & Patterson)
- 📘 *Programming Massively Parallel Processors* (Kirk & Hwu)
- 📘 *Efficient Processing of Deep Neural Networks* (Sze et al.) — free PDF
- 📄 Roofline paper (Williams 2009), TPU paper (Jouppi 2017), Eyeriss (Chen 2016), Quantization (Jacob 2018), FlashAttention (Dao 2022)
- 🎥 MIT 6.5940 *TinyML and Efficient Deep Learning* (Han Song)

📁 **Chi tiết**: [`stage1_Accelerator/README.md`](./stage1_Accelerator//README.md)

---

## 🔧 Giai đoạn 2 — Compiler & code generation cho AI (10 tuần)

> Học MLIR sâu — IR infrastructure đang định hình mọi compiler chip AI hiện đại. Sau đó học TVM để có góc nhìn so sánh.

### Nội dung chính

| Tuần | Chủ đề | Output |
|------|--------|--------|
| 7 | Compiler fundamentals refresher | Toy calculator compiler |
| 8 | Giới thiệu MLIR + setup | LLVM/MLIR build từ source |
| 9 | MLIR Toy tutorial (7 chương) | Toy language compiler hoàn chỉnh |
| 10 | MLIR ML dialects: linalg, tensor, memref, affine | Matmul lowering pipeline tay |
| 11 | Triton — DSL viết kernel GPU dễ hơn CUDA | Flash Attention bằng Triton |
| 12 | TVM — Tensor expressions & auto-tuning | TVM model compile + MetaSchedule tuning |
| 13 | XLA & HLO IR | JAX function → HLO dump & phân tích |
| 14-16 | **Capstone**: Mini compiler PyTorch → systolic simulator | End-to-end compiler với fusion, tiling, codegen |

### Kỹ năng đạt được

- ✅ Đọc và viết MLIR (dialect, op, pass, region)
- ✅ Implement một pass MLIR đơn giản (fusion, tiling, lowering)
- ✅ Viết kernel Triton hiệu năng cao (Flash Attention level)
- ✅ Hiểu TVM Relay → TE → TIR pipeline
- ✅ Build end-to-end mini compiler cho 1 accelerator giả lập

### Tài liệu

- 📘 *Engineering a Compiler* (Cooper & Torczon) — nếu cần ôn compiler cơ bản
- 📄 MLIR paper (Lattner et al. 2021), TVM paper (Chen et al. 2018), Triton paper (Tillet 2019)
- 📺 MLIR Tutorial — LLVM Developer Meeting (Mehdi Amini)
- 🎓 CMU 10-414 *Deep Learning Systems* (Tianqi Chen — tác giả TVM)
- 🔗 Toy Tutorial chính thức: [mlir.llvm.org/docs/Tutorials/Toy](https://mlir.llvm.org/docs/Tutorials/Toy/)

📁 **Chi tiết**: [`stage2_CompilerCodegen/README.md`](./stage2_CompilerCodegen/) *(coming soon)*

---

## ⚙️ Giai đoạn 3 — Runtime, driver, kernel programming (6 tuần)

> Tầng sát phần cứng: cách compiler-generated code thực thi qua runtime, driver, và scheduler.

### Nội dung chính

| Tuần | Chủ đề | Output |
|------|--------|--------|
| 17 | CUDA programming sâu (CUTLASS-style) | Hand-written matmul đạt ~80% cuBLAS |
| 18 | Runtime concepts: command queue, stream, async exec | Mini runtime cho systolic simulator |
| 19 | Driver basics: PCIe, DMA, MMIO, interrupts | Đọc & note Tenstorrent tt-metalium driver |
| 20 | Memory management: pools, allocators, fragmentation | Tensor allocator hỗ trợ multi-stream |
| 21 | Multi-device: NCCL, collective ops | Implement all-reduce ring algorithm |
| 22 | Profiling & debugging | Microbenchmark suite + custom roofline tool |

### Kỹ năng đạt được

- ✅ Viết CUDA kernel đạt 70-80% cuBLAS performance
- ✅ Hiểu cấu trúc một runtime cho accelerator: queue, sync, memory pool
- ✅ Đọc và debug driver-level code (PCIe, DMA, MMIO)
- ✅ Implement collective communication primitives (all-reduce, all-gather)
- ✅ Sử dụng profiler thành thạo (Nsight Systems, Nsight Compute)

### Tài liệu

- 📘 *CUDA C++ Programming Guide* (NVIDIA official)
- 📘 *Programming Massively Parallel Processors* — chapters về advanced patterns
- 🔗 [CUTLASS](https://github.com/NVIDIA/cutlass) — kernel template library của NVIDIA
- 🔗 [Tenstorrent tt-metalium](https://github.com/tenstorrent/tt-metal) — open source full stack cho AI chip thật
- 📄 NCCL paper, ring all-reduce algorithm

📁 **Chi tiết**: [`stage3_Runtime/README.md`](./stage3_Runtime) *(coming soon)*

---

## 🌐 Giai đoạn 4 — Framework integration & deployment (4 tuần)

> Đưa compiler ra thế giới: tích hợp vào PyTorch/ONNX, deploy production-grade, hiểu LLM serving stack hiện đại.

### Nội dung chính

| Tuần | Chủ đề | Output |
|------|--------|--------|
| 23 | PyTorch backend: torch.compile, PT2 export, Dynamo | Custom PT2 backend cho mini compiler tự xây |
| 24 | ONNX Runtime Execution Provider (EP) | EP cho 1 chip giả lập, chạy ResNet50 |
| 25 | Quantization toolkits production: PyTorch quant, ONNX QDQ | Pipeline FP32 → INT8 end-to-end với calibration |
| 26 | LLM serving: TensorRT-LLM, vLLM, paged attention | Phân tích serving stack + benchmark throughput/latency |

### Kỹ năng đạt được

- ✅ Tích hợp compiler riêng vào PyTorch 2.x qua Dynamo
- ✅ Viết ONNX Execution Provider cho 1 backend mới
- ✅ Hiểu pipeline LLM serving đầy đủ (batching, KV cache, paged attention, continuous batching)
- ✅ Deploy production-grade model với quantization tự động

### Tài liệu

- 🔗 [PyTorch 2.x Documentation — torch.compile](https://pytorch.org/docs/stable/torch.compiler.html)
- 🔗 [ONNX Runtime EP Tutorial](https://onnxruntime.ai/docs/execution-providers/)
- 🔗 [vLLM](https://github.com/vllm-project/vllm) — đọc source code paged attention
- 📄 Paged Attention paper (Kwon et al. 2023), Continuous Batching (Yu et al. 2022)
- 🔗 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

📁 **Chi tiết**: [`stage4_Frameworks/README.md`](./stage4_Frameworks/) *(coming soon)*

---

## 🎯 Sau lộ trình — Compiler Engineer cho AI Chip

Sau khi hoàn thành 4 giai đoạn, bạn có đủ nền tảng để apply vào vai trò ML Compiler Engineer. Vài hướng đi tiếp:

### Chọn chuyên sâu (specialization)

**Option A — Compiler infrastructure (broad)**
> Sâu về MLIR, contribute LLVM project, làm việc trên multiple backends.
> Phù hợp: Google XLA, Modular, OpenXLA team

**Option B — Performance/kernel engineer**
> Specialize CUTLASS, Triton, FlashAttention variants. Tay viết kernel level world-class.
> Phù hợp: NVIDIA, AMD, Together AI

**Option C — Chip-specific compiler**
> Pick 1 chip (Tenstorrent, Groq, Cerebras) và đi sâu. Contribute full stack.
> Phù hợp: chính công ty chip đó

### Open source contribution để xây CV

Các project nên contribute (theo độ khó tăng dần):

1. **Tenstorrent tt-metal** — good first issues nhiều, community thân thiện
2. **IREE** — MLIR-based compiler của Google, nhiều issue mở
3. **Triton** — popular, contribute kernel optimizations
4. **TVM** — mature, nhiều surface area để contribute
5. **LLVM/MLIR** — high bar nhưng prestige cao

### Chuẩn bị phỏng vấn

- System design: design XLA-like compiler cho 1 chip giả định
- Coding: LeetCode medium-hard + graph algorithms (DAG scheduling, register allocation)
- Deep dive: chuẩn bị defend được 1-2 paper bạn đọc kỹ
- Behavioral: stories về debug performance, contribute open source

---

## 📊 Tiến độ cá nhân

| Giai đoạn | Trạng thái | Bắt đầu | Kết thúc | Output chính |
|-----------|-----------|---------|----------|--------------|
| 1. Kiến trúc accelerator | 🔄 In progress | TBD | TBD | Systolic simulator + Luckfox NPU deployment |
| 2. Compiler & codegen | ⏳ Pending | - | - | MLIR end-to-end compiler |
| 3. Runtime & driver | ⏳ Pending | - | - | Custom runtime + CUDA kernels |
| 4. Framework integration | ⏳ Pending | - | - | PyTorch backend + ONNX EP |

Legend: ✅ Done · 🔄 In progress · ⏳ Pending · ⏸ Paused

---

## 📂 Cấu trúc repo

```
ml-compiler-roadmap/
├── README.md                          # File này
├── stage1_Accelerator/                       # Nền tảng kiến trúc accelerator
│   ├── README.md                      # Chi tiết giai đoạn (6 tuần)
│   ├── week1-roofline/
│   ├── week2-cpu-gpu/
│   ├── week3-systolic/
│   ├── week4-architectures/
│   ├── week5-quantization/
│   └── week6-memory-tiling-npu/
├── stage2_CompilerCodegen/                       # Compiler & code generation (10 tuần)
│   ├── README.md
│   ├── week7-compiler-basics/
│   ├── week8-mlir-intro/
│   ├── week9-mlir-toy/
│   ├── week10-ml-dialects/
│   ├── week11-triton/
│   ├── week12-tvm/
│   ├── week13-xla-hlo/
│   └── week14-16-capstone/
├── stage3_Runtime/                       # Runtime, driver, kernel (6 tuần)
│   ├── README.md
│   ├── week17-cuda-deep/
│   ├── week18-runtime/
│   ├── week19-driver/
│   ├── week20-memory/
│   ├── week21-multi-device/
│   └── week22-profiling/
├── stage4_Frameworks/                       # Framework integration (4 tuần)
│   ├── README.md
│   ├── week23-pytorch-backend/
│   ├── week24-onnx-ep/
│   ├── week25-quantization-prod/
│   └── week26-llm-serving/
├── blog/                              # Blog posts theo tuần
├── papers/                            # Notes của paper đã đọc
└── resources/                         # Tài liệu chung
```

Mỗi giai đoạn có README riêng với chi tiết từng tuần: lý thuyết, thực hành, code, output, liên hệ HW-SW.

---

## 🎓 Tài liệu reference chung

### Sách nền tảng

- 📘 Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* (6th ed.)
- 📘 Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed.)
- 📘 Cooper & Torczon, *Engineering a Compiler* (3rd ed.)
- 📘 Sze, Chen, Yang, Emer, *Efficient Processing of Deep Neural Networks* — **free PDF**

### Paper kinh điển (must-read)

- Williams et al. 2009 — Roofline Model
- Jouppi et al. 2017 — TPU
- Chen et al. 2016 — Eyeriss
- Lattner et al. 2021 — MLIR
- Chen et al. 2018 — TVM
- Tillet et al. 2019 — Triton
- Dao et al. 2022 — FlashAttention
- Jacob et al. 2018 — Quantization for Integer Inference
- Kwon et al. 2023 — Paged Attention (vLLM)

### Khóa học miễn phí

- 🎓 MIT 6.5940 — TinyML and Efficient Deep Learning (Han Song)
- 🎓 Stanford CS217 — Hardware Accelerators for Machine Learning
- 🎓 CMU 10-414/714 — Deep Learning Systems (Tianqi Chen)
- 🎓 Stanford CS143 — Compilers (cơ bản)

### Cộng đồng

- 💬 [LLVM Discord](https://discord.gg/xS7Z362) — MLIR channel
- 💬 [Triton Discord](https://discord.com/invite/AeFvBV3) — OpenAI Triton
- 💬 [Tenstorrent Discord](https://discord.gg/tenstorrent) — tt-metal contributors
- 💬 r/MachineLearning — Reddit
- 🐦 Twitter/X: theo dõi [@Tianqi_Chen](https://twitter.com/tqchenml), [@tri_dao](https://twitter.com/tri_dao), [@cHHillee](https://twitter.com/chhillee) (Horace He), [@clattner_llvm](https://twitter.com/clattner_llvm)
- 📅 Hội nghị nên theo dõi: MLSys, ASPLOS, PLDI, CGO, HPCA, ISCA

---

## ✍️ Blog posts

Mỗi tuần mình viết 1 blog post (~500-1000 từ) tổng kết bài học. Liệt kê ở đây khi xuất bản:

- *(coming soon)*

---

## 🤝 Đóng góp

Repo này là journey cá nhân, nhưng welcome:

- **Discussion**: mở issue nếu bạn có câu hỏi, gợi ý, hoặc muốn thảo luận về 1 chủ đề
- **Tài liệu**: PR nếu bạn biết resource hay mà mình bỏ sót
- **Đồng hành**: nếu bạn đang theo lộ trình tương tự, ping mình để cùng học!

---

## 📧 Liên hệ

- GitHub: *[your-username]*
- Email: *[your-email]*
- LinkedIn: *[your-linkedin]*

---

## 📜 License

MIT — code và writeup free để fork, học hỏi, chia sẻ. Chỉ cần credit là vui rồi.

---

## ⭐ Lời cuối

> "Compiler không tạo ra phép tính mới — compiler loại bỏ data movement không cần thiết."

Lộ trình này dài (~6 tháng full-time, hoặc 10-12 tháng nếu vừa học vừa làm). Nhưng phần thưởng là một skill set hiếm và có giá trị — đứng giữa AI hardware và software, hiểu cả hai sâu, thiết kế cây cầu giữa chúng.

Nếu bạn tình cờ tìm đến repo này và đang cân nhắc cùng hướng, mình tin: **the field is wide open, the chips keep coming, and someone has to make them sing**.

🚀 *Let's build.*