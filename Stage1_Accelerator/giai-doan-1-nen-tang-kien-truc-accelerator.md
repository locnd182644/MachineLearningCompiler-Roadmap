# GIAI ĐOẠN 1 — NỀN TẢNG KIẾN TRÚC ACCELERATOR

> **Lộ trình ML Compiler Engineer cho AI chip**
> Thời lượng: 6 tuần · ~15-20h/tuần
> Mục tiêu nghề: Compiler Engineer cho AI accelerator (TPU/NPU/ASIC)
> Hardware có sẵn: NVIDIA GPU + Luckfox Pico Plus (RV1103 NPU 0.5 TOPS)

---

## Mục lục

- [Phần 0 — Bức tranh toàn cảnh](#phần-0--bức-tranh-toàn-cảnh-ml-compiler-là-gì-và-tại-sao-tồn-tại)
  - [Vấn đề cốt lõi mà ML compiler giải quyết](#01-vấn-đề-cốt-lõi-mà-ml-compiler-giải-quyết)
  - [Stack đầy đủ và bạn ở đâu trong đó](#02-stack-đầy-đủ-và-bạn-ở-đâu-trong-đó)
  - [Nguyên lý xuyên suốt: liên hệ phần cứng ↔ phần mềm](#03-nguyên-lý-xuyên-suốt-liên-hệ-phần-cứng--phần-mềm)
  - [Roofline Model — Công cụ tư duy quan trọng nhất](#04-roofline-model--công-cụ-tư-duy-quan-trọng-nhất)
  - [Bốn nguyên tắc thiết kế của AI accelerator](#05-bốn-nguyên-tắc-thiết-kế-của-ai-accelerator)
  - [Mục tiêu nghề: Compiler Engineer cho AI chip](#06-mục-tiêu-nghề-compiler-engineer-cho-ai-chip)
- [Tổng quan giai đoạn 1](#tổng-quan-giai-đoạn-1)
- [Tuần 1 — Performance Modeling: Roofline & Memory Wall](#tuần-1--performance-modeling-roofline--memory-wall)
- [Tuần 2 — CPU SIMD & GPU SIMT: Baseline để so sánh](#tuần-2--cpu-simd--gpu-simt-baseline-để-so-sánh)
- [Tuần 3 — TPU & Systolic Array: Case study quan trọng nhất](#tuần-3--tpu--systolic-array-case-study-quan-trọng-nhất)
- [Tuần 4 — Dataflow & Spatial Architectures: Vượt khỏi TPU](#tuần-4--dataflow--spatial-architectures-vượt-khỏi-tpu)
- [Tuần 5 — Numerical Formats & Quantization](#tuần-5--numerical-formats--quantization)
- [Tuần 6 — Memory Hierarchy & Project tích hợp + Luckfox NPU](#tuần-6--memory-hierarchy--project-tích-hợp--luckfox-npu)
- [Tổng kết Giai đoạn 1](#tổng-kết-giai-đoạn-1)
- [Tài liệu reference](#tài-liệu-reference-cho-toàn-giai-đoạn-1)

---

## PHẦN 0 — BỨC TRANH TOÀN CẢNH: ML COMPILER LÀ GÌ VÀ TẠI SAO TỒN TẠI

### 0.1. Vấn đề cốt lõi mà ML compiler giải quyết

Để hiểu ML compiler, phải hiểu **bài toán kinh tế** của nó trước, không phải kỹ thuật.

Có **3 thực tế** va vào nhau tạo ra ngành này:

**Thực tế 1: Mô hình AI ngày càng lớn theo cấp số nhân.** GPT-2 (2019) có 1.5B params. GPT-4 ước tính ~1.7T. Compute demand tăng ~10x mỗi năm. Không một CPU/GPU đa năng nào đáp ứng nổi về chi phí và năng lượng.

**Thực tế 2: Định luật Moore đã chậm lại, nhưng định luật Dennard scaling thì đã chết từ 2006.** Nghĩa là: tăng transistor không còn đồng nghĩa tăng clock speed mà không tăng điện năng. Cách duy nhất để tăng hiệu năng/watt là **specialization** — làm chip chỉ giỏi 1 việc.

**Thực tế 3: Mỗi chip chuyên dụng có ISA (instruction set) riêng.** TPU không chạy được CUDA. Tenstorrent không chạy được PyTorch native. Mỗi chip mới ra đời cần **toàn bộ software stack mới** để các nhà phát triển AI có thể dùng nó mà không phải viết lại model.

Đây là chỗ ML compiler xuất hiện: **nó là cây cầu giữa "PyTorch model" của data scientist và "instruction stream" mà silicon hiểu được**.

```
Data scientist viết:        Compiler engineer làm:        Chip thực thi:

model = ResNet50()    →    [Graph capture]
y = model(x)               [Optimization]            →    Tensor instructions
                           [Scheduling]                    Memory commands
                           [Code generation]               DMA transfers
                                                           Sync primitives
```

Nếu compiler dở, một chip lý thuyết 1000 TFLOPS có thể chỉ đạt 100 TFLOPS thực tế. Compiler engineer là người **biến silicon đắt tiền thành hiệu năng thực**. Đây là lý do lương rất cao và nhu cầu tuyển dụng cực lớn.

### 0.2. Stack đầy đủ và bạn ở đâu trong đó

```
┌──────────────────────────────────────────────────────────────┐
│ USER LAYER                                                   │
│ • PyTorch / JAX / TensorFlow user code                       │
│ • Hugging Face transformers, vLLM, etc.                      │
└──────────────────────────────────────────────────────────────┘
                            ↓ (frontend capture)
┌──────────────────────────────────────────────────────────────┐
│ GRAPH IR LAYER  ← "what to compute"                          │
│ • torch.fx, JAX Jaxpr, TF Graph, ONNX                        │
│ • Stable HLO, Relay (TVM), Torch IR                          │
│ Optimizations: const folding, CSE, dead code, op fusion      │
└──────────────────────────────────────────────────────────────┘
                            ↓ (graph lowering)
┌──────────────────────────────────────────────────────────────┐
│ TENSOR IR LAYER  ← "how to compute, abstractly"              │
│ • Linalg dialect (MLIR), TVM TE/TIR, Triton IR               │
│ Optimizations: tiling, fusion, layout transform, vectorize   │
└──────────────────────────────────────────────────────────────┘
                            ↓ (scheduling + lowering)
┌──────────────────────────────────────────────────────────────┐
│ LOW-LEVEL IR LAYER  ← "concrete loops & memory"              │
│ • Affine/SCF dialects, LLVM IR, PTX, SPIR-V                  │
│ Optimizations: register alloc, instr scheduling, pipelining  │
└──────────────────────────────────────────────────────────────┘
                            ↓ (codegen)
┌──────────────────────────────────────────────────────────────┐
│ MACHINE CODE / ISA                                           │
│ • CUDA SASS, TPU XLA HLO ops, Tenstorrent kernel bins        │
└──────────────────────────────────────────────────────────────┘
                            ↓ (runtime executes)
┌──────────────────────────────────────────────────────────────┐
│ RUNTIME LAYER  ← Giai đoạn 3                                 │
│ • Memory allocator, command queue, stream sync, DMA          │
│ • Multi-device: NCCL, collective ops                         │
└──────────────────────────────────────────────────────────────┘
                            ↓ (hardware control)
┌──────────────────────────────────────────────────────────────┐
│ DRIVER & FIRMWARE                                            │
│ • PCIe, MMIO, interrupts, kernel module                      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ HARDWARE                                                     │
│ • Compute cores (systolic / SIMD / dataflow)                 │
│ • Memory hierarchy (HBM, SRAM, registers)                    │
│ • Interconnect (NoC, NVLink, ICI)                            │
└──────────────────────────────────────────────────────────────┘
```

**ML Compiler Engineer làm việc chủ yếu ở 3 tầng giữa** (Graph IR, Tensor IR, Low-level IR). Nhưng **phải hiểu cả tầng dưới (hardware) lẫn tầng trên (framework)** vì:

- Không hiểu hardware → optimize sai → kernel chậm
- Không hiểu framework → bị developer khiếu nại tại sao model không compile được

### 0.3. Nguyên lý xuyên suốt: liên hệ phần cứng ↔ phần mềm

Đây là **kim chỉ nam** cho toàn bộ lộ trình. Mỗi quyết định compiler là phản chiếu của một ràng buộc phần cứng. Bạn phải tập thói quen luôn hỏi 2 câu khi học bất cứ thứ gì:

> **"Phần cứng có đặc tính gì khiến compiler phải làm thế này?"**
> **"Nếu compiler không làm thế này, phần cứng sẽ bị lãng phí ở đâu?"**

Vài ví dụ minh họa nguyên lý này:

| Compiler optimization | Phần cứng motivation |
|----------------------|---------------------|
| Operator fusion (matmul+relu) | HBM bandwidth thấp hơn compute ~50x → giảm round-trip qua HBM |
| Tiling | SRAM/cache nhỏ hơn tensor → phải chia nhỏ vừa SRAM |
| Layout transformation (NCHW→NHWC) | Memory access pattern phải coalesced để dùng hết bandwidth |
| Quantization (FP32→INT8) | Compute unit có tensor core INT8 4x nhanh hơn FP32 |
| Double buffering | DMA và compute là 2 unit độc lập → chạy song song |
| Software pipelining | Pipeline phần cứng có depth N → nạp đủ work để không stall |
| Sparsity exploitation | Hardware có sparse tensor core (NVIDIA Ampere 2:4 sparsity) |

**Mỗi optimization là một bài toán "matching": tính chất nào của workload phù hợp với tính chất nào của hardware.**

### 0.4. Roofline Model — Công cụ tư duy quan trọng nhất

#### Định nghĩa

Mọi kernel có 2 đại lượng:

- **Compute (FLOPs)**: tổng số phép tính floating-point
- **Memory traffic (Bytes)**: tổng số byte phải đọc/ghi từ memory chậm (HBM/DRAM)

**Arithmetic Intensity (AI) = FLOPs / Bytes** — đơn vị: FLOPs/byte

Mỗi phần cứng có 2 giới hạn:

- **Peak compute**: ví dụ A100 = 312 TFLOPS (FP16)
- **Peak memory bandwidth**: A100 HBM2e = 2 TB/s

**Roofline:** vẽ đồ thị log-log với trục x là AI, trục y là performance (FLOPS):

```
Performance (FLOPS)
       │
Peak ──┼──────────────────────────────
       │     ╱ memory-bound │ compute-bound
       │    ╱                │
       │   ╱                 │
       │  ╱                  │
       │ ╱                   │
       │╱                    │
       └─────────────────────────────── AI (FLOPs/byte)
            AI_critical
```

Điểm giao là **AI_critical = Peak_FLOPS / Peak_BW**. Với A100 FP16: 312e12 / 2e12 = **156 FLOPs/byte**.

Nghĩa là: nếu kernel của bạn có AI < 156, bạn sẽ bị **memory-bound** — chip dù mạnh đến đâu cũng phải chờ memory. Nếu AI > 156, bạn bị **compute-bound** — phần cứng đang chạy hết công suất.

#### Tại sao đây là công cụ quan trọng nhất

Mọi optimization về cơ bản là **dịch chuyển kernel sang phải trên roofline** (tăng AI) hoặc **đẩy nó lên gần roof** (đạt peak hơn).

Ví dụ:

- **Matmul C[M,N] = A[M,K] × B[K,N]**: FLOPs = 2MNK, Bytes (FP16) = 2(MK + KN + MN). Với M=N=K=4096: AI ≈ 1365 → **compute-bound** trên A100. Tốt!
- **Vector add C[N] = A[N] + B[N]**: FLOPs = N, Bytes = 12N. AI ≈ 0.08 → **rất memory-bound**. Không thể tối ưu thêm bằng cách tăng compute.
- **Attention (naive)**: AI thấp do phải materialize ma trận attention NxN → memory-bound → **Flash Attention** ra đời để tăng AI bằng tiling.

**Bài tập ngay lập tức:** mỗi khi đọc một paper kernel hay xem một optimization, hỏi: "Cái này thay đổi AI thế nào? Nó dịch chuyển kernel trên roofline ra sao?"

### 0.5. Bốn nguyên tắc thiết kế của AI accelerator

Mọi chip AI chuyên dụng (TPU, Tenstorrent, Groq, Cerebras, Habana, Ascend) đều dựa trên **4 nguyên tắc giống nhau**, chỉ khác cách thực hiện:

**Nguyên tắc 1: Specialization** — bỏ những thứ CPU/GPU phải có (branch prediction phức tạp, out-of-order, virtual memory đầy đủ) để dồn transistor cho **dense matrix multiplication unit**. TPU dành 95% diện tích cho MXU.

**Nguyên tắc 2: Data locality** — đưa compute đến gần data thay vì ngược lại. Hierarchy: register file → SRAM scratchpad → HBM → host DRAM. Mỗi cấp xa hơn chậm hơn ~10x. Compiler phải maximize việc reuse data ở cấp gần nhất.

**Nguyên tắc 3: Massive parallelism through deterministic dataflow** — không dùng cache coherence phức tạp như CPU/GPU; thay vào đó compiler quyết định mọi data movement tĩnh tại compile time (TPU, Groq) hoặc semi-static (Tenstorrent). Hệ quả: **compiler responsibility tăng vọt**.

**Nguyên tắc 4: Reduced precision** — chấp nhận BF16/FP8/INT8 thay vì FP32 để gấp đôi/gấp bốn compute density mà accuracy chấp nhận được. Đây là chỗ quantization & numerical analysis quan trọng.

**Hệ quả cho compiler engineer:** vì hardware giản lược, **compiler phải gánh phần thông minh**. CPU compiler có thể "lười" vì CPU có cache, OoO, branch predictor lo việc. AI compiler **phải** lo từng byte di chuyển, từng cycle, từng tile placement.

### 0.6. Mục tiêu nghề: Compiler Engineer cho AI chip

**Các công ty tuyển nhiều** (tính đến 2026): Google (XLA, JAX team), NVIDIA (cuDNN, TensorRT, Triton team), AMD (ROCm, MIGraphX), Intel (oneAPI, Habana), Tenstorrent, Groq, SambaNova, Cerebras, Graphcore, Modular (Mojo), AWS (Neuron compiler cho Trainium), Apple (CoreML), Huawei (CANN cho Ascend), Microsoft (Maia compiler), Meta (Glow, AITemplate). Ở VN: FPT Semiconductor, VinAI, một số startup đang hình thành.

**Phân tầng vai trò:**

- **Junior (0-2 năm)**: thêm op mới, fix bug trong existing pipeline, viết test
- **Mid (2-5 năm)**: own một subsystem (ví dụ: tiling pass, fusion pass, một backend)
- **Senior (5+ năm)**: design IR mới, lead toolchain cho chip mới, làm research
- **Staff/Principal**: định hướng kiến trúc compiler-hardware co-design

**Kỹ năng cốt lõi cần có (đầu ra của lộ trình này):**

1. C++ thành thạo (modern C++17/20, template metaprogramming khá)
2. Python tốt (PyTorch, JAX internals)
3. LLVM/MLIR thực hành sâu
4. Hiểu kiến trúc accelerator (đặc biệt 1 chip cụ thể rất sâu)
5. Đọc paper MLSys/ASPLOS/PLDI và áp dụng
6. Performance debugging: profiler, roofline, microbenchmark
7. Open source contribution (cực kỳ quan trọng)

---

## TỔNG QUAN GIAI ĐOẠN 1

### Mục tiêu khi kết thúc

Sau 6 tuần, bạn phải:

1. **Đọc một paper về AI chip** (như TPU, Tenstorrent, Eyeriss) và hiểu được kiến trúc, không bị "ngợp" thuật ngữ.
2. **Dự đoán hiệu năng** của một workload trên một chip cho trước bằng roofline, sai số <2x.
3. **Profile và giải thích** vì sao một kernel chậm/nhanh trên GPU thật.
4. **Viết được simulator systolic array** đơn giản — đây là tài sản dùng cho Giai đoạn 2.
5. **Chạy được model trên NPU của Luckfox** (RKNN toolkit) và hiểu chuyện gì xảy ra dưới hood.

### Bản đồ khái niệm

```
                    KIẾN TRÚC ACCELERATOR
                            │
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
   COMPUTE              MEMORY              DATAFLOW
   (làm tính)         (lưu data)         (di chuyển data)
        │                   │                   │
   • SIMD vs SIMT      • Register/SRAM      • Systolic (TPU)
   • Systolic array    • Cache hierarchy    • Spatial (Groq)
   • Vector unit       • HBM/DDR            • Mesh (Tenstorrent)
   • Tensor core       • Scratchpad         • Wafer-scale (Cerebras)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ↓
                    ROOFLINE MODEL
                  (công cụ phân tích)
                            ↓
                    NUMERICAL FORMATS
              (FP32/BF16/FP8/INT8/INT4)
                            ↓
                  HƯỚNG ĐẾN COMPILER
            (compiler nhìn 4 thứ trên thế nào?)
```

### Triết lý 6 tuần

Mỗi tuần đều xoay quanh **một câu hỏi central** và bạn phải trả lời được nó. Không học theo kiểu "đọc cho biết". Đọc → làm → đo → giải thích.

---

## TUẦN 1 — Performance Modeling: Roofline & Memory Wall

> **Câu hỏi central:** *Tại sao một matmul 1024×1024 chạy nhanh hơn một vector add 10 triệu phần tử, dù vector add ít FLOPs hơn?*

### Lý thuyết (8h)

#### Ngày 1-2: Foundations

Đọc **Hennessy & Patterson** (H&P), *Computer Architecture: A Quantitative Approach* (6th edition):

- Chương 1 toàn bộ (~50 trang). Tập trung: Amdahl's Law, performance equations, technology trends, power wall.
- Đặc biệt section về "Dennard scaling ended" — đây là lý do AI accelerator tồn tại.

#### Ngày 3: Roofline model

Đọc paper gốc: Williams, Waterman, Patterson (2009), *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, Communications of the ACM. ~10 trang, đọc kỹ.

Bài học cốt lõi:

- Performance = min(Peak FLOPS, AI × Peak BW)
- Tăng performance = tăng AI hoặc tăng utilization
- **Compiler optimization 90% là tăng AI**

#### Ngày 4: Memory wall

Đọc Ulrich Drepper, *What Every Programmer Should Know About Memory* (2007) — sections 1, 2, 3 về cache hierarchy. Bài này dài nhưng kinh điển. Đọc lướt phần Linux-specific.

Khái niệm cần nắm:

- Latency: L1 cache ~4 cycles, L2 ~12, L3 ~40, DRAM ~200, HBM cross-die ~400. **Số tương đối quan trọng hơn số tuyệt đối.**
- Bandwidth ≠ Latency. HBM có BW cao nhưng latency vẫn cao.
- **Memory wall** = compute đã tăng nhanh hơn memory ~50 năm → chip ngày càng "đói" data.

### Thực hành (10h)

#### Setup môi trường (1h)

```bash
# Trên máy có GPU NVIDIA
sudo apt install build-essential cmake ninja-build python3-pip
pip install numpy matplotlib torch torchvision

# Cài CUDA toolkit (nếu chưa có)
nvidia-smi  # verify driver
nvcc --version  # nếu chưa có, cài CUDA Toolkit từ NVIDIA

# Nsight Compute (profiler)
sudo apt install nsight-compute
```

#### Bài tập 1.1: Microbenchmark CPU memory bandwidth

Viết một chương trình C++ đo memory bandwidth thực tế của máy bạn:

```cpp
// memcpy_bench.cpp
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

int main() {
    const size_t sizes[] = {
        16 * 1024,      // 16 KB - vừa L1
        256 * 1024,     // 256 KB - vừa L2
        8 * 1024 * 1024, // 8 MB - vừa L3
        512 * 1024 * 1024 // 512 MB - phải vào DRAM
    };

    for (size_t size : sizes) {
        std::vector<char> src(size, 1);
        std::vector<char> dst(size);

        auto start = std::chrono::high_resolution_clock::now();
        const int iters = 100;
        for (int i = 0; i < iters; i++) {
            std::memcpy(dst.data(), src.data(), size);
        }
        auto end = std::chrono::high_resolution_clock::now();

        double seconds = std::chrono::duration<double>(end - start).count();
        double gb_moved = (double)size * iters / 1e9;
        std::cout << "Size: " << size/1024 << " KB, BW: "
                  << gb_moved / seconds << " GB/s\n";
    }
}
```

Compile: `g++ -O3 memcpy_bench.cpp -o memcpy_bench && ./memcpy_bench`

**Bạn sẽ thấy** BW giảm rõ rệt khi size vượt L3. Đây là **memory hierarchy hiện hình**.

#### Bài tập 1.2: Roofline cho 5 kernel

Viết script Python tính lý thuyết AI cho 5 kernel:

```python
def roofline_analysis(name, flops, bytes_accessed, peak_flops, peak_bw):
    ai = flops / bytes_accessed
    perf_compute = peak_flops
    perf_memory = ai * peak_bw
    achievable = min(perf_compute, perf_memory)
    bound = "compute-bound" if perf_compute < perf_memory else "memory-bound"
    print(f"{name}: AI={ai:.2f} FLOPs/B, achievable={achievable/1e12:.2f} TFLOPS, {bound}")

# Giả định GPU của bạn (xem nvidia-smi để lấy thông số thực)
# Ví dụ RTX 3060: ~13 TFLOPS FP32, ~360 GB/s
PEAK_FLOPS = 13e12
PEAK_BW = 360e9

# Kernel 1: Vector add c[N] = a[N] + b[N], FP32
N = 10_000_000
roofline_analysis("VecAdd", N, 3*N*4, PEAK_FLOPS, PEAK_BW)

# Kernel 2: Dot product
roofline_analysis("DotProd", 2*N, 2*N*4, PEAK_FLOPS, PEAK_BW)

# Kernel 3: Matmul MxNxK
M=N=K=4096
roofline_analysis("Matmul 4K", 2*M*N*K, (M*K + K*N + M*N)*4, PEAK_FLOPS, PEAK_BW)

# Kernel 4: Conv2d 256x256, 3x3 kernel, 64->128 channels
H=W=256; C_in=64; C_out=128; KH=KW=3
flops = 2 * H * W * C_in * C_out * KH * KW
bytes_ = (H*W*C_in + KH*KW*C_in*C_out + H*W*C_out) * 4
roofline_analysis("Conv2d", flops, bytes_, PEAK_FLOPS, PEAK_BW)

# Kernel 5: Softmax over [B, S, S] (attention scores)
B, S = 32, 2048
roofline_analysis("Softmax", 3*B*S*S, 2*B*S*S*4, PEAK_FLOPS, PEAK_BW)
```

#### Bài tập 1.3: Đo thực tế và so sánh với lý thuyết

```python
import torch
import time

device = 'cuda'

def bench(fn, name, flops, bytes_):
    # warmup
    for _ in range(10): fn()
    torch.cuda.synchronize()

    t0 = time.time()
    iters = 100
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    t = (time.time() - t0) / iters

    tflops = flops / t / 1e12
    bw = bytes_ / t / 1e9
    print(f"{name}: {t*1000:.2f}ms, {tflops:.2f} TFLOPS, {bw:.1f} GB/s")

# Matmul
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)
bench(lambda: a @ b, "Matmul 4K", 2*4096**3, 3*4096**2*4)

# Vector add
x = torch.randn(10_000_000, device=device)
y = torch.randn(10_000_000, device=device)
bench(lambda: x + y, "VecAdd", 10_000_000, 3*10_000_000*4)
```

**Câu hỏi phải trả lời:**

- Matmul của bạn đạt bao nhiêu % peak FLOPS? (Thường 60-90% nếu PyTorch dùng cuBLAS)
- VecAdd đạt bao nhiêu % peak BW? (Thường 70-90%)
- Tại sao matmul gần peak compute, vecadd gần peak BW? **Roofline trả lời.**

### Liên hệ HW-SW

**Hardware reality → Software consequence:**

- **HW**: A100 có 312 TFLOPS FP16 nhưng HBM BW chỉ 2 TB/s → AI_critical = 156. Mọi kernel có AI < 156 đều "lãng phí" compute.
- **SW**: Compiler tìm mọi cách tăng AI. **Operator fusion** là vũ khí #1: nếu bạn có `softmax(matmul(x, w))`, fusion 2 op lại tránh writing intermediate ra HBM → AI tăng → kernel nhanh hơn.

**Đây là intuition đầu tiên về compiler mà bạn phải nắm:** *compiler không tạo ra phép tính mới, nó loại bỏ data movement không cần thiết.*

### Output cuối tuần

Một repo GitHub `week1-roofline/` chứa:

- `memcpy_bench.cpp` + kết quả đo trên máy bạn
- `roofline_theory.py` + plot roofline cho GPU của bạn
- `roofline_measured.py` + so sánh thực tế vs lý thuyết
- `README.md` giải thích kết quả, đặc biệt **chỗ chênh lệch** giữa lý thuyết và thực tế (PyTorch overhead, warm-up, etc.)

---

## TUẦN 2 — CPU SIMD & GPU SIMT: Baseline để so sánh

> **Câu hỏi central:** *GPU làm AI nhanh hơn CPU vì sao? Cụ thể, không chung chung.*

### Lý thuyết (8h)

#### Ngày 1-2: CPU vector extensions

Đọc H&P chương 4, đặc biệt sections 4.1-4.3 về SIMD và vector processors.

Khái niệm:

- SSE (128-bit), AVX2 (256-bit), AVX-512 (512-bit). 1 instruction xử lý 8/16 floats cùng lúc.
- **Vectorization** là việc compiler/người chuyển scalar loop thành SIMD instruction.
- Intel intrinsics: `_mm256_add_ps`, `_mm256_fmadd_ps`, vv.

#### Ngày 3-4: GPU architecture & CUDA model

Đọc *CUDA C++ Programming Guide*, chương 1-3 (online tại docs.nvidia.com):

- SIMT (Single Instruction Multiple Thread) model
- Thread → warp (32 threads) → block → grid
- SM (Streaming Multiprocessor), tensor core
- Memory hierarchy: register → shared memory → L1/L2 → global (HBM)

Sách quan trọng: **Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed., 2022)**, chương 1-5. Đây là sách CUDA tốt nhất.

#### Ngày 5: Tensor Core

Đọc whitepaper Volta hoặc Ampere của NVIDIA (search "NVIDIA Ampere architecture whitepaper"). Hiểu:

- Tensor core làm matmul 4x4x4 trong 1 instruction
- Khác với CUDA core (làm scalar FMA)
- **Đây là "systolic-like unit" trong GPU** — không phải hardware MXU lớn như TPU, mà nhiều unit nhỏ phân tán trong SMs

### Thực hành (12h)

#### Bài tập 2.1: AVX matmul thủ công

```cpp
// avx_matmul.cpp
#include <immintrin.h>
#include <chrono>
#include <iostream>
#include <vector>

void matmul_naive(float* C, const float* A, const float* B, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

void matmul_avx(float* C, const float* A, const float* B, int N) {
    // Giả định N chia hết cho 8
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = 0; k < N; k++) {
                __m256 a = _mm256_broadcast_ss(&A[i*N + k]);
                __m256 b = _mm256_loadu_ps(&B[k*N + j]);
                acc = _mm256_fmadd_ps(a, b, acc);
            }
            _mm256_storeu_ps(&C[i*N + j], acc);
        }
    }
}

int main() {
    const int N = 512;
    std::vector<float> A(N*N), B(N*N), C(N*N);
    // ... init A, B random

    auto bench = [&](auto fn, const char* name) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double sec = std::chrono::duration<double>(t1-t0).count();
        double gflops = 2.0 * N*N*N / sec / 1e9;
        std::cout << name << ": " << sec*1000 << " ms, " << gflops << " GFLOPS\n";
    };

    bench([&]{ matmul_naive(C.data(), A.data(), B.data(), N); }, "Naive");
    bench([&]{ matmul_avx(C.data(), A.data(), B.data(), N); }, "AVX");
}
```

Compile: `g++ -O3 -mavx2 -mfma avx_matmul.cpp -o avx_matmul`

**Bạn sẽ thấy AVX nhanh hơn naive ~4-8x.**

#### Bài tập 2.2: CUDA matmul, 3 versions

```cuda
// matmul.cu
#include <cuda_runtime.h>
#include <iostream>

// V1: Naive - mỗi thread tính 1 phần tử C[i,j]
__global__ void matmul_naive(float* C, const float* A, const float* B, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0;
        for (int k = 0; k < N; k++)
            sum += A[row*N + k] * B[k*N + col];
        C[row*N + col] = sum;
    }
}

// V2: Shared memory tiling
#define TILE 16
__global__ void matmul_tiled(float* C, const float* A, const float* B, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0;

    for (int t = 0; t < N/TILE; t++) {
        As[threadIdx.y][threadIdx.x] = A[row*N + t*TILE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t*TILE + threadIdx.y)*N + col];
        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row*N + col] = sum;
}

// V3: gọi cuBLAS (để so sánh peak)
// (xem doc cuBLAS, dùng cublasSgemm)
```

Bench cả 3 với N=2048. Bạn sẽ thấy:

- Naive: ~500 GFLOPS (bị memory-bound vì load lặp)
- Tiled: ~3000 GFLOPS (reuse shared memory)
- cuBLAS: ~10000+ GFLOPS (dùng tensor core + nhiều trick khác)

#### Bài tập 2.3: Profile với Nsight Compute

```bash
ncu --set full -o profile_naive ./matmul_naive
ncu --set full -o profile_tiled ./matmul_tiled
ncu-ui profile_naive.ncu-rep
```

Mở Nsight Compute GUI, xem:

- **SM Throughput %**: V1 thấp (~30%), V2 cao hơn (~70%)
- **Memory Throughput %**: V1 cao (bottleneck), V2 thấp hơn
- **Achieved Occupancy**: số warp active / max
- **L1/L2 hit rate**

**Đây là lần đầu bạn "nhìn thấy" hardware đang làm gì.**

### Liên hệ HW-SW

**HW**: GPU có 80+ SMs, mỗi SM có 4 sub-cores chạy warps. Memory: HBM 2TB/s nhưng latency ~500 cycle. Shared memory: 100+ TB/s nhưng chỉ 100KB/SM.

**SW consequence**:

1. **Coalesced memory access**: warp 32 threads phải đọc 32 floats liên tiếp → 1 transaction 128 bytes. Nếu không, 32 transactions → 32x chậm. **Compiler phải đảm bảo layout cho phép coalescing.**
2. **Bank conflicts**: shared memory chia 32 bank, nếu 2 thread truy cập cùng bank → serialize. **Compiler phải thiết kế tiling tránh conflict.**
3. **Tile size**: TILE=16 → 256 thread/block. Quá nhỏ → underutilization. Quá lớn → spill registers. **Compiler chọn tile size dựa trên kiến trúc.**

**Intuition compiler thứ 2:** *compiler phải biết microarchitecture cụ thể. "Optimization cho GPU" là sai — phải nói "optimization cho Ampere" hay "Hopper".*

### Output cuối tuần

Repo `week2-cpu-gpu/`:

- AVX matmul + bench results
- 3 CUDA matmul versions + Nsight reports
- Một file `analysis.md` giải thích từng bottleneck với reference cụ thể đến metric Nsight

---

## TUẦN 3 — TPU & Systolic Array: Case study quan trọng nhất

> **Câu hỏi central:** *Tại sao Google thiết kế TPU dùng systolic array thay vì SIMD như GPU? Compiler cho TPU phải làm gì khác với compiler GPU?*

### Lý thuyết (10h)

#### Ngày 1-2: TPU paper đầu tiên (BẮT BUỘC đọc kỹ)

Jouppi et al. (2017), *In-Datacenter Performance Analysis of a Tensor Processing Unit*, ISCA. ~12 trang.

Đọc 2-3 lần. Lần 1 đọc lướt hiểu big picture. Lần 2 đọc kỹ section 2-4 (architecture). Lần 3 chú ý section 6-7 (performance, energy).

Điểm phải nắm:

- MXU 256×256 systolic array, 65,536 MAC units
- 24MB on-chip Unified Buffer (UB), không có cache truyền thống
- 8-bit integer (INT8), không phải float
- **Software-managed memory** — compiler chịu trách nhiệm hoàn toàn việc đưa data vào UB
- Không có branch predictor, không có speculation — instruction stream phải deterministic

#### Ngày 3: Systolic array sâu

Đọc bài blog "Why is Google's TPU's systolic array so much better than GPU's SIMD?" hoặc tương tự. Hoặc tốt nhất: đọc paper gốc của H.T. Kung (1982), *Why Systolic Architectures?* — kinh điển.

Systolic array hoạt động:

```
Input matrix A bơm vào từ trái, B bơm vào từ trên,
mỗi PE (Processing Element) làm 1 MAC mỗi cycle,
kết quả tích lũy ở dưới (output-stationary) hoặc lan tỏa (weight-stationary).
```

3 dataflows kinh điển (Eyeriss paper sẽ học tuần 4 phân tích sâu):

- **Weight stationary**: weight ở yên trong PE, activation chảy qua (TPU style)
- **Output stationary**: output accumulator ở yên, A và B chảy qua
- **Input stationary**: input ở yên, weight chảy qua

#### Ngày 4: TPU v2/v3/v4 evolution

Jouppi et al. (2020), *A Domain-Specific Supercomputer for Training Deep Neural Networks*, CACM. Hiểu sự tiến hóa:

- v1: inference only, INT8
- v2: training + inference, BF16 ra đời ở đây
- v3: liquid cooling, 4x compute
- v4: optical reconfigurable interconnect, sparse cores

Bonus: paper TPUv4 (Jouppi 2023) nếu có thời gian.

#### Ngày 5: XLA compiler — TPU compiler

Đọc tổng quan về XLA tại openxla.org. Hiểu:

- HLO (High Level Operations) — IR chính của XLA
- Op fusion, layout assignment, memory scheduling
- Backend cho TPU sinh ra "instruction" của MXU

**Đây là lần đầu bạn thấy compiler thực sự cho AI chip thật.**

### Thực hành (12h)

#### Bài tập 3.1: Tay viết systolic array simulator

Đây là project lớn nhất của Giai đoạn 1. Code này sẽ là asset xuyên suốt cho Giai đoạn 2.

```python
# systolic_sim.py
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PE:
    """Processing Element trong systolic array.
    Output-stationary: accumulator ở yên, A bơm từ trái, B bơm từ trên.
    """
    accumulator: float = 0.0
    a_in: float = 0.0  # input từ trái
    b_in: float = 0.0  # input từ trên
    a_out: float = 0.0  # output sang phải (truyền cho PE kế)
    b_out: float = 0.0  # output xuống dưới

    def cycle(self):
        # 1 cycle: nhân a*b cộng vào accumulator, đẩy a,b cho neighbor
        self.accumulator += self.a_in * self.b_in
        self.a_out = self.a_in
        self.b_out = self.b_in

class SystolicArray:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.pes = [[PE() for _ in range(cols)] for _ in range(rows)]
        self.cycle_count = 0
        self.total_macs = 0
        self.active_macs_per_cycle = []

    def reset_accumulators(self):
        for row in self.pes:
            for pe in row:
                pe.accumulator = 0.0

    def step(self, a_inputs: list, b_inputs: list):
        """1 cycle. a_inputs: list dài rows (bơm vào cột trái).
        b_inputs: list dài cols (bơm vào hàng trên).
        """
        # Phase 1: lan truyền a, b vào pes
        for i in range(self.rows):
            for j in range(self.cols):
                if j == 0:
                    self.pes[i][j].a_in = a_inputs[i] if a_inputs[i] is not None else 0
                else:
                    self.pes[i][j].a_in = self.pes[i][j-1].a_out
                if i == 0:
                    self.pes[i][j].b_in = b_inputs[j] if b_inputs[j] is not None else 0
                else:
                    self.pes[i][j].b_in = self.pes[i-1][j].b_out

        # Phase 2: tất cả PE compute đồng thời
        active = 0
        for row in self.pes:
            for pe in row:
                if pe.a_in != 0 or pe.b_in != 0:
                    active += 1
                pe.cycle()

        self.cycle_count += 1
        self.active_macs_per_cycle.append(active)
        self.total_macs += active

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Tính C = A @ B với A: [rows, K], B: [K, cols].
        Giả định A.shape[0] == rows, B.shape[1] == cols.
        """
        assert A.shape[0] == self.rows
        assert B.shape[1] == self.cols
        K = A.shape[1]
        assert B.shape[0] == K

        self.reset_accumulators()

        # Bơm data theo pattern systolic (skewed)
        # Cycle t: bơm A[i, t-i] vào row i (nếu hợp lệ), B[t-j, j] vào col j
        total_cycles = K + self.rows + self.cols - 2

        for t in range(total_cycles):
            a_in = []
            for i in range(self.rows):
                k = t - i
                a_in.append(A[i, k] if 0 <= k < K else None)
            b_in = []
            for j in range(self.cols):
                k = t - j
                b_in.append(B[k, j] if 0 <= k < K else None)
            self.step(a_in, b_in)

        # Đọc accumulators ra
        C = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                C[i, j] = self.pes[i][j].accumulator
        return C

    def stats(self):
        utilization = self.total_macs / (self.cycle_count * self.rows * self.cols)
        print(f"Total cycles: {self.cycle_count}")
        print(f"Total MAC operations: {self.total_macs}")
        print(f"Utilization: {utilization*100:.1f}%")

# Test
if __name__ == "__main__":
    sa = SystolicArray(rows=4, cols=4)
    A = np.random.randn(4, 8)
    B = np.random.randn(8, 4)
    C_sim = sa.matmul(A, B)
    C_ref = A @ B

    print("Max error:", np.abs(C_sim - C_ref).max())
    sa.stats()
```

#### Bài tập 3.2: Mở rộng simulator

- Thêm chế độ **weight-stationary** (B ở yên, A và partial sums chảy)
- Thêm tiling: nếu matmul lớn hơn array (ví dụ matmul 64×64 trên array 16×16), chia thành 4×4 tile và compute lần lượt
- Đếm: số byte đọc từ "DRAM" (mỗi tile load), số byte ghi (output)
- Plot utilization theo cycle — bạn sẽ thấy "ramp-up" và "ramp-down" của systolic

#### Bài tập 3.3: So sánh với GPU

Cho matmul 256×256×256, tính:

- Trên systolic 16×16 của bạn: bao nhiêu cycle? Bao nhiêu byte HBM traffic (giả định weight load 1 lần, không cached)?
- Tỉ lệ "utilization" của bạn vs GPU thực tế (đo bằng PyTorch)?

### Liên hệ HW-SW

**HW**: TPU MXU 256x256 không có scheduler, không có cache, không có branch. Chỉ có "pipe" data chảy qua.

**SW consequence (đây là phần quan trọng nhất tuần này)**:

1. **Compiler phải biết kích thước MXU**. Matmul 1000×1000 phải tile thành chunks 256×256. Tile size = compile-time decision phụ thuộc hardware.

2. **Compiler phải schedule data movement ahead of time**. Vì không có cache, compiler phải emit explicit DMA: "tại cycle X, load tile này từ HBM vào UB". Sai 1 cycle = stall.

3. **Compiler phải lo padding**. Nếu tensor không chia hết 256, phải pad. Padding lãng phí compute → compiler phải minimize padding bằng cách chọn tile sizes thông minh.

4. **Compiler phải lo numerical precision**. TPUv1 INT8 → compiler phải quantize, chèn scale factors. BF16 → compiler phải biết khi nào FP32 accumulator cần thiết.

**Đây là intuition compiler thứ 3:** *cho AI accelerator, compiler không "tối ưu" code — compiler **viết** code. Hardware không thể chạy gì khác.*

### Output cuối tuần

Repo `week3-systolic/`:

- `systolic_sim.py` đầy đủ (output-stationary + weight-stationary)
- Test cases cho matmul 4x4, 16x16, 64x64 (cần tiling)
- Plot utilization
- Markdown phân tích: "vì sao TPU dùng systolic? 5 điểm so sánh với GPU SIMT"

---

## TUẦN 4 — Dataflow & Spatial Architectures: Vượt khỏi TPU

> **Câu hỏi central:** *Có những cách thiết kế AI chip nào khác ngoài systolic? Tradeoff là gì?*

### Lý thuyết (10h)

#### Ngày 1-2: Eyeriss — Dataflow taxonomy

Chen et al. (2016), *Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks*, ISCA. ~13 trang.

Đây là paper định nghĩa terminology dataflow trong AI accelerator. **Bắt buộc đọc.**

Học:

- 4 dataflow: Weight Stationary, Output Stationary, Input Stationary, **Row Stationary** (đóng góp của Eyeriss)
- Phân tích energy: data movement tốn năng lượng hơn compute (DRAM access ~100x register access)
- Reuse types: spatial reuse, temporal reuse, convolutional reuse

#### Ngày 3: Tenstorrent

Vì bạn quan tâm chip chuyên dụng, Tenstorrent rất đáng học vì:

1. Open source toolchain (tt-metalium)
2. Có RISC-V cores embedded (programmable per-tile)
3. Mesh of cores thay vì 1 MXU lớn

Đọc:

- Blog tenstorrent.com về Tensix architecture
- Docs của tt-metalium trên GitHub (`tenstorrent/tt-metal`)
- Paper hoặc talk nếu có (search "Jim Keller Tenstorrent" YouTube)

Kiến trúc:

- Mỗi tile (Tensix core) có: 5 RISC-V baby cores + 1 compute engine với SFPU (scalar) + FPU matrix
- Tiles kết nối qua NoC 2D mesh
- 1 chip = 100+ tiles
- **Compiler phải phân hoạch model lên các tile + route data qua NoC**

#### Ngày 4: Groq — Deterministic dataflow

Abts et al. (2020), *Think Fast: A Tensor Streaming Processor*, ISCA.

Groq's TSP đặc biệt vì:

- 100% deterministic — không có cache miss, không có dynamic scheduling
- Compile time biết chính xác mọi instruction sẽ chạy lúc nào
- 220 MB on-chip SRAM, không HBM
- 750 TOPS INT8 với latency cực thấp

**Đây là ví dụ điển hình của "compiler-centric design": hardware đơn giản hết mức, compiler gánh tất.**

#### Ngày 5: Cerebras — Wafer scale

CS-2 / CS-3 specifications. Wafer-scale engine: cả 1 wafer 300mm thành 1 chip. 850,000 cores.

Đọc blog post hoặc paper Cerebras về memory architecture.

### Thực hành (10h)

#### Bài tập 4.1: Mô phỏng Eyeriss row-stationary

Mở rộng simulator tuần 3 để hỗ trợ **convolution với row-stationary dataflow**. Đây là khó hơn matmul, nhưng phải làm để hiểu spatial computing.

```python
# Pseudo-code khái niệm
# Conv: output[h,w,c_out] = sum_{kh,kw,c_in} input[h+kh, w+kw, c_in] * filter[kh,kw,c_in,c_out]
# Row-stationary: 1 PE compute 1 hàng output
# Filter row của PE stay in PE register
# Input row stream qua PE
# Partial sums tích lũy

class EyerissPE:
    def __init__(self):
        self.filter_row = None  # weight row stay here
        self.psum_acc = 0

    def load_filter_row(self, weights):
        self.filter_row = weights

    def compute_output_row(self, input_row):
        # 1D convolution của filter_row với input_row
        # Trả về row của partial sums
        ...
```

#### Bài tập 4.2: Tự build tt-metalium

```bash
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal
./build_metal.sh  # cần Linux, đọc README cho dependency
```

Bạn không có Tenstorrent card thật, nhưng:

- Đọc examples trong `tt_metal/programming_examples/`
- Đặc biệt `eltwise_binary` và `matmul_single_core` — đọc kỹ kernel C++
- Hiểu cách compiler tile và dispatch lên cores

Đây là **lần đầu bạn đọc code compiler-runtime cho chip AI thực tế, mã nguồn mở**. Rất quý.

#### Bài tập 4.3: Phân tích so sánh

Viết bảng comparison 1 trang:

| Aspect | TPU v4 | Tenstorrent | Groq TSP | Cerebras WSE-2 |
|--------|--------|-------------|----------|----------------|
| Compute organization | 1 lớn MXU/core | Mesh of tiles | Vector lanes | 850K cores |
| On-chip memory | HBM + UB | Per-tile SRAM | 220 MB SRAM | 40 GB SRAM |
| Off-chip memory | HBM | DDR | None | None |
| Programming model | Static schedule (XLA) | Dataflow (Metalium) | Static (Groq compiler) | Streaming (CSL) |
| Compiler complexity | High | Very high | Extreme | Extreme |
| Best workload | Large dense matmul | Flexible | Low latency inference | Massive parallel |

### Liên hệ HW-SW

**Big insight tuần này**: mỗi kiến trúc → mỗi loại compiler khác nhau:

- **TPU (1 big MXU)**: compiler chủ yếu lo tiling cho MXU + memory scheduling
- **Tenstorrent (mesh)**: compiler phải lo placement (op nào ở tile nào) + routing (data qua NoC) → giống compiler cho distributed system
- **Groq (deterministic)**: compiler phải biết **exact cycle** của mọi instruction → cực kỳ nặng về scheduling, không thể có ambiguity
- **Cerebras (wafer-scale)**: compiler phải lo dataflow over 850K cores → giống compile cho supercomputer

**Đây là lý do "compiler engineer cho AI chip" là vai trò chuyên biệt theo từng kiến trúc.** Bạn vào Groq sẽ làm khác Tenstorrent rất nhiều.

### Output cuối tuần

Repo `week4-architectures/`:

- Eyeriss simulator extension
- tt-metalium build success + notes về 1-2 example
- Comparison table + 3-page analysis về tradeoffs

---

## TUẦN 5 — Numerical Formats & Quantization

> **Câu hỏi central:** *Tại sao chip AI không dùng FP64? Khi nào INT8 đủ, khi nào không?*

### Lý thuyết (8h)

#### Ngày 1-2: Floating point fundamentals

Đọc Goldberg (1991), *What Every Computer Scientist Should Know About Floating-Point Arithmetic*. Kinh điển, nhưng dài. Đọc các section đầu về IEEE 754 representation.

Học:

- IEEE 754: sign | exponent | mantissa
- FP32: 1 + 8 + 23
- FP16: 1 + 5 + 10 — exponent range hẹp → dễ overflow/underflow trong training
- BF16: 1 + 8 + 7 — cùng range FP32 nhưng precision thấp hơn FP16
- FP8 (E5M2, E4M3): NVIDIA Hopper, 2 variants

#### Ngày 3: BF16 vs FP16 — Tại sao Google chọn BF16

Đọc paper "A Study of BFLOAT16 for Deep Learning Training" (Kalamkar et al., Intel/Google, 2019).

Insight: trong DL, **dynamic range quan trọng hơn precision**. Gradients có thể rất nhỏ (1e-7) hay rất to. FP16 overflow → loss = NaN. BF16 cùng range FP32 → train stable mà compute fast.

#### Ngày 4: Quantization basics

Đọc Jacob et al. (2018), *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (Google paper, kinh điển).

Học:

- Symmetric vs asymmetric quantization
- Per-tensor vs per-channel scale
- Post-training quantization (PTQ) vs Quantization-aware training (QAT)
- Calibration: tính scale từ representative data

#### Ngày 5: Modern formats

- **FP8**: NVIDIA Transformer Engine paper
- **INT4** và **MX formats** (Microsoft, OCP standard): block floating point
- **Sparsity**: NVIDIA Ampere 2:4 structured sparsity

### Thực hành (10h)

#### Bài tập 5.1: Implement quantization tay

```python
import torch
import numpy as np

def quantize_symmetric(x: torch.Tensor, n_bits: int = 8):
    """Symmetric quantization per-tensor."""
    qmax = 2**(n_bits-1) - 1  # 127 for INT8
    scale = x.abs().max() / qmax
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale

def dequantize(x_q, scale):
    return x_q.float() * scale

# Test trên matmul
A = torch.randn(256, 256)
B = torch.randn(256, 256)

# Reference: FP32
C_fp32 = A @ B

# Quantize A và B
A_q, sA = quantize_symmetric(A)
B_q, sB = quantize_symmetric(B)

# INT8 matmul (đây là điều TPU/NPU làm trong hardware)
C_int = (A_q.float() @ B_q.float()).to(torch.int32)
C_dq = C_int.float() * (sA * sB)

# Sai số
rel_err = (C_dq - C_fp32).abs().mean() / C_fp32.abs().mean()
print(f"Mean relative error: {rel_err*100:.2f}%")
```

#### Bài tập 5.2: Quantize ResNet18 với PyTorch quantization

Dùng `torch.quantization` để PTQ ResNet18 ImageNet:

- Đo accuracy FP32 baseline
- PTQ INT8
- Đo accuracy drop
- Đo inference speedup trên CPU

#### Bài tập 5.3: BF16 vs FP16 stability test

Train MLP nhỏ trên MNIST với 3 precisions: FP32, FP16, BF16. Log gradient magnitude qua các epoch. Bạn sẽ thấy FP16 dễ NaN hơn.

### Liên hệ HW-SW

**HW**: TPU MXU INT8 = 4x throughput FP32 cùng diện tích. NVIDIA Hopper FP8 tensor core = 2x FP16. **Lower precision → more compute density.**

**SW consequence**:

1. **Compiler phải biết hardware support gì**. TPU v1: INT8 only. TPU v2+: BF16. Hopper: FP8. **Lowering pass phải pick đúng dtype.**

2. **Compiler phải insert quantize/dequantize**. Nếu user viết FP32 model, hardware INT8 → compiler tự động chèn `q -> int8_op -> dq`.

3. **Compiler phải lo mixed precision**. Matmul có thể INT8 input nhưng cần INT32 accumulator (vì 256-deep dot product có thể overflow INT8). Đầu ra dequantize về FP16.

4. **Compiler phải tránh numerical pitfall**. Softmax cần FP32 vì exponential → BF16 sẽ underflow. **Op-specific precision** là pattern phổ biến.

**Intuition compiler thứ 4:** *precision không chỉ là "chính xác hay không" — nó là một **resource** mà compiler trade-off như memory hay compute.*

### Output cuối tuần

Repo `week5-quantization/`:

- Quantization implementation từ đầu
- ResNet18 quantization results
- BF16/FP16/FP32 training stability comparison
- Note 1-page về "precision strategies in modern AI chips"

---

## TUẦN 6 — Memory Hierarchy & Project tích hợp + Luckfox NPU

> **Câu hỏi central:** *Toàn bộ kiến thức tuần 1-5 áp dụng thế nào lên 1 chip AI thật?*

Tuần này có 2 mục tiêu: (1) đào sâu memory hierarchy + tiling — chuẩn bị cho Giai đoạn 2; (2) **chạm vào NPU thật của Luckfox**.

### Lý thuyết (6h)

#### Ngày 1: HBM/SRAM/Scratchpad

Đọc về memory technology trong AI chip:

- HBM2/HBM3 chi tiết: 8-12 stacks, mỗi stack 8-channel, TSV interconnect
- SRAM trade-off: tốc độ cao nhưng diện tích lớn, cấp lên 100+ MB là maximum
- Scratchpad vs cache: ai control? Trade-off transparency vs efficiency

#### Ngày 2: Tiling theory

Đọc về polyhedral model (chỉ overview, không đào sâu — Giai đoạn 2 sẽ làm):

- Affine loops
- Loop tiling, loop fusion, loop interchange
- Bài viết "An Introduction to Polyhedral Compilation" của Albert Cohen

#### Ngày 3: Flash Attention — case study tiling cứu memory wall

Đọc Dao et al. (2022), *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*.

Đây là **paper tutorial hoàn hảo** về cách compiler-style tiling giải quyết bài toán thực tế. Standard attention bị memory-bound vì materialize ma trận attention NxN. Flash Attention tile theo cách giữ attention block trong SRAM → giảm HBM traffic → tăng AI → tăng tốc độ.

### Thực hành (14h)

#### Bài tập 6.1: Tiled matmul 3 cấp

Implement CUTLASS-style matmul với 3 cấp tiling:

- Block tile (128x128) → chia work cho thread block
- Warp tile (32x32) → chia trong block
- Thread tile (8x8) → mỗi thread compute 1 sub-block

```cuda
// Pseudo
__global__ void matmul_tiled_3level(...) {
    __shared__ float A_block[128][128]; // block tile in shared mem

    // Loop over K, load block tile to shared
    for (k_block ...) {
        // Cooperative load to shared
        load_block(A_block, ...);
        __syncthreads();

        // Each warp does 32x32 work
        // Each thread does 8x8 work (32 elements in 8x8 = 64 FMAs)
        for (warp k ...) {
            // Load to registers
            // Compute 8x8 outer product, accumulate to register
        }
        __syncthreads();
    }
    // Write back
}
```

Profile với Nsight, so sánh với cuBLAS. Bạn sẽ đạt ~70-80% cuBLAS performance — đây là kết quả tốt cho hand-written kernel.

#### Bài tập 6.2: Flash Attention simple version

Implement Flash Attention naive version bằng Triton (hoặc CUDA). Đọc tutorial Triton chính thức về Flash Attention.

Verify correctness vs PyTorch standard attention. Đo memory usage — bạn sẽ thấy giảm rõ rệt với sequence dài.

#### Bài tập 6.3: LUCKFOX NPU — Chạy model thật

Đây là phần đặc biệt với hardware của bạn. RV1103 có NPU 0.5 TOPS hỗ trợ INT8/INT4.

```bash
# Setup RKNN toolkit trên máy dev
pip install rknn-toolkit2  # phiên bản x86 để convert model
# (Trên Luckfox dùng rknn-toolkit-lite2 để inference)

# Lấy code Luckfox SDK
git clone https://github.com/LuckfoxTECH/luckfox-pico.git
```

Bước thực hiện:

1. **Convert một model PyTorch sang RKNN**:

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)

# Config
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rv1103',
    quantized_algorithm='normal',
    quantized_dtype='asymmetric_quantized-8'
)

# Load ONNX (export từ PyTorch trước)
rknn.load_onnx(model='resnet18.onnx')

# Build với quantization
rknn.build(do_quantization=True, dataset='./calib_dataset.txt')

# Export RKNN format
rknn.export_rknn('./resnet18.rknn')
```

2. **Deploy lên Luckfox, chạy inference**:

```bash
# SCP file .rknn lên board
scp resnet18.rknn root@<luckfox-ip>:/root/

# SSH vào, chạy
ssh root@<luckfox-ip>
# Có example inference C/Python trong SDK
```

3. **Quan sát điều gì xảy ra**:

- Inference time bao nhiêu ms?
- So với CPU-only inference (chạy ONNX runtime trên ARM core)?
- Memory footprint?
- Accuracy trước/sau quantization?

**Đây là lần đầu bạn nắm trong tay TOÀN BỘ stack**: PyTorch model → ONNX → RKNN compiler (đây là 1 compiler thật!) → quantization → NPU instruction → chạy trên silicon thật.

**Bài tập đào sâu (nâng cao)**:

- Dump intermediate output của RKNN compile. Tools có `rknn.list_devices()`, debug mode.
- Đọc về RKNN's IR (giống ONNX nhưng có quantization annotations)
- Tìm hiểu Rockchip NPU instruction set (RKNPU2 driver có public)

### Liên hệ HW-SW (tổng kết giai đoạn 1)

Tại đây bạn đã thấy đủ pattern:

| Hardware feature | Compiler responsibility |
|------------------|------------------------|
| HBM bandwidth << compute | Operator fusion, tiling, layout opt |
| SRAM nhỏ | Tile size selection, double buffering |
| Tensor core specific size (16x16) | Lowering to mma instruction, padding |
| INT8/BF16 hardware | Quantization, mixed precision |
| Systolic array fixed dimension | Hard tiling, matmul decomposition |
| Static scheduled (Groq) | Cycle-accurate scheduling |
| Mesh of cores (Tenstorrent) | Placement + routing |
| NPU on RV1103 | RKNN compiles model + handles all of above |

**Mọi optimization compiler đều là phản ứng với một ràng buộc hardware**. Bạn đã nắm intuition này.

### Output cuối tuần

Repo `week6-memory-tiling-npu/`:

- Tiled matmul 3 cấp + Nsight profile
- Flash Attention Triton implementation
- **Luckfox NPU project**: model gốc → convert → benchmark → analysis. Đây sẽ là portfolio piece.
- Final markdown: "Tổng kết Giai đoạn 1 — 10 insights chính" (tự viết, đây là cách consolidate).

---

## TỔNG KẾT GIAI ĐOẠN 1

### Kiểm tra kiến thức

Bạn phải trả lời được (không nhìn note):

1. Giải thích roofline model. Tính AI_critical cho RTX 4090 (xem spec).
2. Vì sao matmul là kernel "lý tưởng" cho AI accelerator, nhưng vector add thì không?
3. Mô tả 1 cycle của systolic array. Vì sao có "ramp-up" period?
4. So sánh weight-stationary vs output-stationary dataflow. Khi nào dùng cái nào?
5. Vì sao BF16 thay thế FP16 trong training?
6. Quantize asymmetric khác symmetric thế nào? Khi nào dùng?
7. Flash Attention làm gì để tăng AI?
8. Compiler khác nhau thế nào giữa TPU và Tenstorrent?
9. Khi compile một model lên Luckfox NPU, có bao nhiêu transformation? Cái nào quan trọng nhất?
10. Nếu phải design 1 chip AI cho LLM inference, bạn ưu tiên gì: compute density, memory bandwidth, on-chip memory size?

### Câu hỏi phỏng vấn mẫu (Giai đoạn 1 level)

- "Tại sao chip AI có operator fusion?" → roofline + HBM bandwidth
- "What's the difference between SIMD and systolic array?" → control granularity, dataflow pattern
- "Why does quantization help inference performance?" → compute density + memory traffic
- "If your matmul kernel achieves 30% of peak TFLOPS, where would you look first?" → roofline, profile, AI, memory bottleneck

### Output tổng cộng giai đoạn 1

GitHub portfolio sau 6 tuần:

- 6 repo theo tuần
- 1 systolic simulator có thể tái dùng
- 1 working Flash Attention implementation
- 1 Luckfox NPU end-to-end deployment (rất unique, ít người làm)
- Notes/blog posts về 6 paper kinh điển bạn đã đọc

---

## Tài liệu reference cho toàn giai đoạn 1

### Sách

- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* (6th ed., 2017)
- Kirk & Hwu, *Programming Massively Parallel Processors* (4th ed., 2022)
- Sze, Chen, Yang, Emer, *Efficient Processing of Deep Neural Networks* (Morgan & Claypool, 2020) — sách tuyệt vời, free PDF

### Paper bắt buộc

- Williams et al. 2009 (Roofline)
- Jouppi et al. 2017 (TPU)
- Chen et al. 2016 (Eyeriss)
- Jacob et al. 2018 (Quantization)
- Dao et al. 2022 (FlashAttention)

### Khóa học miễn phí

- MIT 6.5940 *TinyML and Efficient Deep Learning* (Han Song) — YouTube
- Stanford CS217 *Hardware Accelerators for Machine Learning* — slides online
- CMU 10-414 *Deep Learning Systems* (Tianqi Chen) — đặc biệt phần system

### Tài nguyên thực hành

- NVIDIA CUDA samples
- tt-metalium (Tenstorrent open source)
- RKNN toolkit (cho Luckfox)
- Triton tutorials (chuẩn bị cho Giai đoạn 2)

---

## Lời khuyên trước khi đi tiếp

**Đừng làm tất cả perfect.** Nếu mỗi bài tập đều phải hoàn hảo, bạn sẽ kẹt mãi. Mục tiêu là **build intuition + portfolio**, không phải nghiên cứu cấp PhD.

**Viết blog/note theo tuần.** Sau mỗi tuần, viết 1 post 500-1000 từ giải thích cho "self của tuần trước". Đây là cách kiểm tra hiểu sâu, và sau này là material phỏng vấn.

**Tham gia community.** Discord MLIR/Triton, r/MachineLearning, Twitter follow Tri Dao (Flash Attention), Horace He, Tianqi Chen, Chris Lattner.

**Khi sang Giai đoạn 2**, bạn sẽ có systolic simulator → bạn sẽ dùng nó như **target backend cho compiler bạn xây**. Đây là lý do mình bắt bạn làm kỹ ở tuần 3.

---

*File này thuộc series lộ trình ML Compiler Engineer. Giai đoạn tiếp theo: **Giai đoạn 2 — Compiler & Code Generation cho AI** (10 tuần, focus MLIR + TVM).*
