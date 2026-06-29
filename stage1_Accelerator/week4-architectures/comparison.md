# Tuần 4 — So sánh kiến trúc AI accelerator (Bài tập 4.3)

## Bảng so sánh

| Aspect | TPU v4 | Tenstorrent | Groq TSP | Cerebras WSE-2 |
|--------|--------|-------------|----------|----------------|
| Compute organization | 1 lớn MXU/core | Mesh of tiles | Vector lanes | 850K cores |
| On-chip memory | HBM + UB | Per-tile SRAM | 220 MB SRAM | 40 GB SRAM |
| Off-chip memory | HBM | DDR | None | None |
| Programming model | Static schedule (XLA) | Dataflow (Metalium) | Static (Groq compiler) | Streaming (CSL) |
| Compiler complexity | High | Very high | Extreme | Extreme |
| Best workload | Large dense matmul | Flexible | Low latency inference | Massive parallel |

## Mỗi kiến trúc → mỗi loại compiler

- **TPU (1 big MXU)**: compiler lo tiling cho MXU + memory scheduling.
- **Tenstorrent (mesh)**: compiler lo placement + routing qua NoC.
- **Groq (deterministic)**: compiler phải biết *exact cycle* mọi instruction.
- **Cerebras (wafer-scale)**: compiler lo dataflow trên 850K core.

## Phân tích tradeoff (tự viết — mục tiêu ~3 trang)

### 1. Compute organization: tập trung vs phân tán

#### TPU v4: Monolithic MXU (Ma trận thống nhất)

**Thiết kế**: 1 khối MXU (Matrix Multiply Unit) khổng lồ 128×128 trên mỗi TensorCore, hoạt động như systolic array.

**Ưu điểm**:
- **Throughput cực cao cho dense matmul**: với batch size lớn, MXU đạt gần 100% utilization
- **Đơn giản về control**: 1 instruction stream điều khiển toàn bộ mảng
- **Power efficiency**: systolic dataflow giảm data movement, năng lượng/MAC thấp
- **Compiler đơn giản hơn**: chỉ cần tile để fit MXU, không cần placement trên nhiều cores

**Nhược điểm**:
- **Utilization thấp với workload nhỏ**: matmul 64×64 chỉ dùng 1/4 MXU → 75% PEs idle
- **Kém linh hoạt**: không thể chạy 2 ops độc lập đồng thời trên 1 MXU
- **Load imbalance**: nếu model có layers khác kích thước (attention vs FFN), phải serialize
- **Bottleneck ở HBM**: với MXU lớn cần bandwidth cực cao; HBM bandwidth giới hạn throughput thực tế

**Khi nào tốt**: LLM training với batch lớn (>= 256), sequence length đều, pure transformer (matmul chiếm >90% FLOPs).

---

#### Tenstorrent: Mesh of Tiles (Phân tán đồng nhất)

**Thiết kế**: 100+ Tensix cores (tiles) nối nhau qua NoC 2D mesh. Mỗi tile = 5 RISC-V + compute engine + 1 MB SRAM.

**Ưu điểm**:
- **Flexibility cực cao**: mỗi tile chạy kernel riêng → có thể run 100 ops nhỏ song song
- **Graceful degradation**: workload nhỏ vẫn dùng hết tiles (ví dụ: 100 attention heads → 100 tiles)
- **Composable**: dễ scale chip size (thêm tiles) mà không đổi ISA
- **Programmability**: mô hình giống cluster computing, dễ port distributed algorithms

**Nhược điểm**:
- **NoC congestion**: nếu nhiều tiles cùng đọc 1 vùng DRAM, NoC thành bottleneck
- **Compiler complexity cực cao**: phải solve placement (op nào ở tile nào) + routing (data path qua NoC) → NP-hard problem
- **Synchronization overhead**: barrier giữa tiles qua NoC chậm hơn systolic implicit sync
- **Underutilization dễ xảy ra**: nếu compiler placement không tối ưu, nhiều tiles idle chờ data

**Khi nào tốt**: Workloads heterogeneous (mixture of experts, sparse models), many small parallel ops, hoặc khi cần flexibility cao (research prototyping).

---

#### Groq TSP: Deterministic Vector Lanes

**Thiết kế**: Không phải systolic array, mà là **vector processor** với nhiều lanes. Compiler phải schedule **exact cycle** cho mọi instruction → deterministic execution.

**Ưu điểm**:
- **Latency cực thấp**: không có cache miss, không có branch misprediction → mỗi token inference < 1ms
- **Throughput/latency tradeoff tốt**: với batch nhỏ (1-8), thắng TPU về latency
- **On-chip SRAM 220 MB**: đủ chứa model 7B → zero DRAM access trong inference pass
- **Power efficiency cao**: deterministic → không waste cycles chờ data

**Nhược điểm**:
- **Compiler extreme complexity**: phải schedule exact cycle, không có dynamic dispatch → compile time rất lâu
- **Brittleness**: thay đổi nhỏ (model arch, input shape) → phải recompile
- **Không scale cho model lớn**: SRAM 220 MB → chỉ fit ≤7B params; model >70B phải multi-chip (mất lợi thế latency)
- **Kém linh hoạt**: không thể chạy arbitrary Python code như GPU

**Khi nào tốt**: LLM inference với latency-critical (chatbot, real-time), batch size nhỏ, model ≤7B.

---

#### Cerebras WSE-2: Wafer-Scale Extreme

**Thiết kế**: **850,000 cores** trên 1 wafer (không cắt thành chips nhỏ). Mỗi core đơn giản (1 FPU + SRAM), kết nối qua mesh 2D cực lớn.

**Ưu điểm**:
- **On-chip memory khổng lồ (40 GB)**: toàn bộ activations + weights cho batch lớn đều on-chip → zero DRAM bottleneck
- **Parallelism tuyệt đối**: map mỗi layer neuron → 1 core → dataflow streaming (weight streaming)
- **Throughput đỉnh cao**: training GPT-3 scale nhanh hơn GPU cluster (ít communication)
- **Không cần gradient accumulation**: batch size lên tới 100K+ nhờ SRAM lớn

**Nhược điểm**:
- **Yield & cost**: 1 wafer chứa hàng nghìn defects → phải có redundancy + routing around faults → giá thành >$2M/unit
- **Compiler nightmare**: place 850K cores, route activations qua mesh khổng lồ → tool chain độc quyền, không public
- **Power & cooling**: 1 wafer tiêu thụ 15-20 kW → cần cooling đặc biệt
- **Vendor lock-in**: không thể chạy trên hardware khác, CSL (Cerebras Software Language) chỉ dùng cho Cerebras

**Khi nào tốt**: Training model cực lớn (>100B params) với budget không giới hạn, khi DRAM bandwidth là bottleneck chính.

---

### 2. Memory: on-chip lớn vs HBM

#### Memory hierarchy tradeoff

| Kiến trúc | On-chip SRAM | Off-chip | Bandwidth | Latency |
|-----------|--------------|----------|-----------|---------|
| TPU v4    | ~few MB (Unified Buffer) | HBM 32 GB | 1.2 TB/s | Medium |
| Tenstorrent | 1 MB/tile × 100 = 100 MB | DDR | ~100 GB/s | High (NoC + DDR) |
| Groq      | 220 MB SRAM | None | N/A | Zero (all on-chip) |
| Cerebras  | 40 GB SRAM | None | N/A | Zero (all on-chip) |

#### HBM-centric (TPU): Throughput-oriented

**Model**: Weights ở HBM, streaming vào MXU qua high-bandwidth link.

**Pros**:
- **Capacity không giới hạn**: HBM 32 GB → fit model lớn (70B+)
- **Cost-effective**: HBM rẻ hơn nhiều so với on-chip SRAM ($/GB)
- **Flexible**: thay đổi model không cần redesign chip

**Cons**:
- **Bandwidth wall**: với MXU 128×128 @ 1 GHz, cần ~100 TB/s nếu không reuse weights → HBM chỉ cho 1.2 TB/s → **phải batch lớn** để amortize weight reads
- **Power hungry**: đọc từ HBM tiêu thụ 200x năng lượng so với on-chip RF (theo Horowitz ISSCC'14)

**Implication cho compiler**:
- **Phải tối ưu weight reuse**: tile theo batch dimension, đảm bảo mỗi weight load dùng cho nhiều activations
- **Pipelining**: overlap HBM fetch với compute (double buffering trong Unified Buffer)

---

#### SRAM-centric (Groq, Cerebras): Latency-oriented

**Model**: Toàn bộ model weights fit on-chip → zero external memory access.

**Pros**:
- **Latency deterministic**: không có DRAM access → mỗi op latency predictable
- **Energy efficiency đỉnh**: chỉ dùng SRAM (energy/access thấp hơn DRAM 100x)
- **Throughput không bị DRAM throttle**: batch size nhỏ vẫn đạt peak FLOPs

**Cons**:
- **Capacity limit nghiêm trọng**:
  - Groq 220 MB → chỉ fit ~7B params (fp16) hoặc 14B (int8)
  - Cerebras 40 GB → fit ~70B params (fp16)
  - Không thể chạy GPT-4 scale (>1T params) trên 1 chip
- **Cost prohibitive**: SRAM >>$$ so với HBM; Groq/Cerebras chips cực đắt
- **Inflexible**: model lớn hơn on-chip SRAM → phải multi-chip (mất lợi thế)

**Implication cho compiler**:
- **Weight placement critical**: phải pack weights vào SRAM sao cho minimize fragmentation
- **Model partitioning**: với multi-chip, compiler phải cut model graph theo memory constraint

---

#### Hybrid (Tenstorrent): Distributed SRAM + DDR

**Model**: Mỗi tile có 1 MB SRAM (local), DDR dùng như "swap space".

**Pros**:
- **Balance**: local SRAM cho reuse, DDR cho capacity
- **Scalable**: thêm tiles → tổng SRAM tăng tuyến tính

**Cons**:
- **NoC bottleneck**: nếu tile A cần data từ tile B, phải qua NoC (latency cao hơn local SRAM)
- **DDR bandwidth thấp**: ~100 GB/s (vs TPU HBM 1.2 TB/s) → bottleneck nếu weights không fit trong tổng SRAM

**Implication cho compiler**:
- **Data locality critical**: compiler phải place ops gần data (minimize NoC hops)
- **Prefetching**: overlap DDR fetch với compute trên tiles khác

---

### 3. Programming model: static vs dataflow vs streaming

#### Static scheduling (TPU, Groq)

**Model**: Compiler tạo ra **static schedule** (sequence of instructions) trước khi chạy. Runtime chỉ execute theo schedule, không có dynamic decisions.

**Ví dụ (XLA trên TPU)**:
```
// Compiler lowers:
C = matmul(A, B)
D = relu(C)

// Thành IR:
%c = hlo.dot(%a, %b)        // → MXU instruction
%d = hlo.max(%c, 0)         // → vector instruction
```
Compiler đã quyết định: tile size, memory addresses, execution order. Runtime chỉ phát lệnh.

**Pros**:
- **Predictable performance**: không có runtime overhead (scheduling, dispatch)
- **Compiler optimization powerful**: có thể fuse ops, reorder, tối ưu toàn cục
- **Deterministic**: cùng input → cùng latency (tốt cho real-time)

**Cons**:
- **Inflexible**: dynamic shapes (variable sequence length) khó xử lý → phải recompile hoặc pad
- **Compile time lâu**: tối ưu toàn cục phức tạp (Groq compiler mất hàng giờ)
- **Không interactive**: không chạy được arbitrary Python (như GPU)

---

#### Dataflow (Tenstorrent, Eyeriss)

**Model**: Programmer/compiler định nghĩa **dataflow graph** (nodes = ops, edges = data dependencies). Runtime schedule dynamically based on data arrival.

**Ví dụ (Metalium)**:
```cpp
// Programmer viết kernels:
Kernel reader = [fetch A_tile from DRAM → CB_in]
Kernel compute = [wait CB_in ready → matmul → CB_out]
Kernel writer = [wait CB_out ready → write DRAM]

// Runtime tự động schedule:
// reader, compute, writer chạy pipeline khi data sẵn sàng
```

**Pros**:
- **Flexibility cao**: dễ handle dynamic shapes, conditional execution
- **Pipeline tự nhiên**: producer-consumer qua circular buffers → overlap I/O và compute
- **Composable**: dễ thêm/bớt ops trong graph

**Cons**:
- **Runtime overhead**: phải check CB status, dispatch kernels → thêm latency
- **Compiler phức tạp hơn**: không kiểm soát exact timing → khó optimize
- **Debugging khó**: execution order không deterministic → race conditions

---

#### Streaming dataflow (Cerebras)

**Model**: Mỗi core = 1 node trong dataflow graph; activations **stream** qua mesh (không buffer ở DRAM).

**Ví dụ (weight-streaming cho FFN)**:
```
Input activations broadcast từ 1 core → fan-out qua mesh
Mỗi core ở layer_i chứa 1 hàng weights W[i, :]
Core tính inner product với activations stream qua
Partial sums flow lên layer_i+1
```

**Pros**:
- **Zero DRAM bottleneck**: activations không write-back, chảy thẳng qua cores
- **Latency thấp**: pipelining tự nhiên (compute layer_i+1 trong khi layer_i chưa xong hết batch)
- **Energy efficiency**: không lưu intermediate activations ra memory

**Cons**:
- **Synchronization phức tạp**: phải đảm bảo activations arrive đúng cycle ở mỗi core
- **Fault tolerance khó**: 1 core chết → toàn pipeline dừng (phải có redundancy)
- **Compiler extreme**: phải map dataflow graph lên 850K cores

---

### Kết luận: chọn kiến trúc nào cho LLM inference?

#### Kịch bản 1: Chatbot real-time (latency < 10ms/token)

**Chọn: Groq TSP**

**Lý do**:
- Latency là yếu tố quan trọng nhất → deterministic execution + on-chip SRAM (220 MB) thắng
- Model size nhỏ (7B) fit đủ trong SRAM
- Batch size nhỏ (1-4 users) → TPU underutilize

**Tradeoff chấp nhận**: compile time lâu (acceptable vì model ít thay đổi), không chạy được model >7B.

---

#### Kịch bản 2: Batch inference cho API service (batch size 128-512)

**Chọn: TPU v4**

**Lý do**:
- Batch lớn → MXU utilization cao (>95%)
- Throughput/$ tốt nhất (TPU rẻ hơn Groq/Cerebras)
- HBM đủ chứa model 70B, dễ upgrade model

**Tradeoff chấp nhận**: latency per token cao hơn (20-50ms), nhưng total throughput thắng.

---

#### Kịch bản 3: Training model 100B+ params

**Chọn: Cerebras WSE-2**

**Lý do**:
- On-chip SRAM 40 GB → fit toàn bộ activations + optimizer states cho batch lớn
- Zero DRAM bottleneck → training throughput gấp 10x GPU cluster tương đương FLOPs
- Batch size lên tới 100K+ → convergence nhanh hơn

**Tradeoff chấp nhận**: giá cực cao ($2M+), vendor lock-in (CSL), không flexible.

---

#### Kịch bản 4: Research prototype với model architectures mới (mixture of experts, sparse)

**Chọn: Tenstorrent**

**Lý do**:
- Flexibility cao: mỗi tile chạy kernel riêng → dễ implement custom ops
- Programmability: viết kernels giống CUDA, debug được
- Mesh architecture tự nhiên cho sparse workloads (mỗi expert → 1 tile)

**Tradeoff chấp nhận**: compiler chưa mature (phải tune placement/routing thủ công), throughput thấp hơn TPU cho dense matmul.

---

### Tổng kết: Không có kiến trúc "tốt nhất"

| Workload | Kiến trúc tốt nhất | Metric quan trọng |
|----------|-------------------|------------------|
| Latency-critical inference (small batch) | Groq | Latency/token |
| Throughput inference (large batch) | TPU | Throughput/$ |
| Training model khổng lồ | Cerebras | Throughput tuyệt đối |
| Flexible/research | Tenstorrent | Programmability |

**Compiler implications**:
- **TPU compiler**: tối ưu tiling cho MXU + memory scheduling (XLA/HLO)
- **Groq compiler**: exact cycle scheduling (SAT solver, constraint programming)
- **Tenstorrent compiler**: placement + routing (graph partitioning, NoC routing algorithms)
- **Cerebras compiler**: dataflow mapping lên 850K cores (streaming dataflow synthesis)

Mỗi kiến trúc → 1 class compiler problems khác nhau, từ "khó" (TPU) tới "extremely hard" (Cerebras).
