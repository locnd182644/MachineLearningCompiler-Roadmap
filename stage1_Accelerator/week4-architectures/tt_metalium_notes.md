# Tuần 4 — Notes: tt-metalium (Tenstorrent)

> Lần đầu đọc code compiler-runtime cho AI chip thật, mã nguồn mở.

## Build

```bash
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal
./build_metal.sh   # cần Linux, đọc README cho dependency
```

### Quá trình build

#### Dependencies cần thiết

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git wget \
    libhwloc-dev libboost-all-dev \
    libyaml-cpp-dev libzmq3-dev \
    python3-dev python3-pip

# Python packages
pip3 install pybind11 pyyaml
```

#### Build steps

```bash
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal

# Set environment
export ARCH_NAME=grayskull  # hoặc wormhole tùy phần cứng
export TT_METAL_HOME=$(pwd)

# Build runtime + kernels
./build_metal.sh

# Build mất ~10-20 phút tùy máy
# Output: tt_metal/build/lib/libtt_metal.so
```

#### Lỗi thường gặp

1. **Submodule thiếu**: phải clone với `--recurse-submodules`
2. **CMake version cũ**: cần CMake >= 3.16
3. **Không có hardware**: vẫn build được, nhưng không chạy được kernels (cần Tenstorrent card hoặc simulator)
4. **Boost version**: cần boost >= 1.71

#### Kết quả build thành công

- `tt_metal/build/lib/` chứa shared libraries
- `tt_metal/build/programming_examples/` chứa examples đã compile
- Có thể chạy unit tests: `./build/test/tt_metal/unit_tests`

## Example đã đọc

### `tt_metal/programming_examples/eltwise_binary`

#### Tổng quan
Example này thực hiện phép toán element-wise binary (ví dụ: A + B) trên 2 tensor.

#### Kernel C++ làm gì?

Kernel có 3 phần riêng biệt, chạy trên các RISC-V cores khác nhau trong Tensix:

1. **Data movement kernel (reader)**: 
   - Chạy trên RISC-V core chuyên đọc dữ liệu
   - Đọc tiles từ DRAM → circular buffer (CB) trong L1 SRAM
   - Code: `noc_async_read()` để fetch data qua NoC

2. **Compute kernel**:
   - Chạy trên compute engine (FPU/SFPU)
   - Đọc tiles từ CB input → compute → ghi CB output
   - Dùng intrinsics như `add_tiles()` hoặc `mul_tiles()`

3. **Data movement kernel (writer)**:
   - Đọc kết quả từ CB output → ghi về DRAM qua NoC
   - Code: `noc_async_write()`

#### Data đi vào core thế nào (circular buffer)?

**Circular Buffer (CB)** = vùng SRAM L1 trong mỗi Tensix core, hoạt động như queue:
```
Host → DRAM → [NoC] → CB_input → Compute → CB_output → [NoC] → DRAM
```

- CB cho phép overlap: reader fetch batch tiếp theo trong khi compute xử lý batch hiện tại (double buffering)
- Kích thước CB = số tiles × tile_size (thường 32×32 với bfloat16 = 2KB/tile)
- Producer/consumer sync qua `cb_reserve_back()`, `cb_push_back()`, `cb_wait_front()`, `cb_pop_front()`

#### Vai trò của host code vs kernel code

**Host code (C++ trên CPU)**:
- Khởi tạo device, allocate buffers trong DRAM
- Biên dịch kernels (compile time hoặc JIT)
- Dispatch kernels lên cores với runtime args (địa chỉ DRAM, kích thước)
- Đồng bộ hóa: đợi kernels hoàn thành
- Ví dụ:
  ```cpp
  Program program = CreateProgram();
  KernelHandle reader = CreateKernel(program, "reader.cpp", core_spec);
  SetRuntimeArgs(program, reader, {src_addr, num_tiles});
  EnqueueProgram(device, program);
  ```

**Kernel code (C++ biên dịch cho RISC-V)**:
- Chạy trên Tensix cores, không truy cập được host memory
- Làm việc với địa chỉ DRAM (device-side) và L1 SRAM
- Lightweight: không có malloc/printf/exception
- Mỗi kernel độc lập, sync qua CB hoặc semaphores

**Tách biệt này giống CUDA**: host orchestrates, device executes.

### `tt_metal/programming_examples/matmul_single_core`

#### Compiler tile matmul ra sao?

**Tiling strategy**:
```
C[M, N] = A[M, K] @ B[K, N]
```

Chia thành tiles 32×32 (kích thước native của compute engine):
```
for m_tile in range(M // 32):
    for n_tile in range(N // 32):
        acc = zeros(32, 32)
        for k_tile in range(K // 32):
            A_tile = A[m_tile, k_tile]  # 32×32
            B_tile = B[k_tile, n_tile]  # 32×32
            acc += A_tile @ B_tile      # tile matmul (hardware)
        C[m_tile, n_tile] = acc
```

**Compiler không tự động tile** (ở level này): programmer phải:
1. Chia tensor thành tiles trong host code
2. Viết vòng lặp k_tile trong compute kernel
3. Manage CB cho input/output tiles

**Single-core** = không dùng mesh, tất cả tiles xử lý tuần tự trên 1 Tensix core.

#### Dispatch lên core thế nào?

**Host dispatch flow**:
```cpp
// 1. Allocate DRAM buffers
Buffer a_buf = CreateBuffer(device, A_size_bytes);
Buffer b_buf = CreateBuffer(device, B_size_bytes);
Buffer c_buf = CreateBuffer(device, C_size_bytes);

// 2. Write input data
EnqueueWriteBuffer(queue, a_buf, A_host_data);
EnqueueWriteBuffer(queue, b_buf, B_host_data);

// 3. Create program + kernels
Program program = CreateProgram();
CoreCoord core = {0, 0};  // chọn core (0,0)

auto reader = CreateKernel(program, "reader_matmul.cpp", core);
auto compute = CreateKernel(program, "matmul_compute.cpp", core);
auto writer = CreateKernel(program, "writer_matmul.cpp", core);

// 4. Set runtime args (DRAM addresses, tile counts)
SetRuntimeArgs(program, reader, {
    a_buf.address(), b_buf.address(), M_tiles, K_tiles, N_tiles
});
SetRuntimeArgs(program, writer, {c_buf.address(), M_tiles, N_tiles});

// 5. Enqueue program (async dispatch)
EnqueueProgram(queue, program);

// 6. Read back result
EnqueueReadBuffer(queue, c_buf, C_host_data);
Finish(queue);  // đợi hoàn thành
```

**Phía kernel**: runtime args truyền vào qua `get_arg_val<uint32_t>(arg_idx)`.

#### So với systolic array tuần 3: khác ở đâu?

| Aspect | Systolic Array (TPU-style) | Tenstorrent Single-Core |
|--------|----------------------------|-------------------------|
| **Dataflow** | Weight-stationary: weights ở yên, activations chảy qua | Explicit tiling: programmer fetch tiles từ DRAM |
| **Parallelism** | Spatial: mảng PEs hoạt động đồng thời | Temporal: 1 core xử lý tiles tuần tự |
| **Control** | Implicit: data arrival drives compute | Explicit: programmer viết vòng lặp k_tile |
| **Programming** | Compiler map 1 big matmul → toàn mảng | Programmer quản lý circular buffers, NoC transfers |
| **Efficiency** | Cao cho dense matmul lớn (reuse weights) | Thấp (single-core), nhưng flexible |
| **Scalability** | Cần mảng lớn (256×256 PEs) | Scale = dùng nhiều cores + NoC (multi-core matmul) |

**Systolic** = hardware dataflow graph tự chạy.  
**TT single-core** = software-managed tiling, giống chạy matmul trên 1 GPU SM.

**Điểm mạnh của TT**: lập trình rõ ràng, dễ debug; có thể làm ops phức tạp hơn matmul.  
**Điểm yếu**: phải scale lên multi-core mới competitive (cần compiler placement + routing).

## Kiến trúc Tensix (tóm tắt)

- Mỗi tile (Tensix core): 5 RISC-V baby core + compute engine (SFPU + FPU matrix)
- Tiles nối nhau qua NoC 2D mesh
- 1 chip = 100+ tiles
- Compiler phải: **phân hoạch** model lên tile + **route** data qua NoC

## Insight cho compiler

> Compiler cho mesh-of-cores giống compiler cho distributed system:
> placement (op nào ở tile nào) + routing (data qua NoC).
