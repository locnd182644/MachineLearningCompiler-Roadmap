# Tuần 4 — Notes: tt-metalium (Tenstorrent)

> Lần đầu đọc code compiler-runtime cho AI chip thật, mã nguồn mở.

## Build

```bash
git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git
cd tt-metal
./build_metal.sh   # cần Linux, đọc README cho dependency
```

### Quá trình build

> Ghi lại: dependency phải cài, lỗi gặp phải, thời gian build, kết quả.

## Example đã đọc

### `tt_metal/programming_examples/eltwise_binary`

> - Kernel C++ làm gì?
> - Data đi vào core thế nào (circular buffer)?
> - Vai trò của host code vs kernel code?

### `tt_metal/programming_examples/matmul_single_core`

> - Compiler tile matmul ra sao?
> - Dispatch lên core thế nào?
> - So với systolic array tuần 3: khác ở đâu?

## Kiến trúc Tensix (tóm tắt)

- Mỗi tile (Tensix core): 5 RISC-V baby core + compute engine (SFPU + FPU matrix)
- Tiles nối nhau qua NoC 2D mesh
- 1 chip = 100+ tiles
- Compiler phải: **phân hoạch** model lên tile + **route** data qua NoC

## Insight cho compiler

> Compiler cho mesh-of-cores giống compiler cho distributed system:
> placement (op nào ở tile nào) + routing (data qua NoC).
