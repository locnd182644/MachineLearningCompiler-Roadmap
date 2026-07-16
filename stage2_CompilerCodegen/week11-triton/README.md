# Tuần 11 — Triton: DSL viết kernel GPU

> **Câu hỏi central:** *Triton tự động hóa được gì mà CUDA bắt bạn làm tay — và cái giá phải trả là gì?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-11--triton-dsl-viết-kernel-gpu)
> 🔗 Tutorials: https://triton-lang.org/main/getting-started/tutorials/

## Cấu trúc dự kiến

```
week11-triton/
├── README.md                 # File này
├── kernels/
│   ├── 01_vector_add.py
│   ├── 02_softmax.py         # fused softmax + benchmark vs PyTorch
│   ├── 03_matmul.py          # autotuned matmul + % cuBLAS
│   └── 04_flash_attention.py # ⭐ project chính
├── benchmarks/
│   ├── results.md            # bảng số liệu tất cả kernel
│   └── flash_attn_scaling.png # seq_len 512→8K
├── ir_dumps/                 # Triton IR / TritonGPU IR / PTX
└── triton_ir_notes.md        # pipeline Triton IR → PTX
```

## TODO

### Lý thuyết (6h)
- [ ] Đọc Triton paper (Tillet et al. 2019) — nắm block-level programming model
- [ ] Note: bảng "CUDA bắt làm tay vs Triton tự lo" (thread mapping, shared mem, coalescing, bank conflict, pipelining)
- [ ] Đọc về Triton compile pipeline: Python AST → Triton IR → TritonGPU IR → LLVM → PTX
- [ ] Ôn lại tuần 2 + tuần 6 stage 1 (CUDA tiled matmul) để so sánh

### Bài tập 11.1 — Vector add + Softmax (tutorial 1-2)
- [ ] Vector add: hiểu `tl.program_id`, `tl.load/store`, masking
- [ ] Fused softmax: chạy + benchmark vs `torch.softmax`
- [ ] Trả lời trong note: vì sao fused softmax nhanh hơn eager? (liên hệ roofline tuần 1 stage 1)

### Bài tập 11.2 — Matmul (tutorial 3)
- [ ] Chạy tutorial matmul với `@triton.autotune` configs
- [ ] Benchmark: đạt bao nhiêu % cuBLAS trên GPU của mình?
- [ ] Dump PTX, xác nhận có `mma.sync` (tensor core) hay không
- [ ] Thử sửa BLOCK_SIZE xấu đi → đo lại → hiểu autotune chọn gì

### Bài tập 11.3 — Flash Attention ⭐ project chính
- [ ] Implement Flash Attention forward: online softmax (rescaling trick) đúng chuẩn
- [ ] Thêm causal masking
- [ ] Verify correctness vs `F.scaled_dot_product_attention` (atol 1e-2 cho fp16)
- [ ] Benchmark seq_len 512 → 8K, plot scaling curve
- [ ] Đo HBM traffic bằng Nsight Compute → chứng minh O(N) memory thay vì O(N²)
- [ ] (Optional nâng cao) Backward pass hoặc so với bản Triton chính thức

### Bài tập 11.4 — Mổ xẻ IR
- [ ] Dump Triton IR, TritonGPU IR, PTX của matmul kernel (`TRITON_KERNEL_DUMP=1`)
- [ ] Note: layout encodings (blocked/mma/dot-operand) nghĩa là gì
- [ ] Note: shared memory được chèn ở tầng nào? software pipelining ở đâu?
- [ ] Viết `triton_ir_notes.md` — đối chiếu với kiến thức MLIR tuần 8-10

### Output cuối tuần
- [ ] 4+ kernels có benchmark trong `results.md`
- [ ] Flash Attention cạnh tranh SDPA + memory analysis
- [ ] `triton_ir_notes.md`
- [ ] (Optional) Blog post tuần 11
