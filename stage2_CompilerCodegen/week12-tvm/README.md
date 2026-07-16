# Tuần 12 — TVM: Tensor Expressions & Auto-tuning

> **Câu hỏi central:** *Thay vì con người viết schedule (Triton) hay pass cố định (MLIR), có thể để máy TÌM KIẾM ra schedule tối ưu không?*
>
> 📖 Chi tiết lý thuyết & bài tập: [README giai đoạn 2](../README.md#tuần-12--tvm-tensor-expressions--auto-tuning)

## Cấu trúc dự kiến

```
week12-tvm/
├── README.md                   # File này
├── 01_resnet_compile.ipynb     # Import + compile + benchmark ResNet18
├── 02_manual_schedule.py       # TE/TIR schedule tay từng bước
├── 03_metaschedule_tune.py     # Auto-tuning matmul + conv2d
├── results/
│   ├── schedule_progression.md # bảng: schedule → GFLOPS
│   └── tuning_comparison.md    # naive / tay / MetaSchedule / vendor
├── best_schedules/             # export schedule MetaSchedule tìm ra
└── three_philosophies.md       # ⭐ essay MLIR vs Triton vs TVM
```

## TODO

### Lý thuyết (6h)
- [ ] Đọc TVM paper (Chen et al. 2018, OSDI)
- [ ] Nắm chắc compute/schedule separation (nguồn gốc: Halide) — note với ví dụ
- [ ] Đọc overview pipeline hiện đại: Relax → TE → TensorIR → codegen
- [ ] Đọc MetaSchedule docs (thay AutoTVM/Ansor)
- [ ] Xem CMU 10-414 các bài giảng về ML compilation (Tianqi Chen)

### Bài tập 12.1 — Cài TVM + first compile
- [ ] Cài TVM (pip `apache-tvm` hoặc build source nếu cần CUDA)
- [ ] Import ResNet18 từ PyTorch (relax frontend / torch.export)
- [ ] Compile target `llvm` (CPU) và `cuda`, chạy đúng kết quả
- [ ] Benchmark vs PyTorch eager, ghi số liệu

### Bài tập 12.2 — TE/TIR schedule tay
- [ ] Viết matmul 1024³ bằng TE, schedule mặc định → đo GFLOPS
- [ ] Apply từng bước, đo sau MỖI bước: `tile` → `reorder` → `vectorize` → `parallel`
- [ ] Trên GPU: thêm `bind` thread/block + `cache_read` shared memory
- [ ] Lập bảng `schedule_progression.md` — tái hiện hành trình naive→tiled tuần 2 stage 1 bằng schedule primitives
- [ ] In TIR sau mỗi primitive (`mod.script()`) — thấy loop transform tương ứng

### Bài tập 12.3 — MetaSchedule auto-tuning
- [ ] Tune matmul với MetaSchedule (~1000 trials)
- [ ] Tune conv2d
- [ ] Bảng so sánh: naive TIR / schedule tay / MetaSchedule / cuBLAS+cuDNN
- [ ] Export + đọc best schedule → trả lời: **máy tìm ra trick nào mình không nghĩ tới?**
- [ ] Note thời gian tuning — cái giá của search

### Bài tập 12.4 — Essay "3 triết lý codegen" ⭐
- [ ] Viết 2 trang so sánh MLIR (pass-based) vs Triton (DSL + autotune nhỏ) vs TVM (search-based)
- [ ] Trục so sánh: ai quyết định schedule / ưu / nhược / ai đang dùng trong industry
- [ ] Kết luận: kiến trúc chip nào hợp với triết lý nào (liên hệ tuần 4 stage 1)

### Output cuối tuần
- [ ] Notebook compile + benchmark ResNet18
- [ ] `schedule_progression.md` + `tuning_comparison.md` với số liệu thật
- [ ] `three_philosophies.md`
- [ ] (Optional) Blog post tuần 12
