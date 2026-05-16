# Stage 1 — Nền tảng Kiến trúc Accelerator

Framework thực hành cho **Giai đoạn 1** của lộ trình ML Compiler Engineer.
Curriculum đầy đủ: [`giai-doan-1-nen-tang-kien-truc-accelerator.md`](./giai-doan-1-nen-tang-kien-truc-accelerator.md).

## Cấu trúc

| Thư mục | Câu hỏi central | Output chính |
|---------|-----------------|--------------|
| `week1-roofline/` | Vì sao matmul nhanh hơn vector add? | Roofline analysis lý thuyết vs đo thực tế |
| `week2-cpu-gpu/` | GPU nhanh hơn CPU vì sao, cụ thể? | AVX matmul + 3 phiên bản CUDA matmul |
| `week3-systolic/` | Vì sao TPU dùng systolic array? | Systolic array simulator (asset cho Stage 2) |
| `week4-architectures/` | Có cách thiết kế nào khác systolic? | Eyeriss sim + bảng so sánh kiến trúc |
| `week5-quantization/` | Khi nào INT8 đủ, khi nào không? | Quantization tự cài + thực nghiệm precision |
| `week6-memory-tiling-npu/` | Áp dụng tất cả lên 1 chip thật? | Tiled matmul + Flash Attention + Luckfox NPU |

## Quy ước code

- **Code đầy đủ**: các file đã có code mẫu trong curriculum — chạy được ngay.
- **Stub có `TODO`**: các bài tập mở rộng (extend simulator, Eyeriss, Luckfox...) —
  bạn tự cài đặt phần lõi. Đây đúng tinh thần *đọc → làm → đo → giải thích*.

Mỗi thư mục tuần có `README.md` riêng liệt kê bài tập, lệnh build/run và checklist output.

## Setup môi trường

### Python (dùng chung cho mọi tuần)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `rknn-toolkit2` (tuần 6) cài riêng — xem `week6-memory-tiling-npu/luckfox/README.md`.

### C++ / CUDA (tuần 1, 2, 6)

```bash
sudo apt install build-essential cmake ninja-build
# CUDA Toolkit cài từ NVIDIA nếu chưa có; kiểm tra: nvidia-smi && nvcc --version
```

Mỗi tuần có C++/CUDA đều kèm `CMakeLists.txt`:

```bash
cd week1-roofline && cmake -B build -G Ninja && cmake --build build
```

## Tiến độ

- [ ] Tuần 1 — Roofline & Memory Wall
- [ ] Tuần 2 — CPU SIMD & GPU SIMT
- [ ] Tuần 3 — TPU & Systolic Array
- [ ] Tuần 4 — Dataflow & Spatial Architectures
- [ ] Tuần 5 — Numerical Formats & Quantization
- [ ] Tuần 6 — Memory Hierarchy & Luckfox NPU
