# Tổng kết Giai đoạn 1 — 10 insights chính

> Tự viết, đây là cách consolidate kiến thức 6 tuần.

1.
2.
3.
4.
5.
6.
7.
8.
9.
10.

---

## Kiểm tra kiến thức (trả lời không nhìn note)

1. Giải thích roofline model. Tính AI_critical cho RTX 4090.
2. Vì sao matmul là kernel "lý tưởng" cho AI accelerator, vector add thì không?
3. Mô tả 1 cycle của systolic array. Vì sao có "ramp-up" period?
4. So sánh weight-stationary vs output-stationary. Khi nào dùng cái nào?
5. Vì sao BF16 thay thế FP16 trong training?
6. Quantize asymmetric khác symmetric thế nào? Khi nào dùng?
7. Flash Attention làm gì để tăng AI?
8. Compiler khác nhau thế nào giữa TPU và Tenstorrent?
9. Compile model lên Luckfox NPU có bao nhiêu transformation? Cái nào quan trọng nhất?
10. Design 1 chip AI cho LLM inference: ưu tiên compute density, memory bandwidth,
    hay on-chip memory size?

## 4 intuition compiler đã nắm

1. Compiler không tạo phép tính mới — nó loại bỏ data movement không cần thiết.
2. Compiler phải biết microarchitecture cụ thể (Ampere ≠ Hopper).
3. Cho AI accelerator, compiler không *tối ưu* code — compiler **viết** code.
4. Precision là một *resource* compiler trade-off như memory/compute.

---

*Tiếp theo: Giai đoạn 2 — Compiler & Code Generation cho AI (MLIR + TVM).
Systolic simulator tuần 3 sẽ là target backend cho compiler bạn xây.*
