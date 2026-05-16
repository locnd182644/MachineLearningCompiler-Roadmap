"""Bài tập 1.3 — Đo thực tế trên GPU và so sánh với lý thuyết.

Cần GPU NVIDIA + PyTorch CUDA. So kết quả với roofline_theory.py.
"""
import time

import torch

device = "cuda"


def bench(fn, name, flops, bytes_):
    # warmup
    for _ in range(10):
        fn()
    torch.cuda.synchronize()

    t0 = time.time()
    iters = 100
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t = (time.time() - t0) / iters

    tflops = flops / t / 1e12
    bw = bytes_ / t / 1e9
    print(f"{name}: {t * 1000:.2f}ms, {tflops:.2f} TFLOPS, {bw:.1f} GB/s")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Cần GPU NVIDIA"
    print("GPU:", torch.cuda.get_device_name(0))

    # Matmul — kỳ vọng gần peak compute
    a = torch.randn(4096, 4096, device=device)
    b = torch.randn(4096, 4096, device=device)
    bench(lambda: a @ b, "Matmul 4K", 2 * 4096 ** 3, 3 * 4096 ** 2 * 4)

    # Vector add — kỳ vọng gần peak bandwidth
    x = torch.randn(10_000_000, device=device)
    y = torch.randn(10_000_000, device=device)
    bench(lambda: x + y, "VecAdd", 10_000_000, 3 * 10_000_000 * 4)

    # Câu hỏi phải trả lời (ghi vào README.md):
    #  - Matmul đạt bao nhiêu % peak FLOPS?
    #  - VecAdd đạt bao nhiêu % peak BW?
    #  - Vì sao matmul gần peak compute còn vecadd gần peak BW? -> roofline.
