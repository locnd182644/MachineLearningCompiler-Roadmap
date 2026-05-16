"""Bài tập 5.1 — Implement quantization tay (symmetric INT8 per-tensor)."""
import torch


def quantize_symmetric(x: torch.Tensor, n_bits: int = 8):
    """Symmetric quantization per-tensor."""
    qmax = 2 ** (n_bits - 1) - 1  # 127 cho INT8
    scale = x.abs().max() / qmax
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale


def dequantize(x_q, scale):
    return x_q.float() * scale


if __name__ == "__main__":
    A = torch.randn(256, 256)
    B = torch.randn(256, 256)

    # Reference: FP32
    C_fp32 = A @ B

    # Quantize A và B
    A_q, sA = quantize_symmetric(A)
    B_q, sB = quantize_symmetric(B)

    # INT8 matmul (đây là điều TPU/NPU làm trong hardware).
    # Accumulator INT32: dot product 256-deep có thể overflow INT8.
    C_int = (A_q.float() @ B_q.float()).to(torch.int32)
    C_dq = C_int.float() * (sA * sB)

    rel_err = (C_dq - C_fp32).abs().mean() / C_fp32.abs().mean()
    print(f"Mean relative error: {rel_err * 100:.2f}%")

    # TODO mở rộng (tùy chọn):
    #  - Thêm asymmetric quantization (zero-point).
    #  - Per-channel scale thay vì per-tensor; so sai số.
