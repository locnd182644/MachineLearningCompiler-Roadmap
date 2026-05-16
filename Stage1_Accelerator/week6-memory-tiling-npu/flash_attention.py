"""Bài tập 6.2 — Flash Attention (naive version) bằng Triton.

STUB: cài đặt các phần TODO. Tham khảo tutorial Triton chính thức về
Flash Attention.

Ý tưởng: standard attention materialize ma trận attention NxN ra HBM
(memory-bound). Flash Attention tile theo cách giữ block attention trong
SRAM → giảm HBM traffic → tăng AI → nhanh hơn, tốn ít memory hơn.
"""
import torch


def attention_reference(q, k, v):
    """Standard attention — dùng làm ground truth."""
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    return attn @ v


# === TODO: Flash Attention kernel bằng Triton ===
# import triton
# import triton.language as tl
#
# @triton.jit
# def flash_attn_kernel(...):
#     # - Chia Q, K, V thành block.
#     # - Vòng lặp trên block của K/V; với mỗi block:
#     #     tính scores block, online-softmax (running max + running sum),
#     #     cập nhật accumulator output.
#     # - Không bao giờ materialize ma trận attention NxN đầy đủ.
#     ...
def flash_attention(q, k, v):
    """Flash Attention — wrapper gọi Triton kernel.

    TODO: cài đặt kernel Triton + launch.
    """
    raise NotImplementedError


if __name__ == "__main__":
    assert torch.cuda.is_available(), "Cần GPU NVIDIA"
    B, H, S, D = 2, 8, 2048, 64
    q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)

    ref = attention_reference(q, k, v)
    # out = flash_attention(q, k, v)
    # err = (out - ref).abs().max()
    # print("Max error:", err.item())
    #
    # TODO: đo peak memory (torch.cuda.max_memory_allocated) cho cả 2 cách,
    #       quan sát Flash Attention giảm rõ rệt với sequence dài.
    print("Cài đặt flash_attention() rồi bỏ comment phần verify.")
