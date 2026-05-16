"""Bài tập 1.2 — Roofline lý thuyết cho 5 kernel.

Sửa PEAK_FLOPS / PEAK_BW theo GPU của bạn (xem nvidia-smi + datasheet).
"""


def roofline_analysis(name, flops, bytes_accessed, peak_flops, peak_bw):
    ai = flops / bytes_accessed
    perf_compute = peak_flops
    perf_memory = ai * peak_bw
    achievable = min(perf_compute, perf_memory)
    bound = "compute-bound" if perf_compute < perf_memory else "memory-bound"
    print(f"{name}: AI={ai:.2f} FLOPs/B, "
          f"achievable={achievable / 1e12:.2f} TFLOPS, {bound}")
    return ai, achievable, bound


# --- Thông số GPU: SỬA THEO MÁY BẠN ---
# Ví dụ RTX 3060: ~13 TFLOPS FP32, ~360 GB/s
PEAK_FLOPS = 13e12
PEAK_BW = 360e9

if __name__ == "__main__":
    results = []

    # Kernel 1: Vector add c[N] = a[N] + b[N], FP32
    N = 10_000_000
    results.append(("VecAdd",
                     roofline_analysis("VecAdd", N, 3 * N * 4, PEAK_FLOPS, PEAK_BW)))

    # Kernel 2: Dot product
    results.append(("DotProd",
                     roofline_analysis("DotProd", 2 * N, 2 * N * 4, PEAK_FLOPS, PEAK_BW)))

    # Kernel 3: Matmul MxNxK
    M = N = K = 4096
    results.append(("Matmul 4K",
                     roofline_analysis("Matmul 4K", 2 * M * N * K,
                                       (M * K + K * N + M * N) * 4,
                                       PEAK_FLOPS, PEAK_BW)))

    # Kernel 4: Conv2d 256x256, 3x3 kernel, 64->128 channels
    H = W = 256
    C_in, C_out, KH, KW = 64, 128, 3, 3
    conv_flops = 2 * H * W * C_in * C_out * KH * KW
    conv_bytes = (H * W * C_in + KH * KW * C_in * C_out + H * W * C_out) * 4
    results.append(("Conv2d",
                     roofline_analysis("Conv2d", conv_flops, conv_bytes,
                                       PEAK_FLOPS, PEAK_BW)))

    # Kernel 5: Softmax over [B, S, S] (attention scores)
    B, S = 32, 2048
    results.append(("Softmax",
                     roofline_analysis("Softmax", 3 * B * S * S, 2 * B * S * S * 4,
                                       PEAK_FLOPS, PEAK_BW)))

    ai_crit = PEAK_FLOPS / PEAK_BW
    print(f"\nAI_critical = {ai_crit:.1f} FLOPs/byte")

    # TODO (bài tập): vẽ plot roofline log-log với matplotlib.
    #   - Trục x: AI (FLOPs/byte), trục y: performance (FLOPS)
    #   - Vẽ 2 "mái": đường nghiêng (AI * peak_bw) và đường ngang (peak_flops)
    #   - Chấm 5 kernel lên đồ thị, lưu thành roofline.png
    #   Gợi ý: dùng plt.loglog, đường roof = [min(ai), ai_crit, max(ai)]
