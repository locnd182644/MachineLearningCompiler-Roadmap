"""Bài tập 1.2 — Roofline lý thuyết cho 5 kernel.

Sửa PEAK_FLOPS / PEAK_BW theo GPU của bạn (xem nvidia-smi + datasheet).
"""

import matplotlib

matplotlib.use("Agg")  # backend không cần màn hình, chỉ ghi ra file
import matplotlib.pyplot as plt


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
# ----- RTX 4050: ~9 TFLOPS FP32, ~192 GB/s
PEAK_FLOPS = 9e12
PEAK_BW = 192e9

if __name__ == "__main__":
    results = []

    # Kernel 1: Vector add c[N] = a[N] + b[N], FP32
    ## FLOPs = N (mỗi phần tử 1 phép cộng)
    ## Bytes = 3 * N * 4 (đọc a[N], b[N], ghi c[N], mỗi phần tử 4 byte)
    N = 10_000_000
    results.append(("VecAdd",
                     roofline_analysis("VecAdd", N, 3 * N * 4, PEAK_FLOPS, PEAK_BW)))

    # Kernel 2: Dot product sum(a[i]*b[i])
    ## FLOPs = 2 * N (N phép nhân + N-1 phép cộng, xấp xỉ 2N)
    ## Bytes = 2 * N * 4 (đọc a[N], b[N], mỗi phần tử 4 byte, kết quả nhỏ gọn chỉ vài byte nên bỏ qua)
    results.append(("DotProd",
                     roofline_analysis("DotProd", 2 * N, 2 * N * 4, PEAK_FLOPS, PEAK_BW)))

    # Kernel 3: Matmul MxNxK
    ### FLOPs = 2 * M * N * K (M*N*K nhân + M*N*(K-1) cộng, xấp xỉ 2MNK)
    ### Bytes = (M*K + K*N + M*N) * 4 (đọc A[M*K], B[K*N], ghi C[M*N], (hai ma trận input + một ma trận output), mỗi phần tử 4 byte)
    M = N = K = 4096
    results.append(("Matmul 4K",
                     roofline_analysis("Matmul 4K", 2 * M * N * K,
                                       (M * K + K * N + M * N) * 4,
                                       PEAK_FLOPS, PEAK_BW)))

    # Kernel 4: Conv2d 256x256, 3x3 kernel, 64->128 channels
    ## FLOPs = 2 * H * W * C_in * C_out * KH * KW (mỗi output pixel = KH*KW*C_in nhân + KH*KW*C_in-1 cộng, xấp xỉ 2*H*W*C_in*C_out*KH*KW)
    ## Bytes = (H*W*C_in + KH*KW*C_in*C_out + H*W*C_out) * 4 (đọc input feature map, đọc kernel weights, ghi output feature map, mỗi phần tử 4 byte)
    H = W = 256
    C_in, C_out, KH, KW = 64, 128, 3, 3
    conv_flops = 2 * H * W * C_in * C_out * KH * KW
    conv_bytes = (H * W * C_in + KH * KW * C_in * C_out + H * W * C_out) * 4
    results.append(("Conv2d",
                     roofline_analysis("Conv2d", conv_flops, conv_bytes,
                                       PEAK_FLOPS, PEAK_BW)))

    # Kernel 5: Softmax over [B, S, S] (attention scores)
    ## FLOPs = 3 * B * S * S (mỗi phần tử: exp + cộng dồn + chia, xấp xỉ 3 FLOPs)
    ## Bytes = 2 * B * S * S * 4 (đọc input và ghi output, mỗi phần tử 4 byte, bỏ qua các hằng số nhỏ khác)
    B, S = 32, 2048
    results.append(("Softmax",
                     roofline_analysis("Softmax", 3 * B * S * S, 2 * B * S * S * 4,
                                       PEAK_FLOPS, PEAK_BW)))

    ai_crit = PEAK_FLOPS / PEAK_BW
    print(f"\nAI_critical = {ai_crit:.1f} FLOPs/byte")

    # --- Vẽ biểu đồ roofline log-log ---
    ais = [ai for _, (ai, _, _) in results]
    x_min, x_max = min(ais) / 4, max(ais) * 4

    # Đường roof: dốc nghiêng (memory-bound) rồi gãy thành ngang (compute-bound)
    roof_x = sorted([x_min, ai_crit, x_max])
    roof_y = [min(PEAK_FLOPS, x * PEAK_BW) for x in roof_x]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(roof_x, roof_y, "k-", lw=2, label="Roofline")
    ax.axvline(ai_crit, color="gray", ls="--", lw=1)
    ax.text(ai_crit, PEAK_FLOPS * 1.15, f"AI_crit = {ai_crit:.1f}",
            ha="center", color="gray")

    seen = set()
    for name, (ai, achievable, bound) in results:
        color = "tab:red" if bound == "compute-bound" else "tab:blue"
        label = bound if bound not in seen else None
        seen.add(bound)
        ax.plot(ai, achievable, "o", ms=9, color=color, label=label)
        ax.annotate(name, (ai, achievable),
                    textcoords="offset points", xytext=(8, 6))

    ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)")
    ax.set_ylabel("Performance (FLOPS)")
    ax.set_title(f"Roofline — peak {PEAK_FLOPS / 1e12:.0f} TFLOPS, "
                 f"{PEAK_BW / 1e9:.0f} GB/s")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig("roofline.png", dpi=120)
    print("Đã lưu roofline.png")
