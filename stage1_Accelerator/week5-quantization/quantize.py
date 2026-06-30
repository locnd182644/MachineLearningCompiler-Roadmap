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


def quantize_asymmetric(x: torch.Tensor, n_bits: int = 8):
    """Asymmetric quantization per-tensor."""
    # 1. Xác định dải giá trị đích cho số nguyên không dấu (UINT8)
    qmin = 0
    qmax = 2 ** n_bits - 1  # 255 cho UINT8
    
    # 2. Tìm giá trị min và max thực tế của tensor đầu vào
    x_min = torch.minimum(x.min(), torch.tensor(0.0, device=x.device))
    x_max = torch.maximum(x.max(), torch.tensor(0.0, device=x.device))
    
    # Đảm bảo không bị chia cho 0 nếu tất cả phần tử trong x bằng nhau
    if x_min == x_max:
        scale = torch.tensor(1.0, device=x.device)
        zero_point = torch.tensor(0, dtype=torch.int32, device=x.device)
        x_q = torch.zeros_like(x, dtype=torch.uint8)
        return x_q, scale, zero_point

    # 3. Tính toán hệ số tỷ lệ (Scale)
    scale = (x_max - x_min) / (qmax - qmin)
    
    # 4. Tính toán Điểm không (Zero-point)
    # Z = round(qmin - x_min / scale) -> vì qmin = 0 nên rút gọn thành round(-x_min / scale)
    zero_point = torch.round(-x_min / scale).to(torch.int32)
    
    # Ép zero_point nằm trong dải [0, 255] hợp lệ của UINT8
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    # 5. Tiến hành lượng tử hóa
    # Công thức: x_q = round(x / scale) + zero_point
    x_q = torch.round(x / scale) + zero_point
    
    # Giới hạn dữ liệu và ép kiểu về số nguyên không dấu 8-bit
    x_q = torch.clamp(x_q, qmin, qmax).to(torch.uint8)
    
    return x_q, scale, zero_point


def dequantize_asymmetric(x_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
    """Dequantize an asymmetrically quantized tensor back to float."""
    # Công thức phục hồi: x_approx = scale * (x_q - zero_point)
    # Cần ép kiểu x_q sang float để thực hiện phép trừ số thực chính xác
    return scale * (x_q.to(torch.float32) - zero_point)


def quantize_symmetric_per_channel(x: torch.Tensor, axis: int = 0, n_bits: int = 8):
    """Symmetric quantization per-channel (mỗi lát theo `axis` có scale riêng).

    Per-tensor dùng chung 1 scale cho toàn bộ tensor nên 1 outlier sẽ làm scale
    lớn, đẩy sai số lên các phần tử nhỏ. Per-channel cấp cho mỗi channel (hàng/cột)
    một scale riêng, giảm ảnh hưởng của outlier => sai số nhỏ hơn.
    """
    qmax = 2 ** (n_bits - 1) - 1  # 127 cho INT8

    # Tìm |max| theo từng channel: giữ nguyên chiều `axis`, gộp các chiều còn lại.
    dims = [d for d in range(x.dim()) if d != axis]
    amax = x.abs().amax(dim=dims, keepdim=True)

    scale = amax / qmax
    scale = torch.clamp(scale, min=1e-12)  # tránh chia 0 khi cả channel bằng 0
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale


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
    print(f"[Per-tensor symmetric] Mean relative error: {rel_err * 100:.2f}%")
    print("-----------------------------------------------------------------")
    
    # Quantize A và B với asymmetric
    A_qa, sA_a, zp_a = quantize_asymmetric(A)
    B_qa, sB_a, zp_b = quantize_asymmetric(B)

    # K là số chiều chung (số cột của A, số hàng của B)
    K = A_qa.shape[1] 
    
    # 1. Thực hiện phép nhân ma trận số nguyên cốt lõi (M, N)
    # INT8 matmul (đây là điều TPU/NPU làm trong hardware).
    # Accumulator INT32: dot product 256-deep có thể overflow INT8.
    C_int = (A_qa.float() @ B_qa.float()).to(torch.int32)
    
    # 2. Tính toán các thành phần bù do Zero-point gây ra
    # Thành phần bù 1: Tổng các cột của A_q nhân với zp_B
    # A_q32.sum(dim=1, keepdim=True) có kích thước (M, 1)
    term_1 = zp_b * A_qa.sum(dim=1, keepdim=True)
    
    # Thành phần bù 2: Tổng các hàng của B_q nhân với zp_A
    # B_q32.sum(dim=0, keepdim=True) có kích thước (1, N)
    term_2 = zp_a * B_qa.sum(dim=0, keepdim=True)
    
    # Thành phần bù 3: Hằng số tích của hai zero_point nhân với chiều K
    term_3 = K * zp_a * zp_b
    
    # 3. Gom các thành phần số nguyên lại (phép tính số nguyên rất nhanh)
    C_quant = C_int - term_1 - term_2 + term_3
    
    # 4. Cuối cùng, nhân với hệ số tỷ lệ tổng hợp để đưa về Float32
    scale_C = sA_a * sB_a
    C_dqa = scale_C * C_quant.to(torch.float32)

    rel_err = (C_dqa - C_fp32).abs().mean() / C_fp32.abs().mean()
    print(f"[Per-tensor asymmetric] Mean relative error: {rel_err * 100:.2f}%")
    print("-----------------------------------------------------------------")

    # --- Per-channel symmetric ---------------------------------------------
    # A: scale theo từng hàng (axis=0) vì C[i,:] dùng hàng i của A.
    # B: scale theo từng cột (axis=1) vì C[:,j] dùng cột j của B.
    # => C[i,j] = sum_k A[i,k] * B[k,j] dequantize bằng sA[i] * sB[j].
    A_qc, sA_c = quantize_symmetric_per_channel(A, axis=0)  # sA_c: (256, 1)
    B_qc, sB_c = quantize_symmetric_per_channel(B, axis=1)  # sB_c: (1, 256)

    C_int_c = (A_qc.float() @ B_qc.float()).to(torch.int32)
    C_dq_c = C_int_c.float() * (sA_c * sB_c)  # broadcast: (256,1)*(1,256) -> (256,256)

    rel_err_c = (C_dq_c - C_fp32).abs().mean() / C_fp32.abs().mean()
    print(f"[Per-channel symmetric] Mean relative error: {rel_err_c * 100:.2f}%")
    print("-----------------------------------------------------------------")
    
    # # --- Asymmetric round-trip ---------------------------------------------
    # # Kiểm tra sai số khôi phục của asymmetric trên 1 tensor (quantize->dequantize).
    # A_qa, sA_a, zp_a = quantize_asymmetric(A)
    # A_dq = dequantize_asymmetric(A_qa, sA_a, zp_a)
    # rt_err = (A_dq - A).abs().mean() / A.abs().mean()
    # print(f"[Asymmetric round-trip]  Mean relative error: {rt_err * 100:.2f}%")
