# Bài tập 6.3 — Luckfox RV1103 NPU (portfolio piece)

RV1103 có NPU 0.5 TOPS, hỗ trợ INT8/INT4. Đây là lần đầu nắm **toàn bộ stack**:
PyTorch → ONNX → RKNN compiler → quantization → NPU instruction → silicon thật.

## 1. Setup (máy dev x86)

```bash
# rknn-toolkit2 — KHÔNG có trên PyPI chuẩn, cài từ wheel của Rockchip:
#   https://github.com/airockchip/rknn-toolkit2
# Tải wheel khớp phiên bản Python rồi:
pip install ./rknn_toolkit2-*.whl

# Luckfox SDK
git clone https://github.com/LuckfoxTECH/luckfox-pico.git
```

## 2. Convert model

```bash
python convert_rknn.py
```

File này: export ResNet18 → ONNX → RKNN (INT8). Cần `calib_dataset.txt`
(mỗi dòng 1 đường dẫn ảnh đại diện) để compiler tính scale quantization.

## 3. Deploy lên board

```bash
scp resnet18.rknn root@<luckfox-ip>:/root/
ssh root@<luckfox-ip>
# Dùng example inference C/Python trong SDK; trên board cài rknn-toolkit-lite2.
```

## 4. Benchmark — điền kết quả

| Chỉ số | NPU (INT8) | CPU-only (ONNX Runtime ARM) |
|--------|------------|------------------------------|
| Inference time (ms) | | |
| Memory footprint | | |
| Accuracy | | (FP32 baseline) |

## 5. Đào sâu (nâng cao)

- Dump intermediate output của RKNN compile (verbose / debug mode).
- Đọc về RKNN IR (giống ONNX + quantization annotations).
- Tìm hiểu Rockchip NPU instruction set (driver RKNPU2 public).

## Phân tích (tự viết)

> Khi compile model lên Luckfox NPU có bao nhiêu transformation?
> Cái nào quan trọng nhất? Quantization ảnh hưởng accuracy/tốc độ thế nào?
