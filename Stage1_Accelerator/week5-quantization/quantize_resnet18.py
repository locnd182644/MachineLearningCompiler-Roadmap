"""Bài tập 5.2 — Post-training quantization (PTQ) ResNet18.

STUB: cài đặt các phần TODO.

Mục tiêu: so accuracy FP32 vs INT8, đo speedup inference trên CPU.
"""
import torch
import torchvision


def load_model():
    """ResNet18 pretrained ImageNet."""
    return torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()


def evaluate(model, dataloader) -> float:
    """Trả về top-1 accuracy."""
    # TODO: vòng lặp qua dataloader, đếm dự đoán đúng
    raise NotImplementedError


def ptq_int8(model, calib_loader):
    """Post-training quantization sang INT8.

    TODO: dùng torch.quantization (fuse modules → set qconfig →
          prepare → calibrate qua calib_loader → convert).
    Gợi ý: torch.ao.quantization.quantize_fx hoặc eager mode API.
    """
    raise NotImplementedError


def bench_cpu(model, sample) -> float:
    """Đo thời gian inference trung bình (ms) trên CPU."""
    # TODO: warmup + timing loop
    raise NotImplementedError


if __name__ == "__main__":
    # TODO:
    #  1. Chuẩn bị ImageNet val subset (vài trăm ảnh đủ để ước lượng accuracy).
    #  2. fp32 = load_model(); đo accuracy + bench_cpu.
    #  3. int8 = ptq_int8(fp32, calib_loader); đo accuracy + bench_cpu.
    #  4. In: accuracy drop (%) và speedup (x).
    print("Cài đặt các TODO trong file này.")
