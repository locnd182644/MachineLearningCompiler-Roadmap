"""Bài tập 5.3 — FP32 vs FP16 vs BF16 training stability.

STUB: cài đặt các phần TODO.

Train MLP nhỏ trên MNIST với 3 precision, log gradient magnitude qua epoch.
Kỳ vọng: FP16 dễ NaN hơn (exponent range hẹp); BF16 ổn định như FP32.
"""
import torch
import torch.nn as nn


def make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def train_one_precision(dtype: torch.dtype, epochs: int = 5):
    """Train MLP với dtype cho trước.

    Trả về: list gradient magnitude trung bình theo step (hoặc epoch).

    TODO:
      - Tải MNIST (torchvision.datasets.MNIST, download=True).
      - Cast model + data sang dtype.
      - Vòng lặp train; sau backward, gom norm gradient của các param.
      - Phát hiện NaN/Inf và ghi nhận.
    """
    raise NotImplementedError


def plot_grad_magnitude(results: dict, path: str = "grad_magnitude.png"):
    """results: {name: [grad_mag...]}. Vẽ chồng 3 đường."""
    # TODO: matplotlib, lưu path
    raise NotImplementedError


if __name__ == "__main__":
    # TODO:
    #   results = {}
    #   for name, dt in [("FP32", torch.float32),
    #                    ("FP16", torch.float16),
    #                    ("BF16", torch.bfloat16)]:
    #       results[name] = train_one_precision(dt)
    #   plot_grad_magnitude(results)
    print("Cài đặt các TODO trong file này.")
