"""Bài tập 5.2 — Post-training quantization (PTQ) ResNet18.

Mục tiêu: so accuracy FP32 vs INT8, đo speedup inference trên CPU.

Pipeline PTQ dùng torch.ao.quantization.quantize_fx (khuyến nghị cho PyTorch >= 1.9):
  1. Load ResNet18 pretrained (FP32).
  2. Tạo qconfig_mapping với backend 'x86' (tương đương fbgemm, tối ưu cho CPU x86-64).
  3. prepare_fx() → dùng FX graph để tự động chèn observer vào TOÀN BỘ graph,
     không cần liệt kê thủ công từng lớp như Eager Mode.
  4. Calibrate → chạy calib_loader qua prepared model để observer ghi min/max activation.
  5. convert_fx() → thay thế các op FP32 bằng INT8 kernels.
  6. So sánh: accuracy drop (%) và speedup (×) trên CPU.

Tại sao dùng quantize_fx thay vì Eager Mode API?
  - Eager Mode yêu cầu model có QuantStub/DeQuantStub bao quanh forward().
    torchvision.models.resnet18 không có → quantized conv nhận float tensor → crash.
  - FX Mode dùng symbolic tracing, tự động thêm quant/dequant nodes vào đúng vị trí.
  - FX Mode hỗ trợ fuse tự động (Conv-BN-ReLU) mà không cần khai báo thủ công.
"""

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torch.utils.data import DataLoader, Subset

# ── Số lần lặp warmup và đo benchmark ────────────────────────────────────────
WARMUP_ITERS = 20
BENCH_ITERS  = 100

# ── Kích thước batch khi calibrate và đánh giá ───────────────────────────────
CALIB_BATCH  = 32
EVAL_BATCH   = 64

# ── Số ảnh tối đa dùng để calibrate (512–1024 là hợp lý) ────────────────────
CALIB_SAMPLES = 512

# ── Số ảnh tối đa để ước lượng accuracy ──────────────────────────────────────
EVAL_SAMPLES  = 2000


# =============================================================================
# 1. Load model
# =============================================================================

def load_model() -> nn.Module:
    """ResNet18 pretrained ImageNet, trả về ở chế độ eval trên CPU."""
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    model.eval().cpu()
    return model


# =============================================================================
# 2. DataLoader
# =============================================================================

def build_imagenet_val_loader(
    data_root: str | Path,
    batch_size: int = 64,
    max_samples: int | None = None,
    num_workers: int = 4,
) -> DataLoader:
    """Tạo DataLoader cho ImageNet validation set.

    Cấu trúc thư mục mong đợi (chuẩn ImageNet):
        <data_root>/val/
            n01440764/   ← synset folder
                ILSVRC2012_val_00000293.JPEG
                ...
            n01443537/
                ...

    Nếu chỉ có subset nhỏ (vài trăm ảnh), hàm vẫn hoạt động đúng vì
    torchvision.datasets.ImageFolder đọc theo cấu trúc thư mục.

    Args:
        data_root:   Đường dẫn đến thư mục gốc chứa thư mục ``val/``.
        batch_size:  Kích thước batch.
        max_samples: Nếu không None, chỉ lấy ``max_samples`` ảnh đầu tiên.
        num_workers: Số worker cho DataLoader.

    Returns:
        DataLoader sẵn sàng dùng.
    """
    # ── Transform chuẩn ImageNet ─────────────────────────────────────────────
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_dir = Path(data_root) / "val"
    if not val_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục val tại: {val_dir}\n"
            "Hãy đặt ImageNet validation set (hoặc subset) vào đúng vị trí.\n"
            "Cấu trúc: <data_root>/val/<synset_id>/*.JPEG"
        )

    dataset = torchvision.datasets.ImageFolder(str(val_dir), transform=transform)

    # Giới hạn số mẫu nếu cần (lấy phần đầu để đảm bảo phân phối đa dạng)
    if max_samples is not None and max_samples < len(dataset):
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,          # không shuffle để kết quả reproducible
        num_workers=num_workers,
        pin_memory=False,       # False vì chạy trên CPU
    )
    return loader


# =============================================================================
# 3. Evaluate — top-1 accuracy
# =============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    """Trả về top-1 accuracy (0–100).

    Logic:
        - Với mỗi batch, lấy argmax của logit → predicted label.
        - Đếm số dự đoán đúng / tổng số mẫu.
        - Model phải ở chế độ eval() và chạy trên CPU.

    Args:
        model:      Model cần đánh giá (FP32 hoặc INT8).
        dataloader: DataLoader trả về (images, labels).

    Returns:
        Top-1 accuracy theo phần trăm (ví dụ: 69.76).
    """
    model.eval()
    correct = 0   # số dự đoán đúng
    total   = 0   # tổng số mẫu đã xử lý

    for images, labels in dataloader:
        # images: (B, 3, 224, 224) — float32
        # labels: (B,)             — int64, class index ImageNet
        images = images.cpu()
        labels = labels.cpu()

        logits = model(images)         # (B, 1000)
        preds  = logits.argmax(dim=1)  # (B,)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


# =============================================================================
# 4. PTQ — Post-training quantization (FX Mode)
# =============================================================================

def ptq_int8(model: nn.Module, calib_loader: DataLoader) -> nn.Module:
    """Chuyển đổi model FP32 sang INT8 bằng Post-training quantization (FX Mode).

    Tại sao dùng quantize_fx (FX Mode) thay vì Eager Mode?
        Eager Mode (torch.quantization.prepare/convert) yêu cầu model tự khai báo
        QuantStub/DeQuantStub trong forward(). torchvision.models.resnet18 không có →
        sau convert(), quantized conv nhận float CPU tensor thay vì QuantizedCPU → crash.

        FX Mode dùng symbolic tracing (torch.fx) để phân tích toàn bộ computation graph,
        tự động:
          - Chèn quant/dequant stubs tại đúng vị trí (trước/sau mỗi quantizable op).
          - Fuse Conv-BN-ReLU mà không cần liệt kê thủ công.
          - Xử lý các pattern phức tạp (skip connection trong ResNet, v.v.).

    Các bước thực hiện:
        1. Deep-copy model để không làm bẩn bản gốc FP32.
        2. Tạo qconfig_mapping với backend 'x86':
             - Activation: HistogramObserver (per-tensor, asymmetric) → scale + zero_point.
             - Weight: PerChannelMinMaxObserver (per-channel, symmetric) → chính xác hơn.
        3. prepare_fx() → trace graph, chèn observer vào mọi activation.
        4. Calibrate: chạy calib_loader để observer thu thống kê min/max thực tế.
        5. convert_fx() → thay thế op FP32 bằng INT8 (QuantizedConv2d, QuantizedLinear…).

    Args:
        model:        Model FP32 gốc (sẽ không bị thay đổi).
        calib_loader: DataLoader dùng để calibrate observer.

    Returns:
        Model INT8 đã quantize, sẵn sàng inference trên CPU.
    """
    # ── Bước 1: Copy để bảo toàn model gốc ──────────────────────────────────
    fp32_copy = copy.deepcopy(model).eval().cpu()

    # ── Bước 2: Tạo qconfig_mapping ──────────────────────────────────────────
    # 'x86' = alias của 'fbgemm' trong PyTorch >= 2.0, tối ưu cho Intel/AMD CPU.
    # QConfigMapping cho phép set qconfig khác nhau cho từng lớp hoặc loại lớp.
    # get_default_qconfig_mapping() dùng cấu hình tốt nhất cho backend đó.
    qconfig_mapping = get_default_qconfig_mapping("x86")

    # ── Bước 3: prepare_fx ───────────────────────────────────────────────────
    # Cần một example input để FX tracing biết shape của tensor (quan trọng với
    # các model có conditional branching phụ thuộc vào shape).
    example_input = torch.randn(1, 3, 224, 224)

    # prepare_fx() thực hiện:
    #   a. torch.fx.symbolic_trace(model) → FX graph
    #   b. Fuse các pattern Conv-BN-ReLU trong graph
    #   c. Chèn FakeQuantize / Observer modules vào đúng vị trí trong graph
    prepared_model = prepare_fx(fp32_copy, qconfig_mapping, example_inputs=(example_input,))

    # ── Bước 4: Calibrate ────────────────────────────────────────────────────
    # Chạy dữ liệu thực qua prepared model để observer học phân phối activation.
    # Observer dùng thuật toán Min-Max hoặc Histogram để xác định scale và zero_point.
    print(f"  [PTQ] Calibrating trên {len(calib_loader.dataset)} mẫu …", flush=True)
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(calib_loader):
            prepared_model(images.cpu())
            if (batch_idx + 1) % 10 == 0:
                print(f"  [PTQ]   batch {batch_idx + 1}/{len(calib_loader)}", flush=True)

    # ── Bước 5: convert_fx ───────────────────────────────────────────────────
    # convert_fx() dùng thống kê đã thu được từ observer để:
    #   - Tính scale và zero_point cho mỗi activation và weight.
    #   - Thay thế FP32 op bằng INT8 op (QuantizedConv2d, QuantizedLinear, v.v.).
    #   - Thêm quantize/dequantize nodes tại ranh giới FP32 ↔ INT8.
    int8_model = convert_fx(prepared_model)
    print("  [PTQ] Hoàn thành convert sang INT8.", flush=True)

    return int8_model


# =============================================================================
# 5. Benchmark inference latency
# =============================================================================

def bench_cpu(model: nn.Module, sample: torch.Tensor) -> float:
    """Đo thời gian inference trung bình (ms) trên CPU.

    Quy trình:
        1. Warmup (WARMUP_ITERS lần) để JIT compilation, CPU cache, kernel initialization
           ổn định — tránh đo nhầm chi phí lần đầu.
        2. Đo BENCH_ITERS lần bằng time.perf_counter() (độ phân giải cao nhất).
        3. Tính trung bình để giảm noise từ OS scheduling.

    Args:
        model:  Model cần đo (FP32 hoặc INT8).
        sample: Tensor đầu vào mẫu, shape (1, 3, 224, 224).

    Returns:
        Thời gian inference trung bình tính bằng milli-giây (ms).
    """
    model.eval()
    sample = sample.cpu()

    # ── Warmup: loại bỏ chi phí khởi tạo lần đầu ────────────────────────────
    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(sample)

    # ── Đo thực sự ───────────────────────────────────────────────────────────
    timings: list[float] = []
    with torch.no_grad():
        for _ in range(BENCH_ITERS):
            t0 = time.perf_counter()
            _ = model(sample)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # → ms

    avg_ms = sum(timings) / len(timings)
    return avg_ms


# =============================================================================
# 6. Main — ghép tất cả lại, in kết quả
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PTQ ResNet18 — FP32 vs INT8")
    parser.add_argument(
        "--data",
        type=str,
        default="data/imagenet",
        help="Thư mục gốc chứa ImageNet val subset (cấu trúc: <data>/val/<synset>/*.JPEG).",
    )
    parser.add_argument(
        "--calib-samples", type=int, default=CALIB_SAMPLES,
        help=f"Số ảnh để calibrate observer (mặc định: {CALIB_SAMPLES}).",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=EVAL_SAMPLES,
        help=f"Số ảnh để đánh giá accuracy (mặc định: {EVAL_SAMPLES}).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Số worker DataLoader (mặc định: 4).",
    )
    args = parser.parse_args()

    # ── Header ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  PTQ ResNet18: FP32 vs INT8  (torch.ao.quantization.quantize_fx)")
    print("=" * 60)

    # ── Bước 1: Chuẩn bị DataLoader ──────────────────────────────────────────
    print(f"\n[1] Đang tải dữ liệu từ: {args.data}")

    # Loader dùng để calibrate (ít mẫu hơn, đủ để observer học thống kê)
    calib_loader = build_imagenet_val_loader(
        data_root=args.data,
        batch_size=CALIB_BATCH,
        max_samples=args.calib_samples,
        num_workers=args.num_workers,
    )

    # Loader dùng để đánh giá accuracy (nhiều mẫu hơn để ước lượng chính xác)
    eval_loader = build_imagenet_val_loader(
        data_root=args.data,
        batch_size=EVAL_BATCH,
        max_samples=args.eval_samples,
        num_workers=args.num_workers,
    )

    print(f"    Calib : {len(calib_loader.dataset)} ảnh  "
          f"({len(calib_loader)} batch × {CALIB_BATCH})")
    print(f"    Eval  : {len(eval_loader.dataset)} ảnh  "
          f"({len(eval_loader)} batch × {EVAL_BATCH})")

    # ── Bước 2: FP32 baseline ────────────────────────────────────────────────
    print("\n[2] Đang đánh giá FP32 baseline …")
    fp32_model = load_model()

    fp32_acc = evaluate(fp32_model, eval_loader)
    print(f"    FP32 accuracy : {fp32_acc:.2f}%")

    # Tạo tensor mẫu đơn lẻ (batch=1) để benchmark giống production inference
    sample_input = torch.randn(1, 3, 224, 224)
    fp32_ms = bench_cpu(fp32_model, sample_input)
    print(f"    FP32 latency  : {fp32_ms:.2f} ms/image  "
          f"(avg {BENCH_ITERS} iters, {WARMUP_ITERS} warmup)")

    # ── Bước 3: PTQ → INT8 ───────────────────────────────────────────────────
    print("\n[3] Đang thực hiện PTQ (FX Mode) …")
    int8_model = ptq_int8(fp32_model, calib_loader)

    # ── Bước 4: Đánh giá INT8 ────────────────────────────────────────────────
    print("\n[4] Đang đánh giá INT8 model …")
    int8_acc = evaluate(int8_model, eval_loader)
    print(f"    INT8 accuracy : {int8_acc:.2f}%")

    int8_ms = bench_cpu(int8_model, sample_input)
    print(f"    INT8 latency  : {int8_ms:.2f} ms/image  "
          f"(avg {BENCH_ITERS} iters, {WARMUP_ITERS} warmup)")

    # ── Bước 5: Kết quả tổng hợp ─────────────────────────────────────────────
    accuracy_drop = fp32_acc - int8_acc
    speedup       = fp32_ms / int8_ms

    print("\n" + "=" * 60)
    print("  KẾT QUẢ SO SÁNH FP32 vs INT8")
    print("=" * 60)
    print(f"  {'Metric':<28} {'FP32':>8} {'INT8':>8} {'Delta':>8}")
    print(f"  {'-' * 54}")
    print(f"  {'Top-1 Accuracy (%)':<28} {fp32_acc:>8.2f} {int8_acc:>8.2f} {-accuracy_drop:>+8.2f}")
    print(f"  {'Latency (ms/img)':<28} {fp32_ms:>8.2f} {int8_ms:>8.2f}")
    print(f"  {'Speedup':<28} {'':>8} {'':>8} {speedup:>7.2f}x")
    print("=" * 60)

    # Nhận xét tự động
    print("\n📌 Nhận xét:")
    if accuracy_drop <= 0.5:
        print(f"  ✅ Accuracy drop chỉ {accuracy_drop:.2f}% — xuất sắc, PTQ không ảnh hưởng đáng kể.")
    elif accuracy_drop <= 1.0:
        print(f"  ✅ Accuracy drop {accuracy_drop:.2f}% — tốt, INT8 có thể deploy ngay.")
    elif accuracy_drop <= 2.0:
        print(f"  ⚠️  Accuracy drop {accuracy_drop:.2f}% — chấp nhận được với nhiều ứng dụng.")
    else:
        print(f"  ❌ Accuracy drop {accuracy_drop:.2f}% — cân nhắc QAT hoặc tăng số mẫu calibrate.")

    if speedup >= 2.0:
        print(f"  🚀 Speedup {speedup:.2f}x — INT8 nhanh hơn đáng kể trên CPU x86.")
    elif speedup >= 1.2:
        print(f"  ✅ Speedup {speedup:.2f}x — INT8 có cải thiện latency đáng chú ý.")
    else:
        print(f"  ℹ️  Speedup {speedup:.2f}x — thấp hơn mong đợi, thường do model nhỏ "
              "và memory bandwidth không phải bottleneck.")

    # ── (Tuỳ chọn) Lưu model INT8 ────────────────────────────────────────────
    save_path = Path("resnet18_int8_fx.pt")
    try:
        # FX-quantized model có thể script được
        scripted = torch.jit.script(int8_model)
        torch.jit.save(scripted, str(save_path))
        print(f"\n💾 Model INT8 (TorchScript) đã lưu tại: {save_path.resolve()}")
    except Exception as e:
        # Một số model FX không script được trực tiếp → dùng torch.save
        fallback_path = Path("resnet18_int8_fx_state.pt")
        torch.save(int8_model, str(fallback_path))
        print(f"\n💾 Model INT8 đã lưu tại: {fallback_path.resolve()}  "
              f"(torch.save, vì TorchScript gặp lỗi: {e})")
