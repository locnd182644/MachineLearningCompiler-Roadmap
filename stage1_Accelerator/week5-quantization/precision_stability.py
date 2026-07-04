"""Bài tập 5.3 — FP32 vs FP16 vs BF16 training stability.

Train MLP nhỏ trên MNIST với 3 precision, log gradient magnitude qua step.

Ý tưởng cốt lõi (vì sao bài này quan trọng cho ML compiler):
  - FP32 : 1 sign + 8 exponent + 23 mantissa. Dải mũ rộng (~1e-38 .. 3e38),
           chính xác cao => baseline ổn định.
  - FP16 : 1 sign + 5 exponent + 10 mantissa. Dải mũ HẸP (~6e-5 .. 65504).
           Gradient nhỏ dễ underflow về 0, activation/loss lớn dễ overflow -> Inf/NaN.
           Đây là lý do training FP16 "thuần" cần loss-scaling (bài này CỐ TÌNH
           không dùng loss-scaling để thấy nó gãy).
  - BF16 : 1 sign + 8 exponent + 7 mantissa. Cùng dải mũ như FP32 (chỉ kém mantissa)
           => hầu như ổn định như FP32, chỉ nhiễu hơn chút. Đây là lý do TPU/NPU
           chọn BF16 làm định dạng train mặc định.

Kỳ vọng: FP16 dễ NaN/underflow hơn; BF16 bám sát FP32.
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )


def _global_grad_norm(model: nn.Module) -> float:
    """Tính L2-norm toàn cục của gradient trên tất cả param.

    Lưu ý: cộng bình phương được thực hiện ở FP32 (`.float()`) để phép ĐO không
    bị overflow/underflow theo dtype đang train — ta muốn đo độ lớn THẬT của
    gradient, không phải đo cái sai số của chính phép đo. Nếu grad đã là NaN/Inh
    trong dtype gốc thì .float() vẫn giữ NaN/Inf nên vẫn phát hiện được.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(g.pow(2).sum())
    return math.sqrt(total_sq)


def train_one_precision(
    dtype: torch.dtype,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 0.05,
    data_root: str = "./data",
):
    """Train MLP với dtype cho trước.

    Trả về dict:
      {
        "grad_mag": [float, ...],   # norm gradient theo từng step
        "loss":     [float, ...],   # loss theo từng step (đã .item())
        "nan_step": int | None,     # step đầu tiên gặp NaN/Inf (nếu có)
        "device":   str,
      }
    """
    # torchvision import cục bộ để file vẫn import được kể cả khi chưa cài tv.
    from torchvision import datasets, transforms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST: chuẩn hóa theo mean/std thống kê sẵn của tập train.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Cast toàn bộ model sang dtype mục tiêu (weight + bias đều theo dtype này).
    model = make_mlp().to(device=device, dtype=dtype)
    criterion = nn.CrossEntropyLoss()
    # SGD + momentum: đủ mạnh để bộc lộ sự bất ổn của FP16, vẫn ổn cho FP32/BF16.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    grad_mag, losses = [], []
    nan_step = None
    step = 0

    for epoch in range(epochs):
        for images, labels in loader:
            # Data cast sang dtype (ảnh); labels giữ int64 cho CrossEntropy.
            images = images.to(device=device, dtype=dtype)
            labels = labels.to(device=device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            # CrossEntropy nội bộ dùng log-softmax; ở FP16 dễ overflow -> Inf/NaN.
            loss = criterion(logits, labels)
            loss.backward()

            gnorm = _global_grad_norm(model)
            loss_val = float(loss.detach().float())

            grad_mag.append(gnorm)
            losses.append(loss_val)

            # Phát hiện NaN/Inf ở loss hoặc gradient => training coi như đã gãy.
            if not math.isfinite(gnorm) or not math.isfinite(loss_val):
                nan_step = step
                print(f"  [!] {dtype} gặp NaN/Inf tại step {step} "
                      f"(loss={loss_val}, grad_norm={gnorm}) -> dừng sớm.")
                return {
                    "grad_mag": grad_mag,
                    "loss": losses,
                    "nan_step": nan_step,
                    "device": str(device),
                }

            optimizer.step()
            step += 1

        print(f"  {dtype} | epoch {epoch + 1}/{epochs} | "
              f"loss={losses[-1]:.4f} | grad_norm={grad_mag[-1]:.4e}")

    return {
        "grad_mag": grad_mag,
        "loss": losses,
        "nan_step": nan_step,
        "device": str(device),
    }


def plot_grad_magnitude(results: dict, path: str = "grad_magnitude.png"):
    """results: {name: result_dict}. Vẽ chồng 3 đường grad-norm theo step."""
    import matplotlib
    matplotlib.use("Agg")  # backend không cần màn hình (chạy headless/CI).
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, res in results.items():
        mags = res["grad_mag"]
        steps = range(len(mags))
        line, = ax.plot(steps, mags, label=name, alpha=0.85, linewidth=1.2)
        # Đánh dấu điểm gãy NaN/Inf (nếu có) bằng chữ X đỏ trên cùng màu đường.
        nan_step = res.get("nan_step")
        if nan_step is not None:
            ax.scatter([nan_step], [mags[nan_step] if math.isfinite(mags[nan_step]) else 0],
                       marker="x", s=90, color=line.get_color(), zorder=5)
            ax.annotate(f"{name}: NaN/Inf @ step {nan_step}",
                        xy=(nan_step, 0), xytext=(nan_step, 0),
                        color=line.get_color(), fontsize=8)

    ax.set_yscale("log")  # grad-norm trải nhiều bậc độ lớn => log dễ đọc.
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient L2-norm (log scale)")
    ax.set_title("Training stability: FP32 vs FP16 vs BF16 (không loss-scaling)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Đã lưu biểu đồ: {path}")


if __name__ == "__main__":
    torch.manual_seed(0)  # cùng khởi tạo/thứ tự để 3 precision so sánh công bằng.

    results = {}
    for name, dt in [("FP32", torch.float32),
                     ("FP16", torch.float16),
                     ("BF16", torch.bfloat16)]:
        print(f"=== Train {name} ({dt}) ===")
        results[name] = train_one_precision(dt)

    plot_grad_magnitude(results)

    # Tóm tắt: precision nào gãy, precision nào sống sót.
    print("\n----- Tóm tắt -----")
    for name, res in results.items():
        if res["nan_step"] is not None:
            print(f"{name:5s}: GÃY (NaN/Inf) tại step {res['nan_step']}")
        else:
            print(f"{name:5s}: OK | loss cuối={res['loss'][-1]:.4f} | "
                  f"grad_norm cuối={res['grad_mag'][-1]:.4e}")
