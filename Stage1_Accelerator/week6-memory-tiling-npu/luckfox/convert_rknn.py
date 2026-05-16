"""Bài tập 6.3 — Convert model PyTorch -> ONNX -> RKNN cho Luckfox RV1103.

Chạy trên máy dev x86 (cần rknn-toolkit2). Inference trên board dùng
rknn-toolkit-lite2. Xem README.md trong thư mục này.

Quy trình: PyTorch -> ONNX -> RKNN compiler (1 compiler thật!) -> quantize
-> NPU instruction -> chạy trên silicon.
"""
import torchvision


def export_onnx(path: str = "resnet18.onnx"):
    """Export ResNet18 pretrained sang ONNX."""
    import torch
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy, path,
                      input_names=["input"], output_names=["output"],
                      opset_version=12)
    print(f"Đã export {path}")


def convert_to_rknn(onnx_path: str = "resnet18.onnx",
                    rknn_path: str = "resnet18.rknn",
                    calib: str = "./calib_dataset.txt"):
    """Convert ONNX -> RKNN với INT8 quantization cho rv1103.

    calib_dataset.txt: mỗi dòng là đường dẫn 1 ảnh đại diện (vài chục - vài trăm
    ảnh) để compiler tính scale factor khi quantize.
    """
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)

    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rv1103",
        quantized_algorithm="normal",
        quantized_dtype="asymmetric_quantized-8",
    )

    rknn.load_onnx(model=onnx_path)
    rknn.build(do_quantization=True, dataset=calib)
    rknn.export_rknn(rknn_path)
    print(f"Đã export {rknn_path}")

    # Bài tập đào sâu: bật verbose, dump intermediate output của RKNN compile,
    # quan sát quantization annotations trong IR của RKNN.
    rknn.release()


if __name__ == "__main__":
    export_onnx()
    convert_to_rknn()
