"""Bài tập 3.2 — Mở rộng systolic simulator.

STUB: bạn tự cài đặt các phần TODO. Tái dùng SystolicArray từ systolic_sim.py.
"""
import numpy as np

from systolic_sim import SystolicArray


# === TODO 1: Weight-stationary dataflow ===
# Trong weight-stationary, ma trận B (weight) được nạp sẵn và ở yên trong PE;
# A (activation) chảy qua, partial sums chảy theo.
# Khác output-stationary ở chỗ: không skew A/B vào, mà preload weight trước.
class WeightStationaryArray:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # TODO: state cho weight đã nạp + partial sum đang chảy
        raise NotImplementedError

    def load_weights(self, B: np.ndarray):
        """Nạp weight B[rows, cols] vào các PE (1 lần, ở yên)."""
        # TODO
        raise NotImplementedError

    def matmul(self, A: np.ndarray) -> np.ndarray:
        """C = A @ B, với B đã nạp sẵn. A: [M, rows]."""
        # TODO: bơm A qua, tích lũy partial sums, trả C[M, cols]
        raise NotImplementedError


# === TODO 2: Tiling — matmul lớn hơn array ===
def tiled_matmul(A: np.ndarray, B: np.ndarray, array_size: int):
    """Matmul C = A @ B khi A, B lớn hơn systolic array array_size x array_size.

    Chia thành các tile array_size x array_size, compute từng tile bằng
    SystolicArray, cộng dồn kết quả.

    Trả về: (C, stats) với stats gồm:
      - dram_bytes_read:  tổng byte đọc từ "DRAM" (mỗi tile load lại)
      - dram_bytes_write: tổng byte ghi output
      - total_cycles:     tổng cycle qua mọi tile
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    # TODO: vòng lặp 3 cấp trên các tile (i, j, k);
    #       với mỗi (i,j) tích lũy qua các tile k.
    #       Đếm dram_bytes: mỗi lần load tile A/B = array_size*array_size*4 byte.
    raise NotImplementedError


# === TODO 3: Plot utilization theo cycle ===
def plot_utilization(sa: SystolicArray, path: str = "utilization.png"):
    """Vẽ sa.active_macs_per_cycle theo cycle.

    Quan sát: ramp-up (data bắt đầu chảy vào) và ramp-down (data chảy ra hết).
    """
    # TODO: dùng matplotlib, lưu thành file path
    raise NotImplementedError


if __name__ == "__main__":
    # Test gợi ý sau khi cài đặt:
    #   - WeightStationaryArray: so với A @ B của numpy
    #   - tiled_matmul: matmul 64x64 trên array 16x16, kiểm tra đúng + xem dram_bytes
    #   - plot_utilization: chạy 1 matmul rồi plot
    print("Cài đặt các TODO trong file này, sau đó viết test ở đây.")
