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
        # Weight B[i][j] nằm yên trong PE(i,j).
        self.weights = np.zeros((rows, cols))
        self.loaded = False
        # Thống kê giống SystolicArray để có thể plot.
        self.cycle_count = 0
        self.total_macs = 0
        self.active_macs_per_cycle = []

    def load_weights(self, B: np.ndarray):
        """Nạp weight B[rows, cols] vào các PE (1 lần, ở yên)."""
        assert B.shape == (self.rows, self.cols), (
            f"weight phải có shape ({self.rows}, {self.cols}), nhận {B.shape}"
        )
        self.weights = B.astype(float).copy()
        self.loaded = True

    def matmul(self, A: np.ndarray) -> np.ndarray:
        """C = A @ B, với B đã nạp sẵn. A: [M, rows].

        Dataflow weight-stationary:
          - activation chảy sang phải (→) qua từng hàng,
          - partial sum chảy xuống dưới (↓) tích lũy theo chiều reduction = rows.
        Activation A[m, i] được bơm vào trái của hàng i tại cycle (m + i) (skew),
        nhờ vậy psum đi xuống cột j gặp đúng activation của cùng output row m.
        Psum ra khỏi đáy cột j tại cycle (m + rows-1 + j) chính là C[m, j].
        """
        assert self.loaded, "phải load_weights() trước khi matmul()"
        M, R = A.shape
        assert R == self.rows, f"A phải có {self.rows} cột, nhận {R}"
        A = A.astype(float)

        rows, cols = self.rows, self.cols
        W = self.weights
        C = np.zeros((M, cols))

        # Register output của mỗi PE (giá trị đẩy cho neighbor ở cycle kế).
        a_out = np.zeros((rows, cols))   # activation đẩy sang phải
        p_out = np.zeros((rows, cols))   # partial sum đẩy xuống dưới

        # Wavefront cuối: cycle lớn nhất = (M-1)+(rows-1)+(cols-1).
        total_cycles = M + rows + cols - 2
        for t in range(total_cycles):
            new_a = np.zeros((rows, cols))
            new_p = np.zeros((rows, cols))
            active = 0
            for i in range(rows):
                # activation bơm vào trái hàng i: A[m, i] với m = t - i.
                m = t - i
                inject = A[m, i] if 0 <= m < M else 0.0
                for j in range(cols):
                    a_in = inject if j == 0 else a_out[i][j - 1]
                    p_in = 0.0 if i == 0 else p_out[i - 1][j]
                    new_p[i][j] = p_in + a_in * W[i][j]
                    new_a[i][j] = a_in
                    if a_in != 0:
                        active += 1
            a_out, p_out = new_a, new_p

            # Thu output từ đáy mảng: C[m, j] = p_out[rows-1][j].
            for j in range(cols):
                m = t - (rows - 1) - j
                if 0 <= m < M:
                    C[m, j] = p_out[rows - 1][j]

            self.cycle_count += 1
            self.active_macs_per_cycle.append(active)
            self.total_macs += active

        return C


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
    S = array_size

    def n_tiles(dim):
        return (dim + S - 1) // S  # ceil chia, hỗ trợ cả khi không chia hết

    ti, tj, tk = n_tiles(M), n_tiles(N), n_tiles(K)

    C = np.zeros((M, N))
    tile_bytes = S * S * 4  # 1 tile S x S float32 = S*S*4 byte
    stats = {"dram_bytes_read": 0, "dram_bytes_write": 0, "total_cycles": 0}

    for i in range(ti):
        r0, r1 = i * S, min((i + 1) * S, M)
        for j in range(tj):
            c0, c1 = j * S, min((j + 1) * S, N)
            acc = np.zeros((r1 - r0, c1 - c0))
            for k in range(tk):
                k0, k1 = k * S, min((k + 1) * S, K)

                # Lấy tile, pad về S x S nếu nằm ở rìa (array cố định S x S).
                a_tile = np.zeros((S, S))
                b_tile = np.zeros((S, S))
                a_tile[: r1 - r0, : k1 - k0] = A[r0:r1, k0:k1]
                b_tile[: k1 - k0, : c1 - c0] = B[k0:k1, c0:c1]

                # Mỗi (i,j,k) load lại tile A và tile B từ "DRAM".
                stats["dram_bytes_read"] += 2 * tile_bytes

                sa = SystolicArray(rows=S, cols=S)
                acc += sa.matmul(a_tile, b_tile)[: r1 - r0, : c1 - c0]
                stats["total_cycles"] += sa.cycle_count

            C[r0:r1, c0:c1] = acc
            # Ghi output tile 1 lần.
            stats["dram_bytes_write"] += tile_bytes

    return C, stats


# === TODO 3: Plot utilization theo cycle ===
def plot_utilization(sa: SystolicArray, path: str = "utilization.png"):
    """Vẽ sa.active_macs_per_cycle theo cycle.

    Quan sát: ramp-up (data bắt đầu chảy vào) và ramp-down (data chảy ra hết).
    """
    import matplotlib
    matplotlib.use("Agg")  # backend không cần display, lưu thẳng ra file
    import matplotlib.pyplot as plt

    util = sa.active_macs_per_cycle
    cycles = range(len(util))
    peak = sa.rows * sa.cols  # số MAC tối đa mỗi cycle

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cycles, util, marker="o", markersize=3, label="active MACs")
    ax.axhline(peak, color="r", linestyle="--", label=f"peak = {peak}")
    ax.fill_between(cycles, util, alpha=0.2)
    ax.set_xlabel("cycle")
    ax.set_ylabel("active MACs")
    ax.set_title(
        f"Systolic utilization ({sa.rows}x{sa.cols}) — "
        f"ramp-up / steady / ramp-down"
    )
    ax.set_ylim(0, peak * 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved utilization plot -> {path}")


if __name__ == "__main__":
    # --- Test 1: WeightStationaryArray so với numpy ---
    print("=== Test 1: WeightStationaryArray ===")
    ws = WeightStationaryArray(rows=4, cols=4)
    B = np.random.randn(4, 4)
    A = np.random.randn(6, 4)  # M=6 activation rows
    ws.load_weights(B)
    C_ws = ws.matmul(A)
    C_ref = A @ B
    print("Max error:", np.abs(C_ws - C_ref).max())
    assert np.allclose(C_ws, C_ref, atol=1e-9)
    print("WeightStationary OK\n")

    # --- Test 2: tiled_matmul 64x64 trên array 16x16 ---
    print("=== Test 2: tiled_matmul (64x64 @ 64x64, array 16x16) ===")
    A2 = np.random.randn(64, 64)
    B2 = np.random.randn(64, 64)
    C2, stats = tiled_matmul(A2, B2, array_size=16)
    print("Max error:", np.abs(C2 - A2 @ B2).max())
    assert np.allclose(C2, A2 @ B2, atol=1e-6)
    print("DRAM bytes read :", stats["dram_bytes_read"])
    print("DRAM bytes write:", stats["dram_bytes_write"])
    print("Total cycles    :", stats["total_cycles"])
    print("tiled_matmul OK\n")

    # --- Test 3: plot_utilization ---
    print("=== Test 3: plot_utilization ===")
    sa = SystolicArray(rows=8, cols=8)
    sa.matmul(np.random.randn(8, 16), np.random.randn(16, 8))
    plot_utilization(sa, "utilization.png")
