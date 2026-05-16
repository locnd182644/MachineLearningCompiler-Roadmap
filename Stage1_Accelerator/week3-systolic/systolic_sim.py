"""Bài tập 3.1 — Systolic array simulator (output-stationary).

Asset xuyên suốt: sẽ dùng làm target backend ở Giai đoạn 2.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class PE:
    """Processing Element trong systolic array.

    Output-stationary: accumulator ở yên, A bơm từ trái, B bơm từ trên.
    """
    accumulator: float = 0.0
    a_in: float = 0.0   # input từ trái
    b_in: float = 0.0   # input từ trên
    a_out: float = 0.0  # output sang phải (truyền cho PE kế)
    b_out: float = 0.0  # output xuống dưới

    def cycle(self):
        # 1 cycle: nhân a*b cộng vào accumulator, đẩy a,b cho neighbor
        self.accumulator += self.a_in * self.b_in
        self.a_out = self.a_in
        self.b_out = self.b_in


class SystolicArray:
    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.pes = [[PE() for _ in range(cols)] for _ in range(rows)]
        self.cycle_count = 0
        self.total_macs = 0
        self.active_macs_per_cycle = []

    def reset_accumulators(self):
        for row in self.pes:
            for pe in row:
                pe.accumulator = 0.0

    def step(self, a_inputs: list, b_inputs: list):
        """1 cycle. a_inputs: list dài rows (bơm vào cột trái).
        b_inputs: list dài cols (bơm vào hàng trên).
        """
        # Phase 1: lan truyền a, b vào pes
        for i in range(self.rows):
            for j in range(self.cols):
                if j == 0:
                    self.pes[i][j].a_in = a_inputs[i] if a_inputs[i] is not None else 0
                else:
                    self.pes[i][j].a_in = self.pes[i][j - 1].a_out
                if i == 0:
                    self.pes[i][j].b_in = b_inputs[j] if b_inputs[j] is not None else 0
                else:
                    self.pes[i][j].b_in = self.pes[i - 1][j].b_out

        # Phase 2: tất cả PE compute đồng thời
        active = 0
        for row in self.pes:
            for pe in row:
                if pe.a_in != 0 or pe.b_in != 0:
                    active += 1
                pe.cycle()

        self.cycle_count += 1
        self.active_macs_per_cycle.append(active)
        self.total_macs += active

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Tính C = A @ B với A: [rows, K], B: [K, cols]."""
        assert A.shape[0] == self.rows
        assert B.shape[1] == self.cols
        K = A.shape[1]
        assert B.shape[0] == K

        self.reset_accumulators()

        # Bơm data theo pattern systolic (skewed)
        total_cycles = K + self.rows + self.cols - 2

        for t in range(total_cycles):
            a_in = []
            for i in range(self.rows):
                k = t - i
                a_in.append(A[i, k] if 0 <= k < K else None)
            b_in = []
            for j in range(self.cols):
                k = t - j
                b_in.append(B[k, j] if 0 <= k < K else None)
            self.step(a_in, b_in)

        C = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                C[i, j] = self.pes[i][j].accumulator
        return C

    def stats(self):
        utilization = self.total_macs / (self.cycle_count * self.rows * self.cols)
        print(f"Total cycles: {self.cycle_count}")
        print(f"Total MAC operations: {self.total_macs}")
        print(f"Utilization: {utilization * 100:.1f}%")


if __name__ == "__main__":
    sa = SystolicArray(rows=4, cols=4)
    A = np.random.randn(4, 8)
    B = np.random.randn(8, 4)
    C_sim = sa.matmul(A, B)
    C_ref = A @ B

    print("Max error:", np.abs(C_sim - C_ref).max())
    sa.stats()
