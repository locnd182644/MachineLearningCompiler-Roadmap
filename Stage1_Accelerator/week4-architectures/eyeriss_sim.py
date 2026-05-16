"""Bài tập 4.1 — Mô phỏng Eyeriss row-stationary dataflow.

STUB: cài đặt các phần TODO.

Row-stationary (đóng góp của Eyeriss paper):
  - Mỗi PE chịu trách nhiệm 1 hàng output.
  - Filter row được nạp và ở yên trong register của PE.
  - Input row stream qua PE.
  - Partial sums tích lũy giữa các PE.

Conv: output[h,w,c_out] = sum_{kh,kw,c_in}
        input[h+kh, w+kw, c_in] * filter[kh,kw,c_in,c_out]
"""
import numpy as np


class EyerissPE:
    def __init__(self):
        self.filter_row = None  # weight row ở yên đây
        self.psum_acc = 0.0

    def load_filter_row(self, weights: np.ndarray):
        """Nạp 1 hàng filter vào PE (stationary)."""
        self.filter_row = weights

    def compute_output_row(self, input_row: np.ndarray) -> np.ndarray:
        """1D convolution filter_row * input_row → row của partial sums.

        TODO: trả về mảng partial sums, dài = len(input_row) - len(filter_row) + 1
        """
        raise NotImplementedError


class EyerissArray:
    """Mảng PE chạy row-stationary cho convolution 2D.

    TODO:
      - Map mỗi hàng filter (theo kh) lên 1 hàng PE.
      - Stream input rows; mỗi PE compute 1 row 1D-conv.
      - Cộng dồn partial sums giữa các hàng PE (theo kh) để ra output row 2D.
      - Lặp qua c_in, c_out.
      - Đếm số lần truy cập "DRAM" vs reuse trong PE (điểm chính của Eyeriss:
        data movement tốn năng lượng hơn compute).
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # TODO: khởi tạo lưới EyerissPE
        raise NotImplementedError

    def conv2d(self, inp: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """inp: [H, W, C_in], filt: [KH, KW, C_in, C_out] → [H', W', C_out]."""
        # TODO
        raise NotImplementedError


if __name__ == "__main__":
    # Sau khi cài đặt: verify với scipy/numpy conv hoặc torch.nn.functional.conv2d.
    print("Cài đặt các TODO, sau đó verify correctness với một conv tham chiếu.")
