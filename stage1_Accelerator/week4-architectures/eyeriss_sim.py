"""Bài tập 4.1 — Mô phỏng Eyeriss row-stationary dataflow.

Row-stationary (đóng góp của Eyeriss paper):
  - Mỗi PE chịu trách nhiệm 1 hàng output (1D convolution của 1 filter row).
  - Filter row được nạp và ở yên trong register của PE (stationary).
  - Input row stream qua PE.
  - Partial sums tích lũy giữa các hàng PE (theo kh) để ra 1 output row 2D.

Conv (cross-correlation, như trong NN):
  output[oh, ow, co] = sum_{kh, kw, ci}
        input[oh+kh, ow+kw, ci] * filter[kh, kw, ci, co]

Điểm chính của Eyeriss: data movement tốn năng lượng hơn nhiều so với compute.
Một MAC ~ 1 đơn vị năng lượng, nhưng đọc 1 toán hạng từ DRAM ~ 200x. Row-stationary
tối đa hoá reuse trên chip (filter ở yên trong PE, input stream + reuse theo chéo,
psum tích lũy giữa các PE) nên giảm mạnh số lần chạm DRAM. `stats()` định lượng
điều này bằng cách so sánh số truy cập DRAM giữa Eyeriss và một accelerator "naive"
(đọc lại toán hạng từ DRAM cho mỗi MAC).
"""
import numpy as np
from dataclasses import dataclass, field


# Mô hình năng lượng tương đối (đơn vị: bội số của 1 MAC), lấy bậc độ lớn từ
# Eyeriss / "Computing's Energy Problem" (Horowitz, ISSCC'14).
ENERGY = {
    "mac": 1.0,        # 1 phép nhân-cộng
    "rf": 1.0,         # đọc/ghi register file trong PE
    "pe_pass": 2.0,    # truyền dữ liệu giữa các PE kề nhau
    "glb": 6.0,        # đọc/ghi global buffer (SRAM on-chip)
    "dram": 200.0,     # đọc/ghi DRAM off-chip
}


@dataclass
class EyerissPE:
    filter_row: np.ndarray | None = None  # weight row ở yên đây (stationary)
    psum_acc: float = 0.0
    rf_reads: int = 0  # đếm số lần đọc filter từ register file (reuse)

    def load_filter_row(self, weights: np.ndarray):
        """Nạp 1 hàng filter vào PE (stationary, dùng lại cho mọi output)."""
        self.filter_row = np.asarray(weights, dtype=np.float64)

    def compute_output_row(self, input_row: np.ndarray) -> np.ndarray:
        """1D cross-correlation filter_row * input_row → row của partial sums.

        Filter ở yên trong RF; input_row stream qua. Mỗi output element dùng lại
        đúng filter row đó (RF read), không chạm lại DRAM.

        Trả về mảng partial sums dài = len(input_row) - len(filter_row) + 1.
        """
        f = self.filter_row
        kw = len(f)
        n = len(input_row) - kw + 1
        out = np.zeros(n, dtype=np.float64)
        for ow in range(n):
            out[ow] = float(np.dot(f, input_row[ow:ow + kw]))
            self.rf_reads += kw  # mỗi tap đọc 1 weight từ RF (đã stationary)
        return out


@dataclass
class _Counters:
    macs: int = 0
    # Truy cập DRAM (off-chip) — cái Eyeriss muốn giảm.
    filter_dram: int = 0   # mỗi weight nạp từ DRAM đúng 1 lần / (ci, co)
    input_dram: int = 0    # mỗi pixel input nạp từ DRAM đúng 1 lần (cache ở GLB)
    psum_dram: int = 0     # mỗi output element ghi ra DRAM đúng 1 lần
    # Reuse trên chip (rẻ) — chỗ Eyeriss "ăn" năng lượng tiết kiệm.
    filter_rf: int = 0     # đọc filter từ RF của PE
    input_pe: int = 0      # input stream/reuse qua PE
    psum_pe: int = 0       # tích lũy psum giữa các PE
    cached_input_pixels: set = field(default_factory=set)


class EyerissArray:
    """Mảng PE chạy row-stationary cho convolution 2D.

    Mapping logic (theo Eyeriss): mỗi filter row kh → 1 hàng PE; trong 1 lần
    xử lý 1 output row oh, các hàng PE (kh = 0..KH-1) cùng đóng góp psum rồi
    cộng dồn theo chiều dọc để ra output row 2D. Bản mô phỏng này không cycle-
    accurate; nó tập trung vào (1) tính đúng conv và (2) đếm data movement để
    minh hoạ vì sao row-stationary tiết kiệm năng lượng.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        # Lưới PE vật lý. rows nên >= KH để map đủ filter rows trong 1 pass.
        self.pes = [[EyerissPE() for _ in range(cols)] for _ in range(rows)]
        self.c = _Counters()

    def conv2d(self, inp: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """inp: [H, W, C_in], filt: [KH, KW, C_in, C_out] → [H', W', C_out]."""
        inp = np.asarray(inp, dtype=np.float64)
        filt = np.asarray(filt, dtype=np.float64)
        H, W, C_in = inp.shape
        KH, KW, C_in_f, C_out = filt.shape
        assert C_in == C_in_f, "C_in của input và filter phải khớp"
        if KH > self.rows:
            raise ValueError(
                f"KH={KH} > rows={self.rows}: mảng PE không đủ hàng để map "
                f"toàn bộ filter rows (cần tiling theo kh, chưa hỗ trợ)."
            )

        OH, OW = H - KH + 1, W - KW + 1
        out = np.zeros((OH, OW, C_out), dtype=np.float64)
        self.c = _Counters()

        for co in range(C_out):
            for ci in range(C_in):
                # --- Nạp filter rows vào PE (stationary) ---
                # Mỗi weight đọc từ DRAM đúng 1 lần cho cặp (ci, co); sau đó ở yên
                # trong RF và được dùng lại cho mọi (oh, ow).
                for kh in range(KH):
                    self.pes[kh][0].load_filter_row(filt[kh, :, ci, co])
                    self.pes[kh][0].psum_acc = 0.0
                    self.pes[kh][0].rf_reads = 0
                    self.c.filter_dram += KW

                # --- Sweep các output row, tích lũy theo kh ---
                for oh in range(OH):
                    acc = np.zeros(OW, dtype=np.float64)
                    for kh in range(KH):
                        r = oh + kh
                        input_row = inp[r, :, ci]

                        # Input pixel chỉ chạm DRAM lần đầu cho mỗi channel ci;
                        # sau đó cache ở GLB và dùng lại theo chéo (nhiều oh) và
                        # qua mọi co. Đây là input reuse của row-stationary.
                        for w in range(W):
                            key = (r, w, ci)
                            if key not in self.c.cached_input_pixels:
                                self.c.cached_input_pixels.add(key)
                                self.c.input_dram += 1
                        self.c.input_pe += W  # stream qua PE (reuse on-chip)

                        pe = self.pes[kh][0]
                        psums = pe.compute_output_row(input_row)
                        acc += psums  # cộng dồn psum giữa các hàng PE

                        self.c.macs += KW * OW
                        self.c.filter_rf += KW * OW
                        self.c.psum_pe += OW  # truyền/tích lũy psum giữa PE

                    out[oh, :, co] += acc  # tích lũy qua các channel ci (on-chip)
                    if ci == C_in - 1:
                        # output element hoàn tất → ghi ra DRAM đúng 1 lần
                        self.c.psum_dram += OW

        return out

    # ---- Báo cáo data movement & năng lượng ----
    def _energy(self, dram, glb, pe_pass, rf, mac):
        return (dram * ENERGY["dram"] + glb * ENERGY["glb"]
                + pe_pass * ENERGY["pe_pass"] + rf * ENERGY["rf"]
                + mac * ENERGY["mac"])

    def stats(self):
        c = self.c
        dram_eyeriss = c.filter_dram + c.input_dram + c.psum_dram

        # Naive: không reuse — mỗi MAC đọc 2 toán hạng + ghi 1 psum từ/ra DRAM.
        dram_naive = 3 * c.macs

        e_eyeriss = self._energy(
            dram=dram_eyeriss, glb=0,
            pe_pass=c.psum_pe, rf=c.filter_rf, mac=c.macs,
        )
        e_naive = self._energy(
            dram=dram_naive, glb=0, pe_pass=0, rf=0, mac=c.macs,
        )

        print(f"Total MACs:            {c.macs}")
        print("--- DRAM access (off-chip, tốn năng lượng) ---")
        print(f"  filter reads:        {c.filter_dram}")
        print(f"  input  reads:        {c.input_dram}")
        print(f"  psum   writes:       {c.psum_dram}")
        print(f"  Eyeriss total:       {dram_eyeriss}")
        print(f"  Naive   total:       {dram_naive}")
        if dram_eyeriss:
            print(f"  → giảm {dram_naive / dram_eyeriss:.1f}x truy cập DRAM")
        print("--- On-chip reuse (rẻ) ---")
        print(f"  filter RF reads:     {c.filter_rf}")
        print(f"  input  PE streams:   {c.input_pe}")
        print(f"  psum   PE accum:     {c.psum_pe}")
        print("--- Năng lượng ước lượng (đơn vị = 1 MAC) ---")
        print(f"  Eyeriss:             {e_eyeriss:,.0f}")
        print(f"  Naive:               {e_naive:,.0f}")
        print(f"  → tiết kiệm {e_naive / e_eyeriss:.1f}x năng lượng")


def _conv2d_reference(inp: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Conv tham chiếu bằng torch.nn.functional.conv2d (cross-correlation)."""
    import torch
    import torch.nn.functional as F

    # inp [H,W,Cin] -> [1,Cin,H,W]; filt [KH,KW,Cin,Cout] -> [Cout,Cin,KH,KW]
    x = torch.from_numpy(inp.transpose(2, 0, 1)[None]).double()
    w = torch.from_numpy(filt.transpose(3, 2, 0, 1)).double()
    y = F.conv2d(x, w)  # [1, Cout, OH, OW]
    return y[0].permute(1, 2, 0).numpy()  # -> [OH, OW, Cout]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, W, C_in, C_out, KH, KW = 8, 8, 3, 4, 3, 3
    inp = rng.standard_normal((H, W, C_in))
    filt = rng.standard_normal((KH, KW, C_in, C_out))

    arr = EyerissArray(rows=KH, cols=W)
    out = arr.conv2d(inp, filt)

    try:
        ref = _conv2d_reference(inp, filt)
        print("Max error vs torch conv2d:", np.abs(out - ref).max())
    except ImportError:
        print("(torch không có — bỏ qua verify)")

    print()
    arr.stats()
