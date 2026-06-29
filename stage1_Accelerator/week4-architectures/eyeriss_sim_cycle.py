"""Bài tập 4.1 (mở rộng) — Eyeriss row-stationary, bản CYCLE-ACCURATE.

Khác với `eyeriss_sim.py` (chỉ đếm data movement, không có khái niệm thời gian),
bản này mô phỏng từng cycle giống `systolic_sim.py` của week3:
  - Mỗi PE có method `cycle()` thực hiện đúng 1 MAC / cycle.
  - Vòng lặp cycle ngoài cùng đẩy toàn mảng tiến 1 nhịp đồng thời.
  - Theo dõi `cycle_count`, `total_macs`, `active_macs_per_cycle` → utilization.

Ánh xạ row-stationary (theo Eyeriss paper, Fig. logical PE array):
  - Hàng PE  = filter row r (kh)         → 1 filter row ở yên trong mỗi PE.
  - Cột PE   = output row e đang tính     → mỗi cột lo 1 output row 2D.
  - PE[r][e] tính 1D conv: filter row r ⊛ input row (e+r) → psum cho output row e.
  - Reuse:
      * filter:  cùng 1 filter row r broadcast cho cả hàng PE (đọc DRAM 1 lần).
      * input :  input row (e+r) dùng chung theo ĐƯỜNG CHÉO giữa các PE.
      * psum  :  cộng dồn theo CHIỀU DỌC qua r = 0..KH-1 → output row 2D.

Mô hình thời gian trong 1 PE: PE đọc filter (stationary trong RF) và stream input
row qua; nó duyệt (ow, kw) làm 1 MAC mỗi cycle → tổng OW*KW cycle / PE. Skew dọc
(start_delay = r) mô phỏng psum chảy xuống cột. Mảng vật lý rows×cols được tái sử
dụng (time-multiplex) cho mỗi cặp (ci, co) và mỗi tile output row.

Conv (cross-correlation):
  output[oh, ow, co] = sum_{kh, kw, ci} input[oh+kh, ow+kw, ci]*filter[kh,kw,ci,co]
"""
import numpy as np
from dataclasses import dataclass, field


# Năng lượng tương đối (đơn vị = 1 MAC), bậc độ lớn từ Horowitz ISSCC'14 / Eyeriss.
ENERGY = {
    "mac": 1.0, "rf": 1.0, "pe_pass": 2.0, "glb": 6.0, "dram": 200.0,
}


@dataclass
class EyerissPE:
    """Processing Element row-stationary, cycle-accurate (1 MAC / cycle).

    filter_row ở yên trong RF; input_row stream qua; psum[] tích lũy tại chỗ.
    `program` là chuỗi (ow, kw): mỗi phần tử = 1 MAC sẽ chạy trong 1 cycle.
    """
    filter_row: np.ndarray | None = None
    input_row: np.ndarray | None = None
    psum: np.ndarray | None = None
    program: list = field(default_factory=list)
    pc: int = 0           # program counter (MAC kế tiếp)
    start_delay: int = 0  # skew: bắt đầu trễ để psum chảy xuống cột

    def load(self, filter_row, input_row, OW, KW, start_delay):
        """Nạp filter (stationary) + input row; dựng lịch MAC cho 1D conv."""
        self.filter_row = filter_row
        self.input_row = input_row
        self.psum = np.zeros(OW, dtype=np.float64)
        self.program = [(ow, kw) for ow in range(OW) for kw in range(KW)]
        self.pc = 0
        self.start_delay = start_delay

    def busy(self, cyc: int) -> bool:
        return cyc >= self.start_delay and self.pc < len(self.program)

    def end_cycle(self) -> int:
        """Cycle (loại trừ) mà PE chạy xong toàn bộ program."""
        return self.start_delay + len(self.program)

    def cycle(self, cyc: int) -> bool:
        """Tiến 1 nhịp. Trả True nếu PE thực sự làm 1 MAC trong cycle này."""
        if not self.busy(cyc):
            return False
        ow, kw = self.program[self.pc]
        self.psum[ow] += self.filter_row[kw] * self.input_row[ow + kw]
        self.pc += 1
        return True


@dataclass
class _Counters:
    macs: int = 0
    filter_dram: int = 0
    input_dram: int = 0
    psum_dram: int = 0
    filter_rf: int = 0
    input_pe: int = 0
    psum_pe: int = 0
    cached_input_pixels: set = field(default_factory=set)


class EyerissArray:
    """Mảng PE vật lý rows×cols chạy row-stationary, cycle-accurate.

    rows phải >= KH (map đủ filter rows trong 1 pass). Output rows (OH) được tile
    theo cols. Mảng vật lý được time-multiplex cho mỗi (ci, co) và mỗi tile.
    """

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.pes = [[EyerissPE() for _ in range(cols)] for _ in range(rows)]
        self.cycle_count = 0
        self.total_macs = 0
        self.active_macs_per_cycle = []
        self.c = _Counters()

    def _run_tile(self, active_pes: list):
        """Chạy 1 tile tới khi mọi PE trong tile xong; ghi nhận utilization.

        active_pes: list các PE đã `load()` cho tile hiện tại. Toàn mảng vật lý
        tiến đồng thời; PE ngoài tile coi như idle (đóng góp 0 vào utilization).
        """
        if not active_pes:
            return
        tile_len = max(pe.end_cycle() for pe in active_pes)
        for cyc in range(tile_len):
            active = 0
            for pe in active_pes:
                if pe.cycle(cyc):
                    active += 1
            self.active_macs_per_cycle.append(active)
            self.total_macs += active
            self.cycle_count += 1

    def conv2d(self, inp: np.ndarray, filt: np.ndarray) -> np.ndarray:
        """inp: [H, W, C_in], filt: [KH, KW, C_in, C_out] → [H', W', C_out]."""
        inp = np.asarray(inp, dtype=np.float64)
        filt = np.asarray(filt, dtype=np.float64)
        H, W, C_in = inp.shape
        KH, KW, C_in_f, C_out = filt.shape
        assert C_in == C_in_f, "C_in của input và filter phải khớp"
        if KH > self.rows:
            raise ValueError(
                f"KH={KH} > rows={self.rows}: cần tiling theo kh (chưa hỗ trợ)."
            )

        OH, OW = H - KH + 1, W - KW + 1
        out = np.zeros((OH, OW, C_out), dtype=np.float64)
        self.cycle_count = 0
        self.total_macs = 0
        self.active_macs_per_cycle = []
        self.c = _Counters()

        for co in range(C_out):
            for ci in range(C_in):
                # Filter row r broadcast cho cả hàng PE → đọc DRAM 1 lần / (r,ci,co).
                for r in range(KH):
                    self.c.filter_dram += KW

                # Tile các output row (e) theo số cột vật lý.
                for e0 in range(0, OH, self.cols):
                    tile_es = list(range(e0, min(e0 + self.cols, OH)))
                    active_pes = []
                    for r in range(KH):
                        for col, e in enumerate(tile_es):
                            pe = self.pes[r][col]
                            input_row = inp[e + r, :, ci]
                            # input pixel chạm DRAM lần đầu cho channel ci, sau đó
                            # reuse theo chéo + qua mọi co (cache ở GLB).
                            for w in range(W):
                                key = (e + r, w, ci)
                                if key not in self.c.cached_input_pixels:
                                    self.c.cached_input_pixels.add(key)
                                    self.c.input_dram += 1
                            pe.load(filt[r, :, ci, co], input_row, OW, KW,
                                    start_delay=r)
                            active_pes.append(pe)

                    self._run_tile(active_pes)

                    # Cộng dồn psum theo chiều dọc (qua r) → output row 2D.
                    for col, e in enumerate(tile_es):
                        acc = np.zeros(OW, dtype=np.float64)
                        for r in range(KH):
                            acc += self.pes[r][col].psum
                            self.c.psum_pe += OW
                        out[e, :, co] += acc  # tích lũy qua channel ci (on-chip)
                        if ci == C_in - 1:
                            self.c.psum_dram += OW  # ghi DRAM 1 lần khi xong

        # Counters dẫn xuất từ tổng MAC (reuse on-chip).
        self.c.macs = KH * KW * OW * OH * C_in * C_out
        self.c.filter_rf = self.c.macs   # mỗi MAC đọc 1 weight từ RF
        self.c.input_pe = self.c.macs    # mỗi MAC đọc 1 activation streaming
        return out

    # ---- Báo cáo ----
    def _energy(self, dram, pe_pass, rf, mac):
        return (dram * ENERGY["dram"] + pe_pass * ENERGY["pe_pass"]
                + rf * ENERGY["rf"] + mac * ENERGY["mac"])

    def stats(self):
        c = self.c
        # --- Cycle / utilization (giống systolic_sim) ---
        denom = self.cycle_count * self.rows * self.cols
        util = self.total_macs / denom if denom else 0.0
        print("=== Cycle-accurate ===")
        print(f"Total cycles:          {self.cycle_count}")
        print(f"Total MAC operations:  {self.total_macs}")
        print(f"PE array:              {self.rows}x{self.cols} = "
              f"{self.rows * self.cols} PEs")
        print(f"Utilization:           {util * 100:.1f}%")

        # --- Data movement / energy (điểm chính của Eyeriss) ---
        dram_eyeriss = c.filter_dram + c.input_dram + c.psum_dram
        dram_naive = 3 * c.macs
        e_eyeriss = self._energy(dram_eyeriss, c.psum_pe, c.filter_rf, c.macs)
        e_naive = self._energy(dram_naive, 0, 0, c.macs)
        print("=== Data movement ===")
        print(f"DRAM (Eyeriss):        {dram_eyeriss} "
              f"(filter {c.filter_dram} + input {c.input_dram} "
              f"+ psum {c.psum_dram})")
        print(f"DRAM (naive):          {dram_naive}")
        if dram_eyeriss:
            print(f"  → giảm {dram_naive / dram_eyeriss:.1f}x truy cập DRAM")
        print(f"Energy Eyeriss/naive:  {e_eyeriss:,.0f} / {e_naive:,.0f}"
              f"  → tiết kiệm {e_naive / e_eyeriss:.1f}x")


def _conv2d_reference(inp: np.ndarray, filt: np.ndarray) -> np.ndarray:
    """Conv tham chiếu bằng torch.nn.functional.conv2d (cross-correlation)."""
    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(inp.transpose(2, 0, 1)[None]).double()
    w = torch.from_numpy(filt.transpose(3, 2, 0, 1)).double()
    y = F.conv2d(x, w)
    return y[0].permute(1, 2, 0).numpy()


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
