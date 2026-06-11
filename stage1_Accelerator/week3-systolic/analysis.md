# Tuần 3 — Phân tích systolic array

## Bài tập 3.3 — So sánh với GPU

Matmul 256×256×256 (FP32). Cột systolic đo bằng `tiled_matmul(A, B, array_size=16)`
trong `systolic_extend.py`; cột GPU đo bằng PyTorch `a @ b` trên **RTX 4050 Laptop**
(20 SM, 2560 CUDA cores, boost-clock max 3.105 GHz).

| Chỉ số | Systolic 16×16 (sim) | GPU thực tế (PyTorch) |
|--------|----------------------|------------------------|
| Số cycle | **188,416** cycle | — (không quy ra cycle; đo ~22.9 µs/matmul) |
| HBM traffic (byte) | **8,650,752 B** (read 8,388,608 + write 262,144) | **≈ 786,432 B** ý tưởng (read A+B 524,288 + write C 262,144) |
| Utilization | **34.8 %** | **≈ 9–12 %** (1.46 TFLOP/s đo ÷ ~12–16 TFLOP/s peak FP32) |

### Cách ra các con số

**Useful MACs** = 256³ = 16,777,216 MAC (cố định cho mọi kiến trúc).

**Systolic — vì sao 188,416 cycle?**
Array 16×16 chỉ tính được tile 16×16×16 mỗi lần. Số tile = (256/16)³ = 16³ = 4096 tile.
Mỗi tile chạy `K + rows + cols − 2 = 16+16+16−2 = 46` cycle (gồm fill + drain).
→ 4096 × 46 = **188,416 cycle**.
Peak slot = 188,416 × (16×16) = 48,234,496 MAC-slot.
Utilization = 16,777,216 / 48,234,496 = **34.8 %**.

> Điểm cốt lõi: tile cạnh 16 mà độ trễ fill/drain ~30 cycle → mỗi tile "lãng phí"
> phần lớn thời gian cho việc bơm-vào / xả-ra. Tile càng nhỏ so với chiều dài pipeline,
> utilization càng tệ. Muốn lên ~90 % phải dùng array lớn hơn **và** K dài (reduction
> dài để amortize fill), đúng triết lý MXU 256×256 của TPU.

**Systolic — vì sao đọc tới 8 MiB?**
Simulator cố tình **load lại mọi tile từ DRAM**, không có reuse on-chip:
mỗi bộ ba (i,j,k) đọc 2 tile (A và B) = 2 × 16×16×4 = 2048 B.
→ 4096 tile × 2048 = **8,388,608 B** đọc + 256 tile output × 1024 = 262,144 B ghi.
GPU/TPU thật giữ lại tile trong SRAM/shared-memory nên đọc mỗi ma trận gần như **1 lần**
(≈ 0.5 MiB). Khoảng cách 8 MiB ↔ 0.75 MiB chính là giá trị của **data reuse** —
và là việc compiler phải tự lo (xem phần HW-SW).

**GPU — vì sao chỉ ~10 % utilization mà vẫn nhanh?**
256×256 quá nhỏ để lấp đầy 20 SM: matmul chạy 22.9 µs đạt 1.46 TFLOP/s, trong khi
peak FP32 lý thuyết ~12–16 TFLOP/s → ~9–12 %. GPU "nhanh" nhờ tần số cao + song song ồ ạt,
nhưng **lãng phí tài nguyên** ở kích thước nhỏ. Đây là lý do benchmark phải dùng matmul
lớn mới thấy GPU bão hòa.

**Kết luận so sánh:** cùng 16.7 M MAC hữu ích, nhưng hai kiến trúc lãng phí theo hai cách
khác nhau — systolic lãng phí ở **pipeline fill trên tile nhỏ** (35 %), GPU lãng phí ở
**không đủ việc để lấp SM** (~10 %). Cả hai đều nhắc: hiệu năng thực = hàm của *kích thước
bài toán so với kích thước phần cứng*, và đó là biến số mà **compiler** điều khiển qua tiling.

## Vì sao TPU dùng systolic array? — 5 điểm so với GPU SIMT

1. **Control granularity — 1 PE vs 1 thread.** GPU SIMT phải fetch/decode lệnh, quản lý
   warp scheduler, register file, predication cho *từng thread*. Trong systolic array, mỗi
   PE chỉ là 1 MAC + vài register; "lệnh" duy nhất là *nhân-cộng-đẩy* lặp lại. Bỏ toàn bộ
   tầng điều khiển per-thread → cùng diện tích silicon nhét được nhiều ALU hơn nhiều lần.

2. **Data reuse — chảy qua PE vs đi vòng qua shared memory.** GPU đọc operand ra register,
   tính, ghi lại; reuse phải dàn xếp qua shared memory + đồng bộ. Systolic để dữ liệu *chảy
   trực tiếp từ PE này sang PE kế* (`a_out → a_in`, `b_out → b_in` trong `PE.cycle()`): một
   activation nạp vào rìa được tái dùng dọc cả hàng PE mà **không** quay lại bộ nhớ. Reuse là
   tính chất *cấu trúc của dây nối*, không phải thứ phải lập lịch.

3. **Năng lượng / diện tích.** Phần lớn năng lượng GPU tiêu vào *di chuyển dữ liệu* (register
   file, cache, NoC) chứ không phải phép nhân. Systolic giữ dữ liệu di chuyển cực ngắn
   (PE↔PE liền kề) và loại bỏ cache/scheduler → MAC/Watt và MAC/mm² cao hơn hẳn. TPU đổi
   *tính linh hoạt* lấy *mật độ tính toán* cho đúng một workload: matmul.

4. **Độ phức tạp scheduler phần cứng ≈ 0.** GPU cần scoreboard, warp scheduler, dynamic
   dependency tracking để giấu latency. Systolic array là **lockstep tuyệt đối**: mọi PE
   bước cùng 1 clock, lịch trình là *tĩnh* và in cứng vào hình học mảng (skew bơm dữ liệu
   trong `matmul()`). Không có nhánh, không có cache miss → timing **xác định hoàn toàn**.

5. **Trách nhiệm chuyển sang compiler.** Vì phần cứng không tự lập lịch, *mọi* quyết định
   (tile size, thứ tự nạp weight, thời điểm DMA, padding) bị đẩy lên compiler ahead-of-time.
   GPU "tha thứ" lỗi lập lịch bằng cache/scheduler; TPU thì không — bù lại nó nhanh và tiết
   kiệm hơn *khi* compiler làm đúng. Đây là điểm bản lề dẫn sang phần dưới.

> Tóm tắt 1 dòng: **GPU đưa độ linh hoạt vào phần cứng; TPU đưa độ linh hoạt vào compiler.**

## Liên hệ HW-SW

TPU MXU không có scheduler, không cache, không branch predictor. Mỗi đặc điểm "thiếu" của
phần cứng biến thành một **nghĩa vụ bắt buộc** của compiler. Dưới đây nối từng đặc điểm HW
với hệ quả SW cụ thể — và với đúng dòng code trong simulator này, vì simulator chính là
mô hình thu nhỏ của ràng buộc đó.

### 1. MXU kích thước cố định → compiler phải tile + pad

**HW:** MXU là lưới PE cố định (TPU thật: 128×128 hoặc 256×256; sim: `array_size`). Nó
*không biết* ma trận của bạn to bao nhiêu — nó chỉ nuốt được đúng một tile bằng kích thước
nó.

**SW:** compiler phải (a) chẻ matmul lớn thành tile đúng cạnh array, (b) sinh vòng lặp ba
chiều i/j/k, (c) **pad** khi chiều không chia hết. Trong `tiled_matmul()`:

```python
a_tile = np.zeros((S, S)); b_tile = np.zeros((S, S))   # pad mặc định = 0
a_tile[:r1-r0, :k1-k0] = A[r0:r1, k0:k1]                # nhét phần thật vào góc
```

Padding bằng 0 không đổi kết quả (0×x=0) nhưng *vẫn tốn cycle*: tile rìa của ma trận 250×250
trên array 16 vẫn chạy đủ 46 cycle dù chỉ 10 hàng có ý nghĩa. → compiler phải cân nhắc
chọn array size / pad sao cho ít lãng phí. **Bài học:** "tensor không chia hết phần cứng" là
vấn đề của compiler, không phải runtime.

### 2. Không scheduler → compiler phải lập DMA ahead-of-time, sai 1 cycle = stall

**HW:** array chạy lockstep. Dữ liệu phải có mặt ở rìa PE *đúng cycle* nó cần. Không có
scoreboard để chờ, không có cache để giấu latency nạp chậm.

**SW:** compiler phải tính trước **wavefront schedule** — bơm phần tử nào vào cycle nào.
Đây chính là phép *skew* trong simulator:

```python
# systolic_sim.py — output-stationary: A[i,k] vào hàng i tại cycle t khi k = t - i
for i in range(self.rows):
    k = t - i
    a_in.append(A[i, k] if 0 <= k < K else None)
```

```python
# systolic_extend.py — weight-stationary: activation A[m,i] bơm vào hàng i tại cycle m+i
m = t - i
inject = A[m, i] if 0 <= m < M else 0.0
```

Cái `t - i` đó *là* lịch DMA. Lệch 1 → partial sum đi xuống cột gặp sai activation → kết quả
sai (chứ không chỉ chậm). Trên phần cứng thật, compiler còn phải xếp lệnh DMA HBM→SRAM
*trước* nhiều cycle để dữ liệu kịp tới rìa khi MXU bắt đầu — không kịp thì array stall, cả
lưới đứng. **Bài học:** timing là một phần của *chương trình*, do compiler sinh, không phải
do phần cứng tự xoay.

### 3. Hai dataflow, hai chiến lược reuse → compiler phải chọn

**HW:** cùng một lưới PE chạy được nhiều dataflow. `SystolicArray` (output-stationary):
accumulator đứng yên, A và B cùng chảy. `WeightStationaryArray`: weight nằm yên trong PE
(`load_weights()`), chỉ activation chảy, partial sum chảy xuống.

**SW:** lựa chọn dataflow là *quyết định của compiler*, đánh đổi reuse:
- **Weight-stationary** (TPU dùng cho lớp có weight tái dùng nhiều): nạp weight 1 lần rồi
  đẩy *cả batch* activation qua → amortize chi phí nạp. Hợp khi M (số activation row) ≫ array.
- **Output-stationary**: hợp khi cần tích lũy K dài tại chỗ.

Compiler phải biết shape thực (batch, in/out features) để chọn, vì sai dataflow = đổ phí
băng thông nạp lại weight mỗi tile. Đây đúng là khoảng cách "8 MiB ↔ 0.75 MiB" ở bài 3.3:
reuse là thứ compiler *mua được* bằng cách chọn dataflow + giữ tile trong SRAM, không phải
thứ phần cứng tự cho.

### 4. Không cache, không reuse tự động → compiler phải quản lý on-chip memory tay

**HW:** TPU có SRAM scratchpad (Unified Buffer) *được địa chỉ tường minh*, không phải cache
tự động. Simulator phơi bày phiên bản tệ nhất: `tiled_matmul` reload mọi tile từ "DRAM"
(`stats["dram_bytes_read"] += 2 * tile_bytes` mỗi vòng k) → 8 MiB.

**SW:** compiler phải tự sinh lịch *staging*: tile nào giữ lại trong SRAM giữa các vòng lặp,
tile nào nạp lại. Ví dụ giữ nguyên một hàng tile của A trong khi quét cột B → cắt mạnh
traffic. Cache GPU làm việc này *tự động* (và đôi khi sai); TPU đẩy nó cho compiler để đổi
lấy tính tất định. **Bài học:** mọi byte qua HBM trong bảng 3.3 là *do compiler quyết*, không
phải hệ quả ngẫu nhiên của cache.

### 5. Không có FPU linh hoạt → compiler lo numeric (INT8/BF16, scale)

**HW:** MXU chạy precision cố định (TPU: BF16/INT8 cho MAC, tích lũy FP32). Không có dải kiểu
số phong phú như GPU.

**SW:** compiler/quantizer phải chèn scale factor, chọn điểm lượng tử, sắp xếp tích lũy để
không tràn — *trước khi* chạy, vì array không có nhánh để xử lý ngoại lệ runtime. Simulator
này dùng FP64 cho đơn giản (max error ~1e-14, thuần lỗi làm tròn), nhưng trên phần cứng thật
đây là cả một pass compiler riêng.

### 6. Lockstep + ramp-up/ramp-down → compiler phải amortize pipeline fill

**HW:** mảng cần thời gian *đổ đầy* (ramp-up) và *xả* (ramp-down); chỉ ở giữa mới đạt peak.
Plot `utilization.png` (từ `plot_utilization`) cho thấy rõ hình thang: tăng dần → bão hòa →
giảm dần. Đây là nguồn gốc con số 34.8 % ở bài 3.3.

**SW:** compiler phải chọn tile *đủ lớn* (đặc biệt chiều reduction K) để phần "bão hòa" lấn
át phần fill/drain. Tile 16 với fill ~30 cycle → 35 % util; tile 256 với K dài → ~90 %. Quyết
định kích thước tile để giấu pipeline-fill là **bài toán tối ưu của compiler**, đọc thẳng ra
từ hình dạng đường utilization.

---

> "Cho AI accelerator, compiler không *tối ưu* code — compiler **viết** code."
>
> Sáu mục trên là minh chứng cụ thể: tiling, padding, lịch DMA (skew), chọn dataflow, staging
> SRAM, lượng tử hóa, chọn tile-size — *không có cái nào* được phần cứng tự lo. Phần cứng chỉ
> hứa một điều: nếu compiler bơm đúng số vào đúng PE ở đúng cycle, nó trả về tích ma trận
> nhanh và tiết kiệm điện. Toàn bộ phần "đúng" đó là chương trình do compiler sinh ra. Đó là
> lý do simulator này được giữ lại làm **target backend cho compiler ở Giai đoạn 2**: mỗi
> ràng buộc HW ở trên sẽ thành một pass compiler bạn phải viết.
