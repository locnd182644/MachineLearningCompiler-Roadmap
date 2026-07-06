# Tuần 5 — Numerical Formats & Quantization

> **Câu hỏi central:** Tại sao chip AI không dùng FP64? Khi nào INT8 đủ,
> khi nào không?

Câu trả lời ngắn: **precision là một trục của design space, không phải "càng cao càng tốt"**.
Mỗi bit mantissa phải trả bằng diện tích silicon, năng lượng, và băng thông bộ nhớ — ba
thứ mà accelerator luôn thiếu. Deep learning lại chịu được nhiễu số học. Nên compiler/hardware
hạ precision xuống đúng mức mà accuracy chưa gãy, và đổi số bit tiết kiệm được lấy nhiều
MAC hơn trên cùng một miếng silicon. Ba bài dưới đây chứng minh điều đó bằng số đo thật.

## Files

| File | Trạng thái | Bài tập |
|------|-----------|---------|
| `quantize.py` | ✅ **Hoàn thành** | 5.1 — Quantization tự cài: symmetric / asymmetric / per-channel INT8 |
| `quantize_resnet18.py` | ✅ **Hoàn thành** | 5.2 — PTQ ResNet18 (FX mode), đo accuracy drop + speedup CPU |
| `precision_stability.py` | ✅ **Hoàn thành** | 5.3 — FP32 vs FP16 vs BF16 training stability |
| `grad_magnitude.png` | ✅ Output | Biểu đồ gradient L2-norm 3 precision (do 5.3 sinh ra) |

## Run

```bash
conda activate mlc

python quantize.py                 # chạy được ngay, in relative error 3 scheme
python precision_stability.py      # tự tải MNIST → grad_magnitude.png
python quantize_resnet18.py \       # cần ImageNet val subset ở data/imagenet/val
    --data data/imagenet \
    --calib-samples 512 --eval-samples 2000
```

## Checklist output cuối tuần

- [x] `quantize.py`: mean relative error của INT8 matmul (3 scheme)
- [x] ResNet18 PTQ: accuracy drop + speedup trên CPU
- [x] Biểu đồ gradient magnitude 3 precision (`grad_magnitude.png`)
- [x] Note "precision strategies in modern AI chips" (phần Phân tích bên dưới)

---

## Nền tảng — giải phẫu các định dạng số

Một số thực dấu phẩy động = `(-1)^sign × 1.mantissa × 2^exponent`. **Exponent quyết định
*dải* (range), mantissa quyết định *độ mịn* (precision).** Đây là chìa khoá của cả tuần:

| Format | Bits (S/E/M) | Dải xấp xỉ | Bytes | Ý nghĩa hardware |
|--------|-------------|------------|-------|------------------|
| FP64 | 1 / 11 / 52 | 1e−308 … 1e308 | 8 | HPC/khoa học. DL không cần. |
| FP32 | 1 / 8 / 23 | 1e−38 … 3e38 | 4 | Baseline "đúng", tham chiếu. |
| **BF16** | 1 / 8 / 7 | **1e−38 … 3e38** | 2 | *Cùng dải mũ FP32*, ít mantissa → ổn định, dùng train trên TPU/NPU. |
| FP16 | 1 / 5 / 10 | 6e−5 … 65504 | 2 | Dải mũ **hẹp** → dễ underflow/overflow, cần loss-scaling. |
| INT8 | integer 8-bit | −128 … 127 | 1 | Không có mũ; cần `scale`(+`zero_point`) để ánh xạ về float. |

> 🔑 **Vì sao BF16 thắng FP16 cho training?** Gradient có thể nhỏ tới `1e−7`. FP16 coi mọi
> thứ dưới `6e−5` là 0 (underflow) → gradient biến mất. BF16 giữ nguyên dải mũ của FP32 nên
> gradient nhỏ vẫn sống, chỉ mất vài chữ số cuối. Hardware chọn BF16 = **đổi mantissa (thứ DL
> chịu được) lấy dải mũ (thứ DL cần)**. Đúng một quyết định co-design HW-SW.

---

## Phân tích — trả lời câu hỏi central

### Phần 1 — Tại sao AI chip *không* dùng FP64 (và hạ tới cả INT8)?

Ba ràng buộc phần cứng, tất cả đều chống lại precision cao:

**(1) Diện tích silicon của multiplier ~ bình phương số bit mantissa.** Một FP32 multiplier
phải nhân mantissa 24×24 bit; FP64 là 53×53 (~4.8× diện tích). INT8 multiplier chỉ 8×8 —
nhỏ hơn FP32 cỡ **một bậc độ lớn**. Cùng một ngân sách transistor, bạn nhét được **rất
nhiều** MAC INT8 thay cho vài MAC FP64. Systolic array/MXU của TPU là biển MAC 8-bit chính
vì lý do này — TOPS/mm² là thứ được tối ưu, không phải độ chính xác 15 chữ số.

**(2) Năng lượng bị thống trị bởi *data movement*, không phải phép tính.** Đọc 1 toán hạng
từ DRAM tốn năng lượng gấp ~100–1000× so với 1 phép nhân. INT8 = 1 byte thay vì FP32 = 4 byte
→ **giảm 4× lưu lượng DRAM/SRAM và băng thông**. Nối lại với roofline tuần 1: phần lớn kernel
ML là *memory-bound*, nên cắt số byte đi một nửa/một phần tư là con đường trực tiếp nhất để
tăng hiệu năng — nhiều khi còn hơn cả tăng số ALU.

**(3) DL chịu được nhiễu số học.** Mạng neural có dư thừa lớn, inference chỉ cần argmax đúng
chứ không cần logit chính xác 7 chữ số. Gradient descent tự sửa sai số nhỏ. Đây là "giấy phép"
để hạ precision — thứ mà mô phỏng vật lý (cần FP64) không có.

→ **Kết luận:** FP64 phí phạm cả ba tài nguyên khan hiếm nhất để mua một độ chính xác mà DL
không dùng đến. Chip AI đi ngược lại: hạ precision đến sát ngưỡng accuracy gãy, đổi bit tiết
kiệm lấy throughput.

### Phần 2 — Cơ chế quantization (bài 5.1) & số đo thật

`quantize.py` cài 3 scheme và đo mean relative error của một INT8 matmul 256×256 so với FP32:

| Scheme | Công thức | Relative error đo được |
|--------|-----------|------------------------|
| Per-tensor symmetric | `scale = |x|max / 127`, zero-centered | **1.61%** |
| Per-tensor asymmetric | `scale = (max−min)/255` + `zero_point` | **1.53%** |
| **Per-channel symmetric** | mỗi hàng/cột một `scale` riêng | **0.98%** |

*(số của một lần chạy; đầu vào `torch.randn` nên dao động nhẹ giữa các lần.)*

**Ba điều rút ra, mỗi điều là một ràng buộc hardware:**

- **INT8 matmul cần accumulator INT32.** Dot product sâu 256 phần tử, mỗi tích INT8×INT8 tới
  ±16k, cộng 256 số → tràn INT8 ngay. Nên `C_int = (A_q @ B_q).to(int32)`. Đây *đúng* là thứ
  MXU của TPU làm trong phần cứng: nhân 8-bit, **cộng dồn trong thanh ghi 32-bit**, chỉ
  requantize về 8-bit ở cuối.

- **Per-channel thắng per-tensor (0.98% < 1.61%) vì outlier.** Per-tensor dùng chung 1 scale;
  một trọng số lớn bất thường kéo `scale` lên, nghiền nát độ phân giải của mọi trọng số nhỏ.
  Per-channel cấp mỗi kênh một scale → outlier bị cô lập. Vì thế hardware hầu như luôn dùng
  **per-channel cho weight** (rẻ, tĩnh) nhưng **per-tensor cho activation** (scale phải tính
  runtime, per-channel quá đắt).

- **Asymmetric không "miễn phí".** Nó khớp phân phối lệch (ví dụ output ReLU ≥ 0) tốt hơn,
  nhưng khai triển `(q−z)` trong matmul đẻ ra 3 số hạng bù zero-point (`term_1/2/3` trong code)
  — thêm phép cộng số nguyên mỗi lần. Đó là lý do phần cứng thích **symmetric cho weight**
  (zero_point = 0, các số hạng bù biến mất) và chỉ dùng asymmetric ở activation khi cần.

### Phần 3 — PTQ ResNet18 (bài 5.2): accuracy vs tốc độ

Pipeline dùng `torch.ao.quantization.quantize_fx` (FX graph mode): `prepare_fx` chèn observer
vào toàn graph → calibrate bằng vài trăm ảnh để observer học min/max activation → `convert_fx`
thay op FP32 bằng INT8 kernel. Dùng FX thay Eager Mode vì `torchvision.resnet18` không có
`QuantStub/DeQuantStub`; FX symbolic-trace tự chèn quant/dequant và tự fuse Conv-BN-ReLU.

**Số đo thật (CPU x86, backend `x86`/fbgemm):**

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Latency (ms/ảnh) | 17.09 | 8.17 | **2.09× nhanh hơn** |
| Top-1 accuracy | — | — | *xem lưu ý* |

> ⚠️ **Trung thực về accuracy:** subset ImageNet ở máy này chỉ 50 ảnh / 5 folder. `ImageFolder`
> gán nhãn 0–4 theo thứ tự thư mục, **không khớp** index 1000-lớp mà ResNet18 xuất ra → cả FP32
> lẫn INT8 đều cho 0.00%, con số này **vô nghĩa** (không phải accuracy thật). Speedup 2.09× thì
> **hợp lệ** vì latency không phụ thuộc nhãn. Accuracy drop tham chiếu từ literature (Jacob 2018,
> torchvision) cho ResNet18 PTQ INT8 là **~0.5–1.0%** — nằm trong ngưỡng "deploy được ngay".
> Muốn số accuracy thật: cần ImageNet val đầy đủ với `val/<synset>` đúng mapping 1000 lớp.

**Vì sao speedup chỉ ~2× chứ không phải 4×** (dù byte giảm 4×): ResNet18 nhỏ, một phần bị
memory-bound trên CPU và INT8 GEMM kernel (fbgemm) không đạt full utilization; ranh giới
quant/dequant cũng thêm overhead. Trên accelerator có MXU INT8 chuyên dụng (TPU/NPU) tỉ lệ
này cao hơn nhiều vì cả throughput MAC lẫn băng thông đều được hưởng lợi.

### Phần 4 — Precision stability khi *training* (bài 5.3)

Train MLP nhỏ trên MNIST với 3 precision, **cố tình không dùng loss-scaling** để lộ điểm gãy.
Log gradient L2-norm (đo ở FP32 để phép đo không tự overflow) → `grad_magnitude.png`.

**Số đo thật (grad-norm cuối mỗi epoch):**

| Precision | epoch 1 | epoch 5 | NaN/Inf? |
|-----------|---------|---------|----------|
| FP32 | 4.05e−1 | 1.68e−1 | Không |
| FP16 | 8.27e−1 | 3.20e−1 | Không (ở scale này) |
| BF16 | 1.34e+0 | 1.60e−1 | Không |

**Đọc kết quả một cách trung thực:** ở bài toán bé và điều kiện tốt này, **cả FP16 cũng sống
sót** — grad-norm không bao giờ chạm dải underflow `6e−5` của FP16, loss hội tụ bình thường.
Điều này *không* bác bỏ luận điểm; nó cho thấy **rủi ro của FP16 mang tính cấu trúc, không phải
luôn hiển thị**: dải mũ hẹp là quả mìn chờ nổ khi (a) model sâu hơn, (b) activation/loss lớn hơn,
hoặc (c) gradient nhỏ hơn — lúc đó FP16 underflow/overflow → NaN, và cần loss-scaling để cứu.
BF16 bám sát FP32 (grad-norm cuối gần trùng: 1.60e−1 vs 1.68e−1) đúng như kỳ vọng vì cùng dải mũ.

> 🔑 **Đây chính là lý do mixed precision ra đời:** train BF16/FP16 để nhanh và tiết kiệm bộ nhớ,
> nhưng giữ **master weights + accumulate ở FP32** để không mất gradient nhỏ. TPU chọn BF16 để
> né hẳn bài toán loss-scaling của FP16.

---

## Note 1 trang — "Precision strategies in modern AI chips"

**Nguyên tắc xuyên suốt:** precision là một *knob* để đổi accuracy lấy diện tích/năng lượng/băng
thông. Compiler và hardware đồng thiết kế để vặn knob đó xuống mức thấp nhất mà accuracy chưa gãy.

- **Inference → INT8 (hoặc thấp hơn).** DL chịu nhiễu; INT8 giảm 4× byte, MAC nhỏ hơn ~1 bậc.
  Weight thường **per-channel symmetric** (chính xác, zero_point=0 nên không đẻ số hạng bù),
  activation thường **per-tensor asymmetric** (scale tính runtime, per-channel quá đắt). Accumulate
  luôn ở **INT32**. Cutting edge đẩy tiếp xuống INT4/FP8/FP4 cho LLM.
- **Training → BF16 + FP32 master (mixed precision).** Cần dải mũ rộng cho gradient nhỏ → BF16
  (cùng mũ FP32) đánh bại FP16 và né được loss-scaling. TPU/NPU đặt BF16 làm mặc định vì thế.
- **FP64 gần như vắng bóng** trong DL accelerator: phí cả silicon, năng lượng lẫn băng thông cho
  độ chính xác không dùng đến. Nó thuộc về HPC/khoa học, không thuộc về AI chip.
- **Việc của ML compiler:** chèn observer, chọn scheme (per-tensor/per-channel, sym/asym), tính
  scale/zero_point, chèn ranh giới quant↔dequant, và fuse để requantize xảy ra càng ít càng tốt.
  Mỗi lựa chọn ở đây là một phản ứng trực tiếp với một ràng buộc phần cứng — đúng triết lý của repo.

---

## Liên hệ HW ↔ SW (tóm tắt)

| Ràng buộc phần cứng | Phản ứng của số học / compiler |
|---------------------|--------------------------------|
| Multiplier ∝ (mantissa bit)² | Hạ precision: FP32→BF16/INT8, nhiều MAC hơn/mm² |
| Data movement thống trị năng lượng | INT8 = ¼ byte → ¼ băng thông (nối roofline tuần 1) |
| Dot product sâu tràn 8-bit | Accumulator INT32 trong MXU |
| Outlier làm hỏng scale chung | Per-channel quantization cho weight |
| Zero_point đẻ số hạng bù | Symmetric cho weight (zero_point = 0) |
| Gradient nhỏ underflow FP16 | BF16 (giữ dải mũ) + FP32 master weights |

## Tài liệu

- 📄 Jacob et al. 2018 — *Quantization and Training of Neural Networks for Efficient
  Integer-Arithmetic-Only Inference* (bài gốc của scheme trong `quantize.py`)
- 📘 Sze et al. — *Efficient Processing of Deep Neural Networks* (ch. numerical precision)
- 🎥 MIT 6.5940 (Han Song) — bài giảng Quantization & mixed precision
- 🔗 PyTorch docs — `torch.ao.quantization.quantize_fx` (FX graph mode quantization)
