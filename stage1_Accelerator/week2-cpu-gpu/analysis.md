# Tuần 2 — Phân tích bottleneck (Bài tập 2.3)

## Kết quả đo

Dưới đây là kết quả đo đạc hiệu năng thực tế trên CPU (Intel Core/AMD tương đương) và GPU (NVIDIA GeForce RTX 4050 Laptop GPU):

| Kernel | Thời gian (ms) | GFLOPS | % so với cuBLAS |
|--------|----------------|--------|-----------------|
| AVX (CPU, N=512) | 15.23 ms | 17.6 GFLOPS | — |
| V1 Naive (GPU, N=2048) | 22.19 ms | 774 GFLOPS | 10.0% |
| V2 Tiled (GPU, N=2048) | 16.84 ms | 1020 GFLOPS | 13.2% |
| V3 cuBLAS (GPU, N=2048) | 2.22 ms | 7741 GFLOPS | 100.0% |

> [!NOTE]
> Phép đo đạc `ncu` (Nsight Compute) trực tiếp bị hạn chế do phân quyền truy cập GPU Performance Counters trên môi trường sandbox (lỗi `driver resource was unavailable`). 
> Bảng dưới đây cung cấp các chỉ số phân tích lý thuyết và dữ liệu đặc trưng ước tính cho kiến trúc Ada Lovelace (RTX 4050) để làm rõ bản chất bottleneck.

## Metric Nsight Compute (Phân tích lý thuyết & ước tính)

| Metric | V1 Naive | V2 Tiled | Nhận xét |
|--------|----------|----------|----------|
| **SM Throughput %** | Thấp (~15% - 25%) | Trung bình (~45% - 55%) | Tiled giải phóng SM khỏi việc đợi bộ nhớ, giúp SM bận rộn tính toán hơn. |
| **Memory Throughput %** | Rất cao (~75% - 85%) | Thấp/Tối ưu (~30% - 40%) | Naive gây nghẽn băng thông bộ nhớ toàn cục (Global Memory). Tiled giảm tải đáng kể nhờ Shared Memory. |
| **Achieved Occupancy** | Cao (~70% - 80%) | Cao (~75% - 85%) | Cả hai đều kích hoạt nhiều warp song song, nhưng Naive bị stall lớn do phụ thuộc dữ liệu (Memory Dependency). |
| **L1 hit rate** | Thấp (~15% - 20%) | N/A (Dùng Shared Memory) | Naive đọc ma trận B không Coalesced nên cache L1 liên tục bị nạp đè. Tiled bỏ qua L1 bằng cách dùng Shared Memory trực tiếp. |
| **L2 hit rate** | Trung bình (~50% - 60%) | Cao (~80% - 90%) | Tiled tận dụng lại dữ liệu theo khối liên tục nên L2 cache giữ được dữ liệu rất tốt. |

---

## Giải thích bottleneck

### V1 Naive — vì sao chậm?

1. **Truy cập bộ nhớ dư thừa quá lớn**: 
   * Để tính ma trận kết quả $C$, mỗi phần tử của $A$ và $B$ bị tải lại từ Global Memory $N$ lần. Tổng số lần đọc bộ nhớ toàn cục là $2 \times N^3$ số float.
   * Với $N=2048$, lượng dữ liệu cần đọc là $2 \times 2048^3 \times 4 \text{ bytes} \approx 68.7 \text{ GB}$. Điều này làm cạn kiệt băng thông của card đồ họa rất nhanh.
2. **Không Coalesced (Gom cụm bộ nhớ)**:
   * Trong vòng lặp `k`, các thread trong cùng một Warp (32 threads) truy cập ma trận B theo cột: `B[k * N + col]`. 
   * Các địa chỉ bộ nhớ cách nhau $N \times 4$ bytes (ví dụ $2048 \times 4 = 8192$ bytes), vượt xa kích thước 1 Cache Line (128 bytes). 
   * Do đó, thay vì 1 giao dịch bộ nhớ phục vụ cho cả Warp, GPU phải phát ra 32 giao dịch bộ nhớ riêng biệt.
3. **Vị trí trên Roofline Model**: 
   * Nằm sâu trong vùng **Memory-Bound** (Giới hạn bởi Băng thông bộ nhớ). SM phần lớn thời gian rảnh rỗi chỉ để đợi dữ liệu được nạp từ VRAM lên.

### V2 Tiled — cải thiện ở đâu?

1. **Tái sử dụng dữ liệu qua Shared Memory**:
   * Bằng cách chia ma trận thành các mảnh con kích thước $\text{TILE} \times \text{TILE}$ ($16 \times 16$), các thread trong một block tải chung dữ liệu vào Shared Memory một lần rồi tính toán lại $\text{TILE}$ lần.
   * Số lần đọc từ Global Memory giảm đi $\text{TILE}$ lần (ở đây giảm 16 lần), từ $2 \times N^3$ xuống còn $\frac{2 \times N^3}{\text{TILE}}$.
2. **Tăng Cường độ tính toán (Arithmetic Intensity - AI)**:
   * AI của thuật toán tăng từ $O(1)$ (Naive) lên $O(\text{TILE})$ (Tiled).
   * Phép tính chuyển dịch dịch gần hơn về phía vùng **Compute-Bound** trên mô hình Roofline.

---

### Khoảng cách Tiled → cuBLAS

Mặc dù Tiled cải thiện đáng kể hiệu năng nhưng vẫn chỉ đạt khoảng **13%** sức mạnh của thư viện NVIDIA cuBLAS vì cuBLAS áp dụng các tối ưu hóa phần cứng chuyên sâu:

1. **Tensor Cores**: 
   * cuBLAS sử dụng các nhân Tensor chuyên dụng trên GPU RTX 4050. Tensor Cores hỗ trợ thực hiện phép nhân ma trận nhỏ (ví dụ $16 \times 16 \times 16$) bằng phần cứng chỉ trong một vài chu kỳ xung nhịp (thông qua chỉ thị lệnh MMA).
2. **Register Tiling (Tối ưu mức thanh ghi)**:
   * Thay vì chỉ lưu trữ trên Shared Memory, cuBLAS nạp tiếp dữ liệu từ Shared Memory vào các **Register (thanh ghi)** của từng Thread để thực hiện phép toán. Thanh ghi có băng thông cao hơn Shared Memory gấp nhiều lần và loại bỏ hoàn toàn nguy cơ **Shared Memory Bank Conflict**.
3. **Double Buffering / Software Pipelining**:
   * Trong khi ALU đang tính toán trên dữ liệu của mảnh $t$, GPU sẽ đồng thời kích hoạt việc tải dữ liệu của mảnh $t+1$ từ Global Memory vào Shared Memory để che giấu hoàn toàn độ trễ bộ nhớ (Latency Hiding).
4. **Vectorized Memory Access (Đọc bộ nhớ dạng Vector)**:
   * Sử dụng lệnh đọc bộ nhớ rộng như `LDG.128` để tải 4 phần tử float (128-bit) cùng một lúc trong một chỉ thị lệnh đơn lẻ.

---

## Liên hệ HW-SW

* **Coalesced Access**: Lập trình viên/Compiler cần tổ chức cấu trúc bộ nhớ sao cho các thread trong cùng warp đọc các ô nhớ kề cạnh nhau (stride-1) để tối ưu băng thông phần cứng.
* **Bank Conflicts**: Bộ nhớ Shared Memory được chia thành 32 bank độc lập. Nếu nhiều thread trong một warp cùng truy cập vào các hàng khác nhau nhưng thuộc cùng một bank, GPU sẽ bị tuần tự hóa truy cập (serialize). Compiler/Lập trình viên cần dùng kỹ thuật padding (đệm thêm cột trống) để tránh hiện tượng này.
* **Tile Size Selection**: Kích thước tile tối ưu phụ thuộc trực tiếp vào dung lượng Shared Memory và số lượng thanh ghi trên mỗi SM của từng kiến trúc phần cứng cụ thể (Ampere - SM 86, Ada Lovelace - SM 89, Hopper - SM 90). Do đó, compiler tối ưu hóa mã nguồn cần có kiến thức sâu rộng về kiến trúc vật lý để đưa ra quyết định tối ưu nhất.

