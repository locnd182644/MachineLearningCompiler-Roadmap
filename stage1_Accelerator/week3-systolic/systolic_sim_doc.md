# Analyzing Systolic Array Code

File systolic_sim.py là một chương trình giả lập bằng Python mô phỏng cơ chế hoạt động của Systolic Array (mạng lưới
tính toán song song dạng "dòng chảy") theo kiến trúc Output-Stationary (kết quả ma trận C đứng yên tại chỗ).       
                                                                                                                    
Mảng Systolic là trái tim của các bộ tăng tốc AI hiện đại như Google TPU, được tối ưu hóa đặc biệt cho phép nhân ma
trận.                                                                                                              
                                                                                                                    
Dưới đây là giải thích chi tiết về cấu trúc và nguyên lý hoạt động của mã nguồn này:                               
──────                                                                                                             
### 1. Processing Element (Lớp systolic_sim.py)                                                                     
                                                                                                                    
Lớp  PE  đại diện cho một ô tính toán phần cứng độc lập trong lưới 2D.                                             
                                                                                                                    
• Cơ chế Output-Stationary: Thanh ghi tích lũy  accumulator  đóng vai trò lưu trữ kết quả tạm thời của một phần tử 
ma trận C và nó nằm cố định tại PE trong suốt quá trình tính toán.                                                 
• Luồng dữ liệu:                                                                                                   
    • Dữ liệu từ ma trận A được bơm từ bên trái qua ( a_in ).                                                      
    • Dữ liệu từ ma trận B được bơm từ bên trên xuống ( b_in ).                                                    
• Hàm  cycle :                                                                                                     
    • Thực hiện phép tính MAC (Multiply-Accumulate):  accumulator += a_in * b_in .                                 
    • Đẩy giá trị hiện tại sang biến đệm đầu ra ( a_out ,  b_out ) để truyền tiếp sang PE láng giềng (PE bên phải  
    và PE bên dưới) ở chu kỳ xung nhịp tiếp theo.

──────
### 2. Mảng Systolic (Lớp systolic_sim.py)

Lớp này quản lý một lưới 2D gồm các PE (kích thước  rows * cols ).

• #### Hàm systolic_sim.py (Mô phỏng 1 chu kỳ xung nhịp):

    • Phase 1 (Lan truyền): Cập nhật các cổng vào  a_in  và  b_in  của tất cả PE.
        • Các PE nằm ở cạnh biên (cột $0$ hoặc hàng $0$) nhận dữ liệu từ dòng đầu vào bên ngoài.
        • Các PE nằm bên trong nhận lại giá trị đầu ra  a_out / b_out  từ PE phía trước ở chu kỳ trước.            
    • Phase 2 (Tính toán đồng thời): Kích hoạt tất cả PE thực hiện tính toán song song ( pe.cycle() ). Đồng thời   
    đếm số lượng PE thực sự làm việc trong chu kỳ đó ( active ) để thống kê hiệu suất phần cứng.

──────
### 3. Thuật toán nhân ma trận và Cơ chế trễ thời gian (Skewing)

Để phép nhân ma trận $A \times B$ diễn ra chính xác, các phần tử tương ứng của A và B phải gặp nhau đúng thời điểm 
tại đúng PE. Điều này đòi hỏi dữ liệu đầu vào phải được bơm lệch pha (skewed) theo thời gian (systolic_sim.py):     

• Hàng $i$ của ma trận A sẽ bị trễ $i$ chu kỳ so với hàng phía trên nó.
• Cột $j$ của ma trận B sẽ bị trễ $j$ chu kỳ so với cột bên trái nó.

    Ví dụ với mảng 4x4 (K=8):
    Chu kỳ t:
    Hàng A0: [ A00, A01, A02, A03, A04, A05, A06, A07 ]
    Hàng A1: [ None, A10, A11, A12, A13, A14, A15, A16, A17 ] (trễ 1 chu kỳ)
    Hàng A2: [ None, None, A20, A21, ...                   ] (trễ 2 chu kỳ)

• Tổng số chu kỳ tính toán được xác định bằng công thức:
$$\text{Total Cycles} = K + \text{rows} + \text{cols} - 2$$
Với ví dụ trong hàm  main  ($K=8$, mảng $4 \times 4$), tổng số chu kỳ cần thiết là $8 + 4 + 4 - 2 = 14$ chu kỳ.    
──────
### 4. Đánh giá hiệu suất sử dụng phần cứng (systolic_sim.py)

Hiệu suất sử dụng lưới PE (Utilization) được tính bằng:
$\text{Utilization} = \frac{\text{Tổng số phép tính MAC thực tế}}{\text{Tổng chu kỳ} \times \text{Tổng số PE}}$

Do quá trình nạp lệch pha (skewed) nên ở giai đoạn đầu (Ramp-up) và giai đoạn cuối (Ramp-down), lưới PE chưa được  
lấp
đầy dữ liệu hoàn toàn nên hiệu suất sử dụng thực tế sẽ thấp hơn 100%. Khi ma trận kích thước cực lớn, hiệu suất này
sẽ tiệm cận 100%.
──────
### 5. Khối kiểm tra ( __main__ )

Chương trình tạo một mảng systolic $4 \times 4$, nhân hai ma trận ngẫu nhiên kích thước $(4 \times 8)$ và $(8      
\times 4)$, so sánh kết quả mô phỏng ( C_sim ) với kết quả chuẩn của NumPy ( C_ref ) để đảm bảo sai số bằng 0      
(chính xác hoàn toàn).