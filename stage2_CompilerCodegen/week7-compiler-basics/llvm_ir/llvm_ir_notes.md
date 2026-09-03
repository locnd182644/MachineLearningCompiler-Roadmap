# Phân tích LLVM IR: Từ cơ bản đến Tối ưu hóa

## 1. Giới thiệu

LLVM Intermediate Representation (LLVM IR) là ngôn ngữ trung gian của kiến trúc LLVM, đóng vai trò như cầu nối giữa frontend (như Clang biên dịch C/C++) và backend (như x86, ARM codegen).

Đặc điểm của LLVM IR:
- **Static Single Assignment (SSA)**: Mỗi biến chỉ được gán (assigned) một lần. Các phép gán mới sẽ tạo ra biến mới (vd: `%1`, `%2`).
- **Infinite Register File**: Không có giới hạn số lượng thanh ghi ảo (virtual registers).
- **Strongly Typed**: Các kiểu dữ liệu được định nghĩa rõ ràng (`i32`, `i64`, `ptr`, ...).
- 3 dạng hiển thị: Trong bộ nhớ (In-memory), bitcode lưu file (`.bc`), và file văn bản người đọc được (`.ll`).

## 2. Cách tạo file .ll

Sử dụng `clang` để emit LLVM IR.
- Để biên dịch không có tối ưu hóa (-O0):
  ```bash
  clang -S -emit-llvm -O0 file.c -o file_O0.ll
  ```
- Để biên dịch có tối ưu hóa (-O2):
  ```bash
  clang -S -emit-llvm -O2 file.c -o file_O2.ll
  ```

---

## 3. Phân tích `add.c`

File `add.c` gồm 2 hàm: `add(int a, int b)` và `add_with_const(int x)`.

### Ở mức -O0 (Không tối ưu):
```llvm
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  ...
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %6 = load i32, ptr %3, align 4
  %7 = load i32, ptr %4, align 4
  %8 = add nsw i32 %6, %7
  ...
```
- Sử dụng cực kỳ nhiều `alloca` (cấp phát trên stack), `load`, và `store`. Mỗi phép tính đều lưu ra bộ nhớ rồi tải lại.
- Pass mem2reg (promote memory to register) chưa được chạy.

### Ở mức -O2 (Có tối ưu):
```llvm
define dso_local i32 @add(i32 noundef %0, i32 noundef %1) {
  %3 = add nsw i32 %1, %0
  ret i32 %3
}

define dso_local i32 @add_with_const(i32 noundef %0) {
  %2 = add nsw i32 %0, 30
  ret i32 %2
}
```
- **Mem2Reg/SROA**: Không còn bất kỳ instruction `alloca`, `load`, `store` nào. Biến được giữ hoàn toàn trên thanh ghi ảo.
- **Constant Folding**: Phép tính `a + b` (10 + 20) trong hàm `add_with_const` đã được tính toán sẵn từ lúc compile thành `30`.

---

## 4. Phân tích `loop_sum.c`

File `loop_sum.c` minh họa vòng lặp.

### Ở mức -O0:
LLVM IR sử dụng một cấu trúc khối lặp (`br label %7`), liên tục nhảy đi nhảy lại:
```llvm
7:                                                ; preds = %22, %2
  %8 = load i32, ptr %6, align 4
  %9 = load i32, ptr %4, align 4
  %10 = icmp slt i32 %8, %9
  br i1 %10, label %11, label %25
...
```
- Liên tục `load` biến đếm `i` và `sum` từ memory, rồi `store` lại. Rất chậm.

### Ở mức -O2:
```llvm
7:                                                ; preds = %5, %7
  %8 = phi i64 [ 0, %5 ], [ %17, %7 ]
  %9 = phi i32 [ 0, %5 ], [ %16, %7 ]
  %10 = getelementptr inbounds i32, ptr %0, i64 %8
...
```
- **Phi Nodes (`phi`)**: Vì SSA yêu cầu mỗi biến gán 1 lần, biến đếm của vòng lặp phải dùng `phi` node. Nó có ý nghĩa: "nếu nhảy tới từ block `%5` thì giá trị là 0, nếu tới từ `%7` thì giá trị là `%17` (giá trị cũ + 1)".
- Các phép load/store dư thừa đã bị loại bỏ.
- *Lưu ý: Tùy phiên bản LLVM mà O2/O3 có thể sinh ra Vectorization (SIMD) mạnh hơn bằng các lệnh `llvm.vector.reduce.add` kết hợp `shufflevector`.* Trong ví dụ này, loop vẫn được giữ dạng scalar tối ưu.

---

## 5. Phân tích `branchy_max.c`

File `branchy_max.c` chứa các hàm với câu lệnh rẽ nhánh (`if/else`).

### Ở mức -O0:
```llvm
  %8 = icmp sgt i32 %6, %7
  br i1 %8, label %9, label %11

9:
  store ...
  br label %13
11:
  store ...
  br label %13
```
- Đúng với bản chất mã C, compiler sinh ra `br` (branch) có điều kiện, chia làm 2 basic block rồi chập lại ở block %13.

### Ở mức -O2:
```llvm
define dso_local i32 @max(i32 noundef %0, i32 noundef %1) {
  %3 = icmp sgt i32 %0, %1
  %4 = select i1 %3, i32 %0, i32 %1
  ret i32 %4
}
```
- **Branch Elimination (SimplifyCFG)**: Cấu trúc rẽ nhánh đã bị xóa bỏ hoàn toàn! Nó được thay thế bằng toán tử `select` (tương đương với phép `? :`).
- Điều này có lợi cho hiệu suất (tránh Branch Prediction Failure ở mức phần cứng).
- Đặc biệt, với hàm `abs_val`, LLVM O2 còn thông minh hơn khi nhận diện đúng pattern toán học và tự động sinh ra lời gọi hàm tối ưu thay thế `tail call i32 @llvm.abs.i32(i32 %0, i1 true)`.

---

## 6. Tổng kết

| Tính năng Tối ưu | Biểu hiện trong LLVM IR |
| :--- | :--- |
| Mem2Reg (SROA) | Xóa bỏ `alloca`, `load`, `store`. Chuyển sang thanh ghi. |
| Constant Folding | Tính trước các phép cộng/nhân hằng số ngay lúc compile. |
| SimplifyCFG | Chuyển `br` (branch) thành lệnh `select` để tránh rẽ nhánh luồng. |
| Loop Optimization | Sử dụng `phi` node cho các vòng lặp SSA, Loop invariant. |
| Pattern Matching | Đổi `if (x < 0) return -x` thành `@llvm.abs`. |

**Liên hệ với AI Compiler:**
Các compiler hiện đại cho ML (như MLIR, XLA, TVM, Triton) đều kế thừa tư duy IR từ LLVM: 
- Tất cả đều dùng SSA.
- Có nhiều tầng (Dialects trong MLIR) để biểu diễn IR.
- Đều thực hiện các optimization pass tương tự như Constant Folding, Dead Code Elimination, nhưng ở quy mô toán học của Tensor thay vì các giá trị vô hướng (scalar).
