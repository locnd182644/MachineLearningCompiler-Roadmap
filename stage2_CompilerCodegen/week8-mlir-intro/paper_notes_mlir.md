# Notes: MLIR Paper (Lattner et al. 2021, CGO)

## 1. Thông tin paper
- **Tiêu đề:** MLIR: Scaling Compiler Infrastructure for Domain Specific Computation
- **Tác giả:** Chris Lattner, Mehdi Amini, Uday Bondhugula, Albert Cohen, Andy Davis, Jacques Pienaar, River Riddle, Tatiana Shpeisman, Nicolas Vasilache, Oleksandr Zinenko
- **Nơi công bố:** CGO (Code Generation and Optimization) 2021
- **Link:** [https://arxiv.org/abs/2002.11054](https://arxiv.org/abs/2002.11054)

## 2. Bối cảnh & Vấn đề (Problem Statement)

Bài báo trình bày một vấn đề nhức nhối trong hệ sinh thái trình biên dịch Machine Learning tại thời điểm đó, được gọi là **"The 'N compilers problem" (Vấn đề N trình biên dịch)**.

- **Sự bùng nổ của các ML Framework:** Mỗi framework hoặc phần cứng tự xây dựng một IR (Intermediate Representation) riêng của mình: TensorFlow Graph, XLA HLO, TorchScript, Glow, nGraph, ONNX... 
- **Hệ quả - Trùng lặp cơ sở hạ tầng:** Do mỗi IR được thiết kế độc lập, chúng phải tự xây dựng lại toàn bộ các hạ tầng cơ bản của một trình biên dịch: Parser (phân tích cú pháp), Printer (in ra string), Verifier (kiểm tra tính hợp lệ), Pass manager (quản lý các bước tối ưu hóa), Location tracking (theo dõi dòng code nguồn để debug), Diagnostics (báo lỗi/cảnh báo). Điều này dẫn đến sự lãng phí tài nguyên phát triển khổng lồ.
- **Pass tối ưu hóa bị cô lập:** Mỗi IR có các tập lệnh (ops) và pass tối ưu hóa riêng. Rất khó hoặc không thể tái sử dụng một pass tối ưu (ví dụ: constant folding, dead code elimination) giữa các hệ thống với nhau.
- **Vấn đề Lowering (Hạ tầng IR):** Việc chuyển đổi (lowering) giữa các tầng IR (từ Graph của framework xuống IR của optimizer, rồi xuống LLVM) thường là một quá trình "full rewrite" (viết lại toàn bộ). Quá trình này đắt đỏ, dễ sinh lỗi và đặc biệt là làm **mất thông tin**.
- **Hạn chế của LLVM IR:** LLVM IR là một tiêu chuẩn vàng cho compiler, nhưng nó quá "low-level". LLVM IR chỉ hiểu các kiểu dữ liệu cơ bản như scalar, vector, và pointer. Nó không có khái niệm về "tensor" hay các phép toán có cấu trúc (structured operations như vòng lặp đa chiều). Do đó, khi hạ một phép toán ML (ví dụ: Conv2D) xuống LLVM IR quá sớm, mọi thông tin về cấu trúc cấp cao (shape, layout, tính song song) đều bị mất, làm cho các tối ưu hóa ở mức cao trở nên bất khả thi.

## 3. Giải pháp: Meta-IR (MLIR)

Để giải quyết vấn đề trên, nhóm tác giả đề xuất **MLIR (Multi-Level Intermediate Representation)**.

- **Ý tưởng cốt lõi:** MLIR **KHÔNG** phải là một IR mới để cạnh tranh với các IR cũ. Thay vào đó, nó là một **FRAMEWORK để xây dựng các IR**. MLIR cung cấp một bộ công cụ "Meta-IR" (IR của các IR).
- **Dialect System (Hệ thống Phương ngữ):** Khái niệm quan trọng nhất của MLIR. Mỗi "dialect" đóng vai trò như một namespace chứa các Operations (lệnh), Types (kiểu dữ liệu) và Attributes (thuộc tính) nằm ở cùng một mức trừu tượng.
- **Cùng tồn tại (Co-existence):** Điều kỳ diệu của MLIR là nhiều dialect khác nhau có thể cùng tồn tại và tương tác bên trong cùng một Module, thậm chí cùng một function. Điều này cho phép **Progressive Lowering** (Hạ tầng dần dần).
- **Dùng chung hạ tầng:** Bằng cách sử dụng MLIR, người dùng có sẵn mọi thứ: Pass manager, verifier, printer/parser, location tracking, diagnostics, và framework viết lại pattern (pattern rewrite framework). Các nhà phát triển chỉ cần tập trung vào logic cốt lõi của compiler mà họ muốn xây dựng.

## 4. Các khái niệm cốt lõi

### 4.1 Operations (Lệnh/Thao tác)
Trong MLIR, mọi thứ đều là Operation (Op). Op là đơn vị cơ bản và có tính vạn năng (universal). 
- Một phép cộng (`arith.addi`) là op.
- Một vòng lặp (`scf.for`) là op.
- Một hàm (`func.func`) là op.
- Thậm chí toàn bộ module (`builtin.module`) cũng là một op chứa các op khác.

**Anatomy (Cấu trúc) của một Operation:**
- **Name:** Tên của op, thường có tiền tố là tên dialect (ví dụ: `linalg.matmul`).
- **Operands:** Các giá trị đầu vào (chuẩn SSA - Static Single Assignment).
- **Results:** Các giá trị đầu ra (chuẩn SSA).
- **Attributes:** Các hằng số tại thời điểm biên dịch (compile-time constants).
- **Regions:** Một op có thể chứa các regions (vùng), cho phép định nghĩa các cấu trúc lồng nhau (nesting) như thân vòng lặp hoặc thân hàm.
- **Successors:** Dùng cho control flow (rẽ nhánh tới các block khác).

MLIR có cả **Generic Format** (in ra mọi thông tin của op, rất rườm rà) và **Custom Format** (ngắn gọn, do người viết dialect định nghĩa).

### 4.2 Regions & Blocks
- **Region:** Là một danh sách chứa các **Blocks**. Region có thể là SSACFG (Control Flow Graph tiêu chuẩn) hoặc Graph region (nơi các op không cần tuân theo thứ tự thực thi tuần tự, phù hợp cho biểu diễn đồ thị như TensorFlow).
- **Block:** Là một danh sách các **Operations** thực thi tuần tự.
- **Block Arguments thay cho Phi Nodes:** Khác với LLVM IR dùng tập lệnh `phi` ở đầu block để giải quyết control flow, MLIR dùng **Block Arguments**. 
  - *Tại sao thanh lịch hơn?* Lệnh `phi` cần tham chiếu trực tiếp đến các block đi vào (predecessor blocks). Block Arguments đẩy trách nhiệm truyền giá trị cho các lệnh nhảy (như `cf.br` - branch). Điều này khiến block trở nên độc lập hơn, giống như một hàm nhận tham số đầu vào.
- **Nesting (Lồng ghép):** Khái niệm lồng ghép sâu là điểm mạnh: `Op` chứa `Region` chứa `Block` chứa `Op`... Điều này cho phép MLIR dễ dàng biểu diễn các cấu trúc có tính chất phân cấp như vòng lặp, if/else, hay song song.

### 4.3 Type System
- MLIR có một hệ thống kiểu dữ liệu mở rộng (extensible). Mỗi dialect có thể tự định nghĩa kiểu của riêng nó.
- **Built-in types:** Số nguyên (`i1`, `i8`, `i32`...), số thực (`f16`, `f32`, `bf16`...), `index` (kiểu dữ liệu dùng cho chỉ số mảng, phụ thuộc vào architecture), `function` type.
- **Tensor types:** 
  - Ranked (có số chiều rõ ràng): `tensor<4x8xf32>`
  - Unranked (không rõ số chiều): `tensor<*xf32>`
- **Memref types:** Đại diện cho một bộ nhớ đã được cấp phát. Rất quan trọng khi chuyển từ semantics "giá trị" (tensor) sang semantics "bộ nhớ".
  - Ví dụ: `memref<4x8xf32, affine_map<(d0, d1) -> (d0, d1)>, 1>` (có layout map và memory space).
- Ngoài ra còn có Tuple, Complex, Vector...

### 4.4 Attributes
- Là các hằng số gắn với Operation, được biết tại thời điểm biên dịch.
- **DenseElementsAttr:** Dùng để lưu trữ hằng số tensor (ví dụ một ma trận trọng số trong Machine Learning).
- Có thể là Affine maps, String, Integer, hoặc Unit attributes (chỉ để đánh dấu, giống boolean cờ).

### 4.5 Dialects
- **Dialect = Namespace chứa Ops + Types + Attributes.**
- Ví dụ về mức độ trừu tượng khác nhau:
  - `linalg.matmul`: Phép nhân ma trận tổng quát (High-level).
  - `arith.addf`: Phép cộng số thực cơ bản.
  - `llvm.fadd`: Phép cộng số thực chuẩn LLVM (Low-level).
- **Dialect Conversion Framework:** MLIR cung cấp một framework mạnh mẽ để chuyển đổi (convert) các op từ dialect này sang dialect khác (hoặc thậm chí trong cùng dialect).
- Framework dựa trên khái niệm **Legal / Illegal Ops**: Target dialect (đích đến) sẽ định nghĩa xem op nào được phép tồn tại (legal), op nào là illegal và bắt buộc phải có pattern để convert.

## 5. Progressive Lowering — Triết lý trung tâm

Thay vì một bước nhảy vọt (giant leap) từ biểu diễn bậc cao (như TensorFlow Graph) trực tiếp xuống mã máy hoặc LLVM IR, MLIR khuyến khích **Progressive Lowering** (Hạ tầng dần dần) qua nhiều tầng dialects.

- Mỗi tầng biểu diễn (dialect) sẽ thực hiện tối ưu hóa những gì nó "nhìn thấy" rõ nhất và phù hợp nhất.
- **Thông tin KHÔNG bị mất quá sớm.** Các bước tối ưu hóa cấp cao được thực hiện trước khi cấu trúc bị phá vỡ thành các con trỏ và lệnh vô hướng.
- **Ví dụ về đường đi của Matmul:**
  1. Từ Graph framework: `tf.MatMul`
  2. Hạ xuống `linalg.matmul` (Dễ dàng thực hiện tiling, fusion ở mức độ toán học).
  3. Hạ xuống `affine.for` (Tối ưu hóa bộ nhớ, vector hóa dựa trên Polyhedral model).
  4. Hạ xuống `scf.for` (Structured Control Flow - vòng lặp chuẩn).
  5. Hạ xuống `cf.br` (Control Flow với các lệnh rẽ nhánh nhảy block).
  6. Hạ xuống `llvm` dialect (Chuẩn bị dịch thành mã máy).
- Mỗi bước lowering chỉ "commit" (cam kết) một phần quyết định kiến trúc.
- **Liên hệ phần cứng:** Quyết định thời điểm và cách thức lowering phụ thuộc chặt chẽ vào đặc điểm của hardware target. Việc lower càng muộn thì trình biên dịch càng giữ được sự linh hoạt, khi hạ xuống là lúc compiler "cam kết" với một chiến lược hardware cụ thể (ví dụ: kích thước cache, thanh ghi vector).

## 6. Infrastructure tái dùng

Khi dùng MLIR, bạn có ngay một bộ công cụ "khổng lồ" mà không phải tự viết:
- **Pass Manager:** Quản lý pipeline tối ưu. Hỗ trợ chạy đa luồng (multi-threaded) cho các Function pass cục bộ.
- **Pattern Rewrite Framework:** Cung cấp `RewritePattern` và Greedy Pattern Rewriter. Đặc biệt là cơ chế **Canonicalization** (Đưa các ops về dạng chuẩn nhất định để tối ưu).
- **Verifier:** Tự động kiểm tra các ràng buộc (invariants) sau mỗi pass để đảm bảo IR không bị hỏng (ví dụ: kiểu trả về của lệnh cộng phải khớp với toán hạng).
- **Printer/Parser:** Có thể parse từ text MLIR vào bộ nhớ và in ngược lại ra text, cực kỳ hữu dụng cho test driven development (FileCheck).
- **Location Tracking:** Mỗi op đều bắt buộc đính kèm location (`loc(...)`). Khi lower, op sinh ra sẽ kế thừa location của op cha. Điều này giúp debug info được xuyên suốt từ code Python gốc xuống tận mã máy.
- **Diagnostics:** Báo cáo lỗi chính xác tới tận dòng code gốc (tương tự như Clang diagnostic).

## 7. So sánh MLIR vs các phương pháp trước đó

| Tiêu chí | MLIR | LLVM IR | Framework/Domain IR (TF, XLA HLO, TorchScript) |
| :--- | :--- | :--- | :--- |
| **Tính mở rộng (Extensibility)** | Rất cao (Dialects không giới hạn) | Thấp (Tập lệnh cố định) | Cố định (Thường ~100-200 ops tuỳ ý framework) |
| **Mức độ trừu tượng** | Đa mức (Từ cao tới thấp) | Thấp (Scalar, Vector, Pointers) | Chỉ ở mức cao (Tensor-level) |
| **Cơ sở hạ tầng (Infra)** | Framework mạnh mẽ, tái sử dụng tối đa | Rất mạnh, nhưng chỉ cho LLVM | Phải tự xây dựng lại từ đầu |
| **Mục đích chính** | Kết nối nhiều tầng abstractions | Sinh mã tối ưu hóa cuối cùng cho CPU/GPU | Tối ưu hóa đồ thị của một framework cụ thể |

## 8. Kết quả & Ảnh hưởng

- **Sự chấp nhận rộng rãi (Adoption):** MLIR đã trở thành tiêu chuẩn de facto cho các ML compilers. Google chuyển XLA dần sang MLIR, NVIDIA phát triển Triton dựa trên MLIR, AMD có IREE, Modular xây dựng ngôn ngữ Mojo hoàn toàn trên nền MLIR, các startup phần cứng như Tenstorrent, SiFive, AWS Trainium đều dùng MLIR.
- **Cộng đồng:** Đã được sáp nhập thành một LLVM subproject chính thức. Cộng đồng trên LLVM Discourse cực kỳ sôi động.
- **Mở ra một Paradigm mới:** Ngành công nghiệp trình biên dịch chuyển từ việc "Viết một trình biên dịch hoàn chỉnh" sang "Lắp ráp các dialect có sẵn và chỉ viết các dialect/pass riêng biệt cần thiết".

## 9. Câu hỏi suy ngẫm (Discussion/Reflection)

- **Tại sao progressive lowering tốt hơn 1 bước nhảy? (Lấy ví dụ Matmul)**
  Nếu dịch một lệnh Matmul trực tiếp xuống LLVM (3 vòng lặp lồng nhau với pointers), ta sẽ làm mất thông tin toán học. Một công cụ tối ưu hóa nhìn vào LLVM IR sẽ không biết đó là phép Matmul mà chỉ thấy một đống vòng lặp và lệnh đọc/ghi bộ nhớ phức tạp. Việc thực hiện Loop Tiling hoặc Fusion trên LLVM IR tốn rất nhiều thuật toán phân tích sự phụ thuộc (dependency analysis). Nếu dùng MLIR, ở tầng `linalg`, compiler chỉ cần biến một lệnh `linalg.matmul` lớn thành các lệnh nhỏ (Tiling) một cách dễ dàng trước khi biến nó thành vòng lặp.

- **Block arguments vs phi nodes: Tradeoff gì?**
  Block arguments mang tính chức năng (functional) cao, sạch sẽ và dễ phân tích theo dạng hàm. Tuy nhiên, `phi` nodes đôi khi giúp truy ngược luồng dữ liệu (def-use) trực tiếp ngay tại lệnh, trong khi Block arguments yêu cầu phân tích thông qua các predecessor blocks (các nhánh rẽ vào block hiện tại). Dù vậy, phần đông cộng đồng đồng ý Block args dễ thao tác cấu trúc lồng ghép hơn phi nodes.

- **Khi nào thì KHÔNG nên dùng MLIR?**
  MLIR có overhead khá lớn, code C++ template phức tạp và learning curve (đường cong học tập) cực kỳ dốc. Nếu bạn chỉ cần viết một trình biên dịch rất đơn giản cho một ngôn ngữ toy không đòi hỏi nhiều tầng tối ưu, việc build AST trực tiếp sinh LLVM IR có thể sẽ nhanh hơn là học cách viết MLIR dialect.

- **MLIR giúp chip startup thế nào?**
  Trước đây, một chip AI startup (ví dụ NPU mới) phải thuê một team compiler lớn để viết trình biên dịch từ TensorFlow/PyTorch xuống phần cứng. Bây giờ, họ có thể sử dụng các MLIR dialect có sẵn (TOSA, Linalg, Vector), và chỉ cần tập trung viết một Backend Dialect map các ops với tập lệnh phần cứng của họ. Nó tiết kiệm hàng năm trời phát triển và hàng triệu đô la.

## 10. Key Takeaways

1. **Không phải một IR, mà là Compiler Framework:** MLIR là công cụ để tạo và thao tác các IR.
2. **Progressive Lowering:** Từng bước hạ thấp trừu tượng, giữ thông tin (semantic) càng lâu càng tốt để tối ưu hóa tốt hơn.
3. **Dialect ecosystem:** Hệ sinh thái Dialect cho phép kết hợp và chia sẻ các mức trừu tượng khác nhau (Tensor, Loop, Vector, LLVM) trong cùng một hệ thống.
4. **Tái sử dụng Infrastructure:** Cung cấp sẵn Pass manager, Verifier, Pattern Rewrite, Location tracking... tiết kiệm khổng lồ chi phí phát triển.
5. **Op là vạn năng, Region là lồng ghép:** Cấu trúc đệ quy Op chứa Region chứa Block chứa Op... giúp biểu diễn Control Flow có cấu trúc (Structured Control Flow) rất thanh lịch.
6. **Block arguments thay thế Phi nodes:** Cung cấp biểu diễn control-flow sạch hơn, độc lập và dễ debug hơn trong biểu diễn SSA.
7. **Cách mạng hóa ML Compilers:** Đang được sử dụng bởi hầu hết các ông lớn phần cứng và phần mềm để xây dựng thế hệ trình biên dịch Machine Learning tiếp theo.
