// =============================================================================
// Bài tập 8.2a — Arithmetic cơ bản trong MLIR
// 
// File này demonstrate:
// - func.func: khai báo hàm
// - arith dialect: các phép toán số học
// - SSA values: mỗi giá trị gán đúng 1 lần
// - Type system: i32, i64, f32, f64, index
//
// Chạy: mlir-opt 01_arith_basic.mlir
// Verify: mlir-opt 01_arith_basic.mlir --verify-diagnostics
// =============================================================================

// --- Hàm 1: Cộng hai số nguyên ---
// Mọi hàm trong MLIR là 1 op `func.func`
// @add_integers là symbol name (giống tên hàm trong C)
// (%a: i32, %b: i32) là block arguments (giống parameters)
// -> i32 là return type
func.func @add_integers(%a: i32, %b: i32) -> i32 {
  // arith.addi = add integers
  // %result là SSA value mới — chỉ được gán ở đây, dùng ở nhiều nơi
  %result = arith.addi %a, %b : i32
  return %result : i32
}

// --- Hàm 2: Biểu thức phức tạp hơn ---
// Tính: (a + b) * (a - b) = a² - b²
// Minh họa chuỗi ops, mỗi op tạo 1 SSA value mới
func.func @diff_of_squares(%a: i32, %b: i32) -> i32 {
  %sum = arith.addi %a, %b : i32       // a + b
  %diff = arith.subi %a, %b : i32      // a - b  
  %result = arith.muli %sum, %diff : i32 // (a+b) * (a-b)
  return %result : i32
}

// --- Hàm 3: Floating point operations ---
// arith dùng suffix khác cho float: addf, mulf, divf (không phải addi, muli)
func.func @float_ops(%x: f32, %y: f32) -> f32 {
  %sum = arith.addf %x, %y : f32
  %prod = arith.mulf %x, %y : f32
  %result = arith.addf %sum, %prod : f32  // x + y + x*y
  return %result : f32
}

// --- Hàm 4: Constants ---
// arith.constant tạo giá trị hằng — tương tự constant folding tuần 7
func.func @with_constants() -> i32 {
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c2 = arith.constant 2 : i32
  %mul = arith.muli %c4, %c2 : i32     // 4 * 2 = 8
  %add = arith.addi %c3, %mul : i32    // 3 + 8 = 11
  return %add : i32
  // Chạy --canonicalize sẽ fold hết thành: return 11
}

// --- Hàm 5: Type casting ---
// Chuyển đổi giữa các types
func.func @type_casts(%i: i32) -> i64 {
  // Sign-extend i32 → i64
  %ext = arith.extsi %i : i32 to i64
  return %ext : i64
}

// --- Hàm 6: Comparison ---
func.func @max_of_two(%a: i32, %b: i32) -> i32 {
  // arith.cmpi: compare integers, trả về i1 (boolean)
  // "sgt" = signed greater than
  %cond = arith.cmpi sgt, %a, %b : i32
  // arith.select: giống ternary operator (cond ? a : b)
  %result = arith.select %cond, %a, %b : i32
  return %result : i32
}
