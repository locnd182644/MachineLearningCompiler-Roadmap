// =============================================================================
// Bài tập 8.2b — Structured Control Flow (scf dialect)
//
// scf = Structured Control Flow — loop và if CÓ CẤU TRÚC
// Khác cf (Control Flow) — jump/branch KHÔNG cấu trúc
//
// scf dễ phân tích + optimize hơn cf (vì compiler biết đây là loop, không phải
// goto spaghetti). MLIR ưu tiên giữ structured càng lâu càng tốt, chỉ lower
// xuống cf ở bước cuối.
//
// Chạy: mlir-opt 02_scf_loop.mlir
// Lower: mlir-opt 02_scf_loop.mlir --convert-scf-to-cf
// =============================================================================

// --- Hàm 1: scf.for cơ bản — tính tổng 0 + 1 + 2 + ... + (n-1) ---
// scf.for %i = %lb to %ub step %step iter_args(%acc = %init) -> (i32) {
//   ... body ...
//   scf.yield %new_acc : i32    // giá trị acc cho iteration tiếp theo
// }
// iter_args = loop-carried values (thay cho phi nodes trong SSA!)
func.func @sum_0_to_n(%n: index) -> i32 {
  %c0 = arith.constant 0 : index       // lower bound
  %c1 = arith.constant 1 : index       // step
  %init = arith.constant 0 : i32       // accumulator ban đầu

  // scf.for trả về giá trị cuối cùng của iter_args
  %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (i32) {
    // Cast index → i32 để cộng
    %i_i32 = arith.index_cast %i : index to i32
    %new_acc = arith.addi %acc, %i_i32 : i32
    scf.yield %new_acc : i32   // yield = "return cho loop body"
  }

  return %result : i32
}

// --- Hàm 2: scf.for lồng nhau — tính tổng i*j cho i,j = 0..3 ---
func.func @nested_loops() -> i32 {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %init = arith.constant 0 : i32

  %outer = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc_i = %init) -> (i32) {
    %inner = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc_j = %acc_i) -> (i32) {
      %ii = arith.index_cast %i : index to i32
      %jj = arith.index_cast %j : index to i32
      %prod = arith.muli %ii, %jj : i32
      %new = arith.addi %acc_j, %prod : i32
      scf.yield %new : i32
    }
    scf.yield %inner : i32   // kết quả inner loop chảy lên outer
  }

  return %outer : i32
}

// --- Hàm 3: scf.if — conditional ---
func.func @abs_value(%x: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %is_neg = arith.cmpi slt, %x, %c0 : i32  // x < 0?

  // scf.if trả về giá trị (như ternary, nhưng powerful hơn)
  %result = scf.if %is_neg -> (i32) {
    %neg = arith.subi %c0, %x : i32    // 0 - x = -x
    scf.yield %neg : i32
  } else {
    scf.yield %x : i32                 // giữ nguyên
  }

  return %result : i32
}

// --- Hàm 4: scf.while — while loop ---
// Tính factorial(n) = n * (n-1) * ... * 1
func.func @factorial(%n: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  // scf.while: 2 regions — "before" (condition) và "after" (body)
  %result, %_ = scf.while (%arg = %n, %acc = %c1) : (i32, i32) -> (i32, i32) {
    %cond = arith.cmpi sgt, %arg, %c0 : i32  // arg > 0?
    scf.condition(%cond) %arg, %acc : i32, i32
  } do {
  ^bb0(%arg: i32, %acc: i32):
    %new_acc = arith.muli %acc, %arg : i32    // acc *= arg
    %new_arg = arith.subi %arg, %c1 : i32     // arg -= 1
    scf.yield %new_arg, %new_acc : i32, i32
  }

  return %result : i32
}
