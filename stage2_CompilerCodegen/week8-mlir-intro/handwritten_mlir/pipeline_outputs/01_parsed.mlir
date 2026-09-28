module {
  func.func @add_integers(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
  func.func @diff_of_squares(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.subi %arg0, %arg1 : i32
    %2 = arith.muli %0, %1 : i32
    return %2 : i32
  }
  func.func @float_ops(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    %1 = arith.mulf %arg0, %arg1 : f32
    %2 = arith.addf %0, %1 : f32
    return %2 : f32
  }
  func.func @with_constants() -> i32 {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = arith.muli %c4_i32, %c2_i32 : i32
    %1 = arith.addi %c3_i32, %0 : i32
    return %1 : i32
  }
  func.func @type_casts(%arg0: i32) -> i64 {
    %0 = arith.extsi %arg0 : i32 to i64
    return %0 : i64
  }
  func.func @max_of_two(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.cmpi sgt, %arg0, %arg1 : i32
    %1 = arith.select %0, %arg0, %arg1 : i32
    return %1 : i32
  }
}

