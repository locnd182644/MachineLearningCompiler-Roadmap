module {
  func.func @sum_0_to_n(%arg0: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0, %c0_i32 : index, i32)
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi slt, %0, %arg0 : index
    cf.cond_br %2, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %3 = arith.index_cast %0 : index to i32
    %4 = arith.addi %1, %3 : i32
    %5 = arith.addi %0, %c1 : index
    cf.br ^bb1(%5, %4 : index, i32)
  ^bb3:  // pred: ^bb1
    return %1 : i32
  }
  func.func @nested_loops() -> i32 {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    cf.br ^bb1(%c0, %c0_i32 : index, i32)
  ^bb1(%0: index, %1: i32):  // 2 preds: ^bb0, ^bb5
    %2 = arith.cmpi slt, %0, %c4 : index
    cf.cond_br %2, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    cf.br ^bb3(%c0, %1 : index, i32)
  ^bb3(%3: index, %4: i32):  // 2 preds: ^bb2, ^bb4
    %5 = arith.cmpi slt, %3, %c4 : index
    cf.cond_br %5, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %6 = arith.index_cast %0 : index to i32
    %7 = arith.index_cast %3 : index to i32
    %8 = arith.muli %6, %7 : i32
    %9 = arith.addi %4, %8 : i32
    %10 = arith.addi %3, %c1 : index
    cf.br ^bb3(%10, %9 : index, i32)
  ^bb5:  // pred: ^bb3
    %11 = arith.addi %0, %c1 : index
    cf.br ^bb1(%11, %4 : index, i32)
  ^bb6:  // pred: ^bb1
    return %1 : i32
  }
  func.func @abs_value(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi slt, %arg0, %c0_i32 : i32
    cf.cond_br %0, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %1 = arith.subi %c0_i32, %arg0 : i32
    cf.br ^bb3(%1 : i32)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%arg0 : i32)
  ^bb3(%2: i32):  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    return %2 : i32
  }
  func.func @factorial(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    cf.br ^bb1(%arg0, %c1_i32 : i32, i32)
  ^bb1(%0: i32, %1: i32):  // 2 preds: ^bb0, ^bb2
    %2 = arith.cmpi sgt, %0, %c0_i32 : i32
    cf.cond_br %2, ^bb2(%0, %1 : i32, i32), ^bb3
  ^bb2(%3: i32, %4: i32):  // pred: ^bb1
    %5 = arith.muli %4, %3 : i32
    %6 = arith.subi %3, %c1_i32 : i32
    cf.br ^bb1(%6, %5 : i32, i32)
  ^bb3:  // pred: ^bb1
    return %0 : i32
  }
}

