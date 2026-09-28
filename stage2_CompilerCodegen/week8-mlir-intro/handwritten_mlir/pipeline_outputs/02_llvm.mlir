module {
  llvm.func @sum_0_to_n(%arg0: i64) -> i32 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1(%0, %2 : i64, i32)
  ^bb1(%3: i64, %4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "slt" %3, %arg0 : i64
    llvm.cond_br %5, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %6 = llvm.trunc %3 : i64 to i32
    %7 = llvm.add %4, %6 : i32
    %8 = llvm.add %3, %1 : i64
    llvm.br ^bb1(%8, %7 : i64, i32)
  ^bb3:  // pred: ^bb1
    llvm.return %4 : i32
  }
  llvm.func @nested_loops() -> i32 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.mlir.constant(1 : i64) : i64
    %3 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb1(%0, %3 : i64, i32)
  ^bb1(%4: i64, %5: i32):  // 2 preds: ^bb0, ^bb5
    %6 = llvm.icmp "slt" %4, %1 : i64
    llvm.cond_br %6, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.br ^bb3(%0, %5 : i64, i32)
  ^bb3(%7: i64, %8: i32):  // 2 preds: ^bb2, ^bb4
    %9 = llvm.icmp "slt" %7, %1 : i64
    llvm.cond_br %9, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %10 = llvm.trunc %4 : i64 to i32
    %11 = llvm.trunc %7 : i64 to i32
    %12 = llvm.mul %10, %11 : i32
    %13 = llvm.add %8, %12 : i32
    %14 = llvm.add %7, %2 : i64
    llvm.br ^bb3(%14, %13 : i64, i32)
  ^bb5:  // pred: ^bb3
    %15 = llvm.add %4, %2 : i64
    llvm.br ^bb1(%15, %8 : i64, i32)
  ^bb6:  // pred: ^bb1
    llvm.return %5 : i32
  }
  llvm.func @abs_value(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.icmp "slt" %arg0, %0 : i32
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %2 = llvm.sub %0, %arg0 : i32
    llvm.br ^bb3(%2 : i32)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%arg0 : i32)
  ^bb3(%3: i32):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %3 : i32
  }
  llvm.func @factorial(%arg0: i32) -> i32 {
    %0 = llvm.mlir.constant(0 : i32) : i32
    %1 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb1(%arg0, %1 : i32, i32)
  ^bb1(%2: i32, %3: i32):  // 2 preds: ^bb0, ^bb2
    %4 = llvm.icmp "sgt" %2, %0 : i32
    llvm.cond_br %4, ^bb2(%2, %3 : i32, i32), ^bb3
  ^bb2(%5: i32, %6: i32):  // pred: ^bb1
    %7 = llvm.mul %6, %5 : i32
    %8 = llvm.sub %5, %1 : i32
    llvm.br ^bb1(%8, %7 : i32, i32)
  ^bb3:  // pred: ^bb1
    llvm.return %2 : i32
  }
}

