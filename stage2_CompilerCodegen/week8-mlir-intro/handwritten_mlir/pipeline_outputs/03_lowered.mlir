module {
  llvm.func @free(!llvm.ptr)
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @printMemrefF32(i64, !llvm.ptr) attributes {sym_visibility = "private"}
  llvm.func @dot_product(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64) -> f32 {
    %0 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg5, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg6, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg7, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg8, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg9, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg0, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg1, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg2, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg3, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg4, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.constant(0 : i64) : i64
    %13 = llvm.mlir.constant(4 : i64) : i64
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    llvm.br ^bb1(%12, %15 : i64, f32)
  ^bb1(%16: i64, %17: f32):  // 2 preds: ^bb0, ^bb2
    %18 = llvm.icmp "slt" %16, %13 : i64
    llvm.cond_br %18, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %19 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.getelementptr inbounds|nuw %19[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %21 = llvm.load %20 : !llvm.ptr -> f32
    %22 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.getelementptr inbounds|nuw %22[%16] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %24 = llvm.load %23 : !llvm.ptr -> f32
    %25 = llvm.fmul %21, %24 : f32
    %26 = llvm.fadd %17, %25 : f32
    %27 = llvm.add %16, %14 : i64
    llvm.br ^bb1(%27, %26 : i64, f32)
  ^bb3:  // pred: ^bb1
    llvm.return %17 : f32
  }
  llvm.func @main() {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.zero : !llvm.ptr
    %3 = llvm.getelementptr %2[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %4 = llvm.ptrtoint %3 : !llvm.ptr to i64
    %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr
    %6 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %5, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.mlir.constant(0 : i64) : i64
    %10 = llvm.insertvalue %9, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %0, %10[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.insertvalue %1, %11[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %13 = llvm.mlir.constant(4 : i64) : i64
    %14 = llvm.mlir.constant(1 : i64) : i64
    %15 = llvm.mlir.zero : !llvm.ptr
    %16 = llvm.getelementptr %15[%13] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %17 = llvm.ptrtoint %16 : !llvm.ptr to i64
    %18 = llvm.call @malloc(%17) : (i64) -> !llvm.ptr
    %19 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %18, %20[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.mlir.constant(0 : i64) : i64
    %23 = llvm.insertvalue %22, %21[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.insertvalue %13, %23[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %25 = llvm.insertvalue %14, %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.mlir.constant(0 : i64) : i64
    %27 = llvm.mlir.constant(1 : i64) : i64
    %28 = llvm.mlir.constant(2 : i64) : i64
    %29 = llvm.mlir.constant(3 : i64) : i64
    %30 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %31 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %32 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %33 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %34 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.getelementptr inbounds|nuw %34[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %30, %35 : f32, !llvm.ptr
    %36 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %37 = llvm.getelementptr inbounds|nuw %36[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %31, %37 : f32, !llvm.ptr
    %38 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %39 = llvm.getelementptr inbounds|nuw %38[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %32, %39 : f32, !llvm.ptr
    %40 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %41 = llvm.getelementptr inbounds|nuw %40[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %33, %41 : f32, !llvm.ptr
    %42 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %43 = llvm.getelementptr inbounds|nuw %42[%26] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %30, %43 : f32, !llvm.ptr
    %44 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %45 = llvm.getelementptr inbounds|nuw %44[%27] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %31, %45 : f32, !llvm.ptr
    %46 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %47 = llvm.getelementptr inbounds|nuw %46[%28] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %32, %47 : f32, !llvm.ptr
    %48 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %49 = llvm.getelementptr inbounds|nuw %48[%29] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %33, %49 : f32, !llvm.ptr
    %50 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %51 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %52 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %53 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %54 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.extractvalue %25[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.extractvalue %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.extractvalue %25[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.extractvalue %25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.extractvalue %25[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %60 = llvm.call @dot_product(%50, %51, %52, %53, %54, %55, %56, %57, %58, %59) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64) -> f32
    %61 = llvm.mlir.constant(1 : i64) : i64
    %62 = llvm.mlir.zero : !llvm.ptr
    %63 = llvm.getelementptr %62[%61] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.call @malloc(%64) : (i64) -> !llvm.ptr
    %66 = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
    %67 = llvm.insertvalue %65, %66[0] : !llvm.struct<(ptr, ptr, i64)> 
    %68 = llvm.insertvalue %65, %67[1] : !llvm.struct<(ptr, ptr, i64)> 
    %69 = llvm.mlir.constant(0 : i64) : i64
    %70 = llvm.insertvalue %69, %68[2] : !llvm.struct<(ptr, ptr, i64)> 
    %71 = llvm.extractvalue %70[1] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.store %60, %71 : f32, !llvm.ptr
    %72 = llvm.mlir.constant(1 : i64) : i64
    %73 = llvm.alloca %72 x !llvm.struct<(ptr, ptr, i64)> : (i64) -> !llvm.ptr
    llvm.store %70, %73 : !llvm.struct<(ptr, ptr, i64)>, !llvm.ptr
    %74 = llvm.mlir.constant(0 : i64) : i64
    %75 = llvm.mlir.poison : !llvm.struct<(i64, ptr)>
    %76 = llvm.insertvalue %74, %75[0] : !llvm.struct<(i64, ptr)> 
    %77 = llvm.insertvalue %73, %76[1] : !llvm.struct<(i64, ptr)> 
    %78 = llvm.extractvalue %77[0] : !llvm.struct<(i64, ptr)> 
    %79 = llvm.extractvalue %77[1] : !llvm.struct<(i64, ptr)> 
    llvm.call @printMemrefF32(%78, %79) : (i64, !llvm.ptr) -> ()
    %80 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%80) : (!llvm.ptr) -> ()
    %81 = llvm.extractvalue %25[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    llvm.call @free(%81) : (!llvm.ptr) -> ()
    %82 = llvm.extractvalue %70[0] : !llvm.struct<(ptr, ptr, i64)> 
    llvm.call @free(%82) : (!llvm.ptr) -> ()
    llvm.return
  }
}

